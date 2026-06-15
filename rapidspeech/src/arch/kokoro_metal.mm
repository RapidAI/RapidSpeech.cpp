#include "kokoro_metal.h"
#include "utils/rs_log.h"
#include "ggml.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Metal shader
//
// One simdgroup (32 lanes) per output element (t, oc).
// Threadgroup grid: (T_out, OC, 1). Threads/threadgroup: (32, 1, 1).
// Each lane sums a stride-32 slice of the IC dimension; simd_sum reduces.
//
// Weight layout on GPU: (IC, K, OC)  IC fastest, so contiguous reads in the
// inner loop. Re-packed from ggml's (K, OC, IC) at init time.
//
// PyTorch ConvTranspose1d output index `t` corresponds to ggml's unpadded
// output index `j = t + pad`. For that j, input positions i contributing are
//   i in [ceil((j - K + 1)/s), floor(j/s)]   intersected with [0, T_in - 1]
// (same bounds as upstream ggml kernel_conv_transpose_1d after PR #1477).
// ---------------------------------------------------------------------------
static const char *kKokoroMetalShaderSrc = R"MTL(
#include <metal_stdlib>
using namespace metal;

struct kokoro_convt1d_args {
    int K;
    int IC;
    int OC;
    int T_in;
    int T_out;
    int stride;
    int pad;
};

kernel void kokoro_convt1d_simdgroup(
    constant kokoro_convt1d_args & args [[buffer(0)]],
    device const half  *           w    [[buffer(1)]],   // (IC, K, OC) IC fastest
    device const float *           bias [[buffer(2)]],   // (OC)
    device const float *           in   [[buffer(3)]],   // (IC, T_in) IC fastest? no: T_in fastest, then IC slow
    device       float *           out  [[buffer(4)]],   // (OC, T_out) T_out fastest, OC slow
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint  tiisg [[thread_index_in_simdgroup]])
{
    const int t  = (int)tgpig.x;
    const int oc = (int)tgpig.y;
    if (t >= args.T_out || oc >= args.OC) return;

    const int j  = t + args.pad;
    const int K  = args.K;
    const int IC = args.IC;
    const int OC = args.OC;
    const int s  = args.stride;
    const int Ti = args.T_in;

    int a = j - K + 1;
    int i_min = (a <= 0) ? 0 : (a + s - 1) / s;
    int i_max = j / s;
    if (i_max > Ti - 1) i_max = Ti - 1;

    float acc = 0.0f;
    if (i_min <= i_max) {
        // simdgroup-stride over IC
        for (int ic = (int)tiisg; ic < IC; ic += 32) {
            // w[ic + k*IC + oc*K*IC]   (IC fastest)
            device const half *w_oc = w + (size_t)oc * (size_t)K * (size_t)IC + (size_t)ic;
            // in[i + ic*Ti]            (Ti fastest, matches ggml (T, C) layout we feed)
            device const float *in_ic = in + (size_t)ic * (size_t)Ti;
            for (int i = i_min; i <= i_max; i++) {
                int k = j - i * s;
                acc += float(w_oc[(size_t)k * (size_t)IC]) * in_ic[i];
            }
        }
    }

    acc = simd_sum(acc);
    if (tiisg == 0) {
        // out[t + oc*T_out]: T_out fastest, OC slow -> matches ggml (T, C) layout
        out[(size_t)oc * (size_t)args.T_out + (size_t)t] = acc + bias[oc];
    }
}
)MTL";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline id<MTLDevice>               dev(void *p)  { return (__bridge id<MTLDevice>)p; }
static inline id<MTLLibrary>              lib(void *p)  { return (__bridge id<MTLLibrary>)p; }
static inline id<MTLCommandQueue>         que(void *p)  { return (__bridge id<MTLCommandQueue>)p; }
static inline id<MTLComputePipelineState> pip(void *p)  { return (__bridge id<MTLComputePipelineState>)p; }
static inline id<MTLBuffer>               buf(void *p)  { return (__bridge id<MTLBuffer>)p; }

static inline void *retain_id(id obj) {
    return (__bridge_retained void *)obj;
}
static inline void release_id(void *p) {
    if (p) {
        id _unused = (__bridge_transfer id)p;
        (void)_unused;
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
KokoroMetalConvT1D::KokoroMetalConvT1D() {}

KokoroMetalConvT1D::~KokoroMetalConvT1D() {
    for (auto &L : layers_) {
        release_id(L.pipeline);
        release_id(L.w_buf);
        release_id(L.b_buf);
        L.pipeline = L.w_buf = L.b_buf = nullptr;
    }
    release_id(queue_);
    release_id(library_);
    release_id(device_);
    queue_ = library_ = device_ = nullptr;
}

void *KokoroMetalConvT1D::alloc_buffer(size_t bytes, const void *data) {
    id<MTLBuffer> b = [dev(device_) newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) return nullptr;
    if (data) memcpy([b contents], data, bytes);
    return retain_id(b);
}

void *KokoroMetalConvT1D::make_pipeline(const char *name) {
    id<MTLFunction> fn = [lib(library_) newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        RS_LOG_ERR("KokoroMetal: kernel '%s' not found", name);
        return nullptr;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev(device_) newComputePipelineStateWithFunction:fn error:&err];
    if (!p) {
        RS_LOG_ERR("KokoroMetal: pipeline '%s': %s", name,
                   err ? [[err localizedDescription] UTF8String] : "unknown");
        return nullptr;
    }
    return retain_id(p);
}

bool KokoroMetalConvT1D::upload_weights(Layer &L, const ggml_tensor *w, const ggml_tensor *b,
                                        int K, int IC, int OC, int stride, int pad) {
    if (!w || !b) {
        RS_LOG_ERR("KokoroMetal: null weight/bias tensor");
        return false;
    }
    if ((int)w->ne[0] != K || (int)w->ne[1] != OC || (int)w->ne[2] != IC) {
        RS_LOG_ERR("KokoroMetal: weight shape mismatch: expected (%d,%d,%d), got (%lld,%lld,%lld)",
                   K, OC, IC, (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2]);
        return false;
    }
    if ((int)b->ne[0] != OC) {
        RS_LOG_ERR("KokoroMetal: bias length mismatch: expected %d, got %lld",
                   OC, (long long)b->ne[0]);
        return false;
    }

    // Pack weights: ggml (K, OC, IC) → Metal (IC, K, OC).
    // Source index:  k + oc*K + ic*K*OC
    // Dest index:    ic + k*IC + oc*K*IC
    const size_t n = (size_t)K * IC * OC;
    std::vector<uint16_t> packed(n);
    if (w->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)w->data;
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                for (int ic = 0; ic < IC; ic++) {
                    packed[(size_t)ic + (size_t)k * IC + (size_t)oc * K * IC] =
                        src[(size_t)k + (size_t)oc * K + (size_t)ic * K * OC];
                }
            }
        }
    } else if (w->type == GGML_TYPE_F32) {
        const float *src = (const float *)w->data;
        for (int oc = 0; oc < OC; oc++) {
            for (int k = 0; k < K; k++) {
                for (int ic = 0; ic < IC; ic++) {
                    float v = src[(size_t)k + (size_t)oc * K + (size_t)ic * K * OC];
                    packed[(size_t)ic + (size_t)k * IC + (size_t)oc * K * IC] =
                        ggml_fp32_to_fp16(v);
                }
            }
        }
    } else {
        RS_LOG_ERR("KokoroMetal: unsupported weight type %d (expect F16 or F32)", (int)w->type);
        return false;
    }
    L.w_buf = alloc_buffer(packed.size() * sizeof(uint16_t), packed.data());
    if (!L.w_buf) return false;

    // Bias: F32 directly.
    std::vector<float> b32(OC);
    if (b->type == GGML_TYPE_F32) {
        memcpy(b32.data(), b->data, (size_t)OC * sizeof(float));
    } else if (b->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)b->data;
        for (int i = 0; i < OC; i++) b32[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        RS_LOG_ERR("KokoroMetal: unsupported bias type %d", (int)b->type);
        return false;
    }
    L.b_buf = alloc_buffer((size_t)OC * sizeof(float), b32.data());
    if (!L.b_buf) return false;

    L.K = K; L.IC = IC; L.OC = OC; L.stride = stride; L.pad = pad;
    return true;
}

bool KokoroMetalConvT1D::init(const ggml_tensor *ups0_w, const ggml_tensor *ups0_b, int stride_0, int pad_0,
                              const ggml_tensor *ups1_w, const ggml_tensor *ups1_b, int stride_1, int pad_1) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) {
            RS_LOG_WARN("KokoroMetal: no Metal device available");
            return false;
        }
        device_ = retain_id(d);

        NSError *err = nil;
        MTLCompileOptions *opts = [MTLCompileOptions new];
        id<MTLLibrary> l = [d newLibraryWithSource:[NSString stringWithUTF8String:kKokoroMetalShaderSrc]
                                           options:opts error:&err];
        if (!l) {
            RS_LOG_ERR("KokoroMetal: shader compile failed: %s",
                       err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }
        library_ = retain_id(l);

        id<MTLCommandQueue> q = [d newCommandQueue];
        if (!q) {
            RS_LOG_ERR("KokoroMetal: command queue creation failed");
            return false;
        }
        queue_ = retain_id(q);

        // Layer 0: K=20, IC=512, OC=256
        // Layer 1: K=12, IC=256, OC=128
        const int K0 = (int)ups0_w->ne[0], OC0 = (int)ups0_w->ne[1], IC0 = (int)ups0_w->ne[2];
        const int K1 = (int)ups1_w->ne[0], OC1 = (int)ups1_w->ne[1], IC1 = (int)ups1_w->ne[2];
        if (!upload_weights(layers_[0], ups0_w, ups0_b, K0, IC0, OC0, stride_0, pad_0)) return false;
        if (!upload_weights(layers_[1], ups1_w, ups1_b, K1, IC1, OC1, stride_1, pad_1)) return false;

        layers_[0].pipeline = make_pipeline("kokoro_convt1d_simdgroup");
        layers_[1].pipeline = layers_[0].pipeline ? retain_id(pip(layers_[0].pipeline)) : nullptr;
        if (!layers_[0].pipeline || !layers_[1].pipeline) return false;

        valid_ = true;
        RS_LOG_INFO("KokoroMetal: initialized (ups0 K=%d IC=%d OC=%d s=%d p=%d; ups1 K=%d IC=%d OC=%d s=%d p=%d)",
                    K0, IC0, OC0, stride_0, pad_0, K1, IC1, OC1, stride_1, pad_1);
        return true;
    }
}

int KokoroMetalConvT1D::output_T(int layer, int T_in) const {
    if (layer < 0 || layer > 1 || !valid_) return 0;
    const Layer &L = layers_[layer];
    return (T_in - 1) * L.stride + L.K - 2 * L.pad;
}

bool KokoroMetalConvT1D::run(int layer, const float *input, int T_in, float *output) {
    if (!valid_ || layer < 0 || layer > 1) return false;
    @autoreleasepool {
        const Layer &L = layers_[layer];
        const int T_out = (T_in - 1) * L.stride + L.K - 2 * L.pad;
        if (T_out <= 0) {
            RS_LOG_ERR("KokoroMetal: T_out=%d invalid for T_in=%d", T_out, T_in);
            return false;
        }

        const size_t in_bytes  = (size_t)L.IC * T_in  * sizeof(float);
        const size_t out_bytes = (size_t)L.OC * T_out * sizeof(float);

        id<MTLBuffer> in_buf  = [dev(device_) newBufferWithLength:in_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf = [dev(device_) newBufferWithLength:out_bytes
                                                          options:MTLResourceStorageModeShared];
        if (!in_buf || !out_buf) {
            RS_LOG_ERR("KokoroMetal: scratch alloc failed");
            return false;
        }
        memcpy([in_buf contents], input, in_bytes);

        struct {
            int K, IC, OC, T_in, T_out, stride, pad;
        } args = { L.K, L.IC, L.OC, T_in, T_out, L.stride, L.pad };

        id<MTLCommandBuffer> cb = [que(queue_) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pip(L.pipeline)];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:buf(L.w_buf) offset:0 atIndex:1];
        [enc setBuffer:buf(L.b_buf) offset:0 atIndex:2];
        [enc setBuffer:in_buf       offset:0 atIndex:3];
        [enc setBuffer:out_buf      offset:0 atIndex:4];

        MTLSize tgs = MTLSizeMake(32, 1, 1);
        MTLSize grid = MTLSizeMake((NSUInteger)T_out, (NSUInteger)L.OC, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        memcpy(output, [out_buf contents], out_bytes);
        return true;
    }
}
