#include "dac_metal.h"
#include "omnivoice.h"
#include "utils/rs_log.h"
#include "ggml.h"
#include <cstring>
#include <cmath>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// =====================================================================
// Metal Shader Library (T-major layout: element(t,c) at t + c*T)
// =====================================================================
static const char *kShaderSrc = R"MTL(
#include <metal_stdlib>
using namespace metal;

// Snake activation (in-place, per-channel alpha/inv_beta)
// f(x) = x + sin(x*alpha)^2 * inv_b  (not x + x*sin^2*inv_b!)
// x: [T, C] packed T-fast.  alpha/inv_b: [C].
kernel void dac_snake(device float *x       [[buffer(0)]],
                      constant float *alpha [[buffer(1)]],
                      constant float *inv_b [[buffer(2)]],
                      constant uint &C      [[buffer(3)]],
                      constant uint &T      [[buffer(4)]],
                      uint idx              [[thread_position_in_grid]]) {
    uint t = idx % T;
    uint c = idx / T;
    float xv = x[idx];                      // x[t + c*T]
    float tv = xv * alpha[c];
    float s = sin(tv);
    x[idx] = xv + s * s * inv_b[c];
}

// Direct Conv1d — weight: [OC, IC, K] (OC fastest, Metal layout from upload_conv_weight)
// Input/output: T-major [T, C]
kernel void dac_conv1d(device float *out        [[buffer(0)]],
                       device const float *inp  [[buffer(1)]],
                       device const float *w    [[buffer(2)]],
                       device const float *b    [[buffer(3)]],
                       constant uint &T_in      [[buffer(4)]],
                       constant uint &C_in      [[buffer(5)]],
                       constant uint &T_out     [[buffer(6)]],
                       constant uint &C_out     [[buffer(7)]],
                       constant uint &K         [[buffer(8)]],
                       constant uint &pad       [[buffer(9)]],
                       constant uint &dilation  [[buffer(10)]],
                       constant uint &stride    [[buffer(11)]],
                       uint2 gid               [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T_out || oc >= C_out) return;
    float sum = b[oc];
    for (uint k = 0; k < K; k++) {
        int t_in = (int)t * (int)stride - (int)pad + (int)k * (int)dilation;
        if (t_in < 0 || t_in >= (int)T_in) continue;
        uint w_base = oc + k * (C_out * C_in);
        for (uint ic = 0; ic < C_in; ic++)
            sum += inp[t_in + ic * T_in] * w[w_base + ic * C_out];
    }
    out[t + oc * T_out] = sum;
}

// Direct ConvTranspose1d — weight: [IC, OC, K] (IC fastest, Metal layout from upload_conv_t_weight)
// Input/output: T-major [T, C]
kernel void dac_conv_transpose_1d(
    device float *out       [[buffer(0)]],
    device const float *inp [[buffer(1)]],
    device const float *w   [[buffer(2)]],
    device const float *b   [[buffer(3)]],
    constant uint &T_in     [[buffer(4)]],
    constant uint &C_in     [[buffer(5)]],
    constant uint &T_out    [[buffer(6)]],
    constant uint &C_out    [[buffer(7)]],
    constant uint &K        [[buffer(8)]],
    constant uint &stride   [[buffer(9)]],
    constant uint &pad      [[buffer(10)]],
    uint2 gid              [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T_out || oc >= C_out) return;
    float sum = b[oc];
    for (uint k = 0; k < K; k++) {
        int t_in_raw = (int)t - (int)k + (int)pad;
        if (t_in_raw < 0 || t_in_raw % (int)stride != 0) continue;
        int t_in = t_in_raw / (int)stride;
        if (t_in < 0 || t_in >= (int)T_in) continue;
        uint w_base = k * (C_in * C_out);
        for (uint ic = 0; ic < C_in; ic++)
            sum += inp[t_in + ic * T_in] * w[ic + oc * C_in + w_base];
    }
    out[t + oc * T_out] = sum;
}

// Fused res_unit part1: snake1(C_in) → conv1d(C_in→C_out, K1, dilated) → snake2(C_out)
// w1: [OC, IC, K1] (OC fastest, Metal layout)
// Input/output: T-major [T, C]
kernel void dac_res_unit_p1(device float *out         [[buffer(0)]],
                            device const float *inp   [[buffer(1)]],
                            device const float *w1    [[buffer(2)]],
                            device const float *b1    [[buffer(3)]],
                            constant float *alpha1    [[buffer(4)]],
                            constant float *inv_b1    [[buffer(5)]],
                            constant float *alpha2    [[buffer(6)]],
                            constant float *inv_b2    [[buffer(7)]],
                            constant uint &T          [[buffer(8)]],
                            constant uint &C_in       [[buffer(9)]],
                            constant uint &C_out      [[buffer(10)]],
                            constant uint &K1         [[buffer(11)]],
                            constant uint &pad1       [[buffer(12)]],
                            constant uint &dilation1  [[buffer(13)]],
                            uint2 gid                [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T || oc >= C_out) return;

    float sum = b1[oc];
    for (uint k = 0; k < K1; k++) {
        int t_in = (int)t - (int)pad1 + (int)k * (int)dilation1;
        if (t_in < 0 || t_in >= (int)T) continue;
        uint w_base = oc + k * (C_out * C_in);
        for (uint ic = 0; ic < C_in; ic++) {
            float xv = inp[t_in + ic * T];
            float t1v = xv * alpha1[ic];
            float s1 = sin(t1v);
            float snaked = xv + s1 * s1 * inv_b1[ic];
            sum += snaked * w1[w_base + ic * C_out];
        }
    }
    float t2v = sum * alpha2[oc];
    float s2 = sin(t2v);
    sum = sum + s2 * s2 * inv_b2[oc];

    out[t + oc * T] = sum;
}

// 1x1 conv2 + residual add (completes a residual unit)
// out: residual buffer [T, C] — read residual, write result
// inp: output of dac_res_unit_p1 [T, C]
// w2: [OC, IC, 1] (1x1 conv, Metal layout)
kernel void dac_conv1x1_add(device float *out       [[buffer(0)]],
                            device const float *inp [[buffer(1)]],
                            device const float *w2  [[buffer(2)]],
                            device const float *b2  [[buffer(3)]],
                            constant uint &T        [[buffer(4)]],
                            constant uint &C        [[buffer(5)]],
                            uint2 gid              [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T || oc >= C) return;
    float sum = b2[oc];
    uint w_base = oc;
    for (uint ic = 0; ic < C; ic++)
        sum += inp[t + ic * T] * w2[w_base + ic * C];
    out[t + oc * T] = out[t + oc * T] + sum;
}

// RVQ decode: embed lookup + project_out matmul, accumulate into output
// emb: [D, V] (D fastest, ggml T-major), proj_w: [D, H] (D fastest)
// Output: T-major [T, H]
kernel void dac_rvq_decode_book(device float *out        [[buffer(0)]],
                                device const int *tokens [[buffer(1)]],
                                device const float *emb  [[buffer(2)]],
                                device const float *proj_w [[buffer(3)]],
                                device const float *proj_b [[buffer(4)]],
                                constant uint &T          [[buffer(5)]],
                                constant uint &k_book     [[buffer(6)]],
                                constant uint &D          [[buffer(7)]],
                                constant uint &V          [[buffer(8)]],
                                constant uint &H          [[buffer(9)]],
                                uint2 gid                [[thread_position_in_grid]]) {
    uint t = gid.x, h = gid.y;
    if (t >= T || h >= H) return;
    int token = tokens[t + k_book * T];
    if (token < 0 || token >= (int)V) return;
    float sum = proj_b[h];
    for (uint d = 0; d < D; d++)
        sum += emb[d + (uint)token * D] * proj_w[d + h * D];
    out[t + h * T] += sum;
}

// Linear layer — weight: [OC, IC] (OC fastest, Metal layout from upload_2d_transposed)
// Input/output: T-major [T, C]
kernel void dac_linear(device float *out        [[buffer(0)]],
                       device const float *inp  [[buffer(1)]],
                       device const float *w    [[buffer(2)]],
                       device const float *b    [[buffer(3)]],
                       constant uint &T         [[buffer(4)]],
                       constant uint &C_in      [[buffer(5)]],
                       constant uint &C_out     [[buffer(6)]],
                       uint2 gid               [[thread_position_in_grid]]) {
    uint t = gid.x, oc = gid.y;
    if (t >= T || oc >= C_out) return;
    float sum = b[oc];
    uint w_base = oc;
    for (uint ic = 0; ic < C_in; ic++)
        sum += inp[t + ic * T] * w[w_base + ic * C_out];
    out[t + oc * T] = sum;
}

// Transpose [T, C] -> [C, T] (both T-major)
kernel void dac_transpose(device float *out       [[buffer(0)]],
                          device const float *inp [[buffer(1)]],
                          constant uint &T        [[buffer(2)]],
                          constant uint &C        [[buffer(3)]],
                          uint2 gid              [[thread_position_in_grid]]) {
    uint t = gid.x, c = gid.y;
    if (t >= T || c >= C) return;
    // inp: [T, C] T-major → inp[t + c*T]
    // out: [C, T] T-major → out[c + t*C]
    out[c + t * C] = inp[t + c * T];
}

// Crop 2D: copy [keep_T, C] from [full_T, C] starting at 'offset', T-major layout
kernel void dac_crop_2d(device float *out       [[buffer(0)]],
                        device const float *inp [[buffer(1)]],
                        constant uint &keep     [[buffer(2)]],
                        constant uint &C        [[buffer(3)]],
                        constant uint &full_T   [[buffer(4)]],
                        constant uint &offset   [[buffer(5)]],
                        uint2 gid              [[thread_position_in_grid]]) {
    uint t = gid.x, c = gid.y;
    if (t >= keep || c >= C) return;
    out[t + c * keep] = inp[(t + offset) + c * full_T];
}

// tanh in-place
kernel void dac_tanh(device float *x [[buffer(0)]],
                     uint idx        [[thread_position_in_grid]]) {
    x[idx] = tanh(x[idx]);
}
)MTL";

// =====================================================================
// Helpers
// =====================================================================
static inline id<MTLDevice>      dev(void *p) { return (__bridge id<MTLDevice>)p; }
static inline id<MTLLibrary>     lib(void *p) { return (__bridge id<MTLLibrary>)p; }
static inline id<MTLCommandQueue> q(void *p)  { return (__bridge id<MTLCommandQueue>)p; }
static inline id<MTLComputePipelineState> ps(void *p) { return (__bridge id<MTLComputePipelineState>)p; }
static inline id<MTLBuffer>      buf(void *p) { return (__bridge id<MTLBuffer>)p; }
static inline void *retain_id(id x) {
#if __has_feature(objc_arc)
    return (__bridge_retained void *)x;
#else
    return (__bridge void *)[x retain];
#endif
}
static inline void release_id(void *p) {
    if (!p) return;
#if __has_feature(objc_arc)
    (void)(__bridge_transfer id)p;
#else
    [(__bridge id)p release];
#endif
}

static void disp2D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe,
                   uint X, uint Y) {
    NSUInteger maxT = pipe.maxTotalThreadsPerThreadgroup;
    NSUInteger tx = std::min((NSUInteger)X, maxT);
    NSUInteger ty = std::min((NSUInteger)Y, maxT / tx);
    if (ty == 0) ty = 1;
    [enc setComputePipelineState:pipe];
    [enc dispatchThreads:MTLSizeMake(X, Y, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
}
static void disp1D(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipe, uint N) {
    NSUInteger t = std::min((NSUInteger)N, pipe.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:pipe];
    [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(t, 1, 1)];
}

// =====================================================================
// Init / Decode
// =====================================================================

DACMetalDecoder::DACMetalDecoder() {}
DACMetalDecoder::~DACMetalDecoder() {
    release_id(pipe_rvq_decode_); release_id(pipe_linear_); release_id(pipe_snake_);
    release_id(pipe_conv1d_); release_id(pipe_conv_transpose_1d_); release_id(pipe_res_unit_);
    release_id(pipe_conv1x1_add_); release_id(pipe_transpose_); release_id(pipe_crop_);
    release_id(pipe_tanh_); release_id(queue_); release_id(library_); release_id(device_);
}

void *DACMetalDecoder::get_fn(const char *name) {
    id<MTLFunction> fn = [lib(library_) newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) { RS_LOG_ERR("DACMetal: kernel '%s' not found", name); return nullptr; }
    NSError *err = nil;
    id<MTLComputePipelineState> p = [dev(device_) newComputePipelineStateWithFunction:fn error:&err];
    if (!p) { RS_LOG_ERR("DACMetal: pipeline '%s': %s", name, [[err localizedDescription] UTF8String]); return nullptr; }
    return retain_id(p);
}

void *DACMetalDecoder::alloc_buffer(size_t bytes, const void *data) {
    id<MTLBuffer> b = [dev(device_) newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (data) memcpy([b contents], data, bytes);
    return retain_id(b);
}
void *DACMetalDecoder::upload_raw(const struct ggml_tensor *t) {
    if (!t || !t->data) return nullptr;
    size_t ne = ggml_nelements(t);
    if (t->type == GGML_TYPE_F16) {
        std::vector<float> f32(ne);
        const ggml_fp16_t *fp16 = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < ne; i++) f32[i] = ggml_fp16_to_fp32(fp16[i]);
        return alloc_buffer(ne * sizeof(float), f32.data());
    }
    return alloc_buffer(ggml_nbytes(t), t->data);
}
void *DACMetalDecoder::upload_vec(const std::vector<float> &v) {
    return v.empty() ? nullptr : alloc_buffer(v.size() * sizeof(float), v.data());
}
void *DACMetalDecoder::upload_2d_transposed(const struct ggml_tensor *t) {
    if (!t || !t->data) return nullptr;
    int CI = (int)t->ne[0], CO = (int)t->ne[1];
    std::vector<float> b((size_t)CI * CO);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *fp = (const ggml_fp16_t *)t->data;
        for (int i = 0; i < CI; i++)
            for (int o = 0; o < CO; o++)
                b[(size_t)o + i * CO] = ggml_fp16_to_fp32(fp[(size_t)i + o * CI]);
    } else {
        const float *f = (const float *)t->data;
        for (int i = 0; i < CI; i++)
            for (int o = 0; o < CO; o++)
                b[(size_t)o + i * CO] = f[(size_t)i + o * CI];
    }
    return alloc_buffer(b.size() * sizeof(float), b.data());
}
void *DACMetalDecoder::upload_conv_weight(const struct ggml_tensor *t, int K, int IC, int OC) {
    // ggml [K,IC,OC](K fast) -> Metal [OC,IC,K](OC fast)
    if (!t || !t->data) return nullptr;
    std::vector<float> b((size_t)K * IC * OC);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *fp = (const ggml_fp16_t *)t->data;
        for (int k = 0; k < K; k++)
            for (int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < OC; oc++)
                    b[(size_t)oc + ic * OC + k * OC * IC] =
                        ggml_fp16_to_fp32(fp[(size_t)k + ic * K + oc * K * IC]);
    } else {
        const float *f = (const float *)t->data;
        for (int k = 0; k < K; k++)
            for (int ic = 0; ic < IC; ic++)
                for (int oc = 0; oc < OC; oc++)
                    b[(size_t)oc + ic * OC + k * OC * IC] =
                        f[(size_t)k + ic * K + oc * K * IC];
    }
    return alloc_buffer(b.size() * sizeof(float), b.data());
}
void *DACMetalDecoder::upload_conv_t_weight(const struct ggml_tensor *t, int K, int IC, int OC) {
    // ggml [K,OC,IC] -> Metal [IC,OC,K] (IC fastest)
    if (!t || !t->data) return nullptr;
    std::vector<float> b((size_t)K * IC * OC);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *fp = (const ggml_fp16_t *)t->data;
        for (int k = 0; k < K; k++)
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < IC; ic++)
                    b[(size_t)ic + oc * IC + k * IC * OC] =
                        ggml_fp16_to_fp32(fp[(size_t)k + oc * K + ic * K * OC]);
    } else {
        const float *f = (const float *)t->data;
        for (int k = 0; k < K; k++)
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < IC; ic++)
                    b[(size_t)ic + oc * IC + k * IC * OC] =
                        f[(size_t)k + oc * K + ic * K * OC];
    }
    return alloc_buffer(b.size() * sizeof(float), b.data());
}

bool DACMetalDecoder::init(const DACDecoder &dac, const struct ggml_tensor *fc2_w,
                           const struct ggml_tensor *fc2_b, const RVQCodec &rvq) {
    if (valid_) return true;
    @autoreleasepool {
        device_ = retain_id(MTLCreateSystemDefaultDevice());
        if (!device_) { RS_LOG_ERR("DACMetal: no Metal device"); return false; }
        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:kShaderSrc];
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> l = [dev(device_) newLibraryWithSource:src options:opts error:&err];
        if (!l) { RS_LOG_ERR("DACMetal: compile failed: %s", [[err localizedDescription] UTF8String]); return false; }
        library_ = retain_id(l);
        queue_ = retain_id([dev(device_) newCommandQueue]);

        pipe_rvq_decode_ = get_fn("dac_rvq_decode_book");
        pipe_linear_ = get_fn("dac_linear");
        pipe_snake_ = get_fn("dac_snake");
        pipe_conv1d_ = get_fn("dac_conv1d");
        pipe_conv_transpose_1d_ = get_fn("dac_conv_transpose_1d");
        pipe_res_unit_ = get_fn("dac_res_unit_p1");
        pipe_conv1x1_add_ = get_fn("dac_conv1x1_add");
        pipe_transpose_ = get_fn("dac_transpose");
        pipe_crop_ = get_fn("dac_crop_2d");
        pipe_tanh_ = get_fn("dac_tanh");

        // verify all loaded
        void *pipes[] = {pipe_rvq_decode_, pipe_linear_, pipe_snake_, pipe_conv1d_,
                         pipe_conv_transpose_1d_, pipe_res_unit_, pipe_conv1x1_add_,
                         pipe_transpose_, pipe_crop_, pipe_tanh_};
        for (size_t i = 0; i < sizeof(pipes)/sizeof(pipes[0]); i++)
            if (!pipes[i]) return false;

        // Upload weights
        if (fc2_w) buf_fc2_w_ = upload_2d_transposed(fc2_w);
        if (fc2_b) buf_fc2_b_ = upload_raw(fc2_b);
        if (dac.c1w) buf_init_conv_w_ = upload_conv_weight(dac.c1w, 7, 256, 1024);
        if (dac.c1b) buf_init_conv_b_ = upload_raw(dac.c1b);

        static const int strides[]={8,5,4,2,3}, in_chs[]={1024,512,256,128,64};
        static const int out_chs[]={512,256,128,64,32}, dils[]={1,3,9};
        for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
            const DACBlockWeights &b = dac.blk[i];
            if (!b.ctw) continue;
            auto &ub = up_blocks_[n_up_blocks_];
            ub.in_ch = in_chs[i]; ub.out_ch = out_chs[i];
            ub.stride = strides[i]; ub.pad = b.pad;
            if (b.s1.a) ub.snake_a = upload_raw(b.s1.a);
            ub.snake_ib = upload_vec(b.s1.inv_b);
            ub.ct_w = upload_conv_t_weight(b.ctw, 2*ub.stride, ub.in_ch, ub.out_ch);
            if (b.ctb) ub.ct_b = upload_raw(b.ctb);
            n_up_blocks_++;
            for (int r = 0; r < DAC_RES_UNITS; r++) {
                const DACResUnitWeights &ru = b.ru[r];
                if (!ru.c1w) continue;
                ResUnit ru_mtl;
                ru_mtl.C_in = ub.out_ch; ru_mtl.C_out = ub.out_ch;
                ru_mtl.K = 7; ru_mtl.pad = 3*dils[r]; ru_mtl.dilation = dils[r];
                if (ru.s1.a) ru_mtl.alpha1 = upload_raw(ru.s1.a);
                ru_mtl.inv_b1 = upload_vec(ru.s1.inv_b);
                ru_mtl.c1w = upload_conv_weight(ru.c1w, ru_mtl.K, ru_mtl.C_in, ru_mtl.C_out);
                if (ru.c1b) ru_mtl.c1b = upload_raw(ru.c1b);
                if (ru.s2.a) ru_mtl.alpha2 = upload_raw(ru.s2.a);
                ru_mtl.inv_b2 = upload_vec(ru.s2.inv_b);
                if (ru.c2w) ru_mtl.c2w = upload_conv_weight(ru.c2w, 1, ru_mtl.C_out, ru_mtl.C_out);
                if (ru.c2b) ru_mtl.c2b = upload_raw(ru.c2b);
                res_units_.push_back(ru_mtl);
            }
        }
        if (dac.s_final.a) buf_final_snake_a_ = upload_raw(dac.s_final.a);
        buf_final_snake_ib_ = upload_vec(dac.s_final.inv_b);
        if (dac.c2w) buf_final_conv_w_ = upload_conv_weight(dac.c2w, 7, 32, 1);
        if (dac.c2b) buf_final_conv_b_ = upload_raw(dac.c2b);

        n_rvq_books_ = std::min(rvq.num_codebooks, 8);
        for (int k = 0; k < n_rvq_books_; k++) {
            if (rvq.cb[k].embed) rvq_books_[k].embed = upload_raw(rvq.cb[k].embed);
            if (rvq.cb[k].project_out_w) rvq_books_[k].proj_w = upload_raw(rvq.cb[k].project_out_w);
            if (rvq.cb[k].project_out_b) rvq_books_[k].proj_b = upload_raw(rvq.cb[k].project_out_b);
        }
        valid_ = true;
        RS_LOG_INFO("DACMetal: ready — %d blocks, %zu res_units", n_up_blocks_, res_units_.size());
        return true;
    }
}

// ---------------------------------------------------------------------------
// Decode (T-major layout throughout)
// ---------------------------------------------------------------------------
bool DACMetalDecoder::decode(const int32_t *tokens, int T, int K,
                             std::vector<float> &audio_out) {
    if (!valid_) return false;
    @autoreleasepool {
        const int H = 1024, D_emb = 64;

        // ---- Step 1: RVQ decode → [T, H] T-major ----
        id<MTLBuffer> buf_x = [dev(device_) newBufferWithLength:(size_t)T * H * sizeof(float)
                               options:MTLResourceStorageModeShared];
        memset([buf_x contents], 0, (size_t)T * H * sizeof(float));
        id<MTLBuffer> buf_tok = [dev(device_) newBufferWithLength:(size_t)T * K * sizeof(int32_t)
                                  options:MTLResourceStorageModeShared];
        memcpy([buf_tok contents], tokens, (size_t)T * K * sizeof(int32_t));

        {
            id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int k = 0; k < std::min(K, n_rvq_books_); k++) {
                auto &bk = rvq_books_[k];
                if (!bk.embed || !bk.proj_w) continue;
                [enc setBuffer:buf_x offset:0 atIndex:0];
                [enc setBuffer:buf_tok offset:0 atIndex:1];
                [enc setBuffer:buf(bk.embed) offset:0 atIndex:2];
                [enc setBuffer:buf(bk.proj_w) offset:0 atIndex:3];
                [enc setBuffer:buf(bk.proj_b) offset:0 atIndex:4];
                { uint v=(uint)T; [enc setBytes:&v length:4 atIndex:5]; }
                { uint v=(uint)k; [enc setBytes:&v length:4 atIndex:6]; }
                { uint v=D_emb;   [enc setBytes:&v length:4 atIndex:7]; }
                { uint v=1024;    [enc setBytes:&v length:4 atIndex:8]; }
                { uint v=(uint)H; [enc setBytes:&v length:4 atIndex:9]; }
                disp2D(enc, ps(pipe_rvq_decode_), T, H);
            }
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }


        // ---- Step 2: fc2 → [T, 256] T-major (no transpose needed; linear already outputs T-fast) ----
        const int C_fc2 = 256;
        id<MTLBuffer> buf_fc2 = [dev(device_) newBufferWithLength:(size_t)T * C_fc2 * sizeof(float)
                                   options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setBuffer:buf_fc2 offset:0 atIndex:0];
            [enc setBuffer:buf_x offset:0 atIndex:1];
            [enc setBuffer:buf(buf_fc2_w_) offset:0 atIndex:2];
            [enc setBuffer:buf(buf_fc2_b_) offset:0 atIndex:3];
            { uint v=(uint)T;   [enc setBytes:&v length:4 atIndex:4]; }
            { uint v=(uint)H;   [enc setBytes:&v length:4 atIndex:5]; }
            { uint v=C_fc2;     [enc setBytes:&v length:4 atIndex:6]; }
            disp2D(enc, ps(pipe_linear_), T, C_fc2);
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }

        // ---- Step 3: initial conv1d (256→1024, k=7, pad=3) ----
        int cur_T = T, cur_C = 1024;
        id<MTLBuffer> buf_cur = [dev(device_) newBufferWithLength:(size_t)cur_T * cur_C * sizeof(float)
                                   options:MTLResourceStorageModeShared];
        {
            id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setBuffer:buf_cur offset:0 atIndex:0];
            [enc setBuffer:buf_fc2 offset:0 atIndex:1];   // fc2 output directly, no transpose
            [enc setBuffer:buf(buf_init_conv_w_) offset:0 atIndex:2];
            [enc setBuffer:buf(buf_init_conv_b_) offset:0 atIndex:3];
            { uint v=(uint)T;       [enc setBytes:&v length:4 atIndex:4]; }  // T_in
            { uint v=(uint)C_fc2;   [enc setBytes:&v length:4 atIndex:5]; }  // C_in=256
            { uint v=(uint)cur_T;   [enc setBytes:&v length:4 atIndex:6]; }  // T_out
            { uint v=(uint)cur_C;   [enc setBytes:&v length:4 atIndex:7]; }  // C_out=1024
            { uint v=7; [enc setBytes:&v length:4 atIndex:8]; }   // K
            { uint v=3; [enc setBytes:&v length:4 atIndex:9]; }   // pad
            { uint v=1; [enc setBytes:&v length:4 atIndex:10]; }  // dilation
            { uint v=1; [enc setBytes:&v length:4 atIndex:11]; }  // stride
            disp2D(enc, ps(pipe_conv1d_), cur_T, cur_C);
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }

        // ---- Step 4: 5 upsampling blocks ----
        int ru_idx = 0;
        for (int bi = 0; bi < n_up_blocks_; bi++) {
            auto &ub = up_blocks_[bi];

            // Snake on in_ch before conv_transpose
            if (ub.snake_a && ub.snake_ib) {
                id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setBuffer:buf_cur offset:0 atIndex:0];
                [enc setBuffer:buf(ub.snake_a) offset:0 atIndex:1];
                [enc setBuffer:buf(ub.snake_ib) offset:0 atIndex:2];
                { uint v=(uint)ub.in_ch; [enc setBytes:&v length:4 atIndex:3]; }
                { uint v=(uint)cur_T;    [enc setBytes:&v length:4 atIndex:4]; }
                disp1D(enc, ps(pipe_snake_), (uint)cur_T * ub.in_ch);
                [enc endEncoding];
                [cb commit]; [cb waitUntilCompleted];
            }

            // ConvTranspose then crop
            int T_raw = cur_T * ub.stride + ub.stride;
            int K_ct = 2 * ub.stride;
            int keep_T = cur_T * ub.stride;
            id<MTLBuffer> buf_ct_raw = [dev(device_) newBufferWithLength:(size_t)T_raw * ub.out_ch * sizeof(float)
                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_ct = [dev(device_) newBufferWithLength:(size_t)keep_T * ub.out_ch * sizeof(float)
                                      options:MTLResourceStorageModeShared];
            {
                id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setBuffer:buf_ct_raw offset:0 atIndex:0];
                [enc setBuffer:buf_cur offset:0 atIndex:1];
                [enc setBuffer:buf(ub.ct_w) offset:0 atIndex:2];
                [enc setBuffer:buf(ub.ct_b) offset:0 atIndex:3];
                { uint v=(uint)cur_T;   [enc setBytes:&v length:4 atIndex:4]; }
                { uint v=(uint)ub.in_ch;  [enc setBytes:&v length:4 atIndex:5]; }
                { uint v=(uint)T_raw;   [enc setBytes:&v length:4 atIndex:6]; }
                { uint v=(uint)ub.out_ch; [enc setBytes:&v length:4 atIndex:7]; }
                { uint v=(uint)K_ct;    [enc setBytes:&v length:4 atIndex:8]; }
                { uint v=(uint)ub.stride; [enc setBytes:&v length:4 atIndex:9]; }
                { uint v=0U;            [enc setBytes:&v length:4 atIndex:10]; }
                disp2D(enc, ps(pipe_conv_transpose_1d_), T_raw, ub.out_ch);

                // Crop: keep [keep_T, out_ch] starting at pad offset
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setBuffer:buf_ct offset:0 atIndex:0];
                [enc setBuffer:buf_ct_raw offset:0 atIndex:1];
                { uint v=(uint)keep_T;     [enc setBytes:&v length:4 atIndex:2]; }
                { uint v=(uint)ub.out_ch;  [enc setBytes:&v length:4 atIndex:3]; }
                { uint v=(uint)T_raw;      [enc setBytes:&v length:4 atIndex:4]; }
                { uint v=(uint)ub.pad;     [enc setBytes:&v length:4 atIndex:5]; }
                disp2D(enc, ps(pipe_crop_), keep_T, ub.out_ch);
                [enc endEncoding];
                [cb commit]; [cb waitUntilCompleted];
            }
            cur_T = keep_T; cur_C = ub.out_ch;
            buf_cur = buf_ct;

            // 3 residual units
            for (int r = 0; r < 3 && ru_idx < (int)res_units_.size(); r++, ru_idx++) {
                auto &ru = res_units_[ru_idx];
                id<MTLBuffer> buf_ru = [dev(device_) newBufferWithLength:(size_t)cur_T * cur_C * sizeof(float)
                                           options:MTLResourceStorageModeShared];

                // Part 1: snake1 → conv1d(dilated) → snake2
                {
                    id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setBuffer:buf_ru offset:0 atIndex:0];
                    [enc setBuffer:buf_cur offset:0 atIndex:1];
                    [enc setBuffer:buf(ru.c1w) offset:0 atIndex:2];
                    [enc setBuffer:buf(ru.c1b) offset:0 atIndex:3];
                    [enc setBuffer:buf(ru.alpha1) offset:0 atIndex:4];
                    [enc setBuffer:buf(ru.inv_b1) offset:0 atIndex:5];
                    [enc setBuffer:buf(ru.alpha2) offset:0 atIndex:6];
                    [enc setBuffer:buf(ru.inv_b2) offset:0 atIndex:7];
                    { uint v=(uint)cur_T;       [enc setBytes:&v length:4 atIndex:8]; }
                    { uint v=(uint)ru.C_in;     [enc setBytes:&v length:4 atIndex:9]; }
                    { uint v=(uint)ru.C_out;    [enc setBytes:&v length:4 atIndex:10]; }
                    { uint v=(uint)ru.K;        [enc setBytes:&v length:4 atIndex:11]; }
                    { uint v=(uint)ru.pad;      [enc setBytes:&v length:4 atIndex:12]; }
                    { uint v=(uint)ru.dilation; [enc setBytes:&v length:4 atIndex:13]; }
                    disp2D(enc, ps(pipe_res_unit_), cur_T, cur_C);
                    [enc endEncoding];
                    [cb commit]; [cb waitUntilCompleted];
                }

                // Part 2: 1x1 conv2 + residual add (in-place on buf_cur)
                {
                    id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setBuffer:buf_cur offset:0 atIndex:0];   // residual IN/OUT
                    [enc setBuffer:buf_ru offset:0 atIndex:1];    // snake2 output IN
                    [enc setBuffer:buf(ru.c2w) offset:0 atIndex:2];
                    [enc setBuffer:buf(ru.c2b) offset:0 atIndex:3];
                    { uint v=(uint)cur_T; [enc setBytes:&v length:4 atIndex:4]; }
                    { uint v=(uint)cur_C; [enc setBytes:&v length:4 atIndex:5]; }
                    disp2D(enc, ps(pipe_conv1x1_add_), cur_T, cur_C);
                    [enc endEncoding];
                    [cb commit]; [cb waitUntilCompleted];
                }
            }
        }

        // ---- Step 5: final snake(C→C) + conv(C→1, k=7) ----
        if (buf_final_snake_a_ && buf_final_snake_ib_) {
            id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setBuffer:buf_cur offset:0 atIndex:0];
            [enc setBuffer:buf(buf_final_snake_a_) offset:0 atIndex:1];
            [enc setBuffer:buf(buf_final_snake_ib_) offset:0 atIndex:2];
            { uint v=(uint)cur_C; [enc setBytes:&v length:4 atIndex:3]; }
            { uint v=(uint)cur_T; [enc setBytes:&v length:4 atIndex:4]; }
            disp1D(enc, ps(pipe_snake_), (uint)cur_T * cur_C);
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }

        id<MTLBuffer> buf_audio = nullptr;
        if (buf_final_conv_w_) {
            int T_audio = cur_T;
            buf_audio = [dev(device_) newBufferWithLength:(size_t)T_audio * sizeof(float)
                             options:MTLResourceStorageModeShared];
            {
                id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setBuffer:buf_audio offset:0 atIndex:0];
                [enc setBuffer:buf_cur offset:0 atIndex:1];
                [enc setBuffer:buf(buf_final_conv_w_) offset:0 atIndex:2];
                [enc setBuffer:buf(buf_final_conv_b_) offset:0 atIndex:3];
                { uint v=(uint)cur_T;    [enc setBytes:&v length:4 atIndex:4]; }
                { uint v=(uint)cur_C;    [enc setBytes:&v length:4 atIndex:5]; }
                { uint v=(uint)T_audio;  [enc setBytes:&v length:4 atIndex:6]; }
                { uint v=1U;             [enc setBytes:&v length:4 atIndex:7]; }
                { uint v=7U;             [enc setBytes:&v length:4 atIndex:8]; }
                { uint v=3U;             [enc setBytes:&v length:4 atIndex:9]; }
                { uint v=1U;             [enc setBytes:&v length:4 atIndex:10]; }
                { uint v=1U;             [enc setBytes:&v length:4 atIndex:11]; }
                disp2D(enc, ps(pipe_conv1d_), T_audio, 1);
                [enc endEncoding];
                [cb commit]; [cb waitUntilCompleted];
            }
            cur_T = T_audio; cur_C = 1;
        } else {
            buf_audio = buf_cur;
        }

        // ---- Step 6: tanh ----
        {
            id<MTLCommandBuffer> cb = [q(queue_) commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setBuffer:buf_audio offset:0 atIndex:0];
            disp1D(enc, ps(pipe_tanh_), cur_T);
            [enc endEncoding];
            [cb commit]; [cb waitUntilCompleted];
        }

        audio_out.resize(cur_T);
        memcpy(audio_out.data(), [buf_audio contents], cur_T * sizeof(float));
        return true;
    }
}
