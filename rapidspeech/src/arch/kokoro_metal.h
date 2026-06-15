#pragma once

#include <cstdint>
#include <cstddef>

struct ggml_tensor;

#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLLibrary;
@protocol MTLCommandQueue;
@protocol MTLComputePipelineState;
@protocol MTLBuffer;
#endif

/**
 * Custom Metal kernel for the two ConvTranspose1d ops inside the Kokoro
 * iSTFTNet generator (dec.gen.ups.0 and dec.gen.ups.1).
 *
 * ggml-metal's kernel_conv_transpose_1d dispatches (OL, OC, 1) threadgroups
 * with a single thread each — the IC reduction runs serially in one thread.
 * For Kokoro's ups0 (K=20, s=10, IC=512, OC=256) and ups1 (K=12, s=6, IC=256,
 * OC=128) this leaves ~96% of GPU SIMD lanes idle.
 *
 * This kernel uses one simdgroup (32 threads) per (t, oc) output element and
 * does a simd_sum reduction over IC. The bias add and PyTorch-style symmetric
 * crop are fused into the same kernel — no separate ggml_add or view+cont.
 *
 * Weight layout stored on the GPU:
 *   ne (logical, ggml) = (K, OC, IC)  K fastest
 *   We re-pack to (IC, K, OC)         IC fastest, so simd_sum over IC reads
 *                                     contiguous F16 lanes.
 */
class KokoroMetalConvT1D {
public:
    KokoroMetalConvT1D();
    ~KokoroMetalConvT1D();

    /**
     * One-time init. Compiles the kernel and uploads weights + biases for the
     * two layers (ups0_w, ups0_b for layer 0; ups1_w, ups1_b for layer 1).
     * `stride_l` and `pad_l` capture the PyTorch hyperparams.
     *
     * Layer 0: K=20, stride=10, pad=5, IC=512, OC=256
     * Layer 1: K=12, stride=6,  pad=3, IC=256, OC=128
     */
    bool init(const ggml_tensor *ups0_w, const ggml_tensor *ups0_b, int stride_0, int pad_0,
              const ggml_tensor *ups1_w, const ggml_tensor *ups1_b, int stride_1, int pad_1);

    bool is_valid() const { return valid_; }

    /**
     * Run ConvTranspose1d + bias + symmetric crop for `layer` ∈ {0, 1}.
     *
     *   input:  F32, shape (IC, T_in), T_in fastest in memory (i.e. for each
     *           ic row, T_in floats contiguous). This is the OPPOSITE of
     *           ggml's normal (C, T) convention where ne[0]=C is fastest —
     *           callers must produce the buffer with a transpose+cont.
     *   output: F32, shape (OC, T_out), T_out fastest in memory.
     *   T_out = (T_in - 1) * stride + K - 2 * pad
     *
     * `input` and `output` are CPU-addressable F32 buffers; on Apple Silicon
     * they are copied into MTLResourceStorageModeShared scratch (effectively a
     * memcpy on unified memory). Output is written directly back.
     */
    bool run(int layer, const float *input, int T_in, float *output);

    /** Last-layer output time-step count given the input time-step count. */
    int output_T(int layer, int T_in) const;

private:
    struct Layer {
        void *w_buf = nullptr;  // F16 weights, packed (IC, K, OC)
        void *b_buf = nullptr;  // F32 bias, length OC
        int   K = 0, IC = 0, OC = 0, stride = 0, pad = 0;
        void *pipeline = nullptr;
    };

    bool   valid_ = false;
    void  *device_   = nullptr;
    void  *library_  = nullptr;
    void  *queue_    = nullptr;
    Layer  layers_[2];

    void *make_pipeline(const char *name);
    void *alloc_buffer(size_t bytes, const void *data);
    bool  upload_weights(Layer &L, const ggml_tensor *w, const ggml_tensor *b,
                         int K, int IC, int OC, int stride, int pad);
};
