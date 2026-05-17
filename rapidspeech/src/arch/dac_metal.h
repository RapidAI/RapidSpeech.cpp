#pragma once

#include <vector>
#include <cstdint>

// Forward declarations (from omnivoice.h — avoid pulling in the full header)
struct DACDecoder;
struct DACBlockWeights;
struct DACResUnitWeights;
struct RVQCodec;
struct ggml_tensor;

#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLLibrary;
@protocol MTLCommandQueue;
@protocol MTLComputePipelineState;
@protocol MTLBuffer;
#endif

/**
 * Custom Metal GPU backend for the DAC vocoder.
 *
 * The DAC decoder contains ~300 small operations (conv1d, snake, conv_transpose)
 * across 5 upsampling blocks. ggml dispatches each as a separate kernel launch,
 * causing GPU overhead to dominate runtime.  This backend replaces the ggml
 * graph with ~30 fused kernel dispatches, letting the GPU actually saturate.
 */
class DACMetalDecoder {
public:
    DACMetalDecoder();
    ~DACMetalDecoder();

    /** One-time init: compile shaders, create pipelines, upload weights to GPU. */
    bool init(const DACDecoder &dac, const struct ggml_tensor *fc2_w,
              const struct ggml_tensor *fc2_b, const RVQCodec &rvq);

    /** Run full DAC decoder on GPU. */
    bool decode(const int32_t *tokens, int T, int K,
                std::vector<float> &audio_out);

    bool is_valid() const { return valid_; }

private:
    bool valid_ = false;

    // Metal objects (void* to keep header ObjC-free)
    void *device_ = nullptr;
    void *library_ = nullptr;
    void *queue_ = nullptr;
    void *pipe_rvq_decode_ = nullptr;
    void *pipe_linear_ = nullptr;
    void *pipe_snake_ = nullptr;
    void *pipe_conv1d_ = nullptr;
    void *pipe_conv_transpose_1d_ = nullptr;
    void *pipe_res_unit_ = nullptr;
    void *pipe_conv1x1_add_ = nullptr;
    void *pipe_transpose_ = nullptr;
    void *pipe_crop_ = nullptr;
    void *pipe_tanh_ = nullptr;

    // Pre-uploaded weight buffers (void* = id<MTLBuffer>)
    void *buf_fc2_w_ = nullptr;    // [256, 1024]
    void *buf_fc2_b_ = nullptr;    // [256]
    void *buf_init_conv_w_ = nullptr;  // [1024, 256, 7]
    void *buf_init_conv_b_ = nullptr;  // [1024]

    struct UpsampleBlock {
        void *snake_a = nullptr, *snake_ib = nullptr;
        void *ct_w = nullptr, *ct_b = nullptr;
        int in_ch, out_ch, stride, pad;
    };
    UpsampleBlock up_blocks_[5];
    int n_up_blocks_ = 0;

    struct ResUnit {
        void *alpha1 = nullptr, *inv_b1 = nullptr, *c1w = nullptr, *c1b = nullptr;
        void *alpha2 = nullptr, *inv_b2 = nullptr, *c2w = nullptr, *c2b = nullptr;
        int C_in, C_out, K, pad, dilation;
    };
    std::vector<ResUnit> res_units_;

    void *buf_final_snake_a_ = nullptr, *buf_final_snake_ib_ = nullptr;
    void *buf_final_conv_w_ = nullptr, *buf_final_conv_b_ = nullptr;

    // RVQ codebooks
    struct RVQBook {
        void *embed = nullptr, *proj_w = nullptr, *proj_b = nullptr;
    };
    RVQBook rvq_books_[8];
    int n_rvq_books_ = 0;

    // Helpers
    void *alloc_buffer(size_t bytes, const void *data = nullptr);
    void *upload_raw(const struct ggml_tensor *t);
    void *upload_2d_transposed(const struct ggml_tensor *t);
    void *upload_conv_weight(const struct ggml_tensor *t, int K, int IC, int OC);
    void *upload_conv_t_weight(const struct ggml_tensor *t, int K, int IC, int OC);
    void *upload_vec(const std::vector<float> &v);
    void *get_fn(const char *name);
};
