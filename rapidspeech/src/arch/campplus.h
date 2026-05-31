// arch/campplus.h — CAMPPlus 192-d speaker encoder, ggml-graph implementation.
//
// Standalone speaker-embedding model (3D-Speaker / FunAudioLLM CosyVoice2/3
// release). The pipeline:
//
//   16 kHz mono float PCM
//     → Kaldi-compatible 80-bin fbank (Povey window, 25 ms / 10 ms,
//       per-utterance mean subtract along time)
//     → FCM head:  Conv2d(1→32) + BN + ReLU + 4 BasicResBlocks
//                  + Conv2d(32→32, s=(2,1)) + BN + ReLU
//                  → reshape (32×10, T) = (320, T)
//     → tdnn (Conv1d 320→128, k=5, s=2) + BN + ReLU
//     → block1 (12 CAMDenseTDNN layers, dilation=1) → 512 channels
//     → transit1 (BN + ReLU + Conv1d 512→256)
//     → block2 (24 CAMDenseTDNN layers, dilation=2) → 1024 channels
//     → transit2 (BN + ReLU + Conv1d 1024→512)
//     → block3 (16 CAMDenseTDNN layers, dilation=2) → 1024 channels
//     → transit3 (BN + ReLU + Conv1d 1024→512)
//     → out_nl (BN + ReLU)
//     → StatsPool (concat mean, std along T) → 1024-d
//     → dense (Conv1d 1024→192, k=1) + BN(affine=False)
//     → L2 normalize → 192-d
//
// The runtime owns this module through a dedicated handle (`rs_speaker_t`),
// not the regular `rs_context_t` ASR/TTS pipeline. The GGUF file must declare
// `general.architecture = "campplus"`.
//
// BN strategy: at Load() time each BN's (running_mean, running_var, weight,
// bias) are folded into a per-channel (gamma_eff, beta_eff) pair so the graph
// only needs `mul + add`. `affine=False` BN (xv.dense.nl.bn) is folded
// identically with weight=1, bias=0.

#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "rapidspeech.h"   // RS_API
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------

struct CAMPPlusHParams {
    // Audio frontend
    int sample_rate = 16000;
    int n_mels      = 80;
    int frame_size  = 400;   // 25 ms @ 16 kHz
    int frame_step  = 160;   // 10 ms @ 16 kHz

    // FCM head
    int fcm_in_channels  = 1;
    int fcm_out_channels = 32;

    // tdnn
    int tdnn_in_channels  = 320;
    int tdnn_out_channels = 128;
    int tdnn_kernel       = 5;
    int tdnn_stride       = 2;

    // CAMDenseTDNN blocks
    int growth_rate = 32;
    int bn_channels = 128;
    int cam_kernel  = 3;

    int block1_layers   = 12;
    int block1_dilation = 1;
    int block2_layers   = 24;
    int block2_dilation = 2;
    int block3_layers   = 16;
    int block3_dilation = 2;

    // transit*
    int transit1_out = 256;
    int transit2_out = 512;
    int transit3_out = 512;

    // Output
    int stats_pool_dim = 1024;
    int embed_dim      = 192;

    // BN epsilon (PyTorch default).
    float bn_eps = 1e-5f;
};

// ---------------------------------------------------------------------------
// Folded BN — at Load() time we replace (mean, var, weight, bias) with
// (gamma_eff, beta_eff) tensors so the graph reduces to mul + add.
// Both stored F32 with shape (C,).
// ---------------------------------------------------------------------------

struct CAMPPlusBN {
    struct ggml_tensor* gamma = nullptr; // (C,) F32
    struct ggml_tensor* beta  = nullptr; // (C,) F32
    int channels = 0;
};

// FCM BasicResBlock (2-D residual unit). When the block down-samples
// (stride along H != 1) it has an extra 1×1 shortcut conv + BN, otherwise
// the shortcut path is identity and `sc_*` stay null.
struct CAMPPlusResBlock {
    struct ggml_tensor* conv1_w = nullptr; // F16 (kw=3, kh=3, in, out)
    CAMPPlusBN bn1;
    struct ggml_tensor* conv2_w = nullptr; // F16 (kw=3, kh=3, out, out)
    CAMPPlusBN bn2;
    bool has_shortcut = false;
    struct ggml_tensor* sc_w = nullptr;    // F16 (1, 1, in, out)
    CAMPPlusBN sc_bn;
    int stride = 1; // along H (mel axis); W (time) always 1
    int in_channels  = 0;
    int out_channels = 0;
};

struct CAMPPlusFCM {
    struct ggml_tensor* conv1_w = nullptr; // F16 (3, 3, 1, 32)
    CAMPPlusBN bn1;
    std::vector<CAMPPlusResBlock> layer1;  // 2 blocks (1st stride=2)
    std::vector<CAMPPlusResBlock> layer2;  // 2 blocks (1st stride=2)
    struct ggml_tensor* conv2_w = nullptr; // F16 (3, 3, 32, 32)
    CAMPPlusBN bn2;
};

struct CAMPPlusDenseLayer {
    // Pre-bottleneck BN: runs on the layer input (in_channels grows
    // monotonically across the dense block).
    CAMPPlusBN nonl1_bn;
    // Bottleneck Conv1d (in → bn_channels=128, k=1, no bias).
    struct ggml_tensor* l1_w = nullptr;    // F16 (1, in, 128)
    // Post-bottleneck BN (channels = bn_channels).
    CAMPPlusBN nonl2_bn;
    // CAM gate: 3 Conv1d sub-layers.
    struct ggml_tensor* cam_ll_w = nullptr; // F16 (k=3, 128, growth=32)
    struct ggml_tensor* cam_l1_w = nullptr; // F16 (1, 128, 64)
    struct ggml_tensor* cam_l1_b = nullptr; // F32 (64,)
    struct ggml_tensor* cam_l2_w = nullptr; // F16 (1, 64, 32)
    struct ggml_tensor* cam_l2_b = nullptr; // F32 (32,)
    int in_channels = 0;
    int dilation    = 1;
};

struct CAMPPlusDenseBlock {
    int num_layers = 0;
    int dilation   = 1;
    std::vector<CAMPPlusDenseLayer> layers;
};

// Conv1d-with-BN unit used by tdnn / transit{1,2,3} / dense.
// transit's order is (BN → ReLU → Conv1d); tdnn is (Conv1d → BN → ReLU).
// The struct just holds the weights — order is enforced by the graph builder.
struct CAMPPlusUnit {
    struct ggml_tensor* lin_w = nullptr; // F16 (kw, in, out)
    CAMPPlusBN bn;
    int kernel_w = 1;
    int in_dim   = 0;
    int out_dim  = 0;
};

struct CAMPPlusWeights {
    CAMPPlusFCM head;
    CAMPPlusUnit tdnn;       // Conv1d → BN → ReLU
    CAMPPlusDenseBlock block1;
    CAMPPlusUnit transit1;   // BN → ReLU → Conv1d
    CAMPPlusDenseBlock block2;
    CAMPPlusUnit transit2;
    CAMPPlusDenseBlock block3;
    CAMPPlusUnit transit3;
    CAMPPlusBN out_nl_bn;    // bare BN (out_nonlinear in upstream)
    CAMPPlusUnit dense;      // Conv1d 1024→192 + BN(affine=False)
};

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

struct CAMPPlusState : public RSState {
    // Final 192-d L2-normalized embedding. Embed() fills this; readers
    // copy out via std::vector::data().
    std::vector<float> embedding;
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

class RS_API CAMPPlusModel : public ISpeechModel {
public:
    CAMPPlusModel();
    ~CAMPPlusModel() override;

    bool Load(const std::unique_ptr<rs_context_t>& ctx,
              ggml_backend_t backend) override;

    // Direct entry — used by the dedicated `rs_speaker_t` C-API handle which
    // owns its own gguf/ggml contexts (not via rs_context_t). The backend
    // is the buffer owner for the GGUF tensors; folded BN tensors are
    // additionally allocated on the same backend.
    bool LoadDirect(ggml_context* gguf_data, gguf_context* ctx_gguf,
                    ggml_backend_t backend);

    std::shared_ptr<RSState> CreateState() override;

    // ISpeechModel boilerplate — CAMPPlus is one-shot, not streaming. We
    // use Encode() as the stateless embedding entry point: input_frames is
    // the raw 16 kHz PCM, output ends up in CAMPPlusState::embedding.
    bool Encode(const std::vector<float>& input_frames, RSState& state,
                ggml_backend_sched_t sched) override;
    bool Decode(RSState& state, ggml_backend_sched_t sched) override;
    std::string GetTranscription(RSState& state) override;
    const RSModelMeta& GetMeta() const override { return meta_; }

    // Convenience entry for callers that already have a state handle.
    // Equivalent to Encode() but more discoverable.
    bool Embed(const float* pcm_16k, int n_samples, RSState& state,
               ggml_backend_sched_t sched);

    int  GetEmbedDim()    const { return hparams_.embed_dim; }
    int  GetSampleRate()  const { return hparams_.sample_rate; }
    ggml_backend_t GetBackend() { return backend_; }

private:
    bool MapTensors(ggml_context* gguf_data);
    bool LoadHParams(gguf_context* ctx_gguf);
    bool FoldAllBN(ggml_context* gguf_data);

    // Internal fbank: 80-bin Kaldi-style + per-utterance mean subtract along
    // time. Returns row-major (T_frames, 80). T_frames_out = 0 on error.
    std::vector<float> ComputeFbank(const float* pcm_16k, int n_samples,
                                    int& T_frames_out) const;

private:
    RSModelMeta      meta_;
    CAMPPlusHParams  hparams_;
    CAMPPlusWeights  weights_;
    ggml_backend_t   backend_ = nullptr;

    // Synthetic ggml context + buffer holding the folded BN tensors. Lives
    // for the lifetime of the model. Allocated lazily during Load().
    ggml_context*         bn_ctx_    = nullptr;
    ggml_backend_buffer_t bn_buffer_ = nullptr;
};
