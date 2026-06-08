#pragma once

#include "cosyvoice3.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

/**
 * CosyVoice3 HiFT (CausalHiFTGenerator) — 80-d mel @ 50 Hz → 24 kHz waveform.
 *
 * Architecture (matches reference `cosyvoice.cpp/src/cosyvoice-graph.cpp:710`):
 *   F0 predictor:   speech_feat → 5× CausalConv1d (condnet) + classifier → f0
 *   Source path:    f0 → nearest-upsample(×scale_factor=hop·∏rates=480)
 *                       → SineGen2 (9 harmonics + noise + uv mask)
 *                       → m_source.l_linear (9→1) + tanh → sine_merge
 *                       → STFT(n_fft=16, hop=4, hann) → s_stft [18, T_stft]
 *   Main path:      conv_pre(k=5, right-causal) → 3 upsample stages
 *                   each stage:
 *                     leaky_relu(0.1) → nearest-upsample(rate) + conv(k=2r, left-causal)
 *                     → add(source_resblocks[i](source_downs[i](s_stft)))
 *                     → mean over 3 ResBlock outputs (num_kernels)
 *                   → leaky_relu(0.01) → conv_post(k=7, left-causal)
 *                   → split into magnitude (top 9 chans, exp+clamp) and
 *                     phase (bottom 9 chans, sin)
 *                   → iSTFT(n_fft=16, hop=4, hann) → 24 kHz waveform
 *                   → clamp(±audio_limit=0.99)
 *
 * The reference keeps the entire HiFT in a single ggml graph by using two
 * custom ops (`ggml_stft`, `ggml_istft`) that stock ggml does not provide. We
 * split the work into THREE ggml graphs + two CPU FFT passes:
 *
 *   graph 1  (ggml) : F0 predictor (speech_feat → f0)
 *   CPU     source  : sinegen + l_linear + tanh (trig math; easy to verify
 *                     against PyTorch). Includes deterministic rand_ini.
 *   CPU     STFT    : sine_merge → s_stft   (n_fft=16, hop=4, hann, center)
 *   graph 2  (ggml) : conv_pre + 3 upsample stages + conv_post + mag/phase
 *   CPU     iSTFT   : (mag, phase) → 24 kHz PCM (n_fft=16, hop=4, hann)
 *
 * The CPU STFT/iSTFT bookends are intentionally simple — n_fft=16 keeps the
 * arithmetic trivial; the dominant cost stays in the upsample tower (graph 2).
 */
class CosyVoice3HiFTModel {
public:
  CosyVoice3HiFTModel() = default;
  ~CosyVoice3HiFTModel() = default;

  bool Load(gguf_context *ctx_gguf, ggml_context *gguf_data,
            ggml_backend_t backend);

  /**
   * Synthesize 24 kHz f32 PCM from `state.mel_output` ([T_mel, 80] row-major).
   * Writes `state.audio_output` and sets `state.hift_done`.
   */
  bool RunHiFT(CosyVoice3State &state, ggml_backend_sched_t sched);

private:
  // -------------------------------------------------------------------------
  // Hparams (read from `cosyvoice3.hift.*` KVs).
  // -------------------------------------------------------------------------
  int sample_rate_         = 24000;
  int n_fft_               = 16;
  int hop_len_             = 4;
  int nb_harmonics_        = 8;
  float nsf_alpha_         = 0.1f;
  float nsf_sigma_         = 0.003f;
  int nsf_voiced_threshold_ = 10;
  float lrelu_slope_       = 0.1f;
  float audio_limit_       = 0.99f;
  int num_kernels_         = 3;
  std::vector<int> upsample_rates_ = {8, 5, 3};
  int scale_factor_        = 480;   // hop_len * ∏rates

  // -------------------------------------------------------------------------
  // Weight tensors (pointers into the GGUF-backed ggml_context).
  // -------------------------------------------------------------------------
  // F0 predictor — condnet has 5 layers at indices 0,2,4,6,8 (even-only;
  // odd indices are ELU in PyTorch).
  ggml_tensor *condnet_w_[5] = {};
  ggml_tensor *condnet_b_[5] = {};
  ggml_tensor *classifier_w_ = nullptr, *classifier_b_ = nullptr;

  // m_source: l_linear is Linear(in=9, out=1).
  ggml_tensor *m_source_lin_w_ = nullptr, *m_source_lin_b_ = nullptr;

  ggml_tensor *conv_pre_w_  = nullptr, *conv_pre_b_  = nullptr;  // k=5 right
  ggml_tensor *conv_post_w_ = nullptr, *conv_post_b_ = nullptr;  // k=7 left

  // 3 upsample stages — nearest-upsample then plain conv k=2r.
  struct UpStage {
    ggml_tensor *w = nullptr, *b = nullptr;
    int rate   = 0;
    int kernel = 0;
  };
  std::vector<UpStage> ups_;

  // source_downs: index 0..N-2 are strided causal conv; the last is stride 1.
  struct SrcDown {
    ggml_tensor *w = nullptr, *b = nullptr;
    int kernel = 0;
    int stride = 0;   // 1 means plain (left-pad k-1, stride 1)
  };
  std::vector<SrcDown> source_downs_;

  // ResBlocks: each has 3 sub-blocks (one per dilation), each is
  // Snake-Conv1d(k=3,d=di)-Snake-Conv1d(k=3,d=1) + residual.
  struct ResBlockW {
    struct Sub {
      ggml_tensor *a1 = nullptr;   // snake alpha (1-D)
      ggml_tensor *c1w = nullptr, *c1b = nullptr;
      ggml_tensor *a2 = nullptr;
      ggml_tensor *c2w = nullptr, *c2b = nullptr;
      int dilation = 1;
    };
    std::array<Sub, 3> subs;
  };
  std::vector<ResBlockW> resblocks_;        // size = 9 (3 stages × num_kernels)
  std::vector<ResBlockW> source_resblocks_; // size = 3 (one per stage)

  // -------------------------------------------------------------------------
  // Precomputed host-side tables.
  // -------------------------------------------------------------------------
  std::vector<float> hann_window_;          // length n_fft_, periodic Hann
  std::vector<float> rand_ini_;             // nb_harmonics+1; rand_ini[0]=0

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------
  bool LoadHparams(gguf_context *ctx_gguf);
  bool LoadTensors(const std::map<std::string, ggml_tensor *> &tensors);
  void PrecomputeWindowAndRandIni(uint64_t seed);

  bool LoadResBlock(const std::map<std::string, ggml_tensor *> &tensors,
                    const std::string &prefix, ResBlockW &out) const;

  // Build sub-graphs.
  ggml_tensor *BuildF0Graph(ggml_context *ctx,
                            ggml_tensor *speech_feat) const;
  ggml_tensor *BuildMainGraph(ggml_context *ctx,
                              ggml_tensor *speech_feat,
                              ggml_tensor *s_stft) const;

  // CPU source path: f0 [T_mel] → sine_merge [T_audio = T_mel * scale_factor].
  void RunSourceCpu(const float *f0_host, int T_mel,
                    std::vector<float> &out_sine_merge,
                    std::mt19937 &rng) const;
  ggml_tensor *BuildResBlock(ggml_context *ctx, const ResBlockW &rb,
                             ggml_tensor *x) const;

  // CPU bookends.
  void StftCpu(const float *signal, int n, std::vector<float> &out_real,
               std::vector<float> &out_imag, int &T_stft) const;
  void IstftCpu(const float *mag, const float *phase, int T_stft,
                std::vector<float> &out_pcm) const;
};
