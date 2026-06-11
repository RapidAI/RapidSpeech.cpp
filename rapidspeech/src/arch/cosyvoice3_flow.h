#pragma once

#include "cosyvoice3.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * CosyVoice3 Flow (CFM + DiT) — speech tokens → 80-dim mel @ 50 Hz.
 *
 * Inputs (per Decode call, supplied by `CosyVoice3State`):
 *   prompt_token   int32[T_pt]                   reference speech tokens
 *   prompt_feat    float[T_pt_mel, 80]           reference mel (80-dim)
 *   embedding      float[192]                    CAMPPlus speaker embedding
 *   speech_token_ids int32[T_gen]                tokens emitted by the LLM
 *
 * Output (written into `CosyVoice3State::mel_output`):
 *   mel            float[T_gen * 2, 80]          generated mel @ 50 Hz
 *
 * Pipeline (matches reference `cosyvoice.cpp/src/cosyvoice-graph.cpp:546+`):
 *   encode: concat(prompt_tok, gen_tok) → input_embedding lookup
 *           → PreLookaheadLayer (2× CausalConv1d residual)
 *           → repeat_interleave by token_mel_ratio=2 (token grid → mel grid)
 *           → pad(prompt_feat, 0, T_gen_mel)                         → mu, cond
 *           l2_norm + Linear(192→80) of embedding                     → spks
 *
 *   solve:  x ← N(0, I) @ shape [80, T_mel]
 *           for step in 1..10:
 *               x_in = repeat(x, batch=2)   (CFG)
 *               dphi_cond, dphi_null = DiT(x_in, mu/spks/cond, t, …)
 *               dphi = (1+cfg)·dphi_cond − cfg·dphi_null
 *               x += dt·dphi
 *           x = x[:, T_pt_mel:]   (drop prompt mel; only on last step)
 *
 *   The 22-block DiT estimator uses AdaLN-Zero modulation with `(1+scale)·norm
 *   + shift`, bidirectional attention, and standard RoPE (θ=10000, head_dim=
 *   64). CFG runs as batch=2 in a single forward.
 *
 * The class owns:
 *   - All flow tensors (loaded once at Load() into the GGUF-backed ggml_context)
 *   - A sinusoidal-position-embedding table (computed at Load(), CPU)
 *   - A scratch ggml_context per RunFlow() call (graph + intermediate tensors)
 */
class CosyVoice3FlowModel {
public:
  CosyVoice3FlowModel() = default;
  ~CosyVoice3FlowModel();

  /**
   * Bind tensors from the unified GGUF and allocate the sinus-embed table on
   * the supplied backend. Returns false if any required tensor is missing.
   *
   * @param ctx_gguf   the loaded gguf metadata context (KVs are read from here)
   * @param gguf_data  the ggml_context holding the tensor data
   * @param backend    backend on which the sinus-embed table will live
   */
  bool Load(gguf_context *ctx_gguf, ggml_context *gguf_data,
            ggml_backend_t backend);

  /**
   * Synthesize a mel spectrogram for the current speech-token sequence. Reads
   * `prompt_token / prompt_feat / embedding / speech_token_ids` from `state`,
   * writes `mel_output` (row-major [T_mel, 80]) and sets `flow_done`.
   */
  bool RunFlow(CosyVoice3State &state, ggml_backend_sched_t sched);

  /** Override the default 10 Euler steps (e.g. from a CLI flag). */
  void SetEulerSteps(int n);

  // Activation-aware quantization hook. The Flow DiT is the most
  // quantization-sensitive part of CV3 (10-step CFM trajectory accumulates
  // error), so imatrix calibration is mainly aimed here.
  void set_imatrix_callback(std::function<void(struct ggml_tensor *)> cb) {
    imatrix_cb_ = std::move(cb);
  }

private:
  // -------------------------------------------------------------------------
  // Bound weight tensors (pointers into the GGUF-backed ggml_context).
  // -------------------------------------------------------------------------

  // CausalMaskedDiffWithDiT
  ggml_tensor *input_embedding_ = nullptr;        // [80, 6561]   tok→mel embed
  ggml_tensor *spk_affine_w_   = nullptr;         // [192, 80]
  ggml_tensor *spk_affine_b_   = nullptr;         // [80]

  // PreLookaheadLayer (kernel=4 then kernel=3)
  ggml_tensor *pre_look_c1_w_  = nullptr;
  ggml_tensor *pre_look_c1_b_  = nullptr;
  ggml_tensor *pre_look_c2_w_  = nullptr;
  ggml_tensor *pre_look_c2_b_  = nullptr;
  int pre_lookahead_len_       = 3;
  int token_mel_ratio_         = 2;

  // DiT::time_embed
  ggml_tensor *time_emb_table_ = nullptr;         // [128]  sinusoidal (CPU)
  ggml_tensor *time_mlp0_w_    = nullptr;
  ggml_tensor *time_mlp0_b_    = nullptr;
  ggml_tensor *time_mlp2_w_    = nullptr;
  ggml_tensor *time_mlp2_b_    = nullptr;

  // DiT::input_embed (Linear proj + CausalConvPositionEmbedding)
  ggml_tensor *input_proj_w_   = nullptr;         // proj.weight [320, 1024]
  ggml_tensor *input_proj_b_   = nullptr;         // proj.bias   [1024]
  ggml_tensor *cpe_c1_w_       = nullptr;         // conv_pos_embed.conv1.0
  ggml_tensor *cpe_c1_b_       = nullptr;
  ggml_tensor *cpe_c2_w_       = nullptr;         // conv_pos_embed.conv2.0
  ggml_tensor *cpe_c2_b_       = nullptr;

  // DiT::norm_out (AdaLayerNorm_Final): Linear(1024→2048) + LayerNorm
  ggml_tensor *norm_out_lin_w_ = nullptr;
  ggml_tensor *norm_out_lin_b_ = nullptr;

  // DiT::proj_out: Linear(1024→80)
  ggml_tensor *proj_out_w_     = nullptr;
  ggml_tensor *proj_out_b_     = nullptr;

  // Per-DiT-block tensors. Each block exposes:
  //   attn_norm.{linear.{w,b}, norm.{w,b?}}  (AdaLN-Zero, Linear 1024→6144)
  //   attn.{to_q,to_k,to_v,to_out.0}.{w,b}    (Linear 1024→1024; biases F32)
  //   ff_norm.{w?,b?}                         (LayerNorm)
  //   ff.{ff.0.0,ff.2}.{w,b}                  (1024→2048→1024 GELU-exact)
  struct DiTBlock {
    ggml_tensor *adaLN_w = nullptr;   // Linear(1024 → 6 × 1024 = 6144)
    ggml_tensor *adaLN_b = nullptr;
    ggml_tensor *norm_w  = nullptr;   // LayerNorm.weight (may be null)
    ggml_tensor *norm_b  = nullptr;
    ggml_tensor *to_q_w  = nullptr, *to_q_b = nullptr;
    ggml_tensor *to_k_w  = nullptr, *to_k_b = nullptr;
    ggml_tensor *to_v_w  = nullptr, *to_v_b = nullptr;
    ggml_tensor *to_out_w = nullptr, *to_out_b = nullptr;
    ggml_tensor *ff_norm_w = nullptr, *ff_norm_b = nullptr;
    ggml_tensor *ff0_w  = nullptr, *ff0_b = nullptr;
    ggml_tensor *ff2_w  = nullptr, *ff2_b = nullptr;
  };
  std::vector<DiTBlock> blocks_;

  // -------------------------------------------------------------------------
  // Hparams (read from `cosyvoice3.flow.*` KVs)
  // -------------------------------------------------------------------------
  int depth_     = 22;
  int dim_       = 1024;
  int n_heads_   = 16;
  int head_dim_  = 64;
  int ff_dim_    = 2048;
  int mel_dim_   = 80;
  int spk_dim_in_ = 192;
  float cfg_rate_ = 0.7f;

  // Cosine schedule: t_span[0..euler_steps_]; populated in Load()/SetEulerSteps.
  std::vector<float> t_span_;
  int euler_steps_ = 5;

  // Backing buffer for the sinus-embed table (so we own its lifetime).
  ggml_backend_buffer_t time_emb_buf_ = nullptr;
  ggml_context *cctx_owned_ = nullptr;

  // imatrix observer — cv3_sched_compute reads &imatrix_cb_ to arm/disarm.
  std::function<void(struct ggml_tensor *)> imatrix_cb_;

  // -------------------------------------------------------------------------
  // Internal helpers
  // -------------------------------------------------------------------------
  bool LoadTensors(const std::map<std::string, ggml_tensor *> &tensors);
  bool LoadHparams(gguf_context *ctx_gguf);
  bool AllocSinusoidalEmbed(ggml_backend_t backend);

  // Build the encode graph: tokens → (mu [80,T_mel], spks [80], cond [80,T_mel])
  struct EncodeOutputs {
    ggml_tensor *mu;
    ggml_tensor *spks;
    ggml_tensor *cond;
    int64_t      cut_len;     // T_pt_mel — number of prompt mel frames
  };
  EncodeOutputs BuildEncodeGraph(ggml_context *ctx,
                                 ggml_tensor *prompt_token,
                                 ggml_tensor *gen_token,
                                 ggml_tensor *prompt_feat,
                                 ggml_tensor *embedding) const;

  // Build a single Euler step. `cut_len` is non-zero only on the final step.
  // `t_in` (a 1-D length-2 tensor) is created inside and returned as an out
  // parameter so the caller can fill it before computing the graph.
  ggml_tensor *BuildCFMStepGraph(ggml_context *ctx,
                                 ggml_tensor *x_init,
                                 ggml_tensor *mu,
                                 ggml_tensor *spks,
                                 ggml_tensor *cond,
                                 int64_t cut_len,
                                 float t_value,
                                 float dt_value,
                                 ggml_tensor *&position_ids_out) const;

  // One DiT block (mirrors cosyvoice-graph.cpp:399).
  ggml_tensor *BuildDiTBlock(ggml_context *ctx, const DiTBlock &b,
                             ggml_tensor *x, ggml_tensor *time_emb,
                             ggml_tensor *position_ids, int64_t cut_len) const;
};
