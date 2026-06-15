#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "cosyvoice3_ras.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llm_graph.h"
#include "llm_model.h"
#include "qwen2.h"
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Arm an imatrix per-node observer (when cb is non-empty) on the given sched,
// run graph_compute, disarm. Mirrors the OmniVoice pattern at
// omnivoice.cpp:3056 — per-node firing keeps src1 live during the callback,
// which is required for activation capture because the sched's buffer reuse
// may overwrite src1 with downstream output once compute returns.
inline enum ggml_status cv3_sched_compute(
    ggml_backend_sched_t sched, struct ggml_cgraph *gf,
    std::function<void(struct ggml_tensor *)> *cb) {
  const bool armed = cb && (bool)*cb;
  if (armed) {
    auto trampoline = [](struct ggml_tensor *t, bool ask, void *ud) -> bool {
      if (ask) return true;
      auto *c = static_cast<std::function<void(struct ggml_tensor *)> *>(ud);
      (*c)(t);
      return true;
    };
    ggml_backend_sched_set_eval_callback(sched, trampoline, cb);
  }
  enum ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
  if (armed) {
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
  }
  return status;
}

// Forward declarations to avoid a header explosion.
class CosyVoice3FlowModel;
class CosyVoice3HiFTModel;
class CosyVoice3SpeechTokenizer;
class CAMPPlusModel;

/**
 * CosyVoice3-LLM (text → speech tokens, autoregressive).
 *
 * Phase 2 scope: only the LLM AR head. Flow / HiFT / voices are deferred to a
 * later PR. The model emits int32 speech-token ids in [0, 6561); a sampled id
 * equal to `stop_token_id` (=6561) terminates decoding.
 *
 * Pipeline:
 *   text → BPE tokens → token_embd lookup
 *   → Qwen2-0.5B body (24L, GQA 14/2, head_dim=64, ff=4864, RoPE θ=1e6)
 *   → speech_lm_head (896 → 6761)
 *   → RAS sample (top_k=25, top_p=0.8, win=10, τ_r=0.1)
 *   → speech_embd[id] back into Qwen2 → repeat.
 *
 * Output is exposed via `GetTranscription` as a CSV string of token ids
 * (e.g. "1234,5678,...") — a Phase-2-only carrier; Phase 3 will replace this
 * with a dedicated token channel once flow/HiFT are wired up.
 */
struct CosyVoice3State : public RSState {
  std::vector<int32_t> text_token_ids;
  std::vector<int32_t> speech_token_ids;

  // Instruct2 mode (matches upstream `inference_instruct2`): when non-empty,
  // these BPE ids replace `prompt_text_token_ids` in the LM prefix AND the
  // LM-side prompt_speech_token rows are suppressed (Flow still gets them).
  // Result: timbre still cloned from ref (Flow path intact), prosody/style
  // guided by the instruct text instead of by the ref speech tokens.
  std::vector<int32_t> instruct_text_token_ids;

  // Host-side KV after prefill, indexed [layer][kv_dim * n_tokens]
  std::vector<std::vector<float>> host_kv_k;
  std::vector<std::vector<float>> host_kv_v;
  uint32_t n_cached = 0;

  std::mt19937 rng{0xC05A3};
  ras_params sampler;
  int32_t max_speech_tokens = 1500;

  // Optional debug dump of step-0 logits (full 6761) before sampling. When
  // set the model writes the raw float32 logits buffer here exactly once.
  std::vector<float> *dump_step0_logits = nullptr;

  // ----- Flow + HiFT pipeline state ----------------------------------------
  // Voice conditioning — populated by PushReferenceAudio (or from the baked
  // default voice when no --ref is supplied). prompt_feat is row-major
  // [T_pt_mel, mel_dim=80].
  std::vector<int32_t> prompt_token;
  std::vector<int32_t> prompt_text_token_ids;  // BPE ids of the ref transcript
  std::vector<float>   prompt_feat;
  std::vector<float>   embedding;          // [192] CAMPPlus output

  // Outputs.
  std::vector<float>   mel_output;         // [T_mel, 80]   from Flow
  std::vector<float>   audio_output;       // 24 kHz f32    from HiFT
  bool flow_done = false;
  bool hift_done = false;

  // ----- Chunk-streaming state (rs-tts-online) -----------------------------
  // Upstream `CosyVoice2Model.tts` chunks the LM token stream and runs
  // Flow+HiFT every `token_hop_len` (25 → 100, doubling) tokens, carrying
  // 8 mel frames + 3840 PCM samples of overlap between chunks for a
  // hamming-window crossfade. These fields are zero in the offline path
  // (`Decode`) and only populated during `DecodeStream`.
  struct HiFTStreamCache {
    std::vector<float> mel;     // last mel_cache_len=8 frames, row-major [8 * n_mel]
    std::vector<float> source;  // last source_cache_len=3840 samples (NSF excitation)
    std::vector<float> speech;  // last source_cache_len=3840 samples (for hamming crossfade)
    bool primed = false;        // false before first chunk emitted
  };
  HiFTStreamCache hift_stream;
  int stream_token_offset = 0;  // tokens already emitted to Flow+HiFT
};

class CosyVoice3LMModel : public ISpeechModel {
public:
  CosyVoice3LMModel();
  ~CosyVoice3LMModel() override;

  // ISpeechModel
  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;

  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override {
    (void)input_frames; (void)state; (void)sched; return true;
  }
  // Full pipeline: RunLM (text → speech tokens) → Flow (→ mel) → HiFT (→ wav).
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;

  // Chunk-streaming pipeline used by `rs-tts-online`. The LM AR loop fires
  // `emit` synchronously every time a token chunk (25 → 100, doubling) has
  // accumulated, after running Flow + HiFT over the new tokens with the
  // upstream pre-lookahead / overlap-add carry semantics. The pcm buffer
  // passed to `emit` is owned by `state.audio_output` and remains valid
  // only until the next chunk is produced.
  using AudioChunkCallback =
      std::function<void(const float *pcm, size_t n)>;
  bool DecodeStream(RSState &state, ggml_backend_sched_t sched,
                    AudioChunkCallback emit);

  // Per-token observer fired inside the LM AR loop right after each newly
  // sampled speech token is appended to `state.speech_token_ids`. Returning
  // `false` aborts decoding cleanly (mapped to a synthetic EOS). Used by
  // `DecodeStream` to detect chunk boundaries.
  using LMTokenCallback = std::function<bool(int32_t token)>;
  void set_lm_token_callback(LMTokenCallback cb) {
    lm_token_cb_ = std::move(cb);
  }
  std::string GetTranscription(RSState &state) override;

  bool PushText(RSState &state, const char *text,
                const char *language = nullptr,
                const char *instruct = nullptr) override;
  bool PushReferenceAudio(RSState &state, const float *samples, int n_samples,
                          int sample_rate,
                          ggml_backend_sched_t sched) override;
  bool PushReferenceText(RSState &state, const char *ref_text) override;
  int  GetAudioOutput(RSState &state, float **out_data) override;

  const RSModelMeta &GetMeta() const override { return meta_; }

  // Tunables for the CLI / tests.
  void SetSampler(const ras_params &p) { default_sampler_ = p; }
  void SetMaxSpeechTokens(int n) { max_tokens_ = n; }
  void SetSeed(uint64_t seed) override { seed_ = seed; }
  int32_t StopTokenId() const { return stop_token_id_; }
  int32_t SpeechVocabSize() const { return speech_vocab_; }

  // Activation-aware quantization hook. Setting a non-empty callback arms the
  // sched eval observer on every LM + Flow compute (HiFT is intentionally
  // excluded: its conv kernels have ne[0]=3..16 and fail K-quant alignment
  // anyway). Pass an empty std::function to disarm.
  void set_imatrix_callback(std::function<void(struct ggml_tensor *)> cb);

private:
  // Original LLM AR loop — implemented in cosyvoice3_decode.cpp.
  bool RunLM(CosyVoice3State &state, ggml_backend_sched_t sched);

  // Voice extraction helpers — fill state.{prompt_token, prompt_feat,
  // embedding} either from a runtime reference wav (when --ref is provided
  // and the encoders are available) or from the GGUF-baked default voice.
  bool PrepareVoice(CosyVoice3State &state, ggml_backend_sched_t sched);
  bool TryLoadDefaultVoiceFromGGUF(gguf_context *ctx_gguf,
                                   ggml_context *gguf_data);
  bool TryLoadExternalCampplus(const char *path);
  bool TryLoadExternalTokenizer(const char *path);

  // Voice bake/reuse: a standalone voice GGUF carries the same
  // `cv3.default_voice.*` tensors as the unified model, so a baked voice can
  // replace the speech-tokenizer + CAMPPlus forward passes entirely.
  bool TryLoadVoiceFile(const char *path);
  bool SaveVoiceFile(const CosyVoice3State &state, const char *path);

private:
  RSModelMeta meta_;
  llm_model_ptr llm_model_;
  ggml_backend_t backend_ = nullptr;

  // CosyVoice3-LLM constants from upstream config.
  int32_t speech_vocab_   = 6761;  // speech_embd / speech_lm_head out_features
  int32_t speech_codebook_ = 6561; // valid token range [0, codebook)
  int32_t stop_token_id_  = 6561;  // = codebook (the first non-codebook id)

  ras_params default_sampler_;
  int32_t max_tokens_ = 1500;
  uint64_t seed_ = 0xC05A3ULL;

  // ----- Pipeline sub-models -----------------------------------------------
  // Flow + HiFT live in the same GGUF as the LLM (loaded in Load).
  std::unique_ptr<CosyVoice3FlowModel> flow_;
  std::unique_ptr<CosyVoice3HiFTModel> hift_;
  bool flow_ready_ = false;
  bool hift_ready_ = false;
  bool voice_saved_ = false;  // RS_CV3_SAVE_VOICE_PATH writes at most once

  // Tokenizer + CAMPPlus live in separate GGUFs, loaded lazily via env vars
  // RS_CV3_SPEECH_TOKENIZER_PATH / RS_CV3_CAMPPLUS_PATH.
  std::shared_ptr<CosyVoice3SpeechTokenizer> speech_tokenizer_;
  std::shared_ptr<CAMPPlusModel> campplus_model_;
  bool speech_tokenizer_ready_ = false;
  bool campplus_ready_ = false;

public:
  // Helper used by load_external_gguf() in cosyvoice3.cpp. Exposed because
  // the file-local loader takes a pointer to one of these.
  struct ExternalGguf {
    gguf_context *ctx_gguf  = nullptr;
    ggml_context *ctx_data  = nullptr;
    ggml_backend_buffer_t buf = nullptr;
  };

private:
  ExternalGguf tokenizer_gguf_;
  ExternalGguf campplus_gguf_;

  // ----- Default voice tuple (baked into the unified GGUF when present) ----
  std::vector<int32_t> default_prompt_text_ids_;
  std::vector<int32_t> default_prompt_token_;
  std::vector<float>   default_prompt_feat_;
  std::vector<float>   default_embedding_;

  // Reference-wav text (set by PushReferenceText, consumed by RunLM via
  // text_token_ids prepending in PushText).
  std::string pending_ref_text_;

  std::string ref_text_;

  // imatrix observer — cv3_sched_compute reads &imatrix_cb_ to arm/disarm.
  std::function<void(struct ggml_tensor *)> imatrix_cb_;

  // Per-token observer fired from RunLM inside the AR loop. Used by
  // DecodeStream to chunk the speech-token stream. Empty in the offline path.
  LMTokenCallback lm_token_cb_;
};
