#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "frontend/whisper_mel.h"
#include "llm_graph.h"
#include "llm_model.h"
#include "qwen3.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

/**
 * Hyperparameters for MOSS-Transcribe-Diarize 0.9B.
 *
 * Encoder side is a stock HF WhisperEncoder (Whisper-Medium config): two
 * Conv1d stem layers (k=3; stride 1 then 2), learned absolute position
 * embeddings over 1500 frames, a Pre-LN bidirectional transformer with
 * GELU(erf) FFNs, and a final LayerNorm. Audio is processed in independent
 * 30 s chunks exactly like HF WhisperFeatureExtractor.
 *
 * Bridge: 4x temporal merge (T,1024)->(T/4,4096) followed by the VQAdaptor
 * MLP Linear(4096->1024) -> SiLU -> Linear(1024->1024) -> LayerNorm.
 *
 * LLM side is Qwen3-0.6B with tied word embeddings (reuses llm_build_qwen3;
 * the lm_head aliases embed_tokens inside llm_model::map_tensors_qwen3).
 */
struct MossTranscribeHParams {
  // Audio frontend (WhisperFeatureExtractor)
  int32_t n_mels        = 80;
  int32_t n_fft         = 400;
  int32_t hop_length    = 160;
  int32_t sample_rate   = 16000;
  int32_t chunk_frames  = 3000;   // mel frames per 30 s chunk

  // Encoder transformer
  int32_t enc_d_model   = 1024;
  int32_t enc_n_head    = 16;
  int32_t enc_n_layer   = 24;
  int32_t enc_ffn_dim   = 4096;
  int32_t enc_max_pos   = 1500;
  float   enc_norm_eps  = 1e-5f;

  // Bridge (time merge + VQAdaptor)
  int32_t merge_size    = 4;
  int32_t adaptor_in    = 4096;
  float   adaptor_norm_eps = 1e-6f;

  // Prompt splicing
  int32_t audio_pad_id   = 151671;
  int32_t audio_start_id = 151669;
  int32_t audio_end_id   = 151670;
  bool    time_marker_enable = true;
  int32_t time_marker_every_seconds = 5;
  float   audio_tokens_per_second   = 12.5f;

  // LLM (Qwen3-0.6B)
  int32_t n_llm_layer   = 0;
  int32_t n_llm_embd    = 0;
};

/**
 * Encoder + adaptor weight bundle. All `ggml_tensor *` are non-owning views
 * into the GGUF context.
 */
struct MossTranscribeWeights {
  // Conv1d stem: conv0 (n_mels->d, s1), conv1 (d->d, s2); both k=3 p=1.
  struct ggml_tensor *conv1d_w[2] = {nullptr, nullptr};
  struct ggml_tensor *conv1d_b[2] = {nullptr, nullptr};

  // Learned position embedding table [d_model, 1500]
  struct ggml_tensor *position_embd = nullptr;

  struct EncLayer {
    struct ggml_tensor *attn_norm_w = nullptr, *attn_norm_b = nullptr;
    struct ggml_tensor *q_w = nullptr, *q_b = nullptr;
    struct ggml_tensor *k_w = nullptr;              // Whisper: no k bias
    struct ggml_tensor *v_w = nullptr, *v_b = nullptr;
    struct ggml_tensor *o_w = nullptr, *o_b = nullptr;
    struct ggml_tensor *ffn_norm_w = nullptr, *ffn_norm_b = nullptr;
    struct ggml_tensor *ffn_up_w = nullptr, *ffn_up_b = nullptr;
    struct ggml_tensor *ffn_dn_w = nullptr, *ffn_dn_b = nullptr;
  };
  std::vector<EncLayer> layers;

  // Final encoder LayerNorm
  struct ggml_tensor *post_norm_w = nullptr;
  struct ggml_tensor *post_norm_b = nullptr;

  // VQAdaptor: Linear -> SiLU -> Linear -> LayerNorm
  struct ggml_tensor *mm_1_w = nullptr, *mm_1_b = nullptr;
  struct ggml_tensor *mm_2_w = nullptr, *mm_2_b = nullptr;
  struct ggml_tensor *mm_norm_w = nullptr, *mm_norm_b = nullptr;
};

/**
 * MOSS-Transcribe-Diarize joint transcription + speaker diarization model.
 *
 * Encode():
 *   - `input_frames` is raw mono PCM @ 16 kHz (use_external_frontend).
 *   - Splits audio into 30 s chunks; each chunk is padded to 30 s, run
 *     through the Whisper mel extractor and one encoder graph, then the
 *     valid frames (4 * token_len) are merged 4x and pushed through the
 *     VQAdaptor — yielding 12.5 LLM-dim audio embeddings per second.
 *
 * Decode():
 *   - Builds the ChatML prompt with the audio span (audio_pad tokens with
 *     digit time anchors every N seconds), scatters the audio embeddings
 *     into the pad positions, prefidxes the Qwen3 LLM and runs greedy
 *     decode until <|im_end|>.
 *   - Output text is the compact `[start][Sxx]text[end]` transcript.
 */
class MossTranscribeModel : public ISpeechModel {
public:
  MossTranscribeModel();
  ~MossTranscribeModel() override;

  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override;
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;
  std::string GetTranscription(RSState &state) override;

  const RSModelMeta &GetMeta() const override { return meta_; }

  void SetUseLLM(bool use) override { (void)use; /* always uses LLM */ }
  bool SupportsTwoPass() const override { return false; }
  void SetCTCPrecheck(bool enable) override { (void)enable; /* no CTC */ }
  void SetUserInputPrompt(const std::string &prompt) override;

private:
  RSModelMeta             meta_;
  MossTranscribeHParams   hparams_;
  MossTranscribeWeights   weights_;
  std::unique_ptr<WhisperMelExtractor> mel_;

  // LLM components
  llm_model_ptr                    llm_model_;
  std::unique_ptr<llm_build_qwen3> llm_graph_builder_;

  // Persistent host KV mirror (mirrors Qwen3ASR's optimised cache).
  std::vector<std::vector<float>>  host_kv_cache_k_;
  std::vector<std::vector<float>>  host_kv_cache_v_;
  uint32_t                         n_cached_tokens_ = 0;
  int                              max_decode_tokens_ = 2048;

  std::string user_input_prompt_;
  int32_t digit_token_ids_[10] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  bool MapEncoderTensors(std::map<std::string, struct ggml_tensor *> &tensors);
  bool LoadLLM(struct gguf_context *ctx_gguf,
               std::map<std::string, struct ggml_tensor *> &tensors,
               ggml_backend_t backend);

  // One 30 s chunk: mel -> encoder -> valid-slice -> merge -> adaptor.
  // Appends `token_len` LLM-dim embedding columns to state.audio_embeds.
  bool RunEncoderChunk(const std::vector<float> &mel_features,
                       int token_len, RSState &state,
                       ggml_backend_sched_t sched);

  // Build the audio span token ids (audio_pad + digit time anchors) exactly
  // like MossTranscribeDiarizeProcessor._audio_span_ids.
  std::vector<int32_t> BuildAudioSpanIds(int n_audio_tokens) const;

  bool DecodeWithLLM(RSState &state, ggml_backend_sched_t sched);
};

/**
 * Per-request state: audio embeddings from the encoder (n_embd fast axis,
 * one column per audio token), decoded token stream and final text.
 */
struct MossTranscribeState : public RSState {
  std::vector<float>   audio_embeds; // [n_embd * T_audio], n_embd fast
  int32_t              T_audio = 0;
  int32_t              n_embd  = 0;
  std::vector<int32_t> token_ids;
  std::vector<std::string> tokens;
};
