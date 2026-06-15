#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward decl — definition in frontend/kokoro_g2p_zh.h.
namespace rs::kokoro_zh { class ZHG2P; }

// Forward decl — definition in frontend/kokoro_g2p_en.h.
namespace rs::kokoro_en { class EnG2P; }

// Forward decl — definition in frontend/wetext_normalizer.h.
namespace rs { class WeTextNormalizer; }

// =====================================================================
// Kokoro / StyleTTS2 (iSTFTNet) TTS — ported from CrispASR src/kokoro.cpp
//
// Forward pass at synthesis time:
//
//   phonemes (IPA, 178-symbol vocab) → token IDs
//     ↓ (pad-wrap: [pad, *raw, pad])
//   text_enc (Embedding → 3× Conv1d k=5 + LN + LeakyReLU → bidir LSTM) → t_enc [L, 512]
//   bert (custom ALBERT, 12 parameter-shared layers)                  → bert_dur [L, 512]
//     ↓
//   ref_s = voice_pack[L-1, 0, :]   split [pred_style 0:128 | dec_style 128:256]
//     ↓
//   ProsodyPredictor: dur_enc (3× alt LSTM/AdaLN) + post-LSTM + dur_proj
//                   + shared LSTM + F0/N AdainResBlk1d stacks
//                                                                      → durations + F0 + N
//     ↓
//   align(t_enc, durations) → en [T_frames, 512]
//     ↓
//   iSTFTNet decoder (encode + 4× decode AdainResBlk1d + asr_res + F0_conv + N_conv
//                   → Generator: HnNSF source + 2× upsample + 6 resblocks averaged
//                   → conv_post → 22 channels → split into mag/phase
//                   → iSTFT (n_fft=20, hop=5, Hann) on CPU)            → 24 kHz audio
//
// Voice packs live in a separate GGUF (arch = "kokoro-voice") containing
// one F32 tensor `voice.pack[max_phon, 1, 256]`. Indexed by phoneme
// length L: ref_s = voice.pack[L-1, 0, :].
// =====================================================================

struct KokoroHParams {
  // Top-level
  uint32_t hidden_dim   = 512;
  uint32_t style_dim    = 128;
  uint32_t max_dur      = 50;
  uint32_t n_token      = 178;
  uint32_t n_mels       = 80;   // unused at synth; kept for completeness
  uint32_t n_layer      = 3;    // duration-encoder depth
  uint32_t text_enc_k   = 5;
  uint32_t sample_rate  = 24000;
  uint32_t vocab_size   = 178;

  // PL-BERT (custom 12-layer parameter-shared ALBERT)
  uint32_t plbert_embd_size = 128;
  uint32_t plbert_hidden    = 768;
  uint32_t plbert_n_layers  = 12;
  uint32_t plbert_n_heads   = 12;
  uint32_t plbert_ff        = 2048;
  uint32_t plbert_max_pos   = 512;
  uint32_t plbert_vocab     = 178;

  // iSTFTNet
  uint32_t istft_init_ch   = 512;
  uint32_t istft_n_fft     = 20;
  uint32_t istft_hop       = 5;
  uint32_t istft_n_dilations = 3;
  std::vector<uint32_t> istft_upsample_rates;        // [10, 6]
  std::vector<uint32_t> istft_upsample_kernel_sizes; // [20, 12]
  std::vector<uint32_t> istft_resblock_kernel_sizes; // [3, 7, 11]
  std::vector<uint32_t> istft_resblock_dilations;    // flat, 3*n_dilations
};

struct KokoroVocab {
  std::vector<std::string> id_to_token;                   // size = vocab_size
  std::unordered_map<std::string, int32_t> token_to_id;
  int32_t pad_id = 0;                                     // "$"
};

// Per-state cache of the loaded voice pack (single F32 tensor). The pack
// is owned by KokoroModel (one pack per LoadVoicePack call); the state
// keeps only the resolved per-utterance style halves.
struct KokoroState : public RSState {
  // Raw phoneme tokens (pre pad-wrap)
  std::vector<int32_t> phoneme_ids;
  // Whether PushText input was already IPA (skip G2P)
  bool phonemes_are_ipa = false;
  // Original text (kept for diagnostics)
  std::string input_text;

  // Per-request style halves (sliced by L from voice pack)
  std::vector<float> ref_s_predictor;  // 128-d
  std::vector<float> ref_s_decoder;    // 128-d

  // Output buffer (24 kHz mono)
  std::vector<float> audio_output;
  int audio_read_cursor = 0;

  KokoroState() = default;
  ~KokoroState() override = default;
};

class KokoroModel : public ISpeechModel {
public:
  KokoroModel();
  ~KokoroModel() override;

  // ISpeechModel
  bool Load(const std::unique_ptr<rs_context_t>& ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;

  // Encode: runs phoneme_ids → BERT → text_enc → predictor → align →
  //         decoder → generator → iSTFT → state.audio_output.
  //         input_frames is ignored for TTS.
  bool Encode(const std::vector<float>& input_frames, RSState& state,
              ggml_backend_sched_t sched) override;

  // Decode: no-op (Encode produces all audio in one shot).
  bool Decode(RSState& state, ggml_backend_sched_t sched) override;

  std::string GetTranscription(RSState& state) override { (void)state; return ""; }
  const RSModelMeta& GetMeta() const override { return meta_; }

  // PushText: parse phonemes from `text`. When the IPA flag is set
  // (via SetPhonemesAreIPA or env RS_KOKORO_PHONEMES=1, or language
  // starts with "ipa"), `text` is tokenised directly. Otherwise
  // PushText returns false (Phase 3 will wire in misaki-zh G2P).
  bool PushText(RSState& state, const char* text,
                const char* language = nullptr,
                const char* instruct = nullptr) override;

  int GetAudioOutput(RSState& state, float** out_data) override;

  // --- Kokoro-specific ---
  bool LoadVoicePack(const char* path);
  void SetLengthScale(float s);
  void SetPhonemesAreIPA(bool b) { force_ipa_input_ = b; }

  // Arm/disarm a per-node observer for activation capture during Encode.
  // Passing an empty std::function disarms (the production path pays no cost
  // beyond a single bool check). Used by rs-imatrix to build an importance
  // matrix for activation-aware quantization.
  void set_imatrix_callback(std::function<void(struct ggml_tensor*)> cb);

private:
  RSModelMeta meta_;
  KokoroHParams hp_;
  KokoroVocab vocab_;

  // Primary weights live in ctx->gguf_data (owned by rs_context_t).
  // We just keep a name→tensor map for fast lookup.
  std::map<std::string, struct ggml_tensor*> tensors_;

  // Voice pack (separate GGUF, separately allocated).
  struct VoicePack {
    std::string name;
    uint32_t max_phonemes = 0;
    uint32_t style_dim = 0;
    struct ggml_tensor* pack = nullptr;    // (max_phon, 1, 256) F32
    struct ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, struct ggml_tensor*> tensors;
  } vp_;
  bool vp_loaded_ = false;

  // Backends. `backend_cpu_` is shared with rs_context; `backend_` is the
  // user-selected backend (may equal CPU). Generator nodes get pinned to
  // CPU via ggml_backend_sched_set_tensor_backend to avoid the known
  // Metal hang on stride-10 ConvTranspose1d.
  ggml_backend_t backend_ = nullptr;
  ggml_backend_t backend_cpu_ = nullptr;
  bool gen_force_metal_ = false;

  // Runtime knobs
  float length_scale_ = 1.0f;
  bool force_ipa_input_ = false;

  // Compute-meta buffers for inline graphs (one per Encode pass).
  std::vector<uint8_t> compute_meta_;
  std::vector<uint8_t> compute_meta_gen_;

  // Tensor lookup helpers
  struct ggml_tensor* try_get(const char* name) const;
  struct ggml_tensor* require(const char* name) const;

  // Tokeniser (greedy UTF-8 longest-match against vocab_.token_to_id).
  std::vector<int32_t> TokenizePhonemes(const char* phonemes) const;

  // Slice voice pack by L (returns false if L-1 OOB or no pack loaded).
  bool ResolveRefS(KokoroState& state, int L_raw) const;

  // Opaque engine handle (CrispASR-port kokoro_context). Implementation
  // lives in kokoro_engine.cpp; the adapter owns it.
  struct Impl;
  std::unique_ptr<Impl> impl_;

  // Lazy-initialised Chinese G2P (misaki[zh] v1.1 port). Built on first
  // language="zh" call via PushText; nullptr until then.
  std::unique_ptr<rs::kokoro_zh::ZHG2P> g2p_zh_;

  // Lazy-initialised English G2P fallback (misaki[en] gold dict). Built on
  // first ZH PushText call alongside g2p_zh_; nullptr if RS_KOKORO_EN=0 or
  // us_gold.bin is missing — Latin segments then fall back to '❓'.
  std::unique_ptr<rs::kokoro_en::EnG2P> g2p_en_;

  // Lazy-initialised WeTextProcessing TN (Chinese). Built on first ZH
  // PushText call; pass-through if FST data is missing.
  std::unique_ptr<rs::WeTextNormalizer> tn_;
};
