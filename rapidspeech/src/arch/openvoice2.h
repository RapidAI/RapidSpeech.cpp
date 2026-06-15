#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "frontend/text_frontend.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

// Forward-declare BertModel so this header doesn't pull in ggml.h.
namespace rapidspeech { class BertModel; }

// Forward decl — definition in frontend/wetext_normalizer.h.
namespace rs { class WeTextNormalizer; }

// =====================================================================
// OpenVoice2 TTS Model (MeloTTS/VITS base + Tone Color Converter)
//
// Architecture:
//   Text → TextFrontend → PhonemeIDs → TextEncoder → DurationPredictor
//     → FlowDecoder (chunked) → HiFi-GAN Vocoder → Audio
//
// Optional voice cloning:
//   RefAudio → ToneColorEncoder → StyleEmbedding → inject into decoder
//
// Streaming: FlowDecoder outputs mel chunks, vocoder converts per-chunk.
// =====================================================================

struct OpenVoice2HParams {
  int32_t hidden_channels = 192;
  int32_t inter_channels  = 192;
  int32_t filter_channels = 768;
  int32_t n_heads         = 2;
  int32_t n_layers        = 6;
  int32_t n_flow_layers   = 4;
  int32_t vocab_size      = 256;
  int32_t sample_rate     = 22050;
  int32_t hop_length      = 256;
  int32_t n_fft           = 1024;
  int32_t n_mels          = 80;
  int32_t num_tones       = 11;
  int32_t num_languages   = 4;
  // Streaming chunk size in mel frames (0 = non-streaming)
  int32_t chunk_mel_frames = 0;  // 0 = non-streaming (whole utterance per vocoder call)
  // HiFiGAN upsample rates (empty = infer from kernel sizes)
  std::vector<int32_t> upsample_rates;
};

// --- OpenVoice2 State ---
struct OpenVoice2State : public RSState {
  // Text frontend output
  std::vector<int32_t> phoneme_ids;
  std::string language;

  // Encoder output (hidden states for all phonemes)
  std::vector<float> encoder_hidden;  // [hidden_channels, T_text]
  int encoder_T = 0;

  // Duration predictor output → mel frame count per phoneme
  std::vector<int32_t> durations;
  int total_mel_frames = 0;

  // Flow decoder output (full mel spectrogram, generated incrementally)
  std::vector<float> mel_spectrogram;  // [n_mels, total_mel_frames]

  // Streaming state
  int mel_chunk_cursor = 0;  // next mel frame to vocoder

  // Audio output buffer (accumulated from vocoder chunks)
  std::vector<float> audio_output;
  int audio_read_cursor = 0;

  // Per-phoneme tone IDs for tone embedding lookup
  std::vector<int32_t> tone_ids;
  // Per-phoneme language IDs (MeloTTS uses 3 for real phonemes, 0 for intersperse blanks)
  std::vector<int32_t> lang_ids;
  int language_id = 0;
  int speaker_id = 1;  // spk2id: {'ZH': 1}

  // Optional BERT features (real or precomputed). Layout: [T_phoneme, 1024] / [T_phoneme, 768].
  // If empty, zeros are fed to bert_proj / ja_bert_proj (matches MeloTTS when BERT disabled).
  // For ZH the model uses ja_bert (multilingual BERT projected via ja_bert_proj).
  std::vector<float> bert_features;     // [T_phoneme, 1024]
  std::vector<float> ja_bert_features;  // [T_phoneme, 768]

  // Prior distribution from text encoder proj (per-phoneme)
  std::vector<float> m_p;    // mean: [hidden, T_text]
  std::vector<float> logs_p; // log-scale: [hidden, T_text]

  // Sampled prior z_p = m_p + noise * exp(logs_p) * noise_scale,
  // expanded to mel frames: [hidden, T_mel]
  std::vector<float> z_p_expanded;

  // Tone color embedding (optional, from reference audio)
  std::vector<float> tone_embedding;
  bool has_tone_embedding = false;

  OpenVoice2State() {}
  ~OpenVoice2State() override = default;
};

// --- OpenVoice2 Weight Structures ---

struct OpenVoice2Weights {
  // Text Encoder (Transformer)
  std::map<std::string, struct ggml_tensor*> text_encoder;

  // Duration Predictor
  std::map<std::string, struct ggml_tensor*> duration_predictor;

  // Flow Decoder
  std::map<std::string, struct ggml_tensor*> flow_decoder;

  // HiFi-GAN Vocoder
  std::map<std::string, struct ggml_tensor*> vocoder;

  // Posterior Encoder (needed for VITS training path, optional at inference)
  std::map<std::string, struct ggml_tensor*> posterior_encoder;

  // Embedding tables
  std::map<std::string, struct ggml_tensor*> embeddings;
};

// --- Tone Color Converter Weights (separate GGUF) ---
struct ToneConverterWeights {
  std::map<std::string, struct ggml_tensor*> all_tensors;
  bool loaded = false;
};

// =====================================================================
// OpenVoice2Model class
// =====================================================================

class OpenVoice2Model : public ISpeechModel {
public:
  OpenVoice2Model();
  ~OpenVoice2Model() override;

  // ISpeechModel interface
  bool Load(const std::unique_ptr<rs_context_t>& ctx, ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;

  // Encode: for TTS, this runs TextEncoder + DurationPredictor + FlowDecoder
  // input_frames is unused for TTS (text is pushed via PushText)
  bool Encode(const std::vector<float>& input_frames, RSState& state,
              ggml_backend_sched_t sched) override;

  // Decode: runs vocoder on next mel chunk, returns true if audio available
  bool Decode(RSState& state, ggml_backend_sched_t sched) override;

  std::string GetTranscription(RSState& state) override { (void)state; return ""; }
  const RSModelMeta& GetMeta() const override { return meta_; }

  // --- TTS-specific methods ---

  /// Push text for synthesis. Must be called before Encode.
  bool PushText(RSState& state, const char* text, const char* language = "zh",
                const char* instruct = nullptr) override;

  /// Push reference audio for voice cloning (optional).
  bool PushReferenceAudio(RSState& state, const float* samples, int n_samples,
                          int sample_rate, ggml_backend_sched_t sched);

  /// Load tone color converter from a separate GGUF file.
  bool LoadConverter(const char* converter_path, ggml_backend_t backend);

  /// Load both BERT models for prosody conditioning. Either path may be null
  /// (the corresponding BERT branch then feeds zeros, matching MeloTTS when
  /// BERT is disabled).
  bool LoadBertModels(const char* zh_bert_path, const char* mbert_path,
                       bool use_gpu = false);

  /// Get audio output chunk. Returns number of samples, 0 if no more.
  int GetAudioOutput(RSState& state, float** out_data);

private:
  RSModelMeta meta_;
  OpenVoice2HParams hparams_;
  OpenVoice2Weights weights_;
  ToneConverterWeights converter_weights_;
  TextFrontend text_frontend_;

  // Optional BERT prosody models. zh_bert_ is 1024-dim, mbert_ is 768-dim;
  // either or both may be null (corresponding branch then feeds zeros).
  std::unique_ptr<rapidspeech::BertModel> zh_bert_;
  std::unique_ptr<rapidspeech::BertModel> mbert_;

  // Lazy-initialised WeTextProcessing TN (Chinese). Built on first ZH
  // PushText call; pass-through if FST data is missing.
  std::unique_ptr<rs::WeTextNormalizer> tn_;

  bool MapTensors(std::map<std::string, struct ggml_tensor*>& all_tensors);

  // Sub-graph builders
  bool RunTextEncoder(OpenVoice2State& state, ggml_backend_sched_t sched);
  bool RunDurationPredictor(OpenVoice2State& state, ggml_backend_sched_t sched);
  bool RunFlowDecoder(OpenVoice2State& state, ggml_backend_sched_t sched);
  bool RunVocoder(OpenVoice2State& state, ggml_backend_sched_t sched,
                  int mel_start, int mel_len);
  bool RunToneColorEncoder(OpenVoice2State& state, const std::vector<float>& mel,
                           ggml_backend_sched_t sched);
};
