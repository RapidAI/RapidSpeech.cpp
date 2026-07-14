#pragma once

#include "frontend/audio_processor.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <memory>
#include <string>
#include <vector>

// Internal State Base Class (Used for KV Cache or RNN States)
struct RSState {
  virtual ~RSState() = default;
};

struct rs_context_t;

// Model Metadata
struct RSModelMeta {
  std::string arch_name;
  int audio_sample_rate = 16000;
  int n_mels = 80;
  int vocab_size = 0;
  // When true, RSProcessor skips its built-in fbank pipeline and passes raw
  // PCM directly to model->Encode() as `input_frames`. Models such as
  // Qwen3-ASR run their own Whisper-style mel extractor instead of the
  // shared Kaldi-style AudioProcessor.
  bool use_external_frontend = false;

  // Analysis window for the shared AudioProcessor frontend. Defaults to
  // Hamming (SenseVoice, FunASR-Nano). Kaldi-style frontends (FireRedVAD,
  // MiMoASR) should set this to WindowType::POVEY so the periodic Hamming
  // doesn't bias the fbank away from the training distribution.
  WindowType window_type = WindowType::HAMMING;
};

// --- Core Model Interface ---
class ISpeechModel {
public:
  virtual ~ISpeechModel() = default;

  // 1. Loading Phase: Parse GGUF, load weights to ggml_backend
  virtual bool Load(const std::unique_ptr<rs_context_t> &ctx,
                    ggml_backend_t backend) = 0;

  // 2. State Creation: Create an independent state for each request
  virtual std::shared_ptr<RSState> CreateState() = 0;

  // 3. Encode/Preprocess: e.g., Audio -> Encoder Hidden States
  // Updated: Pass the backend scheduler to handle memory allocation and compute
  virtual bool Encode(const std::vector<float> &input_frames, RSState &state,
                      ggml_backend_sched_t sched) = 0;

  // 4. Decode/Generate: e.g., Hidden States -> Text Tokens
  virtual bool Decode(RSState &state, ggml_backend_sched_t sched) = 0;

  virtual std::string GetTranscription(RSState &state) = 0;

  // 2-pass support: runtime control of LLM rescoring (FunASRNano only).
  // Default no-ops — only FunASRNanoModel overrides these.
  virtual void SetUseLLM(bool use) { (void)use; }
  virtual bool SupportsTwoPass() const { return false; }

  // CTC pre-check before LLM decode: runs a lightweight CTC pass to
  // detect silence and skip expensive LLM decoding when no speech is
  // present.  Reduces hallucination on silence but adds a small latency
  // overhead.  Disabled by default.  (FunASRNano only.)
  virtual void SetCTCPrecheck(bool enable) { (void)enable; }

  // Set the user input prompt for the LLM decoder (FunASRNano only).
  virtual void SetUserInputPrompt(const std::string &prompt) { (void)prompt; }

  // Set hotwords for RAG contextual biasing (FunASRNano only). The model
  // retrieves the entries matching the CTC hypothesis and injects them into the
  // LLM prompt. An empty list disables biasing.
  virtual void SetHotwords(const std::vector<std::string> &hotwords) {
    (void)hotwords;
  }

  // --- TTS interface (default no-ops; overridden by TTS models) ---

  virtual bool PushText(RSState &state, const char *text,
                        const char *language = nullptr,
                        const char *instruct = nullptr) {
    (void)state; (void)text; (void)language; (void)instruct; return false;
  }

  virtual bool PushReferenceAudio(RSState &state, const float *samples,
                                  int n_samples, int sample_rate,
                                  ggml_backend_sched_t sched) {
    (void)state; (void)samples; (void)n_samples; (void)sample_rate;
    (void)sched; return false;
  }

  virtual bool PushReferenceText(RSState &state, const char *ref_text) {
    (void)state; (void)ref_text; return false;
  }

  virtual int GetAudioOutput(RSState &state, float **out_data) {
    (void)state; (void)out_data; return 0;
  }

  // Independent emotion control (IndexTTS-2). Default no-ops; only emotion-aware
  // TTS models override these.
  //
  // PushEmotionAudio: separate emotion reference audio (distinct from the
  // speaker/timbre reference pushed via PushReferenceAudio).
  virtual bool PushEmotionAudio(RSState &state, const float *samples,
                                int n_samples, int sample_rate,
                                ggml_backend_sched_t sched) {
    (void)state; (void)samples; (void)n_samples; (void)sample_rate;
    (void)sched; return false;
  }

  // SetEmotionControl: select the emotion mode and its parameters.
  //   mode:       0=from speaker, 1=from emo audio, 2=from vector, 3=from text
  //   emo_alpha:  blend weight (emo_weight); 0..1
  //   vec8:       8-d emotion vector [happy, angry, sad, afraid, disgusted,
  //               melancholic, surprised, calm], or nullptr
  //   use_random: random prototype selection in the vector path
  //   apply_bias: apply the bias de-emphasis to vec8 (mode 2)
  //   emo_text:   emotion description text (mode 3), or nullptr
  virtual void SetEmotionControl(RSState &state, int mode, float emo_alpha,
                                 const float *vec8, bool use_random,
                                 bool apply_bias, const char *emo_text) {
    (void)state; (void)mode; (void)emo_alpha; (void)vec8;
    (void)use_random; (void)apply_bias; (void)emo_text;
  }

  // Set number of MaskGIT diffusion steps (OmniVoice TTS only). Default 32.
  virtual void SetDiffusionSteps(int n_steps) { (void)n_steps; }

  // Set RNG seed for sampling-based TTS models. Default no-op; CosyVoice3LM
  // overrides this to seed its RAS sampler. Called by RSProcessor before
  // CreateState() so the new seed propagates into the per-request state.
  virtual void SetSeed(uint64_t seed) { (void)seed; }

  // Get metadata
  virtual const RSModelMeta &GetMeta() const = 0;
};