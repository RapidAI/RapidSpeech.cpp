#pragma once

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

  // Get metadata
  virtual const RSModelMeta &GetMeta() const = 0;
};