#include "core/rs_processor.h"
#include "ggml-backend.h"
#include "utils/rs_log.h"
#include <chrono>
#include <iostream>

RSProcessor::RSProcessor(std::shared_ptr<ISpeechModel> model,
                         ggml_backend_sched_t sched)
    : model_(model), sched_(sched) {

  STFTConfig config;
  if (model_) {
    const auto &meta = model_->GetMeta();
    config.sample_rate = meta.audio_sample_rate;
    config.n_mels = meta.n_mels;
    config.window_type = meta.window_type;

    // SenseVoice specific frontend config
    config.use_lfr = true;
    config.lfr_m = 7;
    config.lfr_n = 6;
  }

  audio_proc_ = std::make_unique<AudioProcessor>(config);
  if (model_) {
    state_ = model_->CreateState();
  }
}

void RSProcessor::PushAudio(const float *data, size_t size) {
  audio_buffer_.Push(data, size);
}

void RSProcessor::SetCMVN(const std::vector<float> &means,
                          const std::vector<float> &vars) {
  if (audio_proc_) {
    audio_proc_->SetCMVN(means, vars);
  }
}

void RSProcessor::SetUserInputPrompt(const std::string &prompt) {
  if (model_) {
    model_->SetUserInputPrompt(prompt);
  }
}

int RSProcessor::Process() {
  if (!model_ || !state_ || !sched_) {
    RS_LOG_ERR("Processor error: Missing model, state, or scheduler.");
    return -1;
  }

  if (audio_buffer_.Size() < static_cast<size_t>(chunk_size_samples_)) {
    return 0;
  }

  std::vector<float> pcm_chunk = audio_buffer_.Pop(audio_buffer_.Size());
  float pcm_duration = pcm_chunk.size() / model_->GetMeta().audio_sample_rate;
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point fist_start =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::vector<float> features;
  if (model_->GetMeta().use_external_frontend) {
    // Architectures like Qwen3-ASR own their mel pipeline and want raw PCM.
    features = std::move(pcm_chunk);
  } else {
    audio_proc_->Compute(pcm_chunk, features);
  }
  end = std::chrono::steady_clock::now();
  RS_LOG_INFO("compute features takes: %f",
              std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                      .count() /
                  1e6);
  if (features.empty())
    return 0;

  // --- CRITICAL FIX FOR Encode Graph ---
  // Reset the scheduler to clear memory assignments before building/allocating
  // the Encode graph.
  ggml_backend_sched_reset(sched_);

  // 3. Model Encoding
  start = std::chrono::steady_clock::now();
  if (!model_->Encode(features, *state_, sched_)) {
    RS_LOG_ERR("Model encoding failed.");
    return -1;
  }
  end = std::chrono::steady_clock::now();
  RS_LOG_INFO("encoder takes: %f",
              std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                      .count() /
                  1e6);

  // --- CRITICAL FIX FOR Decode Graph ---
  // Since Decode builds a NEW ggml_cgraph with its own context, we MUST reset
  // the scheduler again to prevent "GGML_ASSERT(!sched->is_alloc)" failure. The
  // previous graph (Encode) has already finished execution, so it's safe to
  // clear.
  ggml_backend_sched_reset(sched_);

  // 4. Model Decoding
  start = std::chrono::steady_clock::now();
  if (!model_->Decode(*state_, sched_)) {
    RS_LOG_ERR("Model decoding failed.");
    return -1;
  }
  end = std::chrono::steady_clock::now();
  RS_LOG_INFO("decoder takes: %f",
              std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                      .count() /
                  1e6);

  RS_LOG_INFO(
      "RTF is: %f",
      std::chrono::duration_cast<std::chrono::microseconds>(end - fist_start)
              .count() /
          1e6 / pcm_duration);
  return 1;
}

int RSProcessor::DecodeOnly() {
  if (!model_ || !state_ || !sched_) {
    RS_LOG_ERR("DecodeOnly: Missing model, state, or scheduler.");
    return -1;
  }

  if (!model_->SupportsTwoPass()) {
    RS_LOG_ERR("DecodeOnly: model does not support 2-pass rescoring.");
    return -1;
  }

  ggml_backend_sched_reset(sched_);

  auto start = std::chrono::steady_clock::now();
  if (!model_->Decode(*state_, sched_)) {
    RS_LOG_ERR("Model decoding failed in DecodeOnly.");
    return -1;
  }
  auto end = std::chrono::steady_clock::now();
  RS_LOG_INFO("re-decoder takes: %f",
              std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                      .count() /
                  1e6);
  return 1;
}

std::string RSProcessor::GetTextResult() {
  text_accumulator_ = model_->GetTranscription(*state_);
  return text_accumulator_;
}

void RSProcessor::Reset() {
  audio_buffer_ = CircularBuffer();
  text_accumulator_.clear();
  last_token_id_ = -1;
  tts_encoded_ = false;
  tts_decoded_ = false;
  tts_audio_buf_.clear();
  tts_audio_read_pos_ = 0;
  if (model_) {
    state_ = model_->CreateState();
  }
}

// =====================================================================
// TTS methods
// =====================================================================

void RSProcessor::SetTTSParams(const char *instruct, const char *language,
                                int seed) {
  if (instruct) tts_instruct_ = instruct;
  if (language) tts_language_ = language;
  tts_seed_ = seed;
}

void RSProcessor::SetDiffusionSteps(int n_steps) {
  if (model_) model_->SetDiffusionSteps(n_steps);
}

int RSProcessor::PushText(const char *text, const char *language) {
  if (!model_ || !state_) return -1;
  const char *lang = language ? language : tts_language_.c_str();
  return model_->PushText(*state_, text, lang, tts_instruct_.c_str()) ? 0 : -1;
}

int RSProcessor::PushReferenceAudio(const float *samples, int n_samples,
                                     int sample_rate) {
  if (!model_ || !state_ || !sched_) return -1;
  return model_->PushReferenceAudio(*state_, samples, n_samples, sample_rate,
                                    sched_) ? 0 : -1;
}

int RSProcessor::PushReferenceText(const char *ref_text) {
  if (!model_ || !state_) return -1;
  return model_->PushReferenceText(*state_, ref_text) ? 0 : -1;
}

int RSProcessor::ProcessTTS() {
  if (!model_ || !state_ || !sched_) return -1;

  if (!tts_encoded_) {
    ggml_backend_sched_reset(sched_);
    if (!model_->Encode({}, *state_, sched_)) return -1;
    tts_encoded_ = true;
  }

  if (tts_decoded_) {
    return 0;
  }

  ggml_backend_sched_reset(sched_);
  if (!model_->Decode(*state_, sched_)) {
    tts_decoded_ = true;  // No more chunks to decode
    return 0;
  }

  float *chunk_data = nullptr;
  int chunk_n = model_->GetAudioOutput(*state_, &chunk_data);
  if (chunk_n > 0 && chunk_data) {
    tts_audio_buf_.assign(chunk_data, chunk_data + chunk_n);
    tts_audio_read_pos_ = 0;
  }

  return 1;  // More chunks available
}

int RSProcessor::GetAudioOutput(float **out_data) {
  if (tts_audio_buf_.empty() ||
      tts_audio_read_pos_ >= static_cast<int>(tts_audio_buf_.size())) {
    *out_data = nullptr;
    return 0;
  }
  *out_data = tts_audio_buf_.data() + tts_audio_read_pos_;
  int n = static_cast<int>(tts_audio_buf_.size()) - tts_audio_read_pos_;
  tts_audio_read_pos_ = static_cast<int>(tts_audio_buf_.size());
  return n;
}

// CircularBuffer implementation...
void CircularBuffer::Push(const float *data, size_t size) {
  if (data && size > 0) {
    buffer_.insert(buffer_.end(), data, data + size);
  }
}

std::vector<float> CircularBuffer::Pop(size_t size) {
  if (buffer_.size() < size)
    return {};
  std::vector<float> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = buffer_.front();
    buffer_.pop_front();
  }
  return result;
}

size_t CircularBuffer::Size() const { return buffer_.size(); }