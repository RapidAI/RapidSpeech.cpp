#include "core/rs_processor.h"
#include "utils/rs_log.h"
#include "ggml-backend.h"
#include <iostream>
#include <algorithm>

RSProcessor::RSProcessor(std::shared_ptr<ISpeechModel> model, ggml_backend_sched_t sched)
    : model_(model), sched_(sched) {

  STFTConfig config;
  if (model_) {
    const auto& meta = model_->GetMeta();
    config.sample_rate = meta.audio_sample_rate;
    config.n_mels = meta.n_mels;

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

void RSProcessor::PushAudio(const float* data, size_t size) {
  audio_buffer_.Push(data, size);
}

void RSProcessor::SetCMVN(const std::vector<float>& means, const std::vector<float>& vars) {
  if (audio_proc_) {
    audio_proc_->SetCMVN(means, vars);
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
  std::chrono::steady_clock::time_point start = std::chrono::high_resolution_clock::now();
  std::vector<float> features;
  audio_proc_->Compute(pcm_chunk, features);

  if (features.empty()) return 0;

  // --- CRITICAL FIX FOR Encode Graph ---
  // Reset the scheduler to clear memory assignments before building/allocating the Encode graph.
  ggml_backend_sched_reset(sched_);

  // 3. Model Encoding
  if (!model_->Encode(features, *state_, sched_)) {
    RS_LOG_ERR("Model encoding failed.");
    return -1;
  }

  // --- CRITICAL FIX FOR Decode Graph ---
  // Since Decode builds a NEW ggml_cgraph with its own context, we MUST reset the scheduler
  // again to prevent "GGML_ASSERT(!sched->is_alloc)" failure.
  // The previous graph (Encode) has already finished execution, so it's safe to clear.
  ggml_backend_sched_reset(sched_);

  // 4. Model Decoding
  if (!model_->Decode(*state_, sched_)) {
    RS_LOG_ERR("Model decoding failed.");
    return -1;
  }
  std::chrono::steady_clock::time_point end = std::chrono::high_resolution_clock::now();

  text_accumulator_ = model_->GetTranscription(*state_);
  RS_LOG_INFO("RTF is: %f", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6 / pcm_duration);
  return 1;
}

std::string RSProcessor::GetTextResult() {
  return text_accumulator_;
}

// CircularBuffer implementation...
void CircularBuffer::Push(const float* data, size_t size) {
  if (data && size > 0) {
    buffer_.insert(buffer_.end(), data, data + size);
  }
}

std::vector<float> CircularBuffer::Pop(size_t size) {
  if (buffer_.size() < size) return {};
  std::vector<float> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = buffer_.front();
    buffer_.pop_front();
  }
  return result;
}

size_t CircularBuffer::Size() const { return buffer_.size(); }