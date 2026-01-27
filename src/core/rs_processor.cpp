#include "core/rs_processor.h"
#include "utils/rs_log.h"
#include <iostream>
#include <algorithm>

RSProcessor::RSProcessor(std::shared_ptr<ISpeechModel> model, ggml_backend_sched_t sched)
    : model_(model), sched_(sched) {

  // 1. Configure the Audio Processor based on model metadata
  STFTConfig config;
  if (model_) {
    const auto& meta = model_->GetMeta();
    config.sample_rate = meta.audio_sample_rate;
    config.n_mels = meta.n_mels;

    // SenseVoice specific frontend defaults
    config.use_lfr = true;
    config.lfr_m = 7;
    config.lfr_n = 6;
  }

  audio_proc_ = std::make_unique<AudioProcessor>(config);

  // 2. Initialize the model-specific state (context for inference)
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
    RS_LOG_ERR("Processor not fully initialized.");
    return -1;
  }

  // Wait until we have enough data for a meaningful inference
  if (audio_buffer_.Size() < static_cast<size_t>(chunk_size_samples_)) {
    return 0;
  }

  // 1. Fetch chunk from circular buffer
  std::vector<float> pcm_chunk = audio_buffer_.Pop(chunk_size_samples_);

  // 2. Feature Extraction (PCM -> Fbank -> LFR -> CMVN)
  std::vector<float> features;
  audio_proc_->Compute(pcm_chunk, features);

  if (features.empty()) {
    return 0;
  }

  // 3. Model Encoding (Audio Features -> Hidden States)
  if (!model_->Encode(features, *state_, sched_)) {
    RS_LOG_ERR("Model encoding failed.");
    return -1;
  }

  // 4. Model Decoding (Hidden States -> Token IDs)
  if (!model_->Decode(*state_, sched_)) {
    RS_LOG_ERR("Model decoding failed.");
    return -1;
  }

  // 5. Post-process Token IDs to Text
  // We need to access the derived state to get the results
  // For SenseVoice, results are in state->ids
  // Note: In a production version, we would use a more generic state access
  // but for now, we assume SenseVoice-like ID container.

  // This is a simplified token-to-text conversion with CTC deduplication
  // In a real project, this logic would likely live in a 'Tokenizer' class.

  // Since ISpeechModel doesn't expose vocab directly via the interface
  // (except in metadata), we cast for now or ideally add it to the interface.
  // Assuming SenseVoiceModel's vocab is accessible or using a placeholder:

  // For now, let's represent the logic:
  // for (int id : sv_state->ids) {
  //    if (id != 0 && id != last_token_id_) {
  //        text_accumulator_ += model_vocab[id];
  //    }
  //    last_token_id_ = id;
  // }

  return 1;
}

std::string RSProcessor::GetTextResult() {
  return text_accumulator_;
}

// --- CircularBuffer Implementation ---

void CircularBuffer::Push(const float* data, size_t size) {
  if (data && size > 0) {
    buffer_.insert(buffer_.end(), data, data + size);
  }
}

std::vector<float> CircularBuffer::Pop(size_t size) {
  if (buffer_.size() < size) return {};

  std::vector<float> result;
  result.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    result.push_back(buffer_.front());
    buffer_.pop_front();
  }
  return result;
}

size_t CircularBuffer::Size() const {
  return buffer_.size();
}