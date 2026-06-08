#pragma once

// rs_kws: high-level streaming keyword spotter built on top of SenseVoice +
// ContextGraph + KWSDecoder. Owns:
//   - a raw-PCM ring buffer
//   - the SenseVoice encoder/CTC pipeline (driven directly, NOT via
//     RSProcessor — KWS needs raw [T,V] log_probs, not text)
//   - sliding-window timer that fires every `hop_ms` once the buffer has
//     `window_ms` of audio
//   - per-keyword debounce so the same phrase is not reported multiple times
//     in adjacent overlapping windows
//
// The caller drives this synchronously by pushing PCM chunks and calling
// Poll(); each Poll() may emit zero or more KWSHit callbacks.

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "arch/context_graph.h"
#include "arch/kws_decoder.h"
#include "arch/sensevoice.h"
#include "frontend/audio_processor.h"
#include "ggml-backend.h"
#include "rapidspeech.h" // RS_API

namespace rs {

struct KWSHit {
  std::string phrase;
  float avg_prob = 0.0f;
  // Seconds since the first audio sample was pushed.
  double time_s = 0.0;
};

struct RSKwsConfig {
  int sample_rate = 16000;
  // Analysis window in milliseconds. 1600 = 1.6 s — covers 4-character
  // Chinese wake words and most English wake phrases.
  int window_ms = 1600;
  // Hop in milliseconds. 200 ms gives 8x redundancy per window which combined
  // with per-keyword debounce yields stable triggers.
  int hop_ms = 200;
  // Suppress the same phrase for this many ms after a hit. Should be at least
  // (window_ms - hop_ms) so adjacent overlapping windows don't double-trigger.
  int debounce_ms = 1500;

  KWSDecoderConfig decoder;
};

using KWSCallback = std::function<void(const KWSHit &)>;

class RS_API RSKws {
public:
  RSKws(std::shared_ptr<SenseVoiceModel> model, ggml_backend_sched_t sched,
        ContextGraphPtr graph,
        std::unordered_set<int32_t> ignore_ids,
        std::unordered_set<int32_t> punct_ids,
        RSKwsConfig cfg);
  ~RSKws();

  // Append PCM (mono float32 in [-1, 1]) at the configured sample rate.
  void PushAudio(const float *pcm, size_t n);

  // Run as many windows as the buffer supports. Each spotted keyword fires
  // `cb`. Returns the number of windows processed.
  int Poll(const KWSCallback &cb);

  // Reset buffer and debounce state.
  void Reset();

private:
  std::shared_ptr<SenseVoiceModel> model_;
  ggml_backend_sched_t sched_;
  std::shared_ptr<RSState> state_;
  std::unique_ptr<AudioProcessor> audio_proc_;

  ContextGraphPtr graph_;
  std::unique_ptr<KWSDecoder> decoder_;
  RSKwsConfig cfg_;

  std::deque<float> ring_;
  // Number of samples consumed from the head of the ring (since session start).
  size_t consumed_samples_ = 0;

  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      last_hit_;

  int window_samples_ = 0;
  int hop_samples_ = 0;
};

} // namespace rs
