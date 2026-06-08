#pragma once

// MVP CTC keyword spotter.
//
// Algorithm (per analysis window):
//   1. Take greedy argmax over [T, V] log-prob matrix.
//   2. CTC-collapse: drop blanks, repeats, and special tokens listed in
//      `ignore_ids`. Keep a parallel array `avg_log_prob_per_token` recording
//      the log-prob at the frame the token first emerged (used for ac_threshold).
//   3. Walk the collapsed id stream through the Aho-Corasick trie. On each
//      IsMatched() report, gate by (trailing-blank-tail >= num_trailing_blanks)
//      and (per-token avg exp-prob >= node->ac_threshold).
//   4. Emit a KWSMatch per spotted keyword and reset the walker to root so the
//      same window can spot multiple keywords / one keyword multiple times.
//
// The decoder is stateless across windows — rs_kws.cpp owns the per-window
// invocation and applies cross-window debounce on top.

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "context_graph.h"

namespace rs {

struct KWSMatch {
  std::string phrase;
  // Average per-token probability (NOT log) over the matched span.
  float avg_prob = 0.0f;
  // Frame index of the FIRST token of the matched span (in collapsed stream).
  // Caller can map this back to time via frame_shift.
  int32_t first_frame = 0;
  int32_t last_frame = 0;
};

struct KWSDecoderConfig {
  // CTC blank id (always 0 for SenseVoice / FunASR).
  int32_t blank_id = 0;
  // After matching, only emit if this many blank frames follow the last
  // non-blank token in the greedy stream. 0 = emit immediately. For sliding-
  // window CTC KWS the trie match is already deterministic, so 0 is a sane
  // default; raise this to suppress mid-word false positives.
  int32_t num_trailing_blanks = 0;
  // Frame indices (NOT token ids) to skip at the start of the window.
  // SenseVoice prepends 4 prompt frames (LID/SER/AED/ITN).
  int32_t skip_prefix_frames = 4;

  // Beam-search mode (V2). beam_size <= 1 → MVP greedy decoder.
  // beam_size > 1 → CTC prefix beam search with per-hyp ContextGraph state,
  // boost scores from the trie folded into beam log-prob, and tokens in
  // punct_ids / ignore_ids masked out at each frame.
  int32_t beam_size = 1;
  // Per-frame logit candidates considered (beam fan-out cap). Top-K out of
  // |V| non-masked tokens; defaults to 8 which is plenty for short keywords.
  int32_t top_k_per_frame = 8;
};

class KWSDecoder {
public:
  KWSDecoder(ContextGraphPtr graph, KWSDecoderConfig cfg,
             std::unordered_set<int32_t> ignore_ids,
             std::unordered_set<int32_t> punct_ids = {})
      : graph_(std::move(graph)),
        cfg_(cfg),
        ignore_ids_(std::move(ignore_ids)),
        punct_ids_(std::move(punct_ids)) {}

  // Process one window of greedy-CTC-input. `log_probs` is row-major [T, V].
  // Returns all keyword matches found in this window.
  std::vector<KWSMatch> Decode(const float *log_probs, int T, int V) const;

private:
  std::vector<KWSMatch> DecodeGreedy(const float *log_probs, int T,
                                     int V) const;
  std::vector<KWSMatch> DecodeBeam(const float *log_probs, int T, int V) const;

  ContextGraphPtr graph_;
  KWSDecoderConfig cfg_;
  std::unordered_set<int32_t> ignore_ids_;
  std::unordered_set<int32_t> punct_ids_;
};

} // namespace rs
