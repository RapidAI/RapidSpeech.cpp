#pragma once

// Parse a sherpa-onnx style keywords.txt file and build a ContextGraph
// suitable for the CTC-based KWS decoder.
//
// Format (each non-empty line is one keyword):
//   <tok1> <tok2> ... [:boost] [#threshold] [@phrase]
//
//   - tokens are looked up in SenseVoice's id_to_token map (reverse direction)
//   - :boost  (float)  per-keyword boost score override (only used by V2 beam)
//   - #threshold (float in [0,1]) per-keyword avg-probability gate
//   - @phrase (string) human-readable label reported on match
//
// Lines with unknown tokens are skipped with a warning so a single typo
// doesn't kill the whole keyword set.

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "context_graph.h"
#include "rapidspeech.h" // RS_API

namespace rs {

struct KWSLoaderConfig {
  // Default per-token boost when a line has no `:score`.
  float default_score = 1.0f;
  // Default avg-prob threshold when a line has no `#threshold`.
  // 0.0 means "always accept once the trie matches" (still gated by trailing
  // blanks in the decoder).
  float default_threshold = 0.0f;
};

// Build a ContextGraph from an in-memory keywords blob (newline-separated).
// `id_to_token` is the reverse vocabulary loaded from sensevoice.gguf.
// Returns nullptr if no usable line was parsed.
RS_API ContextGraphPtr LoadKeywordsFromString(
    const std::string &content,
    const std::unordered_map<int, std::string> &id_to_token,
    const KWSLoaderConfig &cfg = {});

// Convenience: read file then call LoadKeywordsFromString().
RS_API ContextGraphPtr LoadKeywordsFromFile(
    const std::string &path,
    const std::unordered_map<int, std::string> &id_to_token,
    const KWSLoaderConfig &cfg = {});

} // namespace rs
