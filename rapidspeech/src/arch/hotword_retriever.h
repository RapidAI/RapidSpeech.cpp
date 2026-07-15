#pragma once

// Hotword RAG retriever for FunASR-Nano (方案 B).
//
// The released Fun-ASR-Nano injects hotwords into the LLM prompt. To scale to a
// large hotword database without stuffing every entry into the prompt (which
// induces LLM hallucination), we retrieve only the entries that plausibly match
// the CTC hypothesis, then inject that subset.
//
// Matching is pinyin (toneless) edit-distance so Chinese homophone / near-
// homophone CTC errors still surface the correct hotword (e.g. CTC "章叁" still
// retrieves the hotword "张三"). Non-Han hotwords fall back to character-level
// matching, which emerges naturally from the same per-codepoint token stream.

#include <memory>
#include <string>
#include <vector>

namespace rs {

namespace kokoro_zh {
class PypinyinLite;
}

class HotwordRetriever {
public:
  HotwordRetriever();
  ~HotwordRetriever();

  // Load the pinyin tables (idempotent). Returns false if the kokoro_zh data
  // directory cannot be found; the retriever then degrades to character-level
  // matching (ASCII stays useful, Han chars compare by raw codepoint).
  bool Load();
  bool IsLoaded() const { return loaded_; }

  // (Re)build the internal index from a list of hotword strings.
  void SetHotwords(const std::vector<std::string> &hotwords);
  bool Empty() const { return entries_.empty(); }
  size_t Size() const { return entries_.size(); }

  // Retrieve hotwords whose token sequence approximately matches some substring
  // of the CTC hypothesis. `tau` in [0,1] is the max normalized edit distance
  // (edits / hotword_len). `topk <= 0` means unlimited. Result is deduped and
  // ordered by ascending distance (best match first).
  std::vector<std::string> Retrieve(const std::string &ctc_text, float tau,
                                    int topk) const;

  // Convert text to the normalized token sequence used for matching (toneless
  // pinyin syllables for Han, lowercased chars for ASCII; punctuation dropped).
  // Exposed for logging / tests.
  std::vector<std::string> ToTokens(const std::string &text) const;

  // Deterministic post-substitution: scan `text` and replace any contiguous
  // run of characters whose toneless pinyin EXACTLY equals a hotword's pinyin
  // with that hotword's written form. Forces the exact (possibly rare-character)
  // hotword spelling even when the LLM emitted a homophone. Longer hotwords win
  // on overlap; matches are non-overlapping. Returns the rewritten text.
  std::string ApplySubstitutions(const std::string &text) const;

private:
  struct Entry {
    std::string text;                // original hotword, injected verbatim
    std::vector<std::string> tokens; // normalized match tokens
  };

  std::unique_ptr<kokoro_zh::PypinyinLite> py_;
  std::vector<Entry> entries_;
  bool loaded_ = false;
  bool load_attempted_ = false;
};

} // namespace rs
