// kokoro_zh_tone_sandhi.h — C++ port of misaki/tone_sandhi.py.
//
// All character indexing is by codepoint (UTF-8 character), not byte.
// Receives the shared Jieba instance + the PypinyinLite tables via
// constructor so it can replicate jieba.cut_for_search and lazy_pinyin
// (FINALS_TONE3) lookups internally.

#pragma once

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Forward declarations to keep this header free of cppjieba.
namespace cppjieba { class Jieba; }

namespace rs::kokoro_zh {

class PypinyinLite;

using WordPos    = std::pair<std::string, std::string>;
using WordPosVec = std::vector<WordPos>;

class ToneSandhi {
public:
    ToneSandhi();

    void SetJieba(cppjieba::Jieba* j) { jieba_ = j; }
    void SetPinyin(const PypinyinLite* py) { py_ = py; }

    // Pre-merge passes: _merge_bu, _merge_yi, _merge_reduplication,
    // _merge_continuous_three_tones, _merge_continuous_three_tones_2,
    // _merge_er.
    WordPosVec PreMergeForModify(const WordPosVec& seg) const;

    // Apply _bu_sandhi, _yi_sandhi, _neural_sandhi, _three_sandhi to finals.
    std::vector<std::string> ModifiedTone(const std::string& word,
                                          const std::string& pos,
                                          std::vector<std::string> finals) const;

private:
    // Pre-merge passes
    WordPosVec merge_bu(const WordPosVec& seg) const;
    WordPosVec merge_yi(const WordPosVec& seg) const;
    WordPosVec merge_reduplication(const WordPosVec& seg) const;
    WordPosVec merge_continuous_three_tones(const WordPosVec& seg) const;
    WordPosVec merge_continuous_three_tones_2(const WordPosVec& seg) const;
    WordPosVec merge_er(const WordPosVec& seg) const;

    // modified_tone helpers
    std::vector<std::string> bu_sandhi(const std::string& word,
                                       std::vector<std::string> finals) const;
    std::vector<std::string> yi_sandhi(const std::string& word,
                                       std::vector<std::string> finals) const;
    std::vector<std::string> neural_sandhi(const std::string& word,
                                           const std::string& pos,
                                           std::vector<std::string> finals) const;
    std::vector<std::string> three_sandhi(const std::string& word,
                                          std::vector<std::string> finals) const;

    // split_word (jieba.cut_for_search) — returns up to two subwords.
    std::vector<std::string> split_word(const std::string& word) const;

    // _all_tone_three(finals) — every entry ends with '3'.
    static bool all_tone_three(const std::vector<std::string>& finals);

    // Returns lazy_pinyin(word, style=FINALS_TONE3, neutral_tone_with_five=True)
    // with the '嗯' -> 'n2' fix.
    std::vector<std::string> finals_tone3_for(const std::string& word) const;

    // The two giant frozensets from tone_sandhi.py.
    std::unordered_set<std::string> must_neural_;
    std::unordered_set<std::string> must_not_neural_;

    // ToneSandhi.punc (Python: "、：，；。？！“”‘’':,;.?!").
    std::unordered_set<std::string> punc_chars_;

    cppjieba::Jieba* jieba_ = nullptr;
    const PypinyinLite* py_ = nullptr;
};

} // namespace rs::kokoro_zh
