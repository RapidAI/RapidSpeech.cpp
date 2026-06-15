// kokoro_pinyin.h — pypinyin-lite for Kokoro Chinese G2P.
//
// Reproduces pypinyin.lazy_pinyin(word, style=...) for the three styles
// misaki[zh] v1.1 actually uses (TONE3, FINALS_TONE3, INITIALS) from the
// binary tables dumped by scripts/dump_pypinyin_data.py.
//
// Binary format documented at the top of dump_pypinyin_data.py.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rs::kokoro_zh {

// Initials list (longest-first), matching zh_frontend.py:30-34.
//   ['b','p','m','f','d','t','n','l','g','k','h','zh','ch','sh','r','z','c','s','j','q','x','y','w']
// `y` and `w` count as initials for pypinyin's INITIALS style purposes.
extern const std::vector<std::string> kInitialsLongestFirst;

class PypinyinLite {
public:
    PypinyinLite() = default;

    // Load both .bin files from `data_dir` (e.g. rapidspeech/data/kokoro_zh).
    bool Load(const std::string& data_dir);

    bool IsLoaded() const { return loaded_; }

    // Return one TONE3 syllable per UTF-8 codepoint of `word_utf8`.
    // Non-Han codepoints are returned verbatim as single-character strings.
    // Implements longest-phrase-first greedy lookup à la pypinyin.lazy_pinyin.
    std::vector<std::string> LazyPinyinTone3(const std::string& word_utf8) const;

    // Same scan, but returns the FINAL portion of the TONE3 syllable (with
    // tone digit), e.g. "ni3" -> "i3", "shi4" -> "iii4" (after the i/ii/iii
    // discrimination is applied by the caller), "an1" -> "an1".
    // Special-case: '嗯' (U+55EF) → "n2" (zh_frontend.py:101-103).
    std::vector<std::string> LazyPinyinFinalsTone3(const std::string& word_utf8) const;

    // Same scan, but returns just the INITIAL (e.g. "shi4" -> "sh",
    // "an1" -> "", "ke3" -> "k", "yi1" -> "y").
    std::vector<std::string> LazyPinyinInitials(const std::string& word_utf8) const;

private:
    // Walk the input word and produce one TONE3 syllable per Han codepoint
    // (non-Han codepoints pass through as a one-char string).
    void scan_tone3(const std::string& word, std::vector<std::string>& out) const;

    // Split a TONE3 syllable into (initial, final-with-tone-digit).
    // Matches pypinyin's Style.INITIALS / Style.FINALS_TONE3 conventions.
    static void split_initial_final(const std::string& tone3,
                                    std::string& initial,
                                    std::string& final_out);

    bool loaded_ = false;

    // Single-char readings: codepoint -> list of TONE3 readings.
    // We use only the first reading (matching lazy_pinyin's default).
    std::unordered_map<uint32_t, std::vector<std::string>> single_;

    // Phrase readings: utf8 phrase -> list of TONE3 readings (one per char).
    std::unordered_map<std::string, std::vector<std::string>> phrases_;

    uint32_t max_phrase_chars_ = 0;
};

// UTF-8 helpers ------------------------------------------------------------

// Decode the codepoint starting at `s[i]`, advance `i` past the sequence.
// Returns 0xFFFD on malformed input (and still advances by 1 byte).
uint32_t utf8_decode(const std::string& s, size_t& i);

// Encode `cp` to UTF-8 and append to `out`.
void utf8_encode(uint32_t cp, std::string& out);

// Returns the codepoints of `s` (one per char).
std::vector<uint32_t> utf8_codepoints(const std::string& s);

// Return the number of UTF-8 codepoints in `s`.
size_t utf8_char_count(const std::string& s);

// Return UTF-8 substring [char_start, char_end) by character index.
std::string utf8_substr(const std::string& s, size_t char_start, size_t char_end);

// Convenience: get the i-th codepoint as a UTF-8 string (1..4 bytes).
std::string utf8_char_at(const std::string& s, size_t char_index);

// True iff U+4E00 <= cp <= U+9FFF (CJK Unified Ideographs).
inline bool is_han(uint32_t cp) {
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

} // namespace rs::kokoro_zh
