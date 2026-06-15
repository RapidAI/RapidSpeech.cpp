// kokoro_pinyin.cpp — see kokoro_pinyin.h for overview.

#include "frontend/kokoro_pinyin.h"
#include "utils/rs_log.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rs::kokoro_zh {

const std::vector<std::string> kInitialsLongestFirst = {
    // 2-char first
    "zh", "ch", "sh",
    // 1-char
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "r", "z", "c", "s", "j", "q", "x", "y", "w",
};

// ----------------------------- UTF-8 -----------------------------

uint32_t utf8_decode(const std::string& s, size_t& i) {
    if (i >= s.size()) return 0;
    uint8_t b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) { i += 1; return b0; }
    if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = (uint32_t)(b0 & 0x1F) << 6 | (uint32_t)(s[i+1] & 0x3F);
        i += 2; return cp;
    }
    if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = (uint32_t)(b0 & 0x0F) << 12
                    | (uint32_t)(s[i+1] & 0x3F) << 6
                    | (uint32_t)(s[i+2] & 0x3F);
        i += 3; return cp;
    }
    if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = (uint32_t)(b0 & 0x07) << 18
                    | (uint32_t)(s[i+1] & 0x3F) << 12
                    | (uint32_t)(s[i+2] & 0x3F) << 6
                    | (uint32_t)(s[i+3] & 0x3F);
        i += 4; return cp;
    }
    i += 1;
    return 0xFFFD;
}

void utf8_encode(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back((char)cp);
    } else if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) out.push_back(utf8_decode(s, i));
    return out;
}

size_t utf8_char_count(const std::string& s) {
    size_t n = 0, i = 0;
    while (i < s.size()) { utf8_decode(s, i); ++n; }
    return n;
}

// Build a vector of byte-offsets for each codepoint boundary including end.
static std::vector<size_t> char_offsets(const std::string& s) {
    std::vector<size_t> off;
    off.reserve(s.size() + 1);
    size_t i = 0;
    off.push_back(0);
    while (i < s.size()) {
        utf8_decode(s, i);
        off.push_back(i);
    }
    return off;
}

std::string utf8_substr(const std::string& s, size_t a, size_t b) {
    auto off = char_offsets(s);
    if (a > off.size() - 1) a = off.size() - 1;
    if (b > off.size() - 1) b = off.size() - 1;
    if (a >= b) return "";
    return s.substr(off[a], off[b] - off[a]);
}

std::string utf8_char_at(const std::string& s, size_t idx) {
    size_t i = 0;
    size_t cur = 0;
    while (i < s.size()) {
        size_t start = i;
        utf8_decode(s, i);
        if (cur == idx) return s.substr(start, i - start);
        ++cur;
    }
    return "";
}

// --------------------------- Binary load -------------------------

static bool read_u32(std::ifstream& in, uint32_t& v) {
    return (bool)in.read(reinterpret_cast<char*>(&v), 4);
}
static bool read_u16(std::ifstream& in, uint16_t& v) {
    return (bool)in.read(reinterpret_cast<char*>(&v), 2);
}
static bool read_u8(std::ifstream& in, uint8_t& v) {
    return (bool)in.read(reinterpret_cast<char*>(&v), 1);
}
static bool read_bytes(std::ifstream& in, std::string& s, size_t n) {
    s.resize(n);
    return (bool)in.read(s.data(), (std::streamsize)n);
}

bool PypinyinLite::Load(const std::string& data_dir) {
    std::string single_path = data_dir + "/pinyin_single.bin";
    std::string phrase_path = data_dir + "/pinyin_phrases.bin";

    // ---- single ----
    {
        std::ifstream in(single_path, std::ios::binary);
        if (!in) {
            RS_LOG_ERR("PypinyinLite: cannot open '%s'", single_path.c_str());
            return false;
        }
        uint32_t magic, ver, count;
        if (!read_u32(in, magic) || !read_u32(in, ver) || !read_u32(in, count)) {
            RS_LOG_ERR("PypinyinLite: short header in single dict");
            return false;
        }
        if (magic != 0x53594B50u) {
            RS_LOG_ERR("PypinyinLite: bad magic 0x%08x in '%s' (expected PKYS)",
                       magic, single_path.c_str());
            return false;
        }
        single_.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t cp;
            uint8_t nr;
            if (!read_u32(in, cp) || !read_u8(in, nr)) return false;
            std::vector<std::string> readings;
            readings.reserve(nr);
            for (uint8_t k = 0; k < nr; ++k) {
                uint8_t sl;
                if (!read_u8(in, sl)) return false;
                std::string r;
                if (!read_bytes(in, r, sl)) return false;
                readings.push_back(std::move(r));
            }
            single_.emplace(cp, std::move(readings));
        }
    }

    // ---- phrases ----
    {
        std::ifstream in(phrase_path, std::ios::binary);
        if (!in) {
            RS_LOG_ERR("PypinyinLite: cannot open '%s'", phrase_path.c_str());
            return false;
        }
        uint32_t magic, ver, count, max_chars;
        if (!read_u32(in, magic) || !read_u32(in, ver)
            || !read_u32(in, count) || !read_u32(in, max_chars)) {
            RS_LOG_ERR("PypinyinLite: short header in phrase dict");
            return false;
        }
        if (magic != 0x50594B50u) {
            RS_LOG_ERR("PypinyinLite: bad magic 0x%08x in '%s' (expected PKYP)",
                       magic, phrase_path.c_str());
            return false;
        }
        max_phrase_chars_ = max_chars;
        phrases_.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            uint16_t plen;
            if (!read_u16(in, plen)) return false;
            std::string phrase;
            if (!read_bytes(in, phrase, plen)) return false;
            uint8_t ns;
            if (!read_u8(in, ns)) return false;
            std::vector<std::string> sylls;
            sylls.reserve(ns);
            for (uint8_t k = 0; k < ns; ++k) {
                uint8_t sl;
                if (!read_u8(in, sl)) return false;
                std::string r;
                if (!read_bytes(in, r, sl)) return false;
                sylls.push_back(std::move(r));
            }
            phrases_.emplace(std::move(phrase), std::move(sylls));
        }
    }

    loaded_ = true;
    RS_LOG_INFO(
        "PypinyinLite: loaded %zu singles, %zu phrases (max_chars=%u)",
        single_.size(), phrases_.size(),
        (unsigned)max_phrase_chars_);
    // misaki phrase/single overrides are already baked in by the dump script
    // (see scripts/dump_pypinyin_data.py); nothing to apply here.
    return true;
}

// --------------------------- Lazy pinyin -------------------------

void PypinyinLite::scan_tone3(const std::string& word,
                              std::vector<std::string>& out) const {
    // Build char offsets once
    std::vector<size_t> off = char_offsets(word);
    size_t nchars = off.size() - 1;
    size_t i = 0;
    while (i < nchars) {
        // Try phrases longest first
        bool matched = false;
        size_t max_try = std::min((size_t)max_phrase_chars_, nchars - i);
        for (size_t len = max_try; len >= 2; --len) {
            std::string key = word.substr(off[i], off[i + len] - off[i]);
            auto it = phrases_.find(key);
            if (it != phrases_.end()) {
                for (const auto& s : it->second) out.push_back(s);
                i += len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Single char fallback
        size_t bstart = off[i];
        size_t bend = off[i + 1];
        std::string ch = word.substr(bstart, bend - bstart);
        // Decode the single codepoint
        size_t pos = 0;
        uint32_t cp = utf8_decode(ch, pos);
        if (is_han(cp)) {
            auto it = single_.find(cp);
            if (it != single_.end() && !it->second.empty()) {
                out.push_back(it->second.front());
            } else {
                // Unknown Han char — emit raw codepoint string.
                out.push_back(ch);
            }
        } else {
            // Non-Han: emit raw char unchanged.
            out.push_back(ch);
        }
        i += 1;
    }
}

// Split TONE3 form into initial + final-with-tone-digit. Mirrors pypinyin's
// Style.INITIALS / Style.FINALS_TONE3 conventions, including the glide rules:
//   - 'y' / 'w' are NOT initials; they get rewritten into the final's onset:
//       wo -> uo, wu -> u, wa->ua, wai->uai, wan->uan, wang->uang,
//       wei->uei, wen->uen, weng->ueng
//       yi -> i, ya->ia, yan->ian, yang->iang, yao->iao, ye->ie,
//       yin->in, ying->ing, yong->iong, you->iou
//       yu->v, yue->ve, yuan->van, yun->vn
void PypinyinLite::split_initial_final(const std::string& tone3,
                                       std::string& initial,
                                       std::string& final_out) {
    initial.clear();
    final_out = tone3;
    if (tone3.empty()) return;
    // 2-char initials (zh/ch/sh) first
    for (const auto& ini : {std::string("zh"), std::string("ch"), std::string("sh")}) {
        if (tone3.size() >= ini.size()
            && std::memcmp(tone3.data(), ini.data(), ini.size()) == 0) {
            initial = ini;
            final_out = tone3.substr(ini.size());
            return;
        }
    }
    // Glides: 'y' and 'w' -- empty initial, rewrite final.
    char c0 = tone3[0];
    if (c0 == 'w') {
        // Strip leading 'w'
        std::string rest = tone3.substr(1);
        // Split rest into letters + trailing digits.
        size_t k = 0;
        while (k < rest.size() && (rest[k] < '0' || rest[k] > '9')) ++k;
        std::string letters = rest.substr(0, k);
        std::string digits  = rest.substr(k);
        if (letters == "u") {
            // wu -> final "u"
            final_out = letters + digits;
        } else {
            // wo/wa/wai/wan/wang/wei/wen/weng -> "u" + letters
            final_out = std::string("u") + letters + digits;
        }
        initial.clear();
        return;
    }
    if (c0 == 'y') {
        std::string rest = tone3.substr(1);
        size_t k = 0;
        while (k < rest.size() && (rest[k] < '0' || rest[k] > '9')) ++k;
        std::string letters = rest.substr(0, k);
        std::string digits  = rest.substr(k);
        if (letters == "i") {
            final_out = letters + digits;  // yi -> i
        } else if (letters == "u") {
            final_out = std::string("v") + digits; // yu -> v
        } else if (letters == "ue") {
            final_out = std::string("ve") + digits; // yue -> ve
        } else if (letters == "uan") {
            final_out = std::string("van") + digits; // yuan -> van
        } else if (letters == "un") {
            final_out = std::string("vn") + digits; // yun -> vn
        } else if (letters == "in" || letters == "ing") {
            // yin -> in, ying -> ing (letters already start with 'i')
            final_out = letters + digits;
        } else {
            // ya/yan/yang/yao/ye/yong/you -> i + letters
            final_out = std::string("i") + letters + digits;
        }
        initial.clear();
        return;
    }
    // 1-char standard initials.
    for (const auto& ini : {"b","p","m","f","d","t","n","l","g","k","h",
                            "r","z","c","s","j","q","x"}) {
        size_t L = std::strlen(ini);
        if (tone3.size() >= L
            && std::memcmp(tone3.data(), ini, L) == 0) {
            initial = ini;
            final_out = tone3.substr(L);
            // After j/q/x, pypinyin's Style.FINALS_TONE3 emits ü-form (v_to_u=False
            // → 'v'). The dump stores raw to_tone3 which keeps 'u', so we rewrite
            // ju/qu/xu → jv/qv/xv ; jue → jve ; juan → jvan ; jun → jvn.
            if ((initial == "j" || initial == "q" || initial == "x")
                && !final_out.empty() && final_out[0] == 'u') {
                size_t k = 0;
                while (k < final_out.size()
                       && (final_out[k] < '0' || final_out[k] > '9')) ++k;
                std::string letters = final_out.substr(0, k);
                std::string digits  = final_out.substr(k);
                if (letters == "u")        letters = "v";
                else if (letters == "ue")  letters = "ve";
                else if (letters == "uan") letters = "van";
                else if (letters == "un")  letters = "vn";
                final_out = letters + digits;
            }
            // pypinyin's Style.FINALS_TONE3 (strict=True) expands the
            // shorthand finals iu/ui/un → iou/uei/uen after a consonant
            // initial. to_tone3 keeps the short form, so we patch here.
            // Excludes j/q/x (their 'un' is ün/vn, already handled above).
            {
                size_t k = 0;
                while (k < final_out.size()
                       && (final_out[k] < '0' || final_out[k] > '9')) ++k;
                std::string letters = final_out.substr(0, k);
                std::string digits  = final_out.substr(k);
                if (letters == "iu")      letters = "iou";
                else if (letters == "ui") letters = "uei";
                else if (letters == "un"
                         && initial != "j" && initial != "q"
                         && initial != "x") letters = "uen";
                final_out = letters + digits;
            }
            return;
        }
    }
    // No recognised initial — final is the whole syllable.
}

std::vector<std::string> PypinyinLite::LazyPinyinTone3(
    const std::string& word_utf8) const {
    std::vector<std::string> out;
    scan_tone3(word_utf8, out);
    return out;
}

std::vector<std::string> PypinyinLite::LazyPinyinFinalsTone3(
    const std::string& word_utf8) const {
    std::vector<std::string> tone3;
    scan_tone3(word_utf8, tone3);

    // Identify '嗯' positions for the n2 special-case (zh_frontend.py:101).
    std::vector<uint32_t> cps = utf8_codepoints(word_utf8);

    std::vector<std::string> out;
    out.reserve(tone3.size());
    size_t idx = 0;
    for (const auto& t : tone3) {
        std::string ini, fin;
        split_initial_final(t, ini, fin);
        // Replicate '嗯' n2 only when the syllable looks like the default
        // pypinyin output for U+55EF (no initial, finals would be empty/junk).
        if (idx < cps.size() && cps[idx] == 0x55EFu) {
            fin = "n2";
        }
        out.push_back(fin);
        ++idx;
    }
    return out;
}

std::vector<std::string> PypinyinLite::LazyPinyinInitials(
    const std::string& word_utf8) const {
    std::vector<std::string> tone3;
    scan_tone3(word_utf8, tone3);

    std::vector<uint32_t> cps = utf8_codepoints(word_utf8);

    std::vector<std::string> out;
    out.reserve(tone3.size());
    size_t idx = 0;
    for (const auto& t : tone3) {
        std::string ini, fin;
        split_initial_final(t, ini, fin);
        if (idx < cps.size() && cps[idx] == 0x55EFu) {
            // '嗯' has no initial in pypinyin's Style.INITIALS output.
            ini.clear();
        }
        out.push_back(ini);
        ++idx;
    }
    return out;
}

} // namespace rs::kokoro_zh
