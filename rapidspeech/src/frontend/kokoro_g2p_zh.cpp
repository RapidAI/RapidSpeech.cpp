// kokoro_g2p_zh.cpp — see header for overview.

#include "frontend/kokoro_g2p_zh.h"
#include "frontend/kokoro_pinyin.h"
#include "frontend/kokoro_zh_tone_sandhi.h"
#include "utils/rs_log.h"

#include "cppjieba/Jieba.hpp"

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace rs::kokoro_zh {

ZHG2P::ZHG2P() = default;
ZHG2P::~ZHG2P() = default;

// -------------------- An2cn (simplified) --------------------

std::string ZHG2P::An2cn(const std::string& text) {
    static const char* kDigits[10] = {
        "零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
    std::string out;
    out.reserve(text.size() * 3);
    for (char c : text) {
        if (c >= '0' && c <= '9') out += kDigits[c - '0'];
        else out.push_back(c);
    }
    return out;
}

// -------------------- MapPunctuation --------------------

// Replace `from` with `to` everywhere in s (UTF-8 bytes).
static void replace_all(std::string& s,
                        const std::string& from,
                        const std::string& to) {
    if (from.empty()) return;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string ZHG2P::MapPunctuation(const std::string& text) {
    std::string s = text;
    replace_all(s, "、", ", "); replace_all(s, "，", ", ");
    replace_all(s, "。", ". "); replace_all(s, "．", ". ");
    replace_all(s, "！", "! ");
    replace_all(s, "：", ": ");
    replace_all(s, "；", "; ");
    replace_all(s, "？", "? ");
    replace_all(s, "«", " “"); replace_all(s, "»", "” ");
    replace_all(s, "《", " “"); replace_all(s, "》", "” ");
    replace_all(s, "「", " “"); replace_all(s, "」", "” ");
    replace_all(s, "【", " “"); replace_all(s, "】", "” ");
    replace_all(s, "(", " ("); replace_all(s, ")", ") ");
    // Strip
    size_t a = 0, b = s.size();
    while (a < b && (unsigned char)s[a] <= ' ') ++a;
    while (b > a && (unsigned char)s[b - 1] <= ' ') --b;
    return s.substr(a, b - a);
}

// -------------------- Load --------------------

bool ZHG2P::Load(const std::string& jieba_dict_dir,
                 const std::string& py_data_dir) {
    if (loaded_) return true;
    try {
        jieba_ = std::make_unique<cppjieba::Jieba>(
            jieba_dict_dir + "/jieba.dict.utf8",
            jieba_dict_dir + "/hmm_model.utf8",
            jieba_dict_dir + "/user.dict.utf8",
            jieba_dict_dir + "/idf.utf8",
            jieba_dict_dir + "/stop_words.utf8");
    } catch (const std::exception& e) {
        RS_LOG_ERR("ZHG2P: failed to init cppjieba from '%s': %s",
                   jieba_dict_dir.c_str(), e.what());
        return false;
    }
    py_ = std::make_unique<PypinyinLite>();
    if (!py_->Load(py_data_dir)) {
        RS_LOG_ERR("ZHG2P: failed to load pypinyin tables from '%s'",
                   py_data_dir.c_str());
        return false;
    }
    sandhi_ = std::make_unique<ToneSandhi>();
    sandhi_->SetJieba(jieba_.get());
    sandhi_->SetPinyin(py_.get());

    // ZH_MAP from zh_frontend.py:39.
    static const std::pair<const char*, const char*> kZH[] = {
        {"b","ㄅ"},{"p","ㄆ"},{"m","ㄇ"},{"f","ㄈ"},{"d","ㄉ"},{"t","ㄊ"},
        {"n","ㄋ"},{"l","ㄌ"},{"g","ㄍ"},{"k","ㄎ"},{"h","ㄏ"},{"j","ㄐ"},
        {"q","ㄑ"},{"x","ㄒ"},{"zh","ㄓ"},{"ch","ㄔ"},{"sh","ㄕ"},{"r","ㄖ"},
        {"z","ㄗ"},{"c","ㄘ"},{"s","ㄙ"},{"a","ㄚ"},{"o","ㄛ"},{"e","ㄜ"},
        {"ie","ㄝ"},{"ai","ㄞ"},{"ei","ㄟ"},{"ao","ㄠ"},{"ou","ㄡ"},{"an","ㄢ"},
        {"en","ㄣ"},{"ang","ㄤ"},{"eng","ㄥ"},{"er","ㄦ"},{"i","ㄧ"},{"u","ㄨ"},
        {"v","ㄩ"},{"ii","ㄭ"},{"iii","十"},{"ve","月"},{"ia","压"},{"ian","言"},
        {"iang","阳"},{"iao","要"},{"in","阴"},{"ing","应"},{"iong","用"},
        {"iou","又"},{"ong","中"},{"ua","穵"},{"uai","外"},{"uan","万"},
        {"uang","王"},{"uei","为"},{"uen","文"},{"ueng","瓮"},{"uo","我"},
        {"van","元"},{"vn","云"},
    };
    for (auto& kv : kZH) zh_map_[kv.first] = kv.second;
    // identity-mapped chars: ';:,.!?/—…"()“” 12345R'
    static const char* kIdent[] = {
        ";",":",",",".","!","?","/","—","…","\"","(",")","“","”"," ",
        "1","2","3","4","5","R"};
    for (auto* c : kIdent) zh_map_[c] = c;

    // Frontend punc (zh_frontend.py:47).
    static const char* kPunc[] = {
        ";",":",",",".","!","?","—","…","\"","(",")","“","”"};
    for (auto* c : kPunc) frontend_punc_.insert(c);

    // must_erhua / not_erhua (zh_frontend.py:66-75).
    static const char* kMust[] = {
        "小院儿","胡同儿","范儿","老汉儿","撒欢儿","寻老礼儿","妥妥儿","媳妇儿"};
    for (auto* w : kMust) must_erhua_.insert(w);
    static const char* kNot[] = {
        "虐儿","为儿","护儿","瞒儿","救儿","替儿","有儿","一儿","我儿","俺儿","妻儿",
        "拐儿","聋儿","乞儿","患儿","幼儿","孤儿","婴儿","婴幼儿","连体儿","脑瘫儿",
        "流浪儿","体弱儿","混血儿","蜜雪儿","舫儿","祖儿","美儿","应采儿","可儿","侄儿",
        "孙儿","侄孙儿","女儿","男儿","红孩儿","花儿","虫儿","马儿","鸟儿","猪儿","猫儿",
        "狗儿","少儿"};
    for (auto* w : kNot) not_erhua_.insert(w);

    loaded_ = true;
    return true;
}

// -------------------- helpers --------------------

// Apply z/c/s -> ii, zh/ch/sh/r -> iii to a final that matches r'i\d'.
std::string ZHG2P::discriminate_i(const std::string& c,
                                  const std::string& v) {
    if (v.size() >= 2 && v[0] == 'i' && (v[1] >= '0' && v[1] <= '9')) {
        if (c == "z" || c == "c" || c == "s") {
            return std::string("ii") + v.substr(1);
        }
        if (c == "zh" || c == "ch" || c == "sh" || c == "r") {
            return std::string("iii") + v.substr(1);
        }
    }
    return v;
}

// -------------------- merge_erhua --------------------

void ZHG2P::merge_erhua(std::vector<std::string>& initials,
                        std::vector<std::string>& finals,
                        const std::string& word,
                        const std::string& pos) const {
    const size_t n_chars = utf8_char_count(word);
    // fix er1 -> er2 at end
    if (!finals.empty() && n_chars == finals.size()) {
        size_t i = finals.size() - 1;
        if (utf8_char_at(word, i) == "儿" && finals[i] == "er1") {
            finals[i] = "er2";
        }
    }

    // Skip erhua-merging if word is not_erhua / pos in {a,j,nr}
    bool must = must_erhua_.find(word) != must_erhua_.end();
    if (!must
        && (not_erhua_.find(word) != not_erhua_.end()
            || pos == "a" || pos == "j" || pos == "nr")) {
        return;
    }
    if (finals.size() != n_chars) return;

    std::vector<std::string> new_initials, new_finals;
    for (size_t i = 0; i < finals.size(); ++i) {
        const std::string& phn = finals[i];
        bool last = (i == finals.size() - 1);
        bool ch_is_er = (utf8_char_at(word, i) == "儿");
        bool phn_ok = (phn == "er2" || phn == "er5");
        std::string last2 = utf8_substr(word,
                                        n_chars >= 2 ? n_chars - 2 : 0,
                                        n_chars);
        bool not_last2 = (not_erhua_.find(last2) == not_erhua_.end());
        if (last && ch_is_er && phn_ok && not_last2 && !new_finals.empty()) {
            std::string& prev = new_finals.back();
            // prev[:-1] + 'R' + prev[-1]
            if (!prev.empty()) {
                char tail = prev.back();
                prev.pop_back();
                prev += 'R';
                prev += tail;
            }
        } else {
            new_initials.push_back(initials[i]);
            new_finals.push_back(phn);
        }
    }
    initials = std::move(new_initials);
    finals = std::move(new_finals);
}

// -------------------- ProcessZHSegment --------------------

static bool is_all_han(const std::string& s) {
    auto cps = utf8_codepoints(s);
    if (cps.empty()) return false;
    for (auto cp : cps) if (!is_han(cp)) return false;
    return true;
}

static bool is_all_space(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r')) return false;
    return true;
}

// Replicate re.sub(r'(?=\d)', '_', s).split('_').
static std::vector<std::string> insert_underscore_before_digit_split(
    const std::string& s) {
    std::string buf;
    buf.reserve(s.size() + 4);
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c >= '0' && c <= '9') buf.push_back('_');
        buf.push_back(c);
    }
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= buf.size(); ++i) {
        if (i == buf.size() || buf[i] == '_') {
            out.push_back(buf.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

std::string ZHG2P::ProcessZHSegment(const std::string& text) const {
    if (!loaded_) return unk_;

    // jieba tag -> [(word, pos)]
    std::vector<std::pair<std::string, std::string>> seg_cut;
    jieba_->Tag(text, seg_cut);

    seg_cut = sandhi_->PreMergeForModify(seg_cut);

    // Each token's phonemes + whitespace.
    struct Tok {
        std::string text;
        std::string tag;
        std::string phonemes;        // empty means unk (per misaki)
        bool phonemes_set = false;
        std::string whitespace;
    };
    std::vector<Tok> tokens;
    tokens.reserve(seg_cut.size());

    for (auto& [word, pos_in] : seg_cut) {
        std::string pos = pos_in;
        // pos=='x' and all chars are Han → pos='X'
        if (pos == "x" && is_all_han(word)) {
            pos = "X";
        } else if (pos != "x" && frontend_punc_.count(word)) {
            pos = "x";
        }

        Tok tk;
        tk.text = word;
        tk.tag = pos;

        if (pos == "x" || pos == "eng") {
            if (!is_all_space(word)) {
                if (pos == "x" && frontend_punc_.count(word)) {
                    tk.phonemes = word;
                    tk.phonemes_set = true;
                }
                tokens.push_back(std::move(tk));
            } else if (!tokens.empty()) {
                tokens.back().whitespace += word;
            }
            continue;
        } else if (!tokens.empty()
                   && tokens.back().tag != "x"
                   && tokens.back().tag != "eng"
                   && tokens.back().whitespace.empty()) {
            tokens.back().whitespace = "/";
        }

        // _get_initials_finals
        std::vector<std::string> initials = py_->LazyPinyinInitials(word);
        std::vector<std::string> finals = py_->LazyPinyinFinalsTone3(word);
        // i / ii / iii discrimination
        size_t N = std::min(initials.size(), finals.size());
        for (size_t i = 0; i < N; ++i) {
            finals[i] = discriminate_i(initials[i], finals[i]);
        }

        // tone sandhi
        finals = sandhi_->ModifiedTone(word, pos, std::move(finals));

        // er-hua
        merge_erhua(initials, finals, word, pos);

        // Build phones list
        std::vector<std::string> phones;
        size_t M = std::min(initials.size(), finals.size());
        for (size_t i = 0; i < M; ++i) {
            const auto& c = initials[i];
            const auto& v = finals[i];
            if (!c.empty()) phones.push_back(c);
            if (!v.empty()) {
                // Condition: (v not in punc or v != c) — for our purposes,
                // initials are never punctuation, so include v unconditionally.
                if (!(frontend_punc_.count(v) && v == c)) {
                    phones.push_back(v);
                }
            }
        }

        // Join with '_', then text-level replacements.
        std::string joined;
        for (size_t i = 0; i < phones.size(); ++i) {
            if (i) joined.push_back('_');
            joined += phones[i];
        }
        replace_all(joined, "_eR", "_er");
        replace_all(joined, "R", "_R");
        std::vector<std::string> pieces = insert_underscore_before_digit_split(joined);

        // Map via ZH_MAP, defaulting to unk for unknowns.
        std::string ph;
        for (const auto& p : pieces) {
            if (p.empty()) continue;
            auto it = zh_map_.find(p);
            if (it != zh_map_.end()) ph += it->second;
            else { ph += unk_; }
        }
        tk.phonemes = ph;
        tk.phonemes_set = true;
        tokens.push_back(std::move(tk));
    }

    // Final string concatenation.
    std::string result;
    for (const auto& tk : tokens) {
        if (tk.phonemes_set) result += tk.phonemes;
        else result += unk_;
        result += tk.whitespace;
    }
    return result;
}

// -------------------- Process (top-level __call__) --------------------

// True iff `c` is ASCII A-Z, a-z, space, ', or - (the misaki English
// segment character class).
static bool is_en_char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
           || c == ' ' || c == '\'' || c == '-';
}

// Returns true iff there is at least one A-Z/a-z in [a,b).
static bool has_alpha(const std::string& s, size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
        char c = s[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) return true;
    }
    return false;
}

// Mimic re.findall(r'([A-Za-z \'-]*[A-Za-z][A-Za-z \'-]*)|([^A-Za-z]+)', text).
//   English segment = a run of [A-Za-z \'-] containing at least one A-Za-z.
//   Non-English segment = a maximal run of non-A-Za-z bytes (which may
//   include spaces / apostrophes / hyphens).
// Operates on bytes; UTF-8 multi-byte chars are not in [A-Za-z] so they
// always fall into the non-English bucket.
static std::vector<std::pair<bool /*is_en*/, std::string>> split_en_zh(
    const std::string& text) {
    std::vector<std::pair<bool, std::string>> out;
    size_t i = 0;
    while (i < text.size()) {
        // Try to grow an English candidate
        size_t j = i;
        while (j < text.size() && is_en_char(text[j])) ++j;
        if (j > i && has_alpha(text, i, j)) {
            out.emplace_back(true, text.substr(i, j - i));
            i = j;
            continue;
        }
        // Non-English run: consume bytes that are not A-Za-z.
        j = i;
        while (j < text.size()) {
            char c = text[j];
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) break;
            ++j;
        }
        if (j == i) { ++i; continue; }
        out.emplace_back(false, text.substr(i, j - i));
        i = j;
    }
    return out;
}

static std::string strip_ws(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (unsigned char)s[a] <= ' ') ++a;
    while (b > a && (unsigned char)s[b - 1] <= ' ') --b;
    return s.substr(a, b - a);
}

std::string ZHG2P::Process(const std::string& text_in) const {
    std::string text = text_in;
    // strip
    text = strip_ws(text);
    if (text.empty()) return "";

    text = An2cn(text);
    text = MapPunctuation(text);

    auto parts = split_en_zh(text);
    std::vector<std::string> segments;
    segments.reserve(parts.size());
    for (auto& [is_en, raw] : parts) {
        std::string s = strip_ws(raw);
        if (is_en) {
            if (s.empty()) continue;
            if (en_callable_) {
                std::string ipa = en_callable_(s);
                if (!ipa.empty()) {
                    segments.push_back(std::move(ipa));
                    continue;
                }
            }
            segments.push_back(unk_);
        } else {
            if (s.empty()) continue;
            segments.push_back(ProcessZHSegment(s));
        }
    }
    std::string out;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i) out += ' ';
        out += segments[i];
    }
    return out;
}

} // namespace rs::kokoro_zh
