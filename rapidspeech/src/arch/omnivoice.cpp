#include "omnivoice.h"
#include "core/rs_context.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llm_graph.h"
#include "llm_model.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <unordered_set>

// =====================================================================
// BPE Tokenizer Implementation
// =====================================================================

static void bpe_build_byte_encoder(std::string byte2str[256]) {
    int bs[256], cs[256], n = 0, total = 0;
    for (int b = '!'; b <= '~'; b++) { bs[total] = b; cs[total] = b; total++; }
    for (int b = 0xA1; b <= 0xAC; b++) { bs[total] = b; cs[total] = b; total++; }
    for (int b = 0xAE; b <= 0xFF; b++) { bs[total] = b; cs[total] = b; total++; }
    bool used[256] = {};
    for (int i = 0; i < total; i++) used[bs[i]] = true;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) { bs[total] = b; cs[total] = 256 + n; n++; total++; }
    }
    for (int i = 0; i < 256; i++) {
        int cp = cs[i];
        char buf[4]; int len;
        if (cp < 0x80) { buf[0] = (char)cp; len = 1; }
        else if (cp < 0x800) { buf[0] = (char)(0xC0|(cp>>6)); buf[1] = (char)(0x80|(cp&0x3F)); len = 2; }
        else { buf[0] = (char)(0xE0|(cp>>12)); buf[1] = (char)(0x80|((cp>>6)&0x3F)); buf[2] = (char)(0x80|(cp&0x3F)); len = 3; }
        byte2str[bs[i]] = std::string(buf, len);
    }
}

static int bpe_utf8_codepoint(const char *s, int *advance) {
    unsigned char c = s[0];
    if (c < 0x80) { *advance = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *advance = 2; return ((c&0x1F)<<6)|(s[1]&0x3F); }
    if ((c & 0xF0) == 0xE0) { *advance = 3; return ((c&0x0F)<<12)|((s[1]&0x3F)<<6)|(s[2]&0x3F); }
    if ((c & 0xF8) == 0xF0) { *advance = 4; return ((c&0x07)<<18)|((s[1]&0x3F)<<12)|((s[2]&0x3F)<<6)|(s[3]&0x3F); }
    *advance = 1; return c;
}

static bool bpe_is_letter(int cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    if (cp >= 0xC0 && cp <= 0x024F && cp != 0xD7 && cp != 0xF7) return true;
    if (cp >= 0x0370 && cp <= 0x1FFF) return true;
    if (cp >= 0x2C00 && cp <= 0x2DFF) return true;
    if (cp >= 0x3040 && cp <= 0x9FFF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0x10000) return true;
    return false;
}

static bool bpe_is_digit(int cp) { return cp >= '0' && cp <= '9'; }
static bool bpe_is_whitespace(int cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0xA0 || cp == 0x2000 || cp == 0x2001 || cp == 0x2002 || cp == 0x200B;
}
static bool bpe_is_newline(int cp) { return cp == '\n' || cp == '\r'; }

// GPT-2 pre-tokenizer
static std::vector<std::string> bpe_gpt2_pre_tokenize(const std::string &text) {
    std::vector<std::string> chunks;
    const char *s = text.c_str();
    int len = (int)text.size();
    int i = 0;

    while (i < len) {
        int adv, cp = bpe_utf8_codepoint(s + i, &adv);

        // Contractions
        if ((cp == '\'' || cp == 0x2019) && i + adv < len) {
            const char *rest = s + i + adv;
            int rlen = len - i - adv;
            auto try_match = [&](const char *suffix, int slen) -> bool {
                if (rlen < slen) return false;
                for (int k = 0; k < slen; k++) {
                    char c1 = rest[k], c2 = suffix[k];
                    if (c1 >= 'A' && c1 <= 'Z') c1 = (char)(c1 + 32);
                    if (c1 != c2) return false;
                }
                if (rlen > slen) {
                    int a2, cp2 = bpe_utf8_codepoint(rest + slen, &a2);
                    if (bpe_is_letter(cp2)) return false;
                }
                chunks.push_back(std::string(s + i, adv + slen));
                i += adv + slen;
                return true;
            };
            if (try_match("ll",2)||try_match("re",2)||try_match("ve",2)||
                try_match("s",1)||try_match("t",1)||try_match("m",1)||try_match("d",1)) continue;
        }

        if (bpe_is_letter(cp)) {
            int start = i; i += adv;
            while (i < len) {
                int a2, cp2 = bpe_utf8_codepoint(s + i, &a2);
                if (!bpe_is_letter(cp2)) break;
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        if (!bpe_is_newline(cp) && !bpe_is_letter(cp) && !bpe_is_digit(cp) && !bpe_is_whitespace(cp)) {
            int start = i, after = i + adv;
            if (after < len) {
                int a2, cp2 = bpe_utf8_codepoint(s + after, &a2);
                if (bpe_is_letter(cp2)) {
                    i = after + a2;
                    while (i < len) {
                        int a3, cp3 = bpe_utf8_codepoint(s + i, &a3);
                        if (!bpe_is_letter(cp3)) break;
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
            }
        }

        if (bpe_is_digit(cp)) {
            int start = i;
            while (i < len && bpe_is_digit((unsigned char)s[i])) i++;
            for (int j = start; j < i; j++) {
                chunks.push_back(std::string(s + j, 1));
            }
            continue;
        }

        if (bpe_is_newline(cp)) {
            int start = i;
            while (i < len && bpe_is_newline((unsigned char)s[i])) i++;
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        if (bpe_is_whitespace(cp)) {
            int start = i, ws_end = i + adv;
            while (ws_end < len && bpe_is_whitespace((unsigned char)s[ws_end]) && !bpe_is_newline((unsigned char)s[ws_end])) ws_end++;
            bool followed_by_non_ws = (ws_end < len && !bpe_is_whitespace((unsigned char)s[ws_end]) && !bpe_is_newline((unsigned char)s[ws_end]));
            if (followed_by_non_ws && ws_end - start > 1) {
                chunks.push_back(std::string(s + start, ws_end - 1 - start));
                i = ws_end - 1;
                continue;
            }
            i = start + adv;
            if (i < len) {
                int a2, cp2 = bpe_utf8_codepoint(s + i, &a2);
                if (bpe_is_letter(cp2)) {
                    i += a2;
                    while (i < len) {
                        int a3, cp3 = bpe_utf8_codepoint(s + i, &a3);
                        if (!bpe_is_letter(cp3)) break;
                        i += a3;
                    }
                    chunks.push_back(std::string(s + start, i - start));
                    continue;
                }
                if (bpe_is_digit(cp2)) { chunks.push_back(std::string(s + start, i - start)); continue; }
                if (!bpe_is_whitespace(cp2) && !bpe_is_newline(cp2)) {
                    int pstart = start;
                    while (i < len) {
                        int a3, cp3 = bpe_utf8_codepoint(s + i, &a3);
                        if (bpe_is_whitespace(cp3)||bpe_is_letter(cp3)||bpe_is_digit(cp3)) break;
                        i += a3;
                    }
                    while (i < len && bpe_is_newline((unsigned char)s[i])) i++;
                    chunks.push_back(std::string(s + pstart, i - pstart));
                    continue;
                }
            }
            i = ws_end;
            while (i < len) {
                int a2, cp2 = bpe_utf8_codepoint(s + i, &a2);
                if (!bpe_is_whitespace(cp2)) break;
                i += a2;
            }
            chunks.push_back(std::string(s + start, i - start));
            continue;
        }

        {
            int start = i; i += adv;
            while (i < len) {
                int a2, cp2 = bpe_utf8_codepoint(s + i, &a2);
                if (bpe_is_whitespace(cp2)||bpe_is_letter(cp2)||bpe_is_digit(cp2)||bpe_is_newline(cp2)) break;
                i += a2;
            }
            while (i < len && bpe_is_newline((unsigned char)s[i])) i++;
            chunks.push_back(std::string(s + start, i - start));
        }
    }
    return chunks;
}

// BPE merge
static std::vector<std::string> bpe_merge_symbols(
    const std::unordered_map<std::string, int> &merge_rank,
    const std::vector<std::string> &symbols)
{
    if (symbols.size() <= 1) return symbols;
    std::vector<std::string> work = symbols;
    while (work.size() > 1) {
        int best_rank = INT_MAX, best_pos = -1;
        for (int i = 0; i < (int)work.size() - 1; i++) {
            std::string key = work[i] + " " + work[i + 1];
            auto it = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second; best_pos = i;
            }
        }
        if (best_pos < 0) break;
        work[best_pos] = work[best_pos] + work[best_pos + 1];
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

bool OmniVoiceBPETokenizer::load_from_gguf(const char *gguf_path) {
    bpe_build_byte_encoder(byte2str);

    struct gguf_init_params gp = { true, NULL };
    struct gguf_context *ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) {
        fprintf(stderr, "[BPE] Failed to open %s\n", gguf_path);
        return false;
    }

    int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t mrg_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (tok_key < 0 || mrg_key < 0) {
        fprintf(stderr, "[BPE] Tokenizer not found in %s\n", gguf_path);
        gguf_free(ctx);
        return false;
    }

    int n_tokens = (int)gguf_get_arr_n(ctx, tok_key);
    int n_merges = (int)gguf_get_arr_n(ctx, mrg_key);

    // Use the raw token count from the GGUF array (may include empty-string
    // padding entries to cover the model vocab size including added special
    // tokens).  vocab.size() is smaller when multiple entries share the same
    // string (e.g. empty-string padding).
    n_vocab = n_tokens;
    id_to_str.resize(n_vocab);
    for (int i = 0; i < n_tokens; i++) {
        std::string s = std::string(gguf_get_arr_str(ctx, tok_key, i));
        id_to_str[i] = s;
        vocab[s] = i;
    }
    for (int i = 0; i < n_merges; i++) {
        merges[std::string(gguf_get_arr_str(ctx, mrg_key, i))] = i;
    }
    auto eos_it = vocab.find("<|endoftext|>");
    eos_id = (eos_it != vocab.end()) ? eos_it->second : -1;
    if (eos_id >= 0) specials.emplace_back("<|endoftext|>", eos_id);

    // Read OmniVoice special token IDs from the same GGUF context
    static const char *special_keys[] = {
        "omnivoice.special.denoise", "omnivoice.special.lang_start",
        "omnivoice.special.lang_end", "omnivoice.special.instruct_start",
        "omnivoice.special.instruct_end", "omnivoice.special.text_start",
        "omnivoice.special.text_end",
    };
    for (int i = 0; i < 7; i++) {
        int64_t k = gguf_find_key(ctx, special_keys[i]);
        if (k < 0) continue;
        int id = (int)gguf_get_val_u32(ctx, k);
        if (id >= 0 && id < n_vocab)
            specials.emplace_back(id_to_str[id], id);
    }

    gguf_free(ctx);

    fprintf(stderr, "[BPE] Loaded: %d vocab, %d merges, eos_id=%d specials=%zu\n",
            n_vocab, n_merges, eos_id, specials.size());
    return true;
}

bool OmniVoiceBPETokenizer::load_omnivoice_specials(const char *gguf_path) {
    struct gguf_init_params gp = { true, NULL };
    struct gguf_context *ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) return false;

    static const char *keys[] = {
        "omnivoice.special.denoise", "omnivoice.special.lang_start",
        "omnivoice.special.lang_end", "omnivoice.special.instruct_start",
        "omnivoice.special.instruct_end", "omnivoice.special.text_start",
        "omnivoice.special.text_end",
    };
    const int n_keys = 7;
    for (int i = 0; i < n_keys; i++) {
        int64_t k = gguf_find_key(ctx, keys[i]);
        if (k < 0) continue;
        int id = (int)gguf_get_val_u32(ctx, k);
        if (id < 0 || id >= n_vocab) continue;
        specials.emplace_back(id_to_str[id], id);
    }
    gguf_free(ctx);
    fprintf(stderr, "[BPE] Registered OmniVoice special tokens (total=%zu)\n", specials.size());
    return true;
}

int OmniVoiceBPETokenizer::get_special_id(const std::string &name) const {
    for (auto &sp : specials)
        if (sp.first == name) return sp.second;
    return -1;
}

std::vector<int> OmniVoiceBPETokenizer::encode(const std::string &text, bool add_eos) const {
    std::vector<int> ids;

    auto byte_level_encode = [&](const std::string &t) -> std::string {
        std::string out;
        for (unsigned char c : t) out += byte2str[c];
        return out;
    };

    auto encode_chunk = [&](const std::string &chunk) {
        std::string encoded = byte_level_encode(chunk);
        std::vector<std::string> symbols;
        const char *s = encoded.c_str();
        int len = (int)encoded.size(), i = 0;
        while (i < len) {
            int adv; bpe_utf8_codepoint(s + i, &adv);
            symbols.push_back(std::string(s + i, adv));
            i += adv;
        }
        auto merged = bpe_merge_symbols(merges, symbols);
        for (auto &piece : merged) {
            auto it = vocab.find(piece);
            if (it != vocab.end()) ids.push_back(it->second);
        }
    };

    auto encode_segment = [&](const std::string &seg) {
        if (seg.empty()) return;
        auto chunks = bpe_gpt2_pre_tokenize(seg);
        for (auto &chunk : chunks) encode_chunk(chunk);
    };

    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        int best_idx = -1;
        for (size_t i = 0; i < specials.size(); i++) {
            size_t p = text.find(specials[i].first, pos);
            if (p != std::string::npos && p < best_pos) { best_pos = p; best_idx = (int)i; }
        }
        if (best_idx < 0) { encode_segment(text.substr(pos)); break; }
        if (best_pos > pos) encode_segment(text.substr(pos, best_pos - pos));
        ids.push_back(specials[best_idx].second);
        pos = best_pos + specials[best_idx].first.size();
    }

    if (add_eos && eos_id >= 0) ids.push_back(eos_id);
    return ids;
}

// =====================================================================
// UTF-8 helpers for prompt building
// =====================================================================

static int prompt_utf8_decode(const char *p, size_t avail, uint32_t *cp) {
    if (avail == 0) return 0;
    uint8_t c = (uint8_t)p[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && avail >= 2) { *cp = ((uint32_t)(c&0x1F)<<6)|((uint32_t)(uint8_t)p[1]&0x3F); return 2; }
    if ((c & 0xF0) == 0xE0 && avail >= 3) { *cp = ((uint32_t)(c&0x0F)<<12)|((uint32_t)((uint8_t)p[1]&0x3F)<<6)|((uint32_t)(uint8_t)p[2]&0x3F); return 3; }
    if ((c & 0xF8) == 0xF0 && avail >= 4) { *cp = ((uint32_t)(c&0x07)<<18)|((uint32_t)((uint8_t)p[1]&0x3F)<<12)|((uint32_t)((uint8_t)p[2]&0x3F)<<6)|((uint32_t)(uint8_t)p[3]&0x3F); return 4; }
    return 0;
}

static void prompt_utf8_append(std::string &out, uint32_t cp) {
    if (cp < 0x80) out.push_back((char)cp);
    else if (cp < 0x800) { out.push_back((char)(0xC0|(cp>>6))); out.push_back((char)(0x80|(cp&0x3F))); }
    else if (cp < 0x10000) { out.push_back((char)(0xE0|(cp>>12))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
    else { out.push_back((char)(0xF0|(cp>>18))); out.push_back((char)(0x80|((cp>>12)&0x3F))); out.push_back((char)(0x80|((cp>>6)&0x3F))); out.push_back((char)(0x80|(cp&0x3F))); }
}

static bool prompt_is_cjk(uint32_t cp) { return cp >= 0x4E00 && cp <= 0x9FFF; }

// _combine_text from reference
static std::string prompt_combine_text(const std::string &text, const std::string &ref_text) {
    auto strip = [](const std::string &s) {
        size_t a = 0, b = s.size();
        while (a < b && (s[a]==' '||s[a]=='\t')) a++;
        while (b > a && (s[b-1]==' '||s[b-1]=='\t')) b--;
        return s.substr(a, b - a);
    };
    std::string raw = ref_text.empty() ? strip(text) : (strip(ref_text) + " " + strip(text));

    // Decode to codepoints, drop CR/LF, replace full-width parens
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < raw.size()) {
        uint32_t cp = 0;
        int n = prompt_utf8_decode(raw.data() + i, raw.size() - i, &cp);
        if (n == 0) { i++; continue; }
        i += n;
        if (cp == '\r' || cp == '\n') continue;
        if (cp == 0xFF08) cp = '(';
        else if (cp == 0xFF09) cp = ')';
        cps.push_back(cp);
    }

    // Collapse space/tab runs
    std::vector<uint32_t> collapsed;
    bool in_space = false;
    for (uint32_t cp : cps) {
        bool is_ws = (cp == ' ' || cp == '\t');
        if (is_ws) { if (!in_space) { collapsed.push_back(' '); in_space = true; } }
        else { collapsed.push_back(cp); in_space = false; }
    }

    // Drop spaces adjacent to CJK
    std::string out;
    for (size_t j = 0; j < collapsed.size(); j++) {
        if (collapsed[j] == ' ') {
            bool prev_cjk = (j > 0 && prompt_is_cjk(collapsed[j-1]));
            bool next_cjk = (j + 1 < collapsed.size() && prompt_is_cjk(collapsed[j+1]));
            if (prev_cjk || next_cjk) continue;
        }
        prompt_utf8_append(out, collapsed[j]);
    }
    return out;
}

// =====================================================================
// Non-verbal tag list and language resolution
// =====================================================================

static const char * const PROMPT_NONVERBAL_TAGS[] = {
    "[laughter]",    "[sigh]",        "[confirmation-en]",     "[question-en]", "[question-ah]",
    "[question-oh]", "[question-ei]", "[question-yi]",         "[surprise-ah]", "[surprise-oh]",
    "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
};
static const int PROMPT_NONVERBAL_N = 13;

// Language name -> ISO code table (from reference lang-map.h)
static const std::pair<const char *, const char *> LANG_NAME_TO_ID[] = {
    {"chinese", "zh"}, {"english", "en"}, {"japanese", "ja"}, {"korean", "ko"},
    {"french", "fr"}, {"german", "de"}, {"spanish", "es"}, {"italian", "it"},
    {"portuguese", "pt"}, {"russian", "ru"}, {"arabic", "ar"}, {"hindi", "hi"},
    {"bengali", "bn"}, {"turkish", "tr"}, {"vietnamese", "vi"}, {"thai", "th"},
    {"dutch", "nl"}, {"polish", "pl"}, {"romanian", "ro"}, {"czech", "cs"},
    {"swedish", "sv"}, {"danish", "da"}, {"finnish", "fi"}, {"norwegian", "no"},
    {"hungarian", "hu"}, {"greek", "el"}, {"hebrew", "he"}, {"indonesian", "id"},
    {"malay", "ms"}, {"filipino", "fil"}, {"swahili", "sw"}, {"burmese", "my"},
    {"catalan", "ca"}, {"croatian", "hr"}, {"serbian", "sr"}, {"bulgarian", "bg"},
    {"slovak", "sk"}, {"slovenian", "sl"}, {"estonian", "et"}, {"latvian", "lv"},
    {"lithuanian", "lt"}, {"ukrainian", "uk"}, {"tamil", "ta"}, {"telugu", "te"},
    {"kannada", "kn"}, {"malayalam", "ml"}, {"marathi", "mr"}, {"gujarati", "gu"},
    {"punjabi", "pa"}, {"persian", "fa"}, {"pashto", "ps"}, {"kurdish", "ku"},
    {"turkmen", "tk"}, {"kazakh", "kk"}, {"uzbek", "uz"}, {"tatar", "tt"},
    {"bashkir", "ba"}, {"chuvash", "cv"}, {"mongolian", "mn"}, {"nepali", "ne"},
    {"sinhala", "si"}, {"khmer", "km"}, {"lao", "lo"}, {"burmese", "my"},
    {"tibetan", "bo"}, {"uyghur", "ug"}, {"cantonese", "yue"},
    {"icelandic", "is"}, {"irish", "ga"}, {"welsh", "cy"}, {"basque", "eu"},
    {"galician", "gl"}, {"macedonian", "mk"}, {"albanian", "sq"}, {"bosnian", "bs"},
    {"maltese", "mt"}, {"luxembourgish", "lb"}, {"armenian", "hy"}, {"georgian", "ka"},
    {"azerbaijani", "az"}, {"tajik", "tg"}, {"afrikaans", "af"},
};

static std::string resolve_language(const std::string &language) {
    if (language.empty()) return "";
    std::string low = language;
    for (char &c : low) if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
    if (low == "none") return "";

    // Build lookup on first call
    static std::unordered_map<std::string, std::string> name_to_id;
    static std::unordered_set<std::string> valid_ids;
    if (name_to_id.empty()) {
        int n = sizeof(LANG_NAME_TO_ID) / sizeof(LANG_NAME_TO_ID[0]);
        for (int i = 0; i < n; i++) {
            name_to_id[LANG_NAME_TO_ID[i].first] = LANG_NAME_TO_ID[i].second;
            valid_ids.insert(LANG_NAME_TO_ID[i].second);
        }
    }
    if (valid_ids.count(language)) return language;
    auto it = name_to_id.find(low);
    return (it != name_to_id.end()) ? it->second : "";
}

// Tokenize text that may contain non-verbal tags, matching reference
// prompt_tts_tokenize_nonverbal. Each tag is tokenized independently.
static std::vector<int> prompt_tokenize_nonverbal(const OmniVoiceBPETokenizer *tok,
                                                   const std::string &text) {
    std::vector<int> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best_pos = std::string::npos;
        int best_idx = -1;
        for (int i = 0; i < PROMPT_NONVERBAL_N; i++) {
            size_t p = text.find(PROMPT_NONVERBAL_TAGS[i], pos);
            if (p != std::string::npos && p < best_pos) { best_pos = p; best_idx = i; }
        }
        if (best_idx < 0) {
            auto seg = tok->encode(text.substr(pos), false);
            ids.insert(ids.end(), seg.begin(), seg.end());
            break;
        }
        if (best_pos > pos) {
            auto seg = tok->encode(text.substr(pos, best_pos - pos), false);
            ids.insert(ids.end(), seg.begin(), seg.end());
        }
        auto tag_ids = tok->encode(PROMPT_NONVERBAL_TAGS[best_idx], false);
        ids.insert(ids.end(), tag_ids.begin(), tag_ids.end());
        pos = best_pos + strlen(PROMPT_NONVERBAL_TAGS[best_idx]);
    }
    return ids;
}

// =====================================================================
// OmniVoiceModel Implementation
// =====================================================================

OmniVoiceModel::OmniVoiceModel() { meta_.arch_name = "OmniVoice"; }
OmniVoiceModel::~OmniVoiceModel() {
    if (cpu_backend_) ggml_backend_free(cpu_backend_);
}

bool OmniVoiceModel::Load(const std::unique_ptr<rs_context_t> &ctx, ggml_backend_t backend) {
    backend_ = backend;

    // Create a CPU backend for the DAC vocoder — many small conv ops where
    // GPU kernel launch overhead dwarfs the actual compute (~300x slowdown).
    cpu_backend_ = ggml_backend_cpu_init();
    if (!cpu_backend_) {
        RS_LOG_ERR("OmniVoice: failed to create CPU backend for vocoder");
        return false;
    }

    if (!LoadLM(ctx)) return false;

    // Load codec tensors from the merged GGUF (now in ctx->gguf_data)
    if (!LoadCodec(ctx->gguf_data)) {
        RS_LOG_WARN("OmniVoice: codec load failed, only acoustic token generation available");
        codec_loaded_ = false;
    } else {
        codec_loaded_ = true;
    }

    RS_LOG_INFO("OmniVoice: model loaded successfully (codec=%s)", codec_loaded_ ? "yes" : "no");
    return true;
}

bool OmniVoiceModel::LoadLM(const std::unique_ptr<rs_context_t> &ctx) {
    if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
        RS_LOG_ERR("OmniVoice: Invalid context");
        return false;
    }

    gguf_context *ctx_gguf = ctx->ctx_gguf;
    ggml_context *gguf_data = ctx->gguf_data;

    auto safe_u32 = [&](const char *key, int32_t default_val = 0) -> int32_t {
        int idx = gguf_find_key(ctx_gguf, key);
        return (idx >= 0) ? (int32_t)gguf_get_val_u32(ctx_gguf, idx) : default_val;
    };
    auto safe_f32 = [&](const char *key, float default_val = 0.0f) -> float {
        int idx = gguf_find_key(ctx_gguf, key);
        return (idx >= 0) ? gguf_get_val_f32(ctx_gguf, idx) : default_val;
    };

    // Load hyperparameters from GGUF KV (with fallbacks for all keys)
    hparams_.n_layer    = safe_u32("omnivoice-lm.block_count", 28);
    hparams_.n_embd     = safe_u32("omnivoice-lm.embedding_length", 1024);
    hparams_.n_head     = safe_u32("omnivoice-lm.attention.head_count", 16);
    hparams_.n_head_kv  = safe_u32("omnivoice-lm.attention.head_count_kv", 8);
    hparams_.head_dim   = safe_u32("omnivoice-lm.attention.key_length", 128);
    hparams_.n_ff       = safe_u32("omnivoice-lm.feed_forward_length", 3072);
    hparams_.n_codebooks = safe_u32("omnivoice.num_audio_codebook", 8);
    hparams_.rope_theta = safe_f32("omnivoice-lm.rope.freq_base", 1000000.0f);
    hparams_.eps         = safe_f32("omnivoice-lm.attention.layer_norm_rms_epsilon", 1e-6f);

    // Audio codec parameters
    audio_vocab_size_ = safe_u32("omnivoice.audio_vocab_size", 1025);
    audio_mask_id_    = safe_u32("omnivoice.audio_mask_id", 1024);

    if (audio_vocab_size_ <= 0) audio_vocab_size_ = 1025;
    if (audio_mask_id_ < 0) audio_mask_id_ = 1024;

    // Codebook sizes: from audio_codebook_weights array if available, else uniform
    int cw_idx = gguf_find_key(ctx_gguf, "omnivoice.audio_codebook_weights");
    if (cw_idx >= 0) {
        int n_cw = (int)gguf_get_arr_n(ctx_gguf, cw_idx);
        for (int c = 0; c < std::min(hparams_.n_codebooks, n_cw); ++c)
            hparams_.codebook_sizes[c] = audio_vocab_size_;
    } else {
        for (int c = 0; c < hparams_.n_codebooks; ++c)
            hparams_.codebook_sizes[c] = audio_vocab_size_;
    }

    hparams_.text_vocab_size = safe_u32("omnivoice-lm.vocab_size", 151936);

    meta_.audio_sample_rate = hparams_.audio_sample_rate;
    meta_.n_mels = 0;
    meta_.vocab_size = hparams_.text_vocab_size;

    RS_LOG_INFO("OmniVoice LM: layers=%d embd=%d heads=%d/%d hdim=%d ff=%d codebooks=%d",
                hparams_.n_layer, hparams_.n_embd, hparams_.n_head,
                hparams_.n_head_kv, hparams_.head_dim, hparams_.n_ff, hparams_.n_codebooks);

    // Map tensors from gguf_data
    std::map<std::string, struct ggml_tensor *> tensors;
    int n_tensors = gguf_get_n_tensors(ctx_gguf);
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(ctx_gguf, i);
        struct ggml_tensor *t = ggml_get_tensor(gguf_data, name);
        if (t) tensors[name] = t;
    }

    // Initialize LLM model for Qwen3 backbone
    llm_model_ = std::make_shared<llm_model>();
    if (!llm_model_->load_metadata_from_gguf(ctx_gguf)) {
        RS_LOG_ERR("OmniVoice: failed to load LLM metadata");
        return false;
    }
    if (!llm_model_->map_tensors_qwen3(tensors)) {
        RS_LOG_ERR("OmniVoice: failed to map Qwen3 tensors");
        return false;
    }

    // Verify layer 0 weights
    {
        auto &l0 = llm_model_->layers()[0];
        RS_LOG_INFO("L0 weights: wq=%p(%zu) wk=%p(%zu) wv=%p(%zu) wo=%p(%zu)",
                    (void*)l0.wq, l0.wq ? ggml_nelements(l0.wq) : 0,
                    (void*)l0.wk, l0.wk ? ggml_nelements(l0.wk) : 0,
                    (void*)l0.wv, l0.wv ? ggml_nelements(l0.wv) : 0,
                    (void*)l0.wo, l0.wo ? ggml_nelements(l0.wo) : 0);
        RS_LOG_INFO("L0 weights: ffn_gate=%p(%zu) ffn_up=%p(%zu) ffn_down=%p(%zu)",
                    (void*)l0.ffn_gate, l0.ffn_gate ? ggml_nelements(l0.ffn_gate) : 0,
                    (void*)l0.ffn_up, l0.ffn_up ? ggml_nelements(l0.ffn_up) : 0,
                    (void*)l0.ffn_down, l0.ffn_down ? ggml_nelements(l0.ffn_down) : 0);
        RS_LOG_INFO("L0 norms: attn_norm=%p q_norm=%p k_norm=%p ffn_norm=%p",
                    (void*)l0.attn_norm, (void*)l0.attn_q_norm,
                    (void*)l0.attn_k_norm, (void*)l0.ffn_norm);
        if (l0.wq && l0.wq->data) {
            float *wq0 = (float*)l0.wq->data;
            RS_LOG_INFO("L0 wq[0..3]: %.6f %.6f %.6f %.6f", wq0[0], wq0[1], wq0[2], wq0[3]);
        }
    }

    // Map audio-specific tensors
    if (!MapTensors(tensors)) return false;

    // Load BPE tokenizer from the same GGUF
    // We need the file path to re-open for BPE loading
    // Try to get it from params
    std::string model_path = "";
    if (ctx->params.model_path) model_path = ctx->params.model_path;

    if (!model_path.empty()) {
        if (!bpe_.load_from_gguf(model_path.c_str())) {
            RS_LOG_WARN("OmniVoice: BPE tokenizer load failed, falling back to llm_vocab");
        }
    }

    return true;
}

bool OmniVoiceModel::MapTensors(std::map<std::string, struct ggml_tensor *> &tensors) {
    try {
        // Try multiple naming conventions for text embeddings
        const char *text_embd_names[] = {
            "llm.embed_tokens.weight",         // OmniVoice GGUF convention
            "model.embed_tokens.weight",       // raw safetensors name
            "llm.model.embed_tokens.weight",   // standard Qwen3 GGUF
            nullptr
        };
        for (int i = 0; text_embd_names[i]; ++i) {
            auto it = tensors.find(text_embd_names[i]);
            if (it != tensors.end()) { text_embd_ = it->second; break; }
        }

        // Try multiple naming conventions for output norm
        const char *output_norm_names[] = {
            "llm.norm.weight",                 // OmniVoice GGUF convention
            "model.norm.weight",               // raw safetensors name
            "llm.model.norm.weight",           // standard Qwen3 GGUF
            nullptr
        };
        for (int i = 0; output_norm_names[i]; ++i) {
            auto it = tensors.find(output_norm_names[i]);
            if (it != tensors.end()) { output_norm_ = it->second; break; }
        }

        // Try per-codebook head tensors first
        bool has_per_codebook = false;
        for (int c = 0; c < hparams_.n_codebooks; ++c) {
            char key[64];
            snprintf(key, sizeof(key), "codebook_head.%d.weight", c);
            auto it_w = tensors.find(key);
            if (it_w != tensors.end()) {
                codebook_head_weight_[c] = it_w->second;
                has_per_codebook = true;
            }
            snprintf(key, sizeof(key), "codebook_head.%d.bias", c);
            auto it_b = tensors.find(key);
            if (it_b != tensors.end()) codebook_head_bias_[c] = it_b->second;
        }

        // Try per-codebook embedding tensors
        for (int c = 0; c < hparams_.n_codebooks; ++c) {
            char key[64];
            snprintf(key, sizeof(key), "acoustic_embd.%d.weight", c);
            auto it = tensors.find(key);
            if (it != tensors.end()) acoustic_embeddings_[c] = it->second;
        }

        // Fallback: combined tensors
        if (!has_per_codebook) {
            // Standard reference naming: audio_embeddings.weight + audio_heads.weight
            auto audio_embd = tensors.find("audio_embeddings.weight");
            if (audio_embd != tensors.end()) combined_audio_embeddings_ = audio_embd->second;

            auto audio_heads = tensors.find("audio_heads.weight");
            if (audio_heads != tensors.end()) combined_audio_heads_ = audio_heads->second;

            // OmniVoice GGUF convention: single combined [H, 2*K*V] tensor named
            // omnivoice.audio_codebook_weights. Store it so the forward pass can
            // split it into embeddings (first K*V cols) and heads (last K*V cols).
            if (!combined_audio_embeddings_ && !combined_audio_heads_) {
                auto it = tensors.find("omnivoice.audio_codebook_weights");
                if (it != tensors.end()) combined_codebook_weights_ = it->second;
            }

            RS_LOG_INFO("OmniVoice: using combined audio tensors (embd=%s heads=%s combined=%s)",
                        combined_audio_embeddings_ ? "yes" : "no",
                        combined_audio_heads_ ? "yes" : "no",
                        combined_codebook_weights_ ? "yes" : "no");
        }

        RS_LOG_INFO("OmniVoice: mapped audio tensors (heads=%d, embeddings=%d)",
                    hparams_.n_codebooks, hparams_.n_codebooks);
        return true;
    } catch (const std::out_of_range &e) {
        RS_LOG_ERR("OmniVoice: tensor mapping failed - %s", e.what());
        return false;
    }
}

bool OmniVoiceModel::LoadCodec(struct ggml_context *gguf_data) {
    if (!gguf_data) return false;

    // Create a local CPU ggml_context for codec tensors.
    // The main gguf_data tensors are on the primary backend (GPU), but the
    // DAC vocoder runs on CPU where many small conv ops are much faster.
    // Use no_alloc=true (heap alloc for tensor structs) — the actual tensor
    // data is allocated later via ggml_backend_alloc_ctx_tensors on cpu_backend_.
    const size_t local_ctx_size = ggml_tensor_overhead() * 1024 + (1 << 20);
    struct ggml_init_params lp = { local_ctx_size, nullptr, true };
    struct ggml_context *local_ctx = ggml_init(lp);
    if (!local_ctx) return false;

    // Helper: look up tensor in gguf_data and duplicate it into local_ctx.
    // Returns the local-ctx tensor (or nullptr if not found in gguf_data).
    auto dup = [&](const char *name) -> struct ggml_tensor* {
        struct ggml_tensor *src = ggml_get_tensor(gguf_data, name);
        if (!src) return nullptr;
        struct ggml_tensor *dst = nullptr;
        int nd = ggml_n_dims(src);
        if (nd == 1)
            dst = ggml_new_tensor_1d(local_ctx, src->type, src->ne[0]);
        else if (nd == 2)
            dst = ggml_new_tensor_2d(local_ctx, src->type, src->ne[0], src->ne[1]);
        else if (nd == 3)
            dst = ggml_new_tensor_3d(local_ctx, src->type, src->ne[0], src->ne[1], src->ne[2]);
        else
            dst = ggml_new_tensor_4d(local_ctx, src->type, src->ne[0], src->ne[1], src->ne[2], src->ne[3]);
        ggml_set_name(dst, name);
        return dst;
    };

    // Load codec metadata from main ggml_context
    rvq_.num_codebooks = 8;
    rvq_.codebook_dim = 64;
    // Read actual codebook_size from the embed tensor.
    // After conversion transpose, embed is [dim, vocab_size] in ggml (ne[0]=dim, ne[1]=vocab).
    {
        struct ggml_tensor *emb0 = ggml_get_tensor(gguf_data, "quantizer.quantizers.0.codebook.embed");
        if (emb0 && ggml_n_dims(emb0) == 2) {
            // ne[0] is the smaller dim (embedding dim), ne[1] is vocab size
            rvq_.codebook_dim  = (int)emb0->ne[0];
            rvq_.codebook_size = (int)emb0->ne[1];
        } else {
            rvq_.codebook_size = 1024;
        }
    }
    // RVQ codebook_size is number of real tokens (excludes mask token).
    // The LLM audio vocab includes mask token: V = codebook_size + 1, mask_id = codebook_size.
    audio_vocab_size_ = rvq_.codebook_size + 1;
    audio_mask_id_ = rvq_.codebook_size;

    char tname[256];

    // RVQ codebooks
    for (int k = 0; k < rvq_.num_codebooks; k++) {
        snprintf(tname, sizeof(tname), "quantizer.quantizers.%d.codebook.embed", k);
        rvq_.cb[k].embed = dup(tname);
        snprintf(tname, sizeof(tname), "quantizer.quantizers.%d.project_out.weight", k);
        rvq_.cb[k].project_out_w = dup(tname);
        snprintf(tname, sizeof(tname), "quantizer.quantizers.%d.project_out.bias", k);
        rvq_.cb[k].project_out_b = dup(tname);
    }

    // fc2 (1024 -> 256)
    fc2_w_ = dup("fc2.weight");
    fc2_b_ = dup("fc2.bias");

    // DAC decoder conv1
    dac_.c1w = dup("acoustic_decoder.conv1.weight");
    dac_.c1b = dup("acoustic_decoder.conv1.bias");

    static const int strides[] = {8, 5, 4, 2, 3};
    static const int in_chs[]  = {1024, 512, 256, 128, 64};
    static const int out_chs[] = {512, 256, 128, 64, 32};
    static const int dilations[] = {1, 3, 9};

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACBlockWeights &b = dac_.blk[i];
        b.in_ch = in_chs[i]; b.out_ch = out_chs[i];
        b.stride = strides[i]; b.kernel = 2 * b.stride;
        b.pad = (b.stride + 1) / 2;
        b.output_pad = b.stride % 2;

        std::string pfx = "acoustic_decoder.block." + std::to_string(i);
        snprintf(tname, sizeof(tname), "%s.snake1.alpha", pfx.c_str());
        b.s1.a = dup(tname);
        snprintf(tname, sizeof(tname), "%s.conv_t1.weight", pfx.c_str());
        b.ctw = dup(tname);
        snprintf(tname, sizeof(tname), "%s.conv_t1.bias", pfx.c_str());
        b.ctb = dup(tname);

        for (int r = 0; r < DAC_RES_UNITS; r++) {
            DACResUnitWeights &ru = b.ru[r];
            ru.dilation = dilations[r];
            std::string rp = pfx + ".res_unit" + std::to_string(r + 1);
            snprintf(tname, sizeof(tname), "%s.snake1.alpha", rp.c_str());
            ru.s1.a = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv1.weight", rp.c_str());
            ru.c1w = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv1.bias", rp.c_str());
            ru.c1b = dup(tname);
            snprintf(tname, sizeof(tname), "%s.snake2.alpha", rp.c_str());
            ru.s2.a = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv2.weight", rp.c_str());
            ru.c2w = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv2.bias", rp.c_str());
            ru.c2b = dup(tname);
        }
    }

    dac_.s_final.a = dup("acoustic_decoder.snake1.alpha");
    dac_.c2w = dup("acoustic_decoder.conv2.weight");
    dac_.c2b = dup("acoustic_decoder.conv2.bias");

    // --- Load encoder-side weights ---
    // RVQ project_in (encode path)
    for (int k = 0; k < rvq_.num_codebooks; k++) {
        snprintf(tname, sizeof(tname), "quantizer.quantizers.%d.project_in.weight", k);
        rvq_.cb[k].project_in_w = dup(tname);
        snprintf(tname, sizeof(tname), "quantizer.quantizers.%d.project_in.bias", k);
        rvq_.cb[k].project_in_b = dup(tname);
    }

    // fc (encode path: 1024 -> 1024)
    fc_w_ = dup("fc.weight");
    fc_b_ = dup("fc.bias");

    // DAC encoder
    dac_enc_.c1w = dup("acoustic_encoder.conv1.weight");
    dac_enc_.c1b = dup("acoustic_encoder.conv1.bias");
    dac_enc_.c2w = dup("acoustic_encoder.conv2.weight");
    dac_enc_.c2b = dup("acoustic_encoder.conv2.bias");
    dac_enc_.s_final.a = dup("acoustic_encoder.snake1.alpha");

    static const int enc_strides[] = {8, 5, 4, 2, 3};
    static const int enc_in_chs[]  = {64, 128, 256, 512, 1024};
    static const int enc_out_chs[] = {128, 256, 512, 1024, 2048};
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACEncBlockWeights &b = dac_enc_.blk[i];
        b.in_ch = enc_in_chs[i]; b.out_ch = enc_out_chs[i];
        b.stride = enc_strides[i]; b.kernel = 2 * b.stride;
        b.pad = (b.stride + 1) / 2;
        std::string pfx = "acoustic_encoder.block." + std::to_string(i);
        snprintf(tname, sizeof(tname), "%s.snake1.alpha", pfx.c_str());
        b.s1.a = dup(tname);
        snprintf(tname, sizeof(tname), "%s.conv1.weight", pfx.c_str());
        b.cw = dup(tname);
        snprintf(tname, sizeof(tname), "%s.conv1.bias", pfx.c_str());
        b.cb = dup(tname);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            DACResUnitWeights &ru = b.ru[r];
            ru.dilation = dilations[r];
            std::string rp = pfx + ".res_unit" + std::to_string(r + 1);
            snprintf(tname, sizeof(tname), "%s.snake1.alpha", rp.c_str());
            ru.s1.a = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv1.weight", rp.c_str());
            ru.c1w = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv1.bias", rp.c_str());
            ru.c1b = dup(tname);
            snprintf(tname, sizeof(tname), "%s.snake2.alpha", rp.c_str());
            ru.s2.a = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv2.weight", rp.c_str());
            ru.c2w = dup(tname);
            snprintf(tname, sizeof(tname), "%s.conv2.bias", rp.c_str());
            ru.c2b = dup(tname);
        }
    }

    // HuBERT feature extractor (semantic_model prefix)
    for (int i = 0; i < 7; i++) {
        snprintf(tname, sizeof(tname), "semantic_model.feature_extractor.conv_layers.%d.conv.weight", i);
        hubert_feat_.conv[i].conv_w = dup(tname);
        if (i == 0) {
            snprintf(tname, sizeof(tname), "semantic_model.feature_extractor.conv_layers.%d.layer_norm.weight", i);
            hubert_feat_.conv[i].ln_w = dup(tname);
            snprintf(tname, sizeof(tname), "semantic_model.feature_extractor.conv_layers.%d.layer_norm.bias", i);
            hubert_feat_.conv[i].ln_b = dup(tname);
        }
    }

    // HuBERT feature projection
    hubert_proj_.ln_w   = dup("semantic_model.feature_projection.layer_norm.weight");
    hubert_proj_.ln_b   = dup("semantic_model.feature_projection.layer_norm.bias");
    hubert_proj_.proj_w = dup("semantic_model.feature_projection.projection.weight");
    hubert_proj_.proj_b = dup("semantic_model.feature_projection.projection.bias");

    // HuBERT encoder init (pos_conv_embed + first LayerNorm)
    hubert_enc_init_.pos_conv_w = dup("semantic_model.encoder.pos_conv_embed.conv.weight");
    hubert_enc_init_.pos_conv_b = dup("semantic_model.encoder.pos_conv_embed.conv.bias");
    if (hubert_enc_init_.pos_conv_w) {
        RS_LOG_INFO("Codec encoder: pos_conv_w ne=(%lld,%lld,%lld) type=%d",
                    (long long)hubert_enc_init_.pos_conv_w->ne[0],
                    (long long)hubert_enc_init_.pos_conv_w->ne[1],
                    (long long)hubert_enc_init_.pos_conv_w->ne[2],
                    (int)hubert_enc_init_.pos_conv_w->type);
    }
    hubert_enc_init_.ln_w = dup("semantic_model.encoder.layer_norm.weight");
    hubert_enc_init_.ln_b = dup("semantic_model.encoder.layer_norm.bias");

    // HuBERT encoder layers (12 layers)
    for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
        HubertLayerWeights &l = hubert_layers_[i];
        char pfx[64]; snprintf(pfx, sizeof(pfx), "semantic_model.encoder.layers.%d", i);
        snprintf(tname, sizeof(tname), "%s.layer_norm.weight", pfx);
        l.ln_attn_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.layer_norm.bias", pfx);
        l.ln_attn_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.q_proj.weight", pfx);
        l.attn.q_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.q_proj.bias", pfx);
        l.attn.q_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.k_proj.weight", pfx);
        l.attn.k_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.k_proj.bias", pfx);
        l.attn.k_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.v_proj.weight", pfx);
        l.attn.v_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.v_proj.bias", pfx);
        l.attn.v_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.out_proj.weight", pfx);
        l.attn.o_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.attention.out_proj.bias", pfx);
        l.attn.o_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.final_layer_norm.weight", pfx);
        l.ln_ffn_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.final_layer_norm.bias", pfx);
        l.ln_ffn_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.feed_forward.intermediate_dense.weight", pfx);
        l.ffn.w1_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.feed_forward.intermediate_dense.bias", pfx);
        l.ffn.w1_b = dup(tname);
        snprintf(tname, sizeof(tname), "%s.feed_forward.output_dense.weight", pfx);
        l.ffn.w2_w = dup(tname);
        snprintf(tname, sizeof(tname), "%s.feed_forward.output_dense.bias", pfx);
        l.ffn.w2_b = dup(tname);
    }

    // Semantic encoder (encoder_semantic)
    sem_enc_.c1w = dup("encoder_semantic.conv.weight");
    for (int i = 0; i < 2; i++) {
        char pfx[64]; snprintf(pfx, sizeof(pfx), "encoder_semantic.conv_blocks.%d", i);
        snprintf(tname, sizeof(tname), "%s.conv.weight", pfx);
        sem_enc_.blk[i].cw = dup(tname);
        snprintf(tname, sizeof(tname), "%s.conv.bias", pfx);
        sem_enc_.blk[i].cb = dup(tname);
        for (int r = 0; r < 2; r++) {
            snprintf(tname, sizeof(tname), "%s.res_units.%d.conv1.weight", pfx, r);
            sem_enc_.blk[i].ru[r].c1w = dup(tname);
            snprintf(tname, sizeof(tname), "%s.res_units.%d.conv2.weight", pfx, r);
            sem_enc_.blk[i].ru[r].c2w = dup(tname);
        }
    }

    // Store the local ggml_context for cleanup
    dac_.weight_ctx = local_ctx;

    // ggml_new_tensor_* with no_alloc=true heap-allocates tensor data.
    // ggml_backend_alloc_ctx_tensors only allocates tensors where data==NULL.
    // Clear the heap pointers so the backend buffer is allocated correctly.
    for (struct ggml_tensor *t = ggml_get_first_tensor(local_ctx);
         t != nullptr; t = ggml_get_next_tensor(local_ctx, t)) {
        if (t->data != nullptr) {
            free(t->data);
            t->data = nullptr;
        }
    }

    // Allocate weights on CPU backend — vocoder runs on CPU for speed
    dac_.weight_buf = ggml_backend_alloc_ctx_tensors(local_ctx, cpu_backend_);
    if (!dac_.weight_buf) {
        RS_LOG_ERR("OmniVoice: failed to allocate codec weight buffer");
        ggml_free(local_ctx); dac_.weight_ctx = nullptr;
        return false;
    }

    // Copy tensor data from main gguf_data (GPU) to local_ctx (CPU)
    {
        std::vector<char> rbuf;

        // Build the list of tensor names to copy — same names we just mapped above.
        // We re-generate the name list rather than store pointers because some
        // tensors may be nullptr (skipped during mapping).
        std::vector<std::string> names;

        auto add_name = [&](const char *name) {
            if (ggml_get_tensor(local_ctx, name))
                names.push_back(name);
        };

        // RVQ codebooks (8 codebooks, decode + encode paths)
        for (int k = 0; k < rvq_.num_codebooks; k++) {
            char buf[128];
            snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.codebook.embed", k); add_name(buf);
            snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_out.weight", k); add_name(buf);
            snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_out.bias", k); add_name(buf);
            snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_in.weight", k); add_name(buf);
            snprintf(buf, sizeof(buf), "quantizer.quantizers.%d.project_in.bias", k); add_name(buf);
        }

        // fc / fc2
        add_name("fc.weight"); add_name("fc.bias");
        add_name("fc2.weight"); add_name("fc2.bias");

        // DAC decoder (conv1, conv2, snake final, 5 blocks × 3 res_units)
        add_name("acoustic_decoder.conv1.weight"); add_name("acoustic_decoder.conv1.bias");
        add_name("acoustic_decoder.conv2.weight"); add_name("acoustic_decoder.conv2.bias");
        add_name("acoustic_decoder.snake1.alpha");
        for (int b = 0; b < DAC_NUM_BLOCKS; b++) {
            char pfx[64], buf[128];
            snprintf(pfx, sizeof(pfx), "acoustic_decoder.block.%d", b);
            snprintf(buf, sizeof(buf), "%s.snake1.alpha", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.conv_t1.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.conv_t1.bias", pfx); add_name(buf);
            for (int r = 0; r < DAC_RES_UNITS; r++) {
                snprintf(buf, sizeof(buf), "%s.res_unit%d.snake1.alpha", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv1.weight", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv1.bias", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.snake2.alpha", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv2.weight", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv2.bias", pfx, r+1); add_name(buf);
            }
        }

        // DAC encoder (conv1, conv2, snake final, 5 blocks × 3 res_units)
        add_name("acoustic_encoder.conv1.weight"); add_name("acoustic_encoder.conv1.bias");
        add_name("acoustic_encoder.conv2.weight"); add_name("acoustic_encoder.conv2.bias");
        add_name("acoustic_encoder.snake1.alpha");
        for (int b = 0; b < DAC_NUM_BLOCKS; b++) {
            char pfx[64], buf[128];
            snprintf(pfx, sizeof(pfx), "acoustic_encoder.block.%d", b);
            snprintf(buf, sizeof(buf), "%s.snake1.alpha", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.conv1.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.conv1.bias", pfx); add_name(buf);
            for (int r = 0; r < DAC_RES_UNITS; r++) {
                snprintf(buf, sizeof(buf), "%s.res_unit%d.snake1.alpha", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv1.weight", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv1.bias", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.snake2.alpha", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv2.weight", pfx, r+1); add_name(buf);
                snprintf(buf, sizeof(buf), "%s.res_unit%d.conv2.bias", pfx, r+1); add_name(buf);
            }
        }

        // HuBERT feature extractor (7 conv layers, layer 0 has norm)
        for (int i = 0; i < 7; i++) {
            char buf[128];
            snprintf(buf, sizeof(buf), "semantic_model.feature_extractor.conv_layers.%d.conv.weight", i); add_name(buf);
            if (i == 0) {
                snprintf(buf, sizeof(buf), "semantic_model.feature_extractor.conv_layers.%d.layer_norm.weight", i); add_name(buf);
                snprintf(buf, sizeof(buf), "semantic_model.feature_extractor.conv_layers.%d.layer_norm.bias", i); add_name(buf);
            }
        }

        // HuBERT feature projection, encoder init
        add_name("semantic_model.feature_projection.layer_norm.weight");
        add_name("semantic_model.feature_projection.layer_norm.bias");
        add_name("semantic_model.feature_projection.projection.weight");
        add_name("semantic_model.feature_projection.projection.bias");
        add_name("semantic_model.encoder.pos_conv_embed.conv.weight");
        add_name("semantic_model.encoder.pos_conv_embed.conv.bias");
        add_name("semantic_model.encoder.layer_norm.weight");
        add_name("semantic_model.encoder.layer_norm.bias");

        // HuBERT encoder layers (12 layers)
        for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
            char pfx[64], buf[128];
            snprintf(pfx, sizeof(pfx), "semantic_model.encoder.layers.%d", i);
            snprintf(buf, sizeof(buf), "%s.layer_norm.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.layer_norm.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.q_proj.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.q_proj.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.k_proj.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.k_proj.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.v_proj.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.v_proj.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.out_proj.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.attention.out_proj.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.final_layer_norm.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.final_layer_norm.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.feed_forward.intermediate_dense.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.feed_forward.intermediate_dense.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.feed_forward.output_dense.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.feed_forward.output_dense.bias", pfx); add_name(buf);
        }

        // Semantic encoder
        add_name("encoder_semantic.conv.weight");
        for (int i = 0; i < 2; i++) {
            char pfx[64], buf[128];
            snprintf(pfx, sizeof(pfx), "encoder_semantic.conv_blocks.%d", i);
            snprintf(buf, sizeof(buf), "%s.conv.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.conv.bias", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.res_units.0.conv1.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.res_units.0.conv2.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.res_units.1.conv1.weight", pfx); add_name(buf);
            snprintf(buf, sizeof(buf), "%s.res_units.1.conv2.weight", pfx); add_name(buf);
        }

        // Single copy loop
        for (const auto &name : names) {
            struct ggml_tensor *src = ggml_get_tensor(gguf_data, name.c_str());
            struct ggml_tensor *dst = ggml_get_tensor(local_ctx, name.c_str());
            if (!src || !dst) continue;
            size_t sz_src = ggml_nbytes(src);
            if (sz_src == 0) continue;
            if (rbuf.size() < sz_src) rbuf.resize(sz_src);
            ggml_backend_tensor_get(src, rbuf.data(), 0, sz_src);
            ggml_backend_tensor_set(dst, rbuf.data(), 0, sz_src);
        }
    }

    // Precompute inv_b for all snake activations (handle F16 and F32)
    auto compute_inv_b = [&](struct ggml_tensor *alpha, std::vector<float> &inv_b) {
        if (!alpha) return;
        size_t elem_sz = ggml_type_size(alpha->type);
        int n_elems = (int)(ggml_nbytes(alpha) / elem_sz);
        int C = (int)alpha->ne[1];
        if (C <= 1 && n_elems > 1) C = n_elems;
        inv_b.resize(C);
        std::vector<char> buf(n_elems * elem_sz);
        ggml_backend_tensor_get(alpha, buf.data(), 0, n_elems * elem_sz);
        for (int i = 0; i < C; i++) {
            float val;
            if (alpha->type == GGML_TYPE_F16)
                val = ggml_fp16_to_fp32(((ggml_fp16_t *)buf.data())[i]);
            else
                val = ((float *)buf.data())[i];
            inv_b[i] = 1.0f / (val + 1e-9f);
        }
    };

    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        compute_inv_b(dac_.blk[i].s1.a, dac_.blk[i].s1.inv_b);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            compute_inv_b(dac_.blk[i].ru[r].s1.a, dac_.blk[i].ru[r].s1.inv_b);
            compute_inv_b(dac_.blk[i].ru[r].s2.a, dac_.blk[i].ru[r].s2.inv_b);
        }
    }
    compute_inv_b(dac_.s_final.a, dac_.s_final.inv_b);

    // Debug: check which weights loaded successfully
    RS_LOG_INFO("OmniVoice Codec: RVQ=%dx%d dim=%d, DAC=5 blocks upsample=960x",
                rvq_.num_codebooks, rvq_.codebook_size, rvq_.codebook_dim);
    RS_LOG_INFO("Codec: fc2_w=%s fc2_b=%s c1w=%s c1b=%s c2w=%s c2b=%s sfinal_a=%s",
                fc2_w_ ? "Y" : "N", fc2_b_ ? "Y" : "N",
                dac_.c1w ? "Y" : "N", dac_.c1b ? "Y" : "N",
                dac_.c2w ? "Y" : "N", dac_.c2b ? "Y" : "N",
                dac_.s_final.a ? "Y" : "N");
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACBlockWeights &b = dac_.blk[i];
        int ru_ok = 0;
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            if (b.ru[r].c1w && b.ru[r].c2w) ru_ok++;
        }
        RS_LOG_INFO("Codec: blk%d s1=%s ctw=%s ctb=%s ru_ok=%d/%d in=%d out=%d",
                    i, b.s1.a ? "Y":"N", b.ctw ? "Y":"N", b.ctb ? "Y":"N",
                    ru_ok, DAC_RES_UNITS, b.in_ch, b.out_ch);
    }
    // DAC encoder snake inv_b
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        compute_inv_b(dac_enc_.blk[i].s1.a, dac_enc_.blk[i].s1.inv_b);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            compute_inv_b(dac_enc_.blk[i].ru[r].s1.a, dac_enc_.blk[i].ru[r].s1.inv_b);
            compute_inv_b(dac_enc_.blk[i].ru[r].s2.a, dac_enc_.blk[i].ru[r].s2.inv_b);
        }
    }
    compute_inv_b(dac_enc_.s_final.a, dac_enc_.s_final.inv_b);

    // Precompute RVQ embed_sq (||embed[:,j]||^2 for each codebook entry)
    // embed after conversion: [ne0=dim, ne1=vocab] in ggml
    for (int k = 0; k < rvq_.num_codebooks; k++) {
        if (!rvq_.cb[k].embed) continue;
        int cb_dim = rvq_.codebook_dim;
        int cb_size = rvq_.codebook_size;
        size_t elem_size = ggml_type_size(rvq_.cb[k].embed->type);
        rvq_.cb[k].embed_sq_cpu.resize(cb_size);
        std::vector<char> emb_raw(cb_dim * cb_size * elem_size);
        ggml_backend_tensor_get(rvq_.cb[k].embed, emb_raw.data(), 0, emb_raw.size());
        for (int j = 0; j < cb_size; j++) {
            float s = 0;
            for (int d = 0; d < cb_dim; d++) {
                float val;
                // ggml layout [ne0=dim, ne1=vocab]: element (ne0=d, ne1=j) at flat d + j*cb_dim
                if (rvq_.cb[k].embed->type == GGML_TYPE_F16)
                    val = ggml_fp16_to_fp32(((ggml_fp16_t *)emb_raw.data())[d + j * cb_dim]);
                else
                    val = ((float *)emb_raw.data())[d + j * cb_dim];
                s += val * val;
            }
            rvq_.cb[k].embed_sq_cpu[j] = s;
        }
    }

    // Log encoder status
    int enc_ok = 0;
    for (int k = 0; k < rvq_.num_codebooks; k++)
        if (rvq_.cb[k].project_in_w) enc_ok++;
    RS_LOG_INFO("Codec encoder: RVQ project_in=%d/8 fc=%s dac_enc=%s hubert=%s sem_enc=%s",
                enc_ok, fc_w_ ? "Y":"N", dac_enc_.c1w ? "Y":"N",
                hubert_feat_.conv[0].conv_w ? "Y":"N", sem_enc_.c1w ? "Y":"N");

    for (int k = 0; k < rvq_.num_codebooks; k++) {
        if (!rvq_.cb[k].embed)
            RS_LOG_WARN("Codec: RVQ cb[%d].embed is NULL", k);
        if (!rvq_.cb[k].project_out_w)
            RS_LOG_WARN("Codec: RVQ cb[%d].project_out_w is NULL", k);
    }
    return true;
}

// =====================================================================
// Audio resampling helper (simple linear interpolation)
// =====================================================================

// Hann-windowed sinc resampler matching torchaudio.functional.resample.
// Replaces the linear-interpolation resampler which caused audible aliasing
// artifacts when downsampling 24kHz→16kHz for HuBERT feature extraction.
// The HuBERT model was trained on torchaudio-resampled audio; feeding it
// linear-interpolated audio produces subtly wrong features that degrade
// voice cloning quality.

static int resample_gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

static std::vector<float> audio_resample_linear(const float *in, int n_in,
                                                  int sr_in, int sr_out) {
    if (!in || n_in <= 0) return {};
    if (sr_in == sr_out) return std::vector<float>(in, in + n_in);

    const int    g    = resample_gcd(sr_in, sr_out);
    const int    orig = sr_in / g;   // up/down factors after gcd reduction
    const int    newf = sr_out / g;
    const double rolloff = 0.99;
    const int    lpfw = 6;           // torchaudio default lowpass_filter_width

    // Build Hann-sinc polyphase kernel, matching torchaudio _get_sinc_resample_kernel.
    int    base_int = (orig < newf) ? orig : newf;
    double base     = (double)base_int * rolloff;
    int    width    = (int)std::ceil((double)lpfw * (double)orig / base);
    int    K        = 2 * width + orig;
    double scale    = base / (double)orig;
    double inv_o    = 1.0 / (double)orig;
    double inv_n    = 1.0 / (double)newf;

    std::vector<float> kernel((size_t)newf * (size_t)K);
    for (int j = 0; j < newf; j++) {
        double t_off = (double)(-j) * inv_n;
        for (int k = 0; k < K; k++) {
            double idx_k = (double)(k - width) * inv_o;
            double t     = (t_off + idx_k) * base;
            if (t < -lpfw) t = -lpfw;
            if (t > lpfw) t = lpfw;

            double w = std::cos(t * M_PI / (double)lpfw / 2.0);
            w = w * w;  // Hann window: cos² = (1+cos)/2

            double tp   = t * M_PI;
            double sinc = (tp == 0.0) ? 1.0 : std::sin(tp) / tp;
            kernel[(size_t)j * (size_t)K + (size_t)k] = (float)(sinc * w * scale);
        }
    }

    // Apply: pad (width zeros on both sides + orig for polyphase alignment),
    // then strided conv1d over the padded signal.
    long long target = (long long)std::ceil((double)sr_out * (double)n_in / (double)sr_in);
    int       Np     = n_in + 2 * width + orig;
    std::vector<float> padded((size_t)Np, 0.0f);
    std::memcpy(padded.data() + width, in, (size_t)n_in * sizeof(float));

    int       n_per_chan = (Np - K) / orig + 1;
    long long total      = (long long)n_per_chan * (long long)newf;
    long long out_len    = (target < total) ? target : total;

    std::vector<float> out((size_t)out_len);
    for (long long t_out = 0; t_out < out_len; t_out++) {
        int           chan = (int)(t_out % (long long)newf);
        int           pos  = (int)(t_out / (long long)newf);
        const float * w    = kernel.data() + (size_t)chan * (size_t)K;
        const float * x    = padded.data() + (size_t)pos * (size_t)orig;
        float         sum  = 0.0f;
        for (int k = 0; k < K; k++)
            sum += x[k] * w[k];
        out[(size_t)t_out] = sum;
    }
    return out;
}

// =====================================================================
// RVQ encode: embeddings [1024, T] -> codes [8, T]
// =====================================================================

static bool rvq_encode_cpu(const RVQCodec &rvq, const float *embeddings,
                            int T, std::vector<int32_t> &codes) {
    const int K = rvq.num_codebooks;
    const int H = rvq.hidden;
    const int D = rvq.codebook_dim;
    const int V = rvq.codebook_size;
    codes.resize(K * T);

    // Mutable residual buffer: start with input embeddings, subtract quantized
    // after each codebook.
    std::vector<float> residual(embeddings, embeddings + (size_t)T * H);

    for (int k = 0; k < K; k++) {
        const auto &cb = rvq.cb[k];
        if (!cb.embed || !cb.project_in_w || cb.embed_sq_cpu.empty()) return false;

        // Helper: read tensor data as F32, handling F16→F32 conversion
        auto get_as_f32 = [](const struct ggml_tensor *t, std::vector<float> &dst) {
            size_t n = dst.size();
            size_t ts = ggml_type_size(t->type);
            if (ts == sizeof(float)) {
                ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
            } else if (ts == sizeof(uint16_t)) {
                std::vector<uint16_t> buf(n);
                ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(uint16_t));
                for (size_t i = 0; i < n; i++)
                    dst[i] = ggml_fp16_to_fp32(buf[i]);
            }
        };

        // Download weights to CPU
        std::vector<float> proj_w(H * D), proj_b(D), emb(D * V);
        get_as_f32(cb.project_in_w, proj_w);
        get_as_f32(cb.project_in_b, proj_b);
        get_as_f32(cb.embed, emb);

        // Download project_out weights (for decoding back to residual space)
        std::vector<float> out_w, out_b;
        if (k + 1 < K && cb.project_out_w) {
            out_w.resize(D * H);
            get_as_f32(cb.project_out_w, out_w);
        }
        if (k + 1 < K && cb.project_out_b) {
            out_b.resize(H);
            get_as_f32(cb.project_out_b, out_b);
        }

        for (int t = 0; t < T; t++) {
            const float *r = residual.data() + t * H;

            // Project residual to codebook dim: proj = proj_in @ r + proj_b
            // ggml layout [ne0=H, ne1=D]: element (ne0=h, ne1=d) at flat h + d*H
            // = PT proj_in[d, h]
            std::vector<float> proj(D, 0);
            for (int d = 0; d < D; d++) {
                float s = proj_b[d];
                for (int h = 0; h < H; h++)
                    s += proj_w[h + d * H] * r[h];
                proj[d] = s;
            }

            // Find nearest codebook entry by Euclidean distance
            // ||proj - emb[j]||^2 = ||proj||^2 + ||emb[j]||^2 - 2*proj·emb[j]
            // embed_sq_cpu[j] = ||emb[j]||^2 (precomputed)
            // ggml layout [ne0=D, ne1=V]: element (ne0=d, ne1=j) at flat d + j*D
            int best_j = 0;
            float best_dist = 1e30f;
            for (int j = 0; j < V; j++) {
                float dist = cb.embed_sq_cpu[j];
                for (int d = 0; d < D; d++)
                    dist -= 2.0f * proj[d] * emb[d + j * D];
                if (dist < best_dist) { best_dist = dist; best_j = j; }
            }
            codes[k * T + t] = best_j;
        }

        // Decode quantized vectors and subtract from residual for next codebook
        if (k + 1 < K) {
            for (int t = 0; t < T; t++) {
                int code = codes[k * T + t];
                float *r = residual.data() + t * H;

                if (!out_w.empty()) {
                    // Decode: decoded = project_out_w @ q_emb + project_out_b
                    // ggml layout [ne0=D, ne1=H]: element (ne0=d, ne1=h) at flat d + h*D
                    // embed layout [ne0=D, ne1=V]: element (ne0=d, ne1=code) at flat d + code*D
                    for (int h = 0; h < H; h++) {
                        float s = out_b.empty() ? 0.0f : out_b[h];
                        for (int d = 0; d < D; d++)
                            s += out_w[d + h * D] * emb[d + code * D];
                        r[h] -= s;
                    }
                } else {
                    // No project_out — just use embedding directly (fallback)
                    // Embedding is in D-dim codebook space, residual is H-dim.
                    // This path shouldn't normally be used.
                }
            }
        }
    }
    return true;
}

// Conv1d with F32 im2col — ggml_conv_1d hardcodes F16 im2col, which
// overflows to Inf for values > 65504. The DAC encoder produces intermediate
// feature values exceeding this range, so we must use F32 im2col.
// a = [K, IC, OC]  (weight kernel)
// b = [OW, IW, IC] (input: ne0=OW/output-spatial, ne1=IW/input-spatial, ne2=IC)
static struct ggml_tensor *conv_1d_f32(struct ggml_context *ctx,
                                        struct ggml_tensor *a,
                                        struct ggml_tensor *b,
                                        int s, int p, int d) {
    // CPU backend mul_mat requires both inputs F32 when one is F32
    if (a->type != GGML_TYPE_F32) a = ggml_cast(ctx, a, GGML_TYPE_F32);
    struct ggml_tensor *im2col = ggml_im2col(ctx, a, b, s, 0, p, 0, d, 0, false, GGML_TYPE_F32);
    struct ggml_tensor *result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], (im2col->ne[2] * im2col->ne[1])),
        ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1]), a->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], a->ne[2], im2col->ne[2]);
}

// Forward declaration (defined later in RunVocoder section)
static struct ggml_tensor *dac_snake_graph(struct ggml_context *ctx,
                                           struct ggml_tensor *x,
                                           struct ggml_tensor *alpha,
                                           const std::vector<float> &inv_b_data,
                                           const char *name);

// Helper: reshape a 1D [C] bias/weight to 2D [1, C] for broadcasting with [T, C]
static inline struct ggml_tensor *bias_1d_to_2d(struct ggml_context *ctx,
                                                  struct ggml_tensor *t) {
    if (!t) return nullptr;
    if (t->ne[0] == 1 && t->ne[1] > 1) return t;  // already [1, C]
    int C = (int)t->ne[0];
    return ggml_reshape_2d(ctx, t, 1, C);
}

// =====================================================================
// HuBERT feature extractor graph (7 conv1d layers)
// Input: [T_in] f32, output: [T_out, 512] f32
// Cumulative stride: 5*2*2*2*2*2*2 = 320x
// =====================================================================

// Helper: transpose [T, C] -> [C, T] and back
static inline struct ggml_tensor *feat_to_ct(struct ggml_context *ctx,
                                              struct ggml_tensor *x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}
static inline struct ggml_tensor *feat_to_tc(struct ggml_context *ctx,
                                              struct ggml_tensor *x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

static struct ggml_tensor *hubert_feature_extractor_graph(
    struct ggml_context *ctx, struct ggml_tensor *x,
    const HubertFeatExtractor &feat, int *out_T)
{
    // HF HuBERT uses valid padding (pad=0) and dilation 1 on every conv.
    // Layer 0: conv1d k=10 s=5 pad=0, GroupNorm(G=C, affine), GELU exact.
    // GroupNorm with n_groups==C == InstanceNorm: normalizes each channel
    // independently over the time axis. Input is reshaped to [T, 1, C] so
    // ggml_group_norm reads channels from ne[2].
    if (feat.conv[0].conv_w) {
        struct ggml_tensor *w = feat.conv[0].conv_w;
        if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
        x = ggml_conv_1d(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], 1, 1), 5, 0, 1);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
        if (feat.conv[0].ln_w) {
            int T = (int)x->ne[0];
            int C = (int)x->ne[1];
            x = ggml_reshape_3d(ctx, x, T, 1, C);
            x = ggml_group_norm(ctx, x, C, 1e-5f);
            x = ggml_reshape_2d(ctx, x, T, C);
            struct ggml_tensor *w2 = ggml_reshape_2d(ctx, feat.conv[0].ln_w, 1, C);
            struct ggml_tensor *b2 = ggml_reshape_2d(ctx, feat.conv[0].ln_b, 1, C);
            x = ggml_mul(ctx, x, w2);
            x = ggml_add(ctx, x, b2);
        }
        x = ggml_gelu_erf(ctx, x);
        x = ggml_cont(ctx, x);
    }

    // Layers 1-6: conv1d k=3/2 s=2 pad=0, GELU exact (all in [T, C] layout)
    static const int hubert_kernels[] = {3, 3, 3, 3, 2, 2};
    static const int hubert_strides[] = {2, 2, 2, 2, 2, 2};
    for (int i = 1; i < 7; i++) {
        if (!feat.conv[i].conv_w) continue;
        struct ggml_tensor *w = feat.conv[i].conv_w;
        if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
        int k = hubert_kernels[i - 1];
        int s = hubert_strides[i - 1];
        x = ggml_conv_1d(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1), s, 0, 1);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
        x = ggml_gelu_erf(ctx, x);
        x = ggml_cont(ctx, x);
    }
    // Final transpose to [C, T] for downstream mul_mat / LayerNorm
    x = feat_to_ct(ctx, x);
    if (out_T) *out_T = (int)x->ne[1];
    return x;
}

// =====================================================================
// HuBERT self-attention (bidirectional, no RoPE)
// =====================================================================

static struct ggml_tensor *hubert_attention_graph(
    struct ggml_context *ctx, struct ggml_tensor *x,
    const HubertAttentionWeights &attn, int n_head, int head_dim)
{
    // x layout: [H, T] (ne0=H, ne1=T). H = n_head * head_dim.
    int H = (int)x->ne[0];
    int T = (int)x->ne[1];

    // Q/K/V projections: weight was transposed at conversion time [out,in]→[in,out].
    struct ggml_tensor *q = ggml_mul_mat(ctx, attn.q_w, x);
    if (attn.q_b) q = ggml_add(ctx, q, attn.q_b);
    struct ggml_tensor *k = ggml_mul_mat(ctx, attn.k_w, x);
    if (attn.k_b) k = ggml_add(ctx, k, attn.k_b);
    struct ggml_tensor *v = ggml_mul_mat(ctx, attn.v_w, x);
    if (attn.v_b) v = ggml_add(ctx, v, attn.v_b);

    // Reshape [H, T] -> [head_dim, n_head, T] (ne0=head_dim, ne1=n_head, ne2=T)
    // Then permute to [head_dim, T, n_head] for Q and K (Q^T @ K per head)
    // For V, permute to [T, head_dim, n_head] (ne0=T) so V @ scores works
    q = ggml_reshape_3d(ctx, q, head_dim, n_head, T);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [head_dim, T, n_head]
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, T);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [head_dim, T, n_head]
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, T);
    // ggml_permute args are dst-position per axis:
    // axis0(64=head_dim)->1, axis1(12=n_head)->2, axis2(T)->0, axis3(1)->3
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [T, head_dim, n_head]

    // Scores: Q^T @ K -> [T, T, n_head] (ggml_mul_mat computes K^T @ Q internally)
    struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));
    scores = ggml_soft_max(ctx, scores);

    // Weighted sum: V @ scores -> [head_dim, T, n_head]
    struct ggml_tensor *attn_out = ggml_mul_mat(ctx, v, scores);
    // Permute [head_dim, T, n_head] -> [n_head, head_dim, T]
    // axis0(head_dim)->1, axis1(T)->2, axis2(n_head)->0
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 1, 2, 0, 3));  // [n_head, head_dim, T]
    attn_out = ggml_reshape_2d(ctx, attn_out, H, T);  // [H, T]

    // Output projection: weight was transposed at conversion time [out,in]→[in,out].
    attn_out = ggml_mul_mat(ctx, attn.o_w, attn_out);
    if (attn.o_b) attn_out = ggml_add(ctx, attn_out, attn.o_b);
    return attn_out;
}

// =====================================================================
// HuBERT encoder layer (Post-LN: x = LN(x + attn(x)), x = LN(x + FFN(x)))
// =====================================================================

static struct ggml_tensor *hubert_encoder_layer_graph(
    struct ggml_context *ctx, struct ggml_tensor *x,
    const HubertLayerWeights &l, int n_head, int head_dim)
{
    // x layout: [H, T] (ne0=H, ne1=T).
    // HF HuBERT uses Post-LN (do_stable_layer_norm=false):
    //   x = x + attn(x)  →  LN
    //   x = x + FFN(x)   →  LN

    // Attention with post-norm
    struct ggml_tensor *residual = x;
    struct ggml_tensor *attn_out = hubert_attention_graph(ctx, x, l.attn, n_head, head_dim);
    x = ggml_add(ctx, residual, attn_out);
    x = ggml_norm(ctx, x, 1e-5f);
    if (l.ln_attn_w) x = ggml_mul(ctx, x, l.ln_attn_w);
    if (l.ln_attn_b) x = ggml_add(ctx, x, l.ln_attn_b);

    // FFN with post-norm
    residual = x;
    struct ggml_tensor *ffn = ggml_mul_mat(ctx, l.ffn.w1_w, x);
    if (l.ffn.w1_b) ffn = ggml_add(ctx, ffn, l.ffn.w1_b);
    ffn = ggml_gelu_erf(ctx, ffn);
    ffn = ggml_mul_mat(ctx, l.ffn.w2_w, ffn);
    if (l.ffn.w2_b) ffn = ggml_add(ctx, ffn, l.ffn.w2_b);
    x = ggml_add(ctx, residual, ffn);
    x = ggml_norm(ctx, x, 1e-5f);
    if (l.ln_ffn_w) x = ggml_mul(ctx, x, l.ln_ffn_w);
    if (l.ln_ffn_b) x = ggml_add(ctx, x, l.ln_ffn_b);

    return x;
}

// =====================================================================
// SemanticEncoder graph: conv1d + 2 conv_blocks with res_units
// =====================================================================

static struct ggml_tensor *semantic_encoder_graph(
    struct ggml_context *ctx, struct ggml_tensor *x,
    const SemanticEncoderWeights &sem_enc)
{
    // x is [H, T] (ne0=H, ne1=T). All conv ops need [T, H, 1] input.
    // Reference sem_enc_build_graph expects [T, 768] T-first input/output.
    // We transpose before conv_1d and transpose back afterward, keeping the
    // rest of the pipeline in C-first layout.
    if (!sem_enc.c1w) return x;

    auto conv_in = [&](struct ggml_tensor *f) -> struct ggml_tensor * {
        return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, f)),
                                f->ne[1], f->ne[0], 1);
    };

    // initial conv1d: 768 -> 768, k=3, pad=1, NO bias
    {
        struct ggml_tensor *w = sem_enc.c1w;
        if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
        x = ggml_conv_1d(ctx, w, conv_in(x), 1, 1, 1);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
        x = feat_to_ct(ctx, x);
        x = ggml_cont(ctx, x);
    }

    // 2 blocks, each: 2 res_units (ELU->conv->ELU->conv+skip) -> post-block conv (with bias).
    // No block-level residual. Mirrors reference sem_enc_build_graph.
    for (int i = 0; i < 2; i++) {
        if (!sem_enc.blk[i].cw) continue;

        // 2 res_units: ELU -> conv1(k=3, dil=1, no bias) -> ELU -> conv2(k=1, no bias) + skip
        for (int r = 0; r < 2; r++) {
            if (!sem_enc.blk[i].ru[r].c1w) continue;
            struct ggml_tensor *ru_skip = x;

            // ELU -> conv1 (k=3, pad=dilation, dilation, no bias)
            x = ggml_elu(ctx, x);
            {
                struct ggml_tensor *r1w = sem_enc.blk[i].ru[r].c1w;
                int dil = 1;
                int pad = dil;  // (k-1)*dil/2 = 1 for k=3
                if (r1w->type != GGML_TYPE_F16) r1w = ggml_cast(ctx, r1w, GGML_TYPE_F16);
                x = ggml_conv_1d(ctx, r1w, conv_in(x), 1, pad, dil);
                x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
                x = feat_to_ct(ctx, x);
            }

            // ELU -> conv2 (k=1, pad=0, no bias)
            x = ggml_elu(ctx, x);
            {
                struct ggml_tensor *r2w = sem_enc.blk[i].ru[r].c2w;
                if (r2w->type != GGML_TYPE_F16) r2w = ggml_cast(ctx, r2w, GGML_TYPE_F16);
                x = ggml_conv_1d(ctx, r2w, conv_in(x), 1, 0, 1);
                x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
                x = feat_to_ct(ctx, x);
            }

            x = ggml_add(ctx, ru_skip, x);
            x = ggml_cont(ctx, x);
        }

        // Post-block conv: k=3, pad=1, WITH bias
        {
            struct ggml_tensor *bw = sem_enc.blk[i].cw;
            int k = (int)bw->ne[0];
            int pad = (k - 1) / 2;  // same padding
            if (bw->type != GGML_TYPE_F16) bw = ggml_cast(ctx, bw, GGML_TYPE_F16);
            x = ggml_conv_1d(ctx, bw, conv_in(x), 1, pad, 1);
            x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
            x = feat_to_ct(ctx, x);
            if (sem_enc.blk[i].cb) x = ggml_add(ctx, x, sem_enc.blk[i].cb);
            x = ggml_cont(ctx, x);
        }
    }
    return x;
}

// =====================================================================
// DAC encoder graph: conv1 + 5 downsampling blocks + conv2
// Input: [T_in] f32 audio, output: [T_out, out_ch] features
// =====================================================================

static struct ggml_tensor *dac_encoder_graph(
    struct ggml_context *ctx, struct ggml_tensor *audio,
    const DACEncoder &enc, int *out_T)
{
    // Input: [T_in] f32 -> reshape to [T_in, 1]
    struct ggml_tensor *x = ggml_reshape_2d(ctx, audio, audio->ne[0], 1);

    // conv1: 1 -> in_ch, k=7, s=1
    if (enc.c1w) {
        struct ggml_tensor *w = enc.c1w;
        if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
        x = ggml_conv_1d(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], 1, 1), 1, 3, 1);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
        if (enc.c1b) x = ggml_add(ctx, x, bias_1d_to_2d(ctx, enc.c1b));
        x = ggml_cont(ctx, x);
    }

    // 5 downsampling blocks
    // Block order: residual units (in_ch) → Snake → downsampling conv (in_ch→out_ch)
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        const DACEncBlockWeights &b = enc.blk[i];
        if (!b.cw) continue;

        // 3 residual units (operate on in_ch, before downsampling)
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            const DACResUnitWeights &ru = b.ru[r];
            if (!ru.c1w) continue;
            struct ggml_tensor *skip = x;

            // Snake1 + conv1
            if (ru.s1.a && !ru.s1.inv_b.empty()) {
                int C_alpha = (int)ggml_nelements(ru.s1.a);
                struct ggml_tensor *alpha = ggml_reshape_2d(ctx, ru.s1.a, 1, C_alpha);
                char sname[64];
                snprintf(sname, sizeof(sname), "enc_inv_b_blk%d_ru%d_s1", i, r);
                x = dac_snake_graph(ctx, x, alpha, ru.s1.inv_b, sname);
            }
            {
                int pad1 = 3 * ru.dilation;
                struct ggml_tensor *w = ru.c1w;
                x = conv_1d_f32(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1), 1, pad1, ru.dilation);
                x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
                if (ru.c1b) x = ggml_add(ctx, x, bias_1d_to_2d(ctx, ru.c1b));
                x = ggml_cont(ctx, x);
            }

            // Snake2 + conv2 (1x1)
            if (ru.s2.a && !ru.s2.inv_b.empty()) {
                int C_alpha = (int)ggml_nelements(ru.s2.a);
                struct ggml_tensor *alpha = ggml_reshape_2d(ctx, ru.s2.a, 1, C_alpha);
                char sname[64];
                snprintf(sname, sizeof(sname), "enc_inv_b_blk%d_ru%d_s2", i, r);
                x = dac_snake_graph(ctx, x, alpha, ru.s2.inv_b, sname);
            }
            {
                struct ggml_tensor *w = ru.c2w;
                if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
                x = conv_1d_f32(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1), 1, 0, 1);
                x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
                if (ru.c2b) x = ggml_add(ctx, x, bias_1d_to_2d(ctx, ru.c2b));
            }
            x = ggml_add(ctx, skip, x);
            x = ggml_cont(ctx, x);
        }

        // Snake1
        if (b.s1.a && !b.s1.inv_b.empty()) {
            int C_alpha = (int)ggml_nelements(b.s1.a);
            struct ggml_tensor *alpha = ggml_reshape_2d(ctx, b.s1.a, 1, C_alpha);
            char sname[64];
            snprintf(sname, sizeof(sname), "enc_inv_b_blk%d_s1", i);
            x = dac_snake_graph(ctx, x, alpha, b.s1.inv_b, sname);
        }

        // Conv1d: in_ch -> out_ch, k=2*stride, s=stride
        {
            int K = 2 * b.stride;
            int pad = (b.stride + 1) / 2;
            struct ggml_tensor *w = b.cw;
            if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
            x = conv_1d_f32(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1), b.stride, pad, 1);
            x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
            if (b.cb) x = ggml_add(ctx, x, bias_1d_to_2d(ctx, b.cb));
            x = ggml_cont(ctx, x);
        }
    }
    // Final snake + conv2
    if (enc.s_final.a && !enc.s_final.inv_b.empty()) {
        int C_alpha = (int)ggml_nelements(enc.s_final.a);
        struct ggml_tensor *alpha = ggml_reshape_2d(ctx, enc.s_final.a, 1, C_alpha);
        x = dac_snake_graph(ctx, x, alpha, enc.s_final.inv_b, "enc_inv_b_sfinal");
    }
    if (enc.c2w) {
        struct ggml_tensor *w = enc.c2w;
        if (w->type != GGML_TYPE_F16) w = ggml_cast(ctx, w, GGML_TYPE_F16);
        // pad=1 matches reference dac_enc_build_graph (k=3, pad=1 preserves T)
        x = conv_1d_f32(ctx, w, ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1), 1, 1, 1);
        x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
        if (enc.c2b) x = ggml_add(ctx, x, bias_1d_to_2d(ctx, enc.c2b));
        x = ggml_cont(ctx, x);
    }

    // Transpose to [C, T_out] so memory layout matches CPU [frames, channels]
    x = ggml_cont(ctx, ggml_transpose(ctx, x));
    if (out_T) *out_T = (int)x->ne[1];  // ne1=time after transpose
    return x;
}

// =====================================================================
// EncodeReferenceAudio: full audio -> RVQ codes pipeline
//
// Branch 1 (semantic): audio 24kHz->16kHz -> HuBERT feat extractor (320x)
//                      -> LN+Linear -> 12-layer transformer -> SemanticEncoder
//                      -> [C_s=768, T_s]  (all tensors: ne0=channels, ne1=time)
// Branch 2 (acoustic): audio 24kHz -> DAC encoder (960x) -> [C_a=256, T_a]
// Align: repeat acoustic 2x -> [T_s, C_a]
// Merge: concat -> [T_s, 1024] -> fc -> [T_s, 1024] -> RVQ encode -> [8, T_s]
// =====================================================================

std::vector<int32_t> OmniVoiceModel::EncodeReferenceAudio(const float *audio_24k,
                                                           int n_samples) {
    if (n_samples <= 0) return {};

    // Resample to 16kHz for HuBERT
    std::vector<float> audio_16k = audio_resample_linear(audio_24k, n_samples,
                                                          hparams_.audio_sample_rate, 16000);
    int n_16k = (int)audio_16k.size();

    // Pad 160 zeros on each side
    const int pad = 160;
    std::vector<float> padded(n_16k + 2 * pad, 0.0f);
    memcpy(padded.data() + pad, audio_16k.data(), n_16k * sizeof(float));
    int n_padded = (int)padded.size();

    // Expected output sizes
    int T_s = n_padded / 320;
    if (T_s < 1) T_s = 1;
    int T_a = n_samples / 960;
    if (T_a < 1) T_a = 1;

    RS_LOG_INFO("OmniVoice encode: audio=%d samples 16k_padded=%d T_s=%d T_a=%d",
                n_samples, n_padded, T_s, T_a);

    if (!hubert_feat_.conv[0].conv_w || !hubert_proj_.proj_w ||
        !hubert_layers_[0].attn.q_w) {
        RS_LOG_WARN("OmniVoice: encoder weights not loaded, cannot encode");
        return {};
    }

    const int n_max_nodes = OMNIVOICE_MAX_NODES / 2;

    // === Phase 1: HuBERT semantic features ===
    std::vector<float> semantic_feat;
    {
        struct ggml_init_params gparams = {
            (size_t)n_max_nodes * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(n_max_nodes, false),
            nullptr, true};
        struct ggml_context *gctx = ggml_init(gparams);
        if (!gctx) return {};

        struct ggml_tensor *x_hub = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, n_padded);
        ggml_set_input(x_hub);
        ggml_set_name(x_hub, "audio_16k");

        int T_feat = 0;
        struct ggml_tensor *feat = hubert_feature_extractor_graph(gctx, x_hub, hubert_feat_, &T_feat);
        if (!feat || T_feat == 0) { ggml_free(gctx); return {}; }

        // feat is [C, T] (ne0=C=512, ne1=T). All biases broadcast as 1D [C].
        struct ggml_tensor *sem = ggml_norm(gctx, feat, 1e-5f);
        if (hubert_proj_.ln_w) sem = ggml_mul(gctx, sem, hubert_proj_.ln_w);
        if (hubert_proj_.ln_b) sem = ggml_add(gctx, sem, hubert_proj_.ln_b);
        // proj_w was transposed at conversion time [out,in]→[in,out].
        sem = ggml_mul_mat(gctx, hubert_proj_.proj_w, sem);
        if (hubert_proj_.proj_b) sem = ggml_add(gctx, sem, hubert_proj_.proj_b);
        sem = ggml_cont(gctx, sem);

        // pos_conv_embed: grouped conv1d (mirrors reference hubert_pos_conv_build_graph).
        // 16 groups, each a 48→48 conv with k=128, pad=64.
        if (hubert_enc_init_.pos_conv_w && hubert_enc_init_.pos_conv_b) {
            const int POS_K = 128, POS_IC_PG = 48, POS_OC_PG = 48;
            const int POS_GROUPS = 16, POS_PAD = 64;
            int T = (int)sem->ne[1];  // C-first: ne0=ch=768, ne1=time

            // Transpose to T-first [T, 768]
            struct ggml_tensor *xt = ggml_cont(gctx, ggml_transpose(gctx, sem));
            // Reshape to [T, 48, 16] — 16 groups of 48 channels
            struct ggml_tensor *x3 = ggml_reshape_3d(gctx, xt, T, POS_IC_PG, POS_GROUPS);

            struct ggml_tensor *pw = hubert_enc_init_.pos_conv_w;
            struct ggml_tensor *pb = hubert_enc_init_.pos_conv_b;
            if (pw->type != GGML_TYPE_F16) pw = ggml_cast(gctx, pw, GGML_TYPE_F16);

            struct ggml_tensor *group_outs[POS_GROUPS];
            for (int g = 0; g < POS_GROUPS; g++) {
                // Slice input: [T, 48] at group g
                size_t off_x = (size_t)g * x3->nb[2];
                struct ggml_tensor *xg = ggml_view_2d(gctx, x3, T, POS_IC_PG, x3->nb[1], off_x);

                // Slice weight: [128, 48, 48] at group g
                size_t off_w = (size_t)g * (size_t)POS_OC_PG * pw->nb[2];
                struct ggml_tensor *wg = ggml_view_3d(gctx, pw, POS_K, POS_IC_PG, POS_OC_PG,
                                                       pw->nb[1], pw->nb[2], off_w);

                // Slice bias: [48] at group g
                size_t off_b = (size_t)g * POS_OC_PG * sizeof(float);
                struct ggml_tensor *bg = ggml_view_1d(gctx, pb, POS_OC_PG, off_b);

                // conv1d: [T, 48, 1] -> [T+1, 48, 1] (k=128 even, pad=64 -> T_out=T+1)
                struct ggml_tensor *cg = ggml_conv_1d(gctx, wg,
                    ggml_reshape_3d(gctx, xg, T, POS_IC_PG, 1), 1, POS_PAD, 1);
                cg = ggml_reshape_2d(gctx, cg, cg->ne[0], POS_OC_PG);
                // bias_1d_to_2d broadcast
                cg = ggml_add(gctx, cg, ggml_reshape_2d(gctx, bg, 1, POS_OC_PG));
                group_outs[g] = cg;
            }

            // Concat 16 outputs on channel axis -> [T+1, 768]
            struct ggml_tensor *y = group_outs[0];
            for (int g = 1; g < POS_GROUPS; g++)
                y = ggml_concat(gctx, y, group_outs[g], 1);

            // Trim trailing sample (k even, T_out = T+1 -> T)
            y = ggml_cont(gctx, ggml_view_2d(gctx, y, T, HUBERT_HIDDEN, y->nb[1], 0));
            y = ggml_gelu_erf(gctx, y);

            // Transpose back to C-first [768, T] and add as residual
            struct ggml_tensor *pos = ggml_cont(gctx, ggml_transpose(gctx, y));
            sem = ggml_add(gctx, sem, pos);
            sem = ggml_cont(gctx, sem);
        }

        if (hubert_enc_init_.ln_w) {
            sem = ggml_norm(gctx, sem, 1e-5f);
            sem = ggml_mul(gctx, sem, hubert_enc_init_.ln_w);
            if (hubert_enc_init_.ln_b) sem = ggml_add(gctx, sem, hubert_enc_init_.ln_b);
            sem = ggml_cont(gctx, sem);
        }

        // Capture 13 hidden states for mean-pool (mirrors reference:
        // post enc_init + post layer 0..11).
        const int n_states = HUBERT_NUM_LAYERS + 1;
        std::vector<struct ggml_tensor *> states(n_states);
        states[0] = sem;
        for (int i = 0; i < HUBERT_NUM_LAYERS; i++) {
            sem = hubert_encoder_layer_graph(gctx, sem, hubert_layers_[i],
                                              HUBERT_NUM_HEADS, HUBERT_HIDDEN / HUBERT_NUM_HEADS);
            states[i + 1] = sem;
        }

        // Mean-pool the 13 states
        struct ggml_tensor *sum = states[0];
        for (int i = 1; i < n_states; i++)
            sum = ggml_add(gctx, sum, states[i]);
        struct ggml_tensor *mean = ggml_scale(gctx, sum, 1.0f / (float)n_states);

        // Downsample time axis by 2 (semantic_downsample_factor)
        // mean is [C, T_h] (ne0=channels, ne1=time). Strided view picks
        // every other column starting at 0, then cont for host copy.
        int T_h = (int)mean->ne[1];
        int T_ds = T_h / 2;
        struct ggml_tensor *strided = ggml_view_2d(gctx, mean, HUBERT_HIDDEN, T_ds,
                                                    2 * mean->nb[1], 0);
        struct ggml_tensor *features = ggml_cont(gctx, strided);

        sem = semantic_encoder_graph(gctx, features, sem_enc_);
        // sem is [C, T] layout: ne0=channels, ne1=time
        T_s = (int)sem->ne[1];
        ggml_set_name(sem, "semantic_out");
        ggml_set_output(sem);

        struct ggml_cgraph *gf = ggml_new_graph_custom(gctx, n_max_nodes, false);
        ggml_build_forward_expand(gf, sem);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cpu_backend_);
        if (!buf) { ggml_free(gctx); return {}; }

        ggml_backend_tensor_set(ggml_get_tensor(gctx, "audio_16k"), padded.data(), 0,
                                n_padded * sizeof(float));
        ggml_backend_cpu_set_n_threads(cpu_backend_, 4);
        if (ggml_backend_graph_compute(cpu_backend_, gf) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("OmniVoice: HuBERT graph compute failed");
            ggml_backend_buffer_free(buf);
            ggml_free(gctx);
            return {};
        }

        struct ggml_tensor *sem_out = ggml_get_tensor(gctx, "semantic_out");
        if (!sem_out) {
            ggml_backend_buffer_free(buf);
            ggml_free(gctx);
            return {};
        }
        // [C, T] layout: ne0=channels=768, ne1=time
        T_s = (int)sem_out->ne[1];
        int C_sem = (int)sem_out->ne[0];
        semantic_feat.resize((size_t)T_s * C_sem);
        ggml_backend_tensor_get(sem_out, semantic_feat.data(), 0,
                                semantic_feat.size() * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);

        // Diagnostic: semantic feature statistics
        {
            double sum = 0, sq = 0;
            float mn = semantic_feat[0], mx = semantic_feat[0];
            for (size_t i = 0; i < semantic_feat.size(); i++) {
                float v = semantic_feat[i];
                sum += v; sq += (double)v * v;
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            double mean = sum / semantic_feat.size();
            double std = sqrt(sq / semantic_feat.size() - mean * mean);
            RS_LOG_INFO("OmniVoice DIAG semantic_feat: min=%.4f max=%.4f mean=%.4f std=%.4f rms=%.4f",
                        mn, mx, mean, std, sqrt(sq / semantic_feat.size()));
        }
    }

    RS_LOG_INFO("OmniVoice: HuBERT -> %d frames x 768", T_s);

    // === Phase 2: DAC encoder acoustic features ===
    // Mirror reference: conditionally pad audio when DAC output length != T_s.
    // Without padding, temporal misalignment degrades RVQ code quality.
    const float *dac_audio = audio_24k;
    int dac_n_samples = n_samples;
    std::vector<float> dac_audio_padded;
    {
        // Compute analytical DAC output length (mirrors compute_dac_output_length)
        auto dac_output_len = [](int n) -> int {
            const int K[5] = {16, 10, 8, 4, 6};
            const int S[5] = {8, 5, 4, 2, 3};
            const int P[5] = {4, 3, 2, 1, 2};
            int T = n;
            for (int i = 0; i < 5; i++)
                T = (T + 2 * P[i] - K[i]) / S[i] + 1;
            return T;
        };
        int T_a_expected = dac_output_len(n_samples);
        if (T_a_expected != T_s) {
            int p = hparams_.hop_length / 2;  // 480
            dac_audio_padded.resize(n_samples + 2 * p, 0.0f);
            memcpy(dac_audio_padded.data() + p, audio_24k, n_samples * sizeof(float));
            dac_audio = dac_audio_padded.data();
            dac_n_samples = (int)dac_audio_padded.size();
            RS_LOG_INFO("OmniVoice: DAC pad %d samples (T_a_expected=%d != T_s=%d)",
                        dac_n_samples, T_a_expected, T_s);
        }
    }

    std::vector<float> acoustic_feat;
    {
        struct ggml_init_params gparams = {
            (size_t)n_max_nodes * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(n_max_nodes, false),
            nullptr, true};
        struct ggml_context *gctx = ggml_init(gparams);
        if (!gctx) return {};

        struct ggml_tensor *x_dac = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, dac_n_samples);
        ggml_set_input(x_dac);
        ggml_set_name(x_dac, "audio_24k");

        int T_a_out = 0;
        struct ggml_tensor *acoustic = dac_encoder_graph(gctx, x_dac, dac_enc_, &T_a_out);
        if (T_a_out == 0 || !acoustic) { ggml_free(gctx); return {}; }
        ggml_set_name(acoustic, "acoustic_out");
        ggml_set_output(acoustic);

        struct ggml_cgraph *gf = ggml_new_graph_custom(gctx, n_max_nodes, false);
        ggml_build_forward_expand(gf, acoustic);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, cpu_backend_);
        if (!buf) { ggml_free(gctx); return {}; }

        ggml_backend_tensor_set(ggml_get_tensor(gctx, "audio_24k"), dac_audio, 0,
                                dac_n_samples * sizeof(float));

        // Set snake inv_b tensors
        auto set_enc_inv_b = [&](const std::vector<float> &data, const char *name) {
            if (data.empty()) return;
            struct ggml_tensor *t = ggml_get_tensor(gctx, name);
            if (t) ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        };
        for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
            char name[64];
            snprintf(name, sizeof(name), "enc_inv_b_blk%d_s1", i);
            set_enc_inv_b(dac_enc_.blk[i].s1.inv_b, name);
            for (int r = 0; r < DAC_RES_UNITS; r++) {
                snprintf(name, sizeof(name), "enc_inv_b_blk%d_ru%d_s1", i, r);
                set_enc_inv_b(dac_enc_.blk[i].ru[r].s1.inv_b, name);
                snprintf(name, sizeof(name), "enc_inv_b_blk%d_ru%d_s2", i, r);
                set_enc_inv_b(dac_enc_.blk[i].ru[r].s2.inv_b, name);
            }
        }
        set_enc_inv_b(dac_enc_.s_final.inv_b, "enc_inv_b_sfinal");

        ggml_backend_cpu_set_n_threads(cpu_backend_, 4);
        if (ggml_backend_graph_compute(cpu_backend_, gf) != GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("OmniVoice: DAC encoder graph compute failed");
            ggml_backend_buffer_free(buf);
            ggml_free(gctx);
            return {};
        }

        struct ggml_tensor *ac_out = ggml_get_tensor(gctx, "acoustic_out");
        if (!ac_out) {
            ggml_backend_buffer_free(buf);
            ggml_free(gctx);
            return {};
        }
        // [C, T] layout after transpose: ne0=channels, ne1=time
        int C_a = (int)ac_out->ne[0];
        T_a = (int)ac_out->ne[1];
        acoustic_feat.resize((size_t)T_a * C_a);
        ggml_backend_tensor_get(ac_out, acoustic_feat.data(), 0,
                                acoustic_feat.size() * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(gctx);

        // Diagnostic: acoustic feature statistics
        {
            double sum = 0, sq = 0;
            float mn = acoustic_feat[0], mx = acoustic_feat[0];
            for (size_t i = 0; i < acoustic_feat.size(); i++) {
                float v = acoustic_feat[i];
                sum += v; sq += (double)v * v;
                if (v < mn) mn = v; if (v > mx) mx = v;
            }
            double mean = sum / acoustic_feat.size();
            double std = sqrt(sq / acoustic_feat.size() - mean * mean);
            RS_LOG_INFO("OmniVoice DIAG acoustic_feat: min=%.4f max=%.4f mean=%.4f std=%.4f rms=%.4f",
                        mn, mx, mean, std, sqrt(sq / acoustic_feat.size()));
        }
    }

    RS_LOG_INFO("OmniVoice: DAC encoder -> %d frames x %zu", T_a, acoustic_feat.size() / std::max(1, T_a));

    // === Phase 3: Temporal alignment + concat + fc on CPU ===
    // Repeat acoustic frames to match T_s
    int C_a = (int)acoustic_feat.size() / std::max(1, T_a);
    int C_s = 768;
    std::vector<float> combined((size_t)T_s * (C_a + C_s), 0.0f);

    for (int t = 0; t < T_s; t++) {
        // Repeat acoustic: map t -> t * T_a / T_s
        int ta = t * T_a / T_s;
        if (ta >= T_a) ta = T_a - 1;
        for (int c = 0; c < C_a; c++)
            combined[(size_t)t * (C_a + C_s) + c] = acoustic_feat[(size_t)ta * C_a + c];
        // Copy semantic
        for (int c = 0; c < C_s; c++)
            combined[(size_t)t * (C_a + C_s) + C_a + c] = semantic_feat[(size_t)t * C_s + c];
    }

    // fc projection on CPU: fc_w [C_out, C_in] @ combined [C_in, T_s] -> [C_out, T_s]
    int C_in = C_a + C_s;
    // fc_w in GGML: ne0=C_in, ne1=C_out. PT element fc_weight[out=o, in=i] stored at
    // flat position i + o*C_in (ggml convention: ne0 varies fastest, ne0=in dimension).
    int C_out = (fc_w_) ? (int)fc_w_->ne[1] : C_in;

    // Diagnostic: combined features before fc
    {
        double sum = 0, sq = 0;
        float mn = combined[0], mx = combined[0];
        for (size_t i = 0; i < combined.size(); i++) {
            float v = combined[i];
            sum += v; sq += (double)v * v;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        double mean = sum / combined.size();
        double std = sqrt(sq / combined.size() - mean * mean);
        RS_LOG_INFO("OmniVoice DIAG combined(pre-fc): min=%.4f max=%.4f mean=%.4f std=%.4f rms=%.4f",
                    mn, mx, mean, std, sqrt(sq / combined.size()));
    }

    std::vector<float> projected;
    if (fc_w_ && C_in == (int)fc_w_->ne[0]) {
        C_out = (int)fc_w_->ne[1];
        projected.resize((size_t)T_s * C_out, 0.0f);

        // Read fc_w, converting from F16 if needed
        size_t n_weights = (size_t)C_in * C_out;
        size_t type_sz = ggml_type_size(fc_w_->type);
        std::vector<float> fc_w_cpu(n_weights);

        // The fc weight has acoustic column RMS ~7.6x larger than semantic columns
        // (verified against original PyTorch safetensors). This partially compensates
        // for the acoustic feature scale (~15611) being much larger than semantic (~0.3),
        // but acoustic features still dominate the projection by ~6500x.
        if (type_sz == sizeof(float)) {
            ggml_backend_tensor_get(fc_w_, fc_w_cpu.data(), 0, n_weights * sizeof(float));
        } else if (type_sz == sizeof(uint16_t)) {
            std::vector<uint16_t> f16_buf(n_weights);
            ggml_backend_tensor_get(fc_w_, f16_buf.data(), 0, n_weights * sizeof(uint16_t));
            for (size_t i = 0; i < n_weights; i++)
                fc_w_cpu[i] = ggml_fp16_to_fp32(f16_buf[i]);
        } else {
            RS_LOG_ERR("OmniVoice: fc_w unsupported type size %zu", type_sz);
            return {};
        }

        std::vector<float> fc_b_cpu;
        if (fc_b_) {
            fc_b_cpu.resize(C_out);
            // fc_b is typically F32 (1D tensor), but handle F16 defensively
            size_t b_type_sz = ggml_type_size(fc_b_->type);
            if (b_type_sz == sizeof(float)) {
                ggml_backend_tensor_get(fc_b_, fc_b_cpu.data(), 0, C_out * sizeof(float));
            } else if (b_type_sz == sizeof(uint16_t)) {
                std::vector<uint16_t> f16_buf(C_out);
                ggml_backend_tensor_get(fc_b_, f16_buf.data(), 0, C_out * sizeof(uint16_t));
                for (size_t i = 0; i < (size_t)C_out; i++)
                    fc_b_cpu[i] = ggml_fp16_to_fp32(f16_buf[i]);
            }
        }

        // fc_w in ggml ne0=C_in: fc_weight[out=o, in=i] at flat i + o*C_in
        for (int t = 0; t < T_s; t++) {
            for (int o = 0; o < C_out; o++) {
                float s = fc_b_cpu.empty() ? 0.0f : fc_b_cpu[o];
                for (int i = 0; i < C_in; i++)
                    s += fc_w_cpu[(size_t)o * C_in + i] * combined[(size_t)t * C_in + i];
                projected[(size_t)t * C_out + o] = s;
            }
        }
    } else {
        // No fc or dimension mismatch — use combined directly
        projected = std::move(combined);
        C_out = C_in;
    }

    RS_LOG_INFO("OmniVoice: projected -> %d frames x %d", T_s, C_out);

    // Diagnostic: projected features after fc
    {
        double sum = 0, sq = 0;
        float mn = projected[0], mx = projected[0];
        for (size_t i = 0; i < projected.size(); i++) {
            float v = projected[i];
            sum += v; sq += (double)v * v;
            if (v < mn) mn = v; if (v > mx) mx = v;
        }
        double mean = sum / projected.size();
        double std = sqrt(sq / projected.size() - mean * mean);
        RS_LOG_INFO("OmniVoice DIAG projected(post-fc): min=%.4f max=%.4f mean=%.4f std=%.4f rms=%.4f",
                    mn, mx, mean, std, sqrt(sq / projected.size()));
    }

    // === Phase 4: RVQ encode ===
    std::vector<int32_t> codes;
    if (!rvq_encode_cpu(rvq_, projected.data(), T_s, codes)) return {};

    // Diagnostic: RVQ code distribution per codebook
    {
        std::string cdist;
        for (int k = 0; k < rvq_.num_codebooks && k < 8; k++) {
            std::vector<int> hist(rvq_.codebook_size, 0);
            for (int t = 0; t < T_s; t++) {
                int c = codes[k * T_s + t];
                if (c >= 0 && c < rvq_.codebook_size) hist[c]++;
            }
            // count non-zero bins and top-5 codes
            int nz = 0;
            std::vector<std::pair<int,int>> top;
            for (int j = 0; j < rvq_.codebook_size; j++) {
                if (hist[j] > 0) { nz++; top.push_back({hist[j], j}); }
            }
            std::sort(top.begin(), top.end(), std::greater<>());
            char buf[256];
            snprintf(buf, sizeof(buf), " cb%d:nz=%d", k, nz);
            cdist += buf;
            int show = std::min(5, (int)top.size());
            for (int j = 0; j < show; j++) {
                snprintf(buf, sizeof(buf), " #%d=%d", top[j].second, top[j].first);
                cdist += buf;
            }
        }
        RS_LOG_INFO("OmniVoice DIAG RVQ codes: T=%d%s", T_s, cdist.c_str());
    }

    RS_LOG_INFO("OmniVoice: encoded reference audio %d samples -> T_s=%d T_a=%d -> %zu codes",
                n_samples, T_s, T_a, codes.size());
    return codes;
}

std::shared_ptr<RSState> OmniVoiceModel::CreateState() {
    return std::make_shared<OmniVoiceState>();
}

// =====================================================================
// Text tokenization
// =====================================================================

bool OmniVoiceModel::PushText(RSState &state, const char *text, const char *language,
                              const char *instruct) {
    auto &s = static_cast<OmniVoiceState &>(state);
    if (language && language[0] != '\0') s.language = language;
    if (instruct && instruct[0] != '\0') s.instruct = instruct;
    s.text_original = text ? text : "";

    // Use BPE tokenizer if available, otherwise fall back to llm_vocab
    if (bpe_.n_vocab > 0) {
        s.text_tokens = bpe_.encode(std::string(text), false);
    } else if (llm_model_) {
        s.text_tokens = llm_model_->vocab().tokenize(std::string(text), false);
    }

    RS_LOG_INFO("OmniVoice: tokenized '%s' -> %zu tokens", text, s.text_tokens.size());
    return !s.text_tokens.empty();
}

// =====================================================================
// Reference audio -> acoustic prompt tokens
// =====================================================================

bool OmniVoiceModel::PushReferenceAudio(RSState &state, const float *samples,
                                        int n_samples, int sample_rate,
                                        ggml_backend_sched_t sched) {
    auto &s = static_cast<OmniVoiceState &>(state);
    (void)sched;

    if (!samples || n_samples <= 0) {
        RS_LOG_ERR("OmniVoice: invalid reference audio");
        return false;
    }

    // Resample to 24kHz if needed
    std::vector<float> audio_24k;
    const float *ref_data = samples;
    int ref_len = n_samples;
    if (sample_rate != hparams_.audio_sample_rate) {
        RS_LOG_INFO("OmniVoice: resampling reference %d Hz -> %d Hz",
                    sample_rate, hparams_.audio_sample_rate);
        audio_24k = audio_resample_linear(samples, n_samples, sample_rate,
                                          hparams_.audio_sample_rate);
        ref_data = audio_24k.data();
        ref_len = (int)audio_24k.size();
    }

    // Run full encode pipeline (HuBERT + SemanticEncoder + DAC encoder + RVQ)
    std::vector<int32_t> codes = EncodeReferenceAudio(ref_data, ref_len);
    if (codes.empty()) {
        RS_LOG_WARN("OmniVoice: encode failed, using placeholder prompt");
        s.n_prompt_frames = 16;
        s.prompt_tokens.resize(s.n_prompt_frames * hparams_.n_codebooks, audio_mask_id_);
        return true;  // continue without voice cloning
    }

    int K = hparams_.n_codebooks;
    s.n_prompt_frames = (int)codes.size() / K;
    s.prompt_tokens = std::move(codes);

    RS_LOG_INFO("OmniVoice: reference audio encoded -> %d frames x %d codebooks",
                s.n_prompt_frames, K);
    return true;
}

bool OmniVoiceModel::PushReferenceText(RSState &state, const char *ref_text) {
    auto &s = static_cast<OmniVoiceState &>(state);
    s.ref_text = ref_text ? ref_text : "";
    RS_LOG_INFO("OmniVoice: reference text set -> '%s'", s.ref_text.c_str());
    return true;
}

// =====================================================================
// Prompt Building
// =====================================================================

bool OmniVoiceModel::BuildPrompt(OmniVoiceState &state, OmniVoicePrompt &prompt) {
    const int K = hparams_.n_codebooks;
    const int Stgt = state.n_target_frames;
    const int Sref = state.n_prompt_frames;
    const int mask_id = audio_mask_id_;

    // Build style text with language resolution (English -> en, etc.)
    std::string style_text;
    std::string lang_resolved = resolve_language(state.language);
    std::string lang_str = lang_resolved.empty() ? "None" : lang_resolved;
    std::string instruct_str = state.instruct.empty() ? "None" : state.instruct;

    style_text += "<|lang_start|>" + lang_str + "<|lang_end|>";
    style_text += "<|instruct_start|>" + instruct_str + "<|instruct_end|>";

    std::vector<int> style_ids;
    if (bpe_.n_vocab > 0) {
        style_ids = bpe_.encode(style_text, false);
    } else {
        style_ids = llm_model_->vocab().tokenize(style_text, false);
    }

    // Do NOT combine ref_text with target text for voice cloning.
    // Combining causes prompt speech leakage where reference audio content
    // bleeds into synthesized target audio (model generates tokens for both
    // ref_text and target_text in the target window).
    // For pure TTS (no ref), ref_text is empty so combining would be a no-op.
    std::string full_text = state.text_original;
    std::string wrapped = "<|text_start|>" + full_text + "<|text_end|>";

    std::vector<int> text_ids;
    if (bpe_.n_vocab > 0) {
        text_ids = prompt_tokenize_nonverbal(&bpe_, wrapped);
    } else {
        text_ids = llm_model_->vocab().tokenize(wrapped, false);
    }

    const int N1 = (int)style_ids.size();
    const int N2 = (int)text_ids.size();
    const int c_len = N1 + N2 + Sref + Stgt;
    const int u_len = Stgt;

    prompt.B = 1;
    prompt.B_prime = 2;
    prompt.K = K;
    prompt.S_max = c_len;
    prompt.c_len = c_len;
    prompt.u_len = u_len;

    prompt.input_ids.assign((size_t)prompt.B_prime * K * c_len, mask_id);
    prompt.audio_mask.assign((size_t)prompt.B_prime * c_len, 0);
    prompt.attention_mask.assign((size_t)prompt.B_prime * c_len * c_len, 0);

    auto cond_at = [&](int k, int s) -> int32_t& {
        return prompt.input_ids[((size_t)0 * K + k) * c_len + s];
    };
    auto uncond_at = [&](int k, int s) -> int32_t& {
        return prompt.input_ids[((size_t)1 * K + k) * c_len + s];
    };

    // Cond row: style + text tokens duplicated across all K codebooks
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N1; n++) cond_at(k, n) = (int32_t)style_ids[n];
        for (int n = 0; n < N2; n++) cond_at(k, N1 + n) = (int32_t)text_ids[n];
    }

    // Reference audio tokens
    const int ref_start = N1 + N2;
    if (Sref > 0) {
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < Sref; t++) {
                cond_at(k, ref_start + t) = (int32_t)state.prompt_tokens[k * Sref + t];
            }
        }
    }

    // Audio mask covers ref + target
    const int audio_start = ref_start;
    for (int s = audio_start; s < c_len; s++)
        prompt.audio_mask[(size_t)0 * c_len + s] = 1;

    // Uncond row: only target window
    for (int k = 0; k < K; k++) {
        for (int s = 0; s < u_len; s++)
            uncond_at(k, s) = cond_at(k, c_len - Stgt + s);
    }
    for (int s = 0; s < u_len; s++)
        prompt.audio_mask[(size_t)1 * c_len + s] = 1;

    // Attention masks
    auto attn_at = [&](int b, int sq, int skv) -> int32_t& {
        return prompt.attention_mask[((size_t)b * c_len + sq) * c_len + skv];
    };
    // Cond: full bidirectional
    for (int sq = 0; sq < c_len; sq++)
        for (int skv = 0; skv < c_len; skv++)
            attn_at(0, sq, skv) = 1;
    // Uncond: bidirectional on target window, diagonal on padding
    for (int sq = 0; sq < u_len; sq++)
        for (int skv = 0; skv < u_len; skv++)
            attn_at(1, sq, skv) = 1;
    for (int sq = u_len; sq < c_len; sq++)
        attn_at(1, sq, sq) = 1;

    RS_LOG_INFO("OmniVoice Prompt: B'=%d K=%d S=%d N1=%d N2=%d Sref=%d Stgt=%d",
                prompt.B_prime, K, c_len, N1, N2, Sref, Stgt);
    // Debug: print prompt token IDs
    {
        std::string sd = "style_ids:";
        for (int n = 0; n < N1; n++) sd += " " + std::to_string(style_ids[n]);
        RS_LOG_INFO("%s", sd.c_str());
        std::string td = "text_ids:";
        for (int n = 0; n < N2; n++) td += " " + std::to_string(text_ids[n]);
        RS_LOG_INFO("%s", td.c_str());
    }
    // Diagnostic: reference code statistics
    if (Sref > 0) {
        std::string ref_info = "ref_codes:";
        for (int k = 0; k < K; k++) {
            int mn = 99999, mx = -1;
            size_t nonzero = 0;
            for (int t = 0; t < Sref; t++) {
                int32_t v = prompt.input_ids[((size_t)0 * K + k) * c_len + (ref_start + t)];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                if (v != 0 && v != audio_mask_id_) nonzero++;
            }
            char buf[128];
            snprintf(buf, sizeof(buf), " cb%d:[%d,%d] nz=%zu/%d", k, mn, mx, nonzero, Sref);
            ref_info += buf;
        }
        RS_LOG_INFO("%s", ref_info.c_str());
    }
    return true;
}

// =====================================================================
// Batched LLM Forward (CFG: cond + uncond)
// =====================================================================

std::vector<float> OmniVoiceModel::RunLLMForwardBatched(
    const OmniVoicePrompt &prompt, ggml_backend_sched_t sched, int T_audio)
{
    const int B_prime = prompt.B_prime;
    const int K = prompt.K;
    const int S = prompt.S_max;
    const int V = audio_vocab_size_;
    const int H = hparams_.n_embd;

    // Batch forward pass (CFG: cond + uncond)
    const int n_head = hparams_.n_head;
    const int n_head_kv = hparams_.n_head_kv;
    const int head_dim = hparams_.head_dim;
    const int n_layer = hparams_.n_layer;

    if (B_prime <= 0 || K <= 0 || S <= 0) return {};

    // CPU pre-compute: shifted_ids, text_ids, masks
    std::vector<int32_t> shifted((size_t)K * B_prime * S);
    std::vector<int32_t> text_ids_buf((size_t)B_prime * S);
    for (int b = 0; b < B_prime; b++) {
        for (int s = 0; s < S; s++) {
            int m = (prompt.audio_mask[(size_t)b * S + s] != 0) ? 1 : 0;
            text_ids_buf[(size_t)b * S + s] = prompt.input_ids[((size_t)b * K + 0) * S + s];
            for (int k = 0; k < K; k++) {
                shifted[((size_t)k * B_prime + b) * S + s] =
                    prompt.input_ids[((size_t)b * K + k) * S + s] * m + k * V;
            }
        }
    }

    // Attention mask as F16
    std::vector<uint16_t> attn_f16;
    if (!prompt.attention_mask.empty()) {
        attn_f16.resize((size_t)B_prime * S * S);
        const uint16_t f16_one = ggml_fp32_to_fp16(1.0f);
        const uint16_t f16_zero = ggml_fp32_to_fp16(0.0f);
        for (size_t i = 0; i < (size_t)B_prime * S * S; i++)
            attn_f16[i] = (prompt.attention_mask[i] != 0) ? f16_one : f16_zero;
    }

    // Build computation graph
    const int n_max_nodes = OMNIVOICE_MAX_NODES;
    struct ggml_init_params gparams = {
        (size_t)n_max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(n_max_nodes, false),
        nullptr, true};
    struct ggml_context *gctx = ggml_init(gparams);
    if (!gctx) return {};

    // Input tensors
    struct ggml_tensor *t_text_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, B_prime * S);
    ggml_set_name(t_text_ids, "text_ids"); ggml_set_input(t_text_ids);

    struct ggml_tensor *t_shifted = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, B_prime * S, K);
    ggml_set_name(t_shifted, "shifted_ids"); ggml_set_input(t_shifted);

    struct ggml_tensor *t_mask = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, S, B_prime);
    ggml_set_name(t_mask, "mask"); ggml_set_input(t_mask);

    struct ggml_tensor *t_inv_mask = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, S, B_prime);
    ggml_set_name(t_inv_mask, "inv_mask"); ggml_set_input(t_inv_mask);

    struct ggml_tensor *t_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_name(t_positions, "positions"); ggml_set_input(t_positions);

    struct ggml_tensor *t_attn = nullptr;
    if (!attn_f16.empty()) {
        t_attn = ggml_new_tensor_4d(gctx, GGML_TYPE_F16, S, S, 1, B_prime);
        ggml_set_name(t_attn, "attn_mask"); ggml_set_input(t_attn);
    }

    // Custom embed
    // text_embd_ and audio_embeddings/heads are already in ggml layout from conversion.
    struct ggml_tensor *text_embeds_flat = ggml_get_rows(gctx, text_embd_, t_text_ids);
    struct ggml_tensor *audio_embeds_flat = nullptr;
    struct ggml_tensor *embd_table = combined_audio_embeddings_;
    struct ggml_tensor *head_table = combined_audio_heads_;
    if (!embd_table && !head_table && combined_codebook_weights_) {
        int64_t H_dim = combined_codebook_weights_->ne[0];
        int64_t KV = (int64_t)K * V;
        if (combined_codebook_weights_->ne[1] == 2 * KV) {
            embd_table = ggml_view_2d(gctx, combined_codebook_weights_, H_dim, KV,
                                      combined_codebook_weights_->nb[1], 0);
            head_table = ggml_view_2d(gctx, combined_codebook_weights_, H_dim, KV,
                                      combined_codebook_weights_->nb[1], KV * combined_codebook_weights_->nb[1]);
        }
    }
    for (int k = 0; k < K; k++) {
        struct ggml_tensor *idx_k = ggml_view_1d(gctx, t_shifted, B_prime * S,
            (size_t)k * (size_t)(B_prime * S) * sizeof(int32_t));
        struct ggml_tensor *emb_w = acoustic_embeddings_[k];
        if (!emb_w) emb_w = embd_table;
        struct ggml_tensor *emb_k = ggml_get_rows(gctx, emb_w, idx_k);
        audio_embeds_flat = (k == 0) ? emb_k : ggml_add(gctx, audio_embeds_flat, emb_k);
    }

    struct ggml_tensor *text_embeds = ggml_reshape_3d(gctx, ggml_cont(gctx, text_embeds_flat), H, S, B_prime);
    struct ggml_tensor *audio_embeds = ggml_reshape_3d(gctx, ggml_cont(gctx, audio_embeds_flat), H, S, B_prime);
    struct ggml_tensor *text_branch = ggml_mul(gctx, text_embeds, t_inv_mask);
    struct ggml_tensor *audio_branch = ggml_mul(gctx, audio_embeds, t_mask);
    struct ggml_tensor *hidden = ggml_add(gctx, text_branch, audio_branch);

    // Helper: cast tensor to F32 (like reference qwen3_f32).
    // OmniVoice norm weights may be F16 in the GGUF; running ggml_mul in F16
    // loses ~3 decimal digits per layer and compound error across 28 layers
    // causes logit divergence that produces different MaskGIT tokens.
    auto to_f32 = [&](struct ggml_tensor *t) -> struct ggml_tensor * {
        if (!t) return nullptr;
        if (t->type == GGML_TYPE_F32) return t;
        return ggml_cast(gctx, t, GGML_TYPE_F32);
    };

    // Transformer layers (bidirectional)
    const float rope_theta = hparams_.rope_theta;
    const float attn_scale = 1.0f / sqrtf((float)head_dim);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *residual = hidden;

        hidden = ggml_rms_norm(gctx, hidden, hparams_.eps);
        hidden = ggml_mul(gctx, hidden, to_f32(llm_model_->layers()[il].attn_norm));

        struct ggml_tensor *q = ggml_mul_mat(gctx, llm_model_->layers()[il].wq, hidden);
        struct ggml_tensor *k = ggml_mul_mat(gctx, llm_model_->layers()[il].wk, hidden);
        struct ggml_tensor *v = ggml_mul_mat(gctx, llm_model_->layers()[il].wv, hidden);

        // Q/K norm (per-head)
        if (llm_model_->layers()[il].attn_q_norm) {
            q = ggml_reshape_4d(gctx, q, head_dim, n_head, S, B_prime);
            q = ggml_rms_norm(gctx, q, hparams_.eps);
            q = ggml_mul(gctx, q, to_f32(llm_model_->layers()[il].attn_q_norm));
            q = ggml_reshape_3d(gctx, q, head_dim * n_head, S, B_prime);
        }
        if (llm_model_->layers()[il].attn_k_norm) {
            k = ggml_reshape_4d(gctx, k, head_dim, n_head_kv, S, B_prime);
            k = ggml_rms_norm(gctx, k, hparams_.eps);
            k = ggml_mul(gctx, k, to_f32(llm_model_->layers()[il].attn_k_norm));
            k = ggml_reshape_3d(gctx, k, head_dim * n_head_kv, S, B_prime);
        }

        q = ggml_reshape_4d(gctx, q, head_dim, n_head, S, B_prime);
        k = ggml_reshape_4d(gctx, k, head_dim, n_head_kv, S, B_prime);
        v = ggml_reshape_4d(gctx, v, head_dim, n_head_kv, S, B_prime);

        // RoPE — NEOX half-split (mode=2) matching Qwen3 reference.
        // Pairs element [i] with [i + D/2] per head, not consecutive [2i, 2i+1].
        // n_ctx=0 disables freq scaling (matching reference qwen3_build_self_attn).
        q = ggml_rope_ext_inplace(gctx, q, t_positions, nullptr, head_dim, 2, 0,
                                  rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext_inplace(gctx, k, t_positions, nullptr, head_dim, 2, 0,
                                  rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute to flash_attn_ext layout: [D, Nh/Nkv, S, B] -> [D, S, Nh/Nkv, B]
        q = ggml_permute(gctx, q, 0, 2, 1, 3);
        k = ggml_permute(gctx, k, 0, 2, 1, 3);
        v = ggml_permute(gctx, v, 0, 2, 1, 3);

        // Flash attention or manual
        struct ggml_tensor *attn_out;
        if (t_attn) {
            // Clamp V to FP16 range before flash_attn to prevent overflow on
            // sub-Ampere tensor cores that accumulate in FP16.
            v = ggml_clamp(gctx, v, -65504.0f, 65504.0f);
            attn_out = ggml_flash_attn_ext(gctx, q, k, v, t_attn, attn_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        } else {
            // Manual attention: q,k,v are already [D, S, Nh/Nkv, B]
            // GQA: repeat KV heads to match query heads
            if (n_head != n_head_kv) {
                int n_rep = n_head / n_head_kv;
                k = ggml_repeat(gctx, k, ggml_new_tensor_4d(gctx, k->type, head_dim, S, n_head, B_prime));
                k = ggml_reshape_4d(gctx, k, head_dim, S, n_head, B_prime);
                v = ggml_repeat(gctx, v, ggml_new_tensor_4d(gctx, v->type, head_dim, S, n_head, B_prime));
                v = ggml_reshape_4d(gctx, v, head_dim, S, n_head, B_prime);
            }

            // scores = k^T @ q  ->  [Skv, Sq, Nh, B]
            struct ggml_tensor *scores = ggml_mul_mat(gctx, k, q);
            scores = ggml_scale_inplace(gctx, scores, attn_scale);
            struct ggml_tensor *probs = ggml_soft_max_inplace(gctx, scores);

            // out = v @ probs  ->  [D, Sq, Nh, B]
            struct ggml_tensor *v_tr = ggml_cont(gctx, ggml_transpose(gctx, v));
            struct ggml_tensor *ctx_out = ggml_mul_mat(gctx, v_tr, probs);
            attn_out = ggml_cont(gctx, ggml_permute(gctx, ctx_out, 0, 2, 1, 3));
        }
        attn_out = ggml_reshape_3d(gctx, attn_out, head_dim * n_head, S, B_prime);
        attn_out = ggml_mul_mat(gctx, llm_model_->layers()[il].wo, attn_out);

        hidden = ggml_add(gctx, residual, attn_out);

        // FFN (SwiGLU)
        struct ggml_tensor *ffn_residual = hidden;
        hidden = ggml_rms_norm(gctx, hidden, hparams_.eps);
        hidden = ggml_mul(gctx, hidden, to_f32(llm_model_->layers()[il].ffn_norm));

        struct ggml_tensor *gate = ggml_mul_mat(gctx, llm_model_->layers()[il].ffn_gate, hidden);
        struct ggml_tensor *up = ggml_mul_mat(gctx, llm_model_->layers()[il].ffn_up, hidden);
        gate = ggml_silu_inplace(gctx, gate);
        struct ggml_tensor *ffn_out = ggml_mul(gctx, gate, up);
        ffn_out = ggml_mul_mat(gctx, llm_model_->layers()[il].ffn_down, ffn_out);

        hidden = ggml_add(gctx, ffn_residual, ffn_out);
    }

    // Final norm
    hidden = ggml_rms_norm(gctx, hidden, hparams_.eps);
    hidden = ggml_mul(gctx, hidden, to_f32(output_norm_));

    // Audio heads
    struct ggml_tensor *logits_flat = nullptr;
    for (int c = 0; c < K; c++) {
        struct ggml_tensor *head_w = codebook_head_weight_[c];
        struct ggml_tensor *head_b = codebook_head_bias_[c];
        if (!head_w && head_table) {
            head_w = ggml_view_2d(gctx, head_table, head_table->ne[0], V,
                                  head_table->nb[1], c * V * head_table->nb[1]);
        }
        struct ggml_tensor *head_logits = ggml_mul_mat(gctx, head_w, hidden);
        if (head_b)
            head_logits = ggml_add(gctx, head_logits, head_b);
        head_logits = ggml_reshape_4d(gctx, head_logits, V, 1, S, B_prime);
        logits_flat = (c == 0) ? head_logits : ggml_concat(gctx, logits_flat, head_logits, 1);
    }
    // logits_flat is [V, K, S, B_prime]

    // Output selection: if T_audio > 0, extract audio window only
    struct ggml_tensor *cond_audio = nullptr, *uncond_audio = nullptr;
    if (T_audio > 0 && T_audio < S) {
        // Cond audio: positions [S - T_audio, S)
        struct ggml_tensor *cond_view = ggml_view_4d(gctx, logits_flat, V, K, T_audio, 1,
            logits_flat->nb[1], logits_flat->nb[2], logits_flat->nb[3],
            (size_t)(S - T_audio) * logits_flat->nb[2] + 0 * logits_flat->nb[3]);
        cond_audio = ggml_cont(gctx, cond_view);
        ggml_set_name(cond_audio, "cond_audio_logits");
        ggml_set_output(cond_audio);

        // Uncond audio: positions [0, T_audio)
        struct ggml_tensor *uncond_view = ggml_view_4d(gctx, logits_flat, V, K, T_audio, 1,
            logits_flat->nb[1], logits_flat->nb[2], logits_flat->nb[3],
            0 * logits_flat->nb[2] + 1 * logits_flat->nb[3]);
        uncond_audio = ggml_cont(gctx, uncond_view);
        ggml_set_name(uncond_audio, "uncond_audio_logits");
        ggml_set_output(uncond_audio);
    } else {
        ggml_set_name(logits_flat, "audio_logits");
        ggml_set_output(logits_flat);
    }

    // Build and execute graph
    struct ggml_cgraph *graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    if (T_audio > 0) {
        ggml_build_forward_expand(graph, cond_audio);
        ggml_build_forward_expand(graph, uncond_audio);
    } else {
        ggml_build_forward_expand(graph, logits_flat);
    }

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        RS_LOG_ERR("OmniVoice: sched_alloc_graph failed (B'=%d K=%d S=%d)", B_prime, K, S);
        ggml_free(gctx);
        return {};
    }

    // Set input data
    ggml_backend_tensor_set(t_text_ids, text_ids_buf.data(), 0, text_ids_buf.size() * sizeof(int32_t));
    ggml_backend_tensor_set(t_shifted, shifted.data(), 0, shifted.size() * sizeof(int32_t));

    std::vector<float> mask_data((size_t)B_prime * S);
    std::vector<float> inv_mask_data((size_t)B_prime * S);
    for (int b = 0; b < B_prime; b++) {
        for (int s = 0; s < S; s++) {
            int m = (prompt.audio_mask[(size_t)b * S + s] != 0) ? 1 : 0;
            mask_data[(size_t)b * S + s] = (float)m;
            inv_mask_data[(size_t)b * S + s] = (float)(1 - m);
        }
    }
    ggml_backend_tensor_set(t_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_inv_mask, inv_mask_data.data(), 0, inv_mask_data.size() * sizeof(float));

    std::vector<int32_t> pos_data(S);
    for (int i = 0; i < S; i++) pos_data[i] = i;
    ggml_backend_tensor_set(t_positions, pos_data.data(), 0, S * sizeof(int32_t));

    if (t_attn)
        ggml_backend_tensor_set(t_attn, attn_f16.data(), 0, attn_f16.size() * sizeof(uint16_t));

    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("OmniVoice: LLM forward compute failed");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return {};
    }

    // Read output
    std::vector<float> out;
    if (T_audio > 0) {
        size_t per = (size_t)V * K * T_audio;
        out.resize(2 * per);
        ggml_backend_tensor_get(cond_audio, out.data(), 0, per * sizeof(float));
        ggml_backend_tensor_get(uncond_audio, out.data() + per, 0, per * sizeof(float));
    } else {
        size_t n = ggml_nelements(logits_flat);
        out.resize(n);
        ggml_backend_tensor_get(logits_flat, out.data(), 0, n * sizeof(float));
    }

    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    return out;
}

// =====================================================================
// MaskGIT helpers
// =====================================================================

void OmniVoiceModel::MaskGITLogSoftmax(float *x, int V) {
    float m = x[0];
    for (int v = 1; v < V; v++) if (x[v] > m) m = x[v];
    float sum = 0.0f;
    for (int v = 0; v < V; v++) sum += expf(x[v] - m);
    float lse = m + logf(sum);
    for (int v = 0; v < V; v++) x[v] -= lse;
}

void OmniVoiceModel::MaskGITTopKFilter(float *x, int V, float ratio) {
    int k = (int)ceilf(ratio * V);
    if (k <= 0 || k >= V) return;
    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    std::nth_element(idx.begin(), idx.begin() + (V - k), idx.end(),
                     [&](int a, int b) { return x[a] < x[b]; });
    float threshold = x[idx[V - k]];
    for (int v = 0; v < V; v++) if (x[v] < threshold) x[v] = -INFINITY;
}

void OmniVoiceModel::MaskGITGumbel(float *x, int n, float temperature, uint64_t seed, uint32_t &ctr_lo) {
    // Simplified Gumbel sampling using standard C++ random
    std::mt19937_64 rng(seed + ctr_lo);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < n; i++) {
        float u = dist(rng) + 1e-10f;
        float g = -logf(-logf(u));
        x[i] = x[i] * inv_t + g;
    }
    ctr_lo += 1;
}

// =====================================================================
// Diffusion Decode: MaskGIT iterative unmasking
// =====================================================================

bool OmniVoiceModel::RunDiffusionDecode(OmniVoiceState &state, ggml_backend_sched_t sched) {
    const int K = hparams_.n_codebooks;
    const int T = state.n_target_frames;
    const int Sref = state.n_prompt_frames;
    const int V = audio_vocab_size_;
    const int mask_id = audio_mask_id_;
    const int n_steps = n_diff_steps_;
    const float tau = hparams_.diffusion_tau;

    // Build prompt
    OmniVoicePrompt prompt;
    if (!BuildPrompt(state, prompt)) return false;

    const int S = prompt.S_max;
    const int B_prime = prompt.B_prime;
    const int audio_start_cond = S - T;

    // Initialize token matrix
    state.acoustic_tokens.resize((size_t)K * T, mask_id);
    state.n_total_frames = Sref + T;

    // Cosine timestep schedule
    std::vector<float> timesteps(n_steps + 1);
    for (int i = 0; i <= n_steps; i++) {
        float t = (float)i / n_steps;
        timesteps[i] = tau * t / (1.0f + (tau - 1.0f) * t);
    }

    // Demask schedule
    int total_mask = T * K;
    std::vector<int> demask_sched(n_steps);
    int rem = total_mask;
    for (int step = 0; step < n_steps; step++) {
        if (step == n_steps - 1) demask_sched[step] = rem;
        else {
            double frac = (double)(timesteps[step + 1] - timesteps[step]);
            int target = (int)ceil((double)total_mask * frac);
            demask_sched[step] = std::min(target, rem);
        }
        rem -= demask_sched[step];
    }
    RS_LOG_INFO("MaskGIT schedule: total=%d steps=%d sum=%d", total_mask, n_steps, total_mask - rem);

    MaskgitConfig mg_cfg;
    mg_cfg.num_step = n_steps;
    mg_cfg.guidance_scale = 2.0f;  // reference always uses 2.0
    mg_cfg.t_shift = tau;

    uint32_t ctr_lo = 0;

    RS_LOG_INFO("MaskGIT: T=%d K=%d S=%d V=%d steps=%d", T, K, S, V, n_steps);

    for (int step = 0; step < n_steps; step++) {
        int k_demask = demask_sched[step];
        if (k_demask <= 0) continue;

        state.diff_step = step + 1;

        // Run batched LLM forward
        std::vector<float> logits_full = RunLLMForwardBatched(prompt, sched, T);
        if (logits_full.empty()) return false;

        size_t per_audio = (size_t)V * K * T;
        if (logits_full.size() != 2 * per_audio) {
            RS_LOG_ERR("MaskGIT: expected %zu logits, got %zu", 2 * per_audio, logits_full.size());
            return false;
        }

        // Debug: save first-step raw logits for analysis
        if (step == 0) {
            float *sc = logits_full.data();
            float *su = logits_full.data() + per_audio;
            double cd = 0.0, cs = 0.0, us = 0.0;
            for (size_t i = 0; i < per_audio; i++) {
                double d = (double)sc[i] - (double)su[i];
                cd += d * d;
                cs += (double)sc[i] * (double)sc[i];
                us += (double)su[i] * (double)su[i];
            }
            RS_LOG_INFO("Step0 logit stats: cond_rms=%.4f uncond_rms=%.4f diff_rms=%.4f ratio=%.4f",
                        sqrt(cs/per_audio), sqrt(us/per_audio), sqrt(cd/per_audio),
                        sqrt(cd / (cs + 1e-30)));
        }

        // Extract cond and uncond logits, layout [K, T, V]
        std::vector<float> c_log((size_t)K * T * V);
        std::vector<float> u_log((size_t)K * T * V);
        float *src_cond = logits_full.data();
        float *src_uncond = logits_full.data() + per_audio;
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                size_t src_off = ((size_t)t * K + k) * V;
                size_t dst_off = ((size_t)k * T + t) * V;
                memcpy(c_log.data() + dst_off, src_cond + src_off, V * sizeof(float));
                memcpy(u_log.data() + dst_off, src_uncond + src_off, V * sizeof(float));
            }
        }

        // CFG + log_softmax
        std::vector<float> log_probs((size_t)K * T * V);
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                size_t off = ((size_t)k * T + t) * V;
                float *c = c_log.data() + off;
                float *u = u_log.data() + off;
                float *lp = log_probs.data() + off;

                if (mg_cfg.guidance_scale != 0.0f) {
                    MaskGITLogSoftmax(c, V);
                    MaskGITLogSoftmax(u, V);
                    for (int v = 0; v < V; v++)
                        lp[v] = c[v] + mg_cfg.guidance_scale * (c[v] - u[v]);
                    MaskGITLogSoftmax(lp, V);
                } else {
                    for (int v = 0; v < V; v++) lp[v] = c[v];
                    MaskGITLogSoftmax(lp, V);
                }
                lp[mask_id] = -INFINITY;
            }
        }

        // Predict tokens and confidence
        std::vector<int32_t> pred_tokens((size_t)K * T);
        std::vector<float> confidence((size_t)K * T);
        for (int k = 0; k < K; k++) {
            for (int t = 0; t < T; t++) {
                size_t off = ((size_t)k * T + t) * V;
                float *lp = log_probs.data() + off;

                int best_v = 0;
                float best = lp[0];
                for (int v = 1; v < V; v++) {
                    if (lp[v] > best) { best = lp[v]; best_v = v; }
                }
                pred_tokens[(size_t)k * T + t] = (int32_t)best_v;
                confidence[(size_t)k * T + t] = best;
            }
        }

        // Layer penalty
        for (int k = 0; k < K; k++)
            for (int t = 0; t < T; t++)
                confidence[(size_t)k * T + t] -= (float)k * mg_cfg.layer_penalty_factor;

        // Mask scores of already-decoded slots
        for (int k = 0; k < K; k++)
            for (int t = 0; t < T; t++)
                if (state.acoustic_tokens[(size_t)k * T + t] != mask_id)
                    confidence[(size_t)k * T + t] = -INFINITY;

        // Top-k selection
        int N = K * T;
        std::vector<int> idx(N);
        for (int i = 0; i < N; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k_demask, idx.end(),
                         [&](int a, int b) { return confidence[a] > confidence[b]; });

        // Apply predictions
        for (int n = 0; n < k_demask; n++) {
            int i = idx[n];
            int k = i / T, t = i % T;
            int32_t v = pred_tokens[(size_t)k * T + t];
            state.acoustic_tokens[(size_t)k * T + t] = v;

            // Update prompt for next iteration
            prompt.input_ids[((size_t)0 * K + k) * S + (audio_start_cond + t)] = v;
            prompt.input_ids[((size_t)1 * K + k) * S + t] = v;
        }

        int remaining = (int)std::count(state.acoustic_tokens.begin(), state.acoustic_tokens.end(), mask_id);
    }

    // Safety net: force-demask any residual mask tokens
    int remaining = (int)std::count(state.acoustic_tokens.begin(), state.acoustic_tokens.end(), mask_id);
    if (remaining > 0) {
        RS_LOG_INFO("MaskGIT: force-demasking %d residual tokens", remaining);
        std::vector<float> logits_full = RunLLMForwardBatched(prompt, sched, T);
        if (logits_full.empty()) return false;

        size_t per_audio = (size_t)V * K * T;
        if (logits_full.size() >= 2 * per_audio) {
            for (int k = 0; k < K; k++) {
                for (int t = 0; t < T; t++) {
                    size_t idx = (size_t)k * T + t;
                    if (state.acoustic_tokens[idx] != mask_id) continue;

                    // Use cond logits for argmax
                    size_t src_off = ((size_t)t * K + k) * V;
                    float *src = logits_full.data() + src_off;
                    float best = src[0];
                    int best_v = 0;
                    for (int v = 1; v < V - 1; v++) { // skip mask_id
                        if (src[v] > best) { best = src[v]; best_v = v; }
                    }
                    state.acoustic_tokens[idx] = (int32_t)best_v;
                }
            }
        }
        remaining = (int)std::count(state.acoustic_tokens.begin(), state.acoustic_tokens.end(), mask_id);
        if (remaining > 0)
            RS_LOG_ERR("MaskGIT: still %d residual masks after force-demask!", remaining);
    }

    return true;
}

// =====================================================================
// Encode: text -> acoustic tokens via diffusion
// =====================================================================

bool OmniVoiceModel::Encode(const std::vector<float> &input_frames,
                            RSState &state, ggml_backend_sched_t sched) {
    auto &s = static_cast<OmniVoiceState &>(state);
    (void)input_frames;

    if (s.text_tokens.empty()) {
        RS_LOG_ERR("OmniVoice: no text pushed for synthesis");
        return false;
    }

    // Duration estimation — byte-perfect mirror of duration_estimate_tokens()
    // in omnivoice/utils/duration.py. Uses ref-based speed factor when both
    // ref_text and ref_audio_tokens are available; falls back to the canonical
    // anchor "Nice to meet you." at 25 tokens otherwise.
    auto char_weight = [](uint32_t cp) -> float {
        if (cp >= 0x4E00 && cp <= 0x9FFF) return 3.0f;          // CJK
        if (cp >= 0x3400 && cp <= 0x4DBF) return 3.0f;          // CJK Ext-A
        if (cp >= 0xF900 && cp <= 0xFAFF) return 3.0f;          // CJK compat
        if (cp >= 0xAC00 && cp <= 0xD7AF) return 2.5f;          // Hangul
        if (cp >= 0x3040 && cp <= 0x30FF) return 2.2f;          // Kana
        if (cp >= 0x2E80 && cp <= 0x2FDF) return 3.0f;          // CJK radicals
        if (cp >= 0x3000 && cp <= 0x312F) return 3.0f;          // CJK symbols
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return 1.0f;
        if (cp >= '0' && cp <= '9') return 3.5f;
        if (cp == ' ') return 0.2f;
        return 0.5f;  // punctuation, other
    };
    auto text_weight = [&](const std::string &t) -> float {
        float w = 0.0f;
        size_t p = 0;
        while (p < t.size()) {
            uint32_t cp = 0;
            int n = prompt_utf8_decode(t.data() + p, t.size() - p, &cp);
            if (n == 0) { p++; continue; }
            w += char_weight(cp);
            p += n;
        }
        return w;
    };

    // Mirror duration_estimate_tokens(): use ref_text + ref_audio_tokens when
    // available, otherwise fall back to anchor.
    std::string est_ref_text;
    int est_ref_duration;
    if (s.n_prompt_frames > 0 && !s.ref_text.empty()) {
        est_ref_text = s.ref_text;
        est_ref_duration = s.n_prompt_frames;
    } else {
        est_ref_text = "Nice to meet you.";
        est_ref_duration = 25;
    }

    float ref_weight = text_weight(est_ref_text);
    float speed_factor = (ref_weight > 0.0f && est_ref_duration > 0)
        ? ref_weight / (float)est_ref_duration
        : 16.9f / 25.0f;

    float target_weight = text_weight(s.text_original);
    float estimated = (target_weight > 0.0f) ? target_weight / speed_factor : 16.0f;
    // Power-law boost for short sequences (mirrors RuleDurationEstimator)
    {
        float low_threshold = 50.0f;
        float boost_strength = 3.0f;
        if (estimated < low_threshold) {
            float alpha = 1.0f / boost_strength;
            estimated = low_threshold * powf(estimated / low_threshold, alpha);
        }
    }
    s.n_target_frames = std::max(1, (int)estimated);

    RS_LOG_INFO("OmniVoice: encoding '%s' weight=%.1f speed=%.4f -> %d target frames (ref_frames=%d ref='%s')",
                s.text_original.c_str(), target_weight, speed_factor, s.n_target_frames,
                s.n_prompt_frames, est_ref_text.c_str());

    return RunDiffusionDecode(s, sched);
}

// =====================================================================
// Decode: acoustic tokens -> audio via vocoder
// =====================================================================

// Snake activation: y = x + sin^2(alpha * x) * inv_b
// inv_b is precomputed as 1/(alpha + 1e-9)
// name: unique name for the inv_b tensor (used to set data after graph allocation)
// Snake activation: y = x + sin^2(alpha * x) * inv_b
// x is [T, C] 2D, alpha and inv_b are [1, C] 2D
// Broadcast happens automatically via ggml_mul repeat mechanism
static struct ggml_tensor *dac_snake_graph(struct ggml_context *ctx,
                                           struct ggml_tensor *x,
                                           struct ggml_tensor *alpha,
                                           const std::vector<float> &inv_b_data,
                                           const char *name) {
    int64_t T = x->ne[0], C = x->ne[1];

    // Ensure alpha is F32 for binary ops (may be F16 from GGUF)
    if (alpha->type != GGML_TYPE_F32)
        alpha = ggml_cast(ctx, alpha, GGML_TYPE_F32);

    struct ggml_tensor *t = ggml_mul(ctx, x, alpha);      // [T, C] * [1, C] -> [T, C]
    t = ggml_sin(ctx, t);
    t = ggml_sqr(ctx, t);
    struct ggml_tensor *inv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    ggml_set_name(inv_b, name);
    ggml_set_input(inv_b);
    t = ggml_mul(ctx, t, inv_b);                          // [T, C] * [1, C] -> [T, C]
    return ggml_add(ctx, x, t);
}

bool OmniVoiceModel::RunVocoder(OmniVoiceState &state, ggml_backend_sched_t sched) {
    const int T = state.n_target_frames;
    const int K = hparams_.n_codebooks;

    if (!codec_loaded_) {
        RS_LOG_ERR("OmniVoice: codec not loaded, cannot decode audio");
        return false;
    }

    if (state.vocoder_done) return false;

    // Verify no residual mask tokens
    for (size_t i = 0; i < state.acoustic_tokens.size(); i++) {
        if (state.acoustic_tokens[i] == audio_mask_id_) {
            RS_LOG_ERR("OmniVoice: residual mask token at index %zu, refusing to decode", i);
            return false;
        }
    }

    // Build vocoder graph
    struct ggml_init_params params = {
        (size_t)(OMNIVOICE_MAX_NODES / 2) * ggml_tensor_overhead() + (1 << 20),
        nullptr, true};
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return false;

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, OMNIVOICE_MAX_NODES / 2, false);

    // --- RVQ Decode ---
    // Input tokens: [T, K] (T fast, K slow)
    struct ggml_tensor *tokens_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, T, K);
    ggml_set_input(tokens_inp);

    struct ggml_tensor *rvq_out = nullptr;
    for (int c = 0; c < std::min(K, rvq_.num_codebooks); c++) {
        struct ggml_tensor *cb_tokens = ggml_view_1d(ctx, tokens_inp, T,
            (size_t)c * T * sizeof(int32_t));
        // Embed table already in ggml [dim, vocab_size] = [64, 1024] from conversion
        struct ggml_tensor *emb = ggml_get_rows(ctx, rvq_.cb[c].embed, cb_tokens);
        if (rvq_.cb[c].project_out_w) {
            // project_out_w already in ggml [in=64, out=1024] from conversion
            struct ggml_tensor *proj = ggml_mul_mat(ctx, rvq_.cb[c].project_out_w, emb);
            if (rvq_.cb[c].project_out_b)
                proj = ggml_add(ctx, proj, rvq_.cb[c].project_out_b);
            rvq_out = (c == 0) ? proj : ggml_add(ctx, rvq_out, proj);
        } else {
            rvq_out = (c == 0) ? emb : ggml_add(ctx, rvq_out, emb);
        }
    }

    if (!rvq_out) {
        RS_LOG_ERR("OmniVoice: RVQ decode produced no output");
        ggml_free(ctx);
        return false;
    }
    ggml_set_name(rvq_out, "rvq_out");
    ggml_set_output(rvq_out);

    // --- fc2: 1024 -> 256 ---
    // fc2_w_ already in ggml [in=1024, out=256] from conversion
    struct ggml_tensor *fc2_out = rvq_out;
    if (fc2_w_) {
        fc2_out = ggml_mul_mat(ctx, fc2_w_, fc2_out);
        if (fc2_b_) fc2_out = ggml_add(ctx, fc2_out, fc2_b_);
    }
    // fc2_out is [256, T], transpose to [T, 256] for conv
    fc2_out = ggml_cont(ctx, ggml_transpose(ctx, fc2_out));

    // --- DAC Decoder ---
    struct ggml_tensor *cur = fc2_out;

    // Initial conv1: 256 -> 1024, k=7, pad=3
    if (dac_.c1w) {
        struct ggml_tensor *c1w_f16 = dac_.c1w;
        if (dac_.c1w->type != GGML_TYPE_F16)
            c1w_f16 = ggml_cast(ctx, dac_.c1w, GGML_TYPE_F16);
        cur = ggml_conv_1d(ctx, c1w_f16, ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[1], 1), 1, 3, 1);
        if (dac_.c1b) {
            struct ggml_tensor *b2d = ggml_reshape_2d(ctx, dac_.c1b, 1, dac_.c1b->ne[0]);
            cur = ggml_add(ctx, cur, b2d);
        }
    }

    // 5 upsampling blocks
    // Decoder block: Snake(in_ch) → ConvTranspose(in_ch→out_ch) → ResUnits(out_ch)
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        DACBlockWeights &b = dac_.blk[i];
        if (!b.ctw) continue;

        // Snake1 (on in_ch, before conv_transpose)
        if (b.s1.a && !b.s1.inv_b.empty()) {
            int C_alpha = (int)ggml_nelements(b.s1.a);
            struct ggml_tensor *alpha = ggml_reshape_2d(ctx, b.s1.a, 1, C_alpha);
            char snake_name[64];
            snprintf(snake_name, sizeof(snake_name), "inv_b_blk%d_s1", i);
            cur = dac_snake_graph(ctx, cur, alpha, b.s1.inv_b, snake_name);
        }

        // ConvTranspose1d: ggml output is (T_in-1)*s + K = T_in*s + s (K=2s, p=0)
        // PyTorch with padding=(s+1)//2 and output_padding=s%2 gives T_in*s
        // The total excess is 's' samples, split: pad_left = (s+1)//2 from LHS
        // and the rest from RHS. No output_pad needed — it's already in the formula.
        if (b.ctw && b.ctb) {
            int K = 2 * b.stride;
            struct ggml_tensor *ctw_raw = ggml_cont(ctx, b.ctw);
            if (ctw_raw->type != GGML_TYPE_F16)
                ctw_raw = ggml_cast(ctx, ctw_raw, GGML_TYPE_F16);
            struct ggml_tensor *ctw_3d = ggml_reshape_3d(ctx, ctw_raw, K, b.out_ch, b.in_ch);
            cur = ggml_conv_transpose_1d(ctx, ctw_3d, cur, b.stride, 0, 1);

            // Crop: remove pad_left from LHS, keep T_in*s samples
            int T_cur = (int)cur->ne[0];
            int pad_left = b.pad;  // (stride+1)/2
            int keep_len = T_cur - b.stride;  // = T_in * stride
            if (keep_len > 0 && pad_left + keep_len <= T_cur) {
                cur = ggml_view_2d(ctx, cur, keep_len, cur->ne[1],
                                   cur->nb[1], (size_t)pad_left * sizeof(float));
            }

            struct ggml_tensor *b2d = ggml_reshape_2d(ctx, b.ctb, 1, b.ctb->ne[0]);
            cur = ggml_add(ctx, cur, b2d);

            cur = ggml_cont(ctx, cur);
        }

        // 3 residual units (operate on out_ch, after conv_transpose)
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            DACResUnitWeights &ru = b.ru[r];
            if (!ru.c1w) continue;
            struct ggml_tensor *skip = cur;

            // Snake1 + conv1
            if (ru.s1.a && !ru.s1.inv_b.empty()) {
                int C_alpha = (int)ggml_nelements(ru.s1.a);
                struct ggml_tensor *alpha = ggml_reshape_2d(ctx, ru.s1.a, 1, C_alpha);
                char snake_name[64];
                snprintf(snake_name, sizeof(snake_name), "inv_b_blk%d_ru%d_s1", i, r);
                cur = dac_snake_graph(ctx, cur, alpha, ru.s1.inv_b, snake_name);
            }

            int pad1 = 3 * ru.dilation;
            struct ggml_tensor *c1w_f16 = ru.c1w;
            if (ru.c1w->type != GGML_TYPE_F16)
                c1w_f16 = ggml_cast(ctx, ru.c1w, GGML_TYPE_F16);
            cur = ggml_conv_1d(ctx, c1w_f16, ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[1], 1), 1, pad1, ru.dilation);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
            if (ru.c1b) {
                struct ggml_tensor *b2d = ggml_reshape_2d(ctx, ru.c1b, 1, ru.c1b->ne[0]);
                cur = ggml_add(ctx, cur, b2d);
            }
            cur = ggml_cont(ctx, cur);

            // Snake2 + conv2 (1x1)
            if (ru.s2.a && !ru.s2.inv_b.empty()) {
                int C_alpha = (int)ggml_nelements(ru.s2.a);
                struct ggml_tensor *alpha = ggml_reshape_2d(ctx, ru.s2.a, 1, C_alpha);
                char snake_name[64];
                snprintf(snake_name, sizeof(snake_name), "inv_b_blk%d_ru%d_s2", i, r);
                cur = dac_snake_graph(ctx, cur, alpha, ru.s2.inv_b, snake_name);
            }

            struct ggml_tensor *c2w_f16 = ru.c2w;
            if (ru.c2w->type != GGML_TYPE_F16)
                c2w_f16 = ggml_cast(ctx, ru.c2w, GGML_TYPE_F16);
            cur = ggml_conv_1d(ctx, c2w_f16, ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[1], 1), 1, 0, 1);
            cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
            if (ru.c2b) {
                struct ggml_tensor *b2d = ggml_reshape_2d(ctx, ru.c2b, 1, ru.c2b->ne[0]);
                cur = ggml_add(ctx, cur, b2d);
            }

            cur = ggml_add(ctx, skip, cur);
        }

    }

    // Final snake + conv2: 32 -> 1
    if (dac_.s_final.a && !dac_.s_final.inv_b.empty()) {
        int C_alpha = (int)ggml_nelements(dac_.s_final.a);
        struct ggml_tensor *alpha = ggml_reshape_2d(ctx, dac_.s_final.a, 1, C_alpha);
        cur = dac_snake_graph(ctx, cur, alpha, dac_.s_final.inv_b, "inv_b_sfinal");
    }
    if (dac_.c2w) {
        struct ggml_tensor *c2w_f16 = dac_.c2w;
        if (dac_.c2w->type != GGML_TYPE_F16)
            c2w_f16 = ggml_cast(ctx, dac_.c2w, GGML_TYPE_F16);
        cur = ggml_conv_1d(ctx, c2w_f16, ggml_reshape_3d(ctx, cur, cur->ne[0], cur->ne[1], 1), 1, 3, 1);
        cur = ggml_reshape_2d(ctx, cur, cur->ne[0], cur->ne[1]);
        if (dac_.c2b) {
            struct ggml_tensor *b2d = ggml_reshape_2d(ctx, dac_.c2b, 1, dac_.c2b->ne[0]);
            cur = ggml_add(ctx, cur, b2d);
        }
    }
    // Output: [T_out, 1] -> flat audio, apply tanh as in reference DAC decode()
    cur = ggml_tanh(ctx, cur);
    cur = ggml_cont(ctx, ggml_view_1d(ctx, cur, ggml_nelements(cur), 0));
    ggml_set_name(cur, "audio_out");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    // Allocate all tensors on CPU backend — bypasses the GPU scheduler entirely.
    // The DAC vocoder has many small conv ops where GPU kernel launch overhead
    // dominates (~300x slower than CPU).
    ggml_backend_buffer_t voc_buf = ggml_backend_alloc_ctx_tensors(ctx, cpu_backend_);
    if (!voc_buf) {
        RS_LOG_ERR("OmniVoice: vocoder CPU alloc failed");
        ggml_free(ctx);
        return false;
    }

    // Set inv_b tensor data (precomputed snake activation parameters)
    auto set_inv_b = [&](const std::vector<float> &data, const char *name) {
        if (data.empty()) return;
        struct ggml_tensor *t = ggml_get_tensor(ctx, name);
        if (t) ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    };
    for (int i = 0; i < DAC_NUM_BLOCKS; i++) {
        char name[64];
        snprintf(name, sizeof(name), "inv_b_blk%d_s1", i);
        set_inv_b(dac_.blk[i].s1.inv_b, name);
        for (int r = 0; r < DAC_RES_UNITS; r++) {
            snprintf(name, sizeof(name), "inv_b_blk%d_ru%d_s1", i, r);
            set_inv_b(dac_.blk[i].ru[r].s1.inv_b, name);
            snprintf(name, sizeof(name), "inv_b_blk%d_ru%d_s2", i, r);
            set_inv_b(dac_.blk[i].ru[r].s2.inv_b, name);
        }
    }
    set_inv_b(dac_.s_final.inv_b, "inv_b_sfinal");

    // Set input tokens [T, K] (T fast)
    // Set input tokens [T, K]: tokens_inp ne0=T (fast), ne1=K (slow) -> data[t + k*T]
    // acoustic_tokens layout: [k*T + t]
    std::vector<int32_t> tokens_data((size_t)T * K);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++)
            tokens_data[(size_t)t + k * T] = state.acoustic_tokens[(size_t)k * T + t];

    ggml_backend_tensor_set(tokens_inp, tokens_data.data(), 0, tokens_data.size() * sizeof(int32_t));

    // Compute vocoder graph on CPU
    ggml_backend_cpu_set_n_threads(cpu_backend_, 4);
    if (ggml_backend_graph_compute(cpu_backend_, gf) != GGML_STATUS_SUCCESS) {
        RS_LOG_ERR("OmniVoice: vocoder CPU compute failed");
        ggml_backend_buffer_free(voc_buf);
        ggml_free(ctx);
        return false;
    }

    struct ggml_tensor *audio_out = ggml_get_tensor(ctx, "audio_out");
    if (audio_out) {
        int n_samples = (int)ggml_nelements(audio_out);
        state.audio_output.resize(n_samples);
        ggml_backend_tensor_get(audio_out, state.audio_output.data(), 0, ggml_nbytes(audio_out));

        // Debug: audio statistics
        float a_min = 0, a_max = 0, a_mean = 0, a_rms = 0;
        double hf_sum = 0, lf_sum = 0;
        if (n_samples > 0) {
            a_min = a_max = state.audio_output[0];
            for (int i = 0; i < n_samples; i++) {
                float v = state.audio_output[i];
                if (v < a_min) a_min = v;
                if (v > a_max) a_max = v;
                a_mean += v;
                a_rms += (double)v * v;
                if (i > 0) {
                    double d = (double)v - state.audio_output[i-1];
                    hf_sum += d * d;
                }
                lf_sum += (double)v * v;
            }
            a_mean /= n_samples;
            a_rms = (float)sqrt(a_rms / n_samples);
        }
        RS_LOG_INFO("OmniVoice: vocoder output %d samples (%.2fs at %d Hz) [%.4f, %.4f] rms=%.4f hf/lf=%.4f",
                    n_samples, (double)n_samples / hparams_.audio_sample_rate, hparams_.audio_sample_rate,
                    a_min, a_max, a_rms,
                    hf_sum / std::max(lf_sum, 1e-30));

    }

    ggml_backend_buffer_free(voc_buf);
    ggml_free(ctx);
    bool ok = !state.audio_output.empty();
    if (ok) state.vocoder_done = true;
    return ok;
}

bool OmniVoiceModel::Decode(RSState &state, ggml_backend_sched_t sched) {
    auto &s = static_cast<OmniVoiceState &>(state);
    return RunVocoder(s, sched);
}

// =====================================================================
// Audio output
// =====================================================================

int OmniVoiceModel::GetAudioOutput(RSState &state, float **out_data) {
    auto &s = static_cast<OmniVoiceState &>(state);
    if (s.audio_output.empty()) {
        *out_data = nullptr;
        return 0;
    }
    *out_data = s.audio_output.data();
    int n = (int)s.audio_output.size();
    // Don't clear — caller may need multiple reads
    return n;
}

// =====================================================================
// Static model registration
// =====================================================================

extern void rs_register_model_arch(const std::string &arch,
                                   std::function<std::shared_ptr<ISpeechModel>()> creator);
namespace {
struct OmniVoiceRegistrar {
    OmniVoiceRegistrar() {
        rs_register_model_arch("OmniVoice", []() {
            return std::make_shared<OmniVoiceModel>();
        });
        // Also register omnivoice-lm as architecture name used in GGUF
        rs_register_model_arch("omnivoice-lm", []() {
            return std::make_shared<OmniVoiceModel>();
        });
    }
} global_omnivoice_reg;
}  // namespace
