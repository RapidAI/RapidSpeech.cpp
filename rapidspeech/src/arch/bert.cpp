#include "bert.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rapidspeech {

// =====================================================================
// UTF-8 helpers
// =====================================================================
namespace {

// Decode one UTF-8 codepoint from s starting at pos; advances pos.
// Returns 0xFFFD on malformed input.
uint32_t utf8_next(const std::string &s, size_t &pos) {
    if (pos >= s.size()) return 0;
    unsigned char c = (unsigned char)s[pos++];
    if (c < 0x80) return c;
    int extra;
    uint32_t cp;
    if ((c & 0xE0) == 0xC0) { extra = 1; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { extra = 2; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { extra = 3; cp = c & 0x07; }
    else return 0xFFFD;
    for (int i = 0; i < extra; ++i) {
        if (pos >= s.size()) return 0xFFFD;
        unsigned char cc = (unsigned char)s[pos++];
        if ((cc & 0xC0) != 0x80) return 0xFFFD;
        cp = (cp << 6) | (cc & 0x3F);
    }
    return cp;
}

void utf8_append(std::string &out, uint32_t cp) {
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

bool is_whitespace_cp(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') return true;
    // HF BertBasicTokenizer treats Unicode category Zs as whitespace.
    if (cp == 0x00A0 || cp == 0x1680) return true;
    if (cp >= 0x2000 && cp <= 0x200A) return true;
    if (cp == 0x202F || cp == 0x205F || cp == 0x3000) return true;
    return false;
}

bool is_control_cp(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r') return false;
    if (cp < 0x20) return true;
    if (cp == 0x7F) return true;
    if (cp >= 0x80 && cp <= 0x9F) return true;
    return false;
}

bool is_punct_cp(uint32_t cp) {
    // HF BertBasicTokenizer treats: ASCII punctuation OR Unicode P* categories.
    if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
        (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
        return true;
    // Approximate Unicode P* with common ranges (CJK punctuation etc.).
    if (cp >= 0x2000 && cp <= 0x206F) return true;  // General Punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return true;  // CJK Symbols and Punctuation
    if (cp >= 0xFF00 && cp <= 0xFF0F) return true;  // Fullwidth punctuation pt1
    if (cp >= 0xFF1A && cp <= 0xFF20) return true;  // Fullwidth pt2
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;  // Fullwidth pt3
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;  // Fullwidth pt4
    return false;
}

bool is_chinese_cp(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0x2F800 && cp <= 0x2FA1F);
}

uint32_t to_lower_cp(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp - 'A' + 'a';
    // Latin-1 supplement uppercase → lowercase (basic coverage).
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) return cp + 32;
    return cp;
}

// Light NFD-strip-accents pass for common Latin-with-diacritics: remove
// combining-mark codepoints (U+0300..U+036F). For full HF parity we'd need
// real NFD; this covers the cases that show up for our test corpus.
bool is_combining_mark_cp(uint32_t cp) {
    return cp >= 0x0300 && cp <= 0x036F;
}

}  // namespace

// =====================================================================
// WordPiece tokenizer
// =====================================================================

bool WordPieceTokenizer::LoadFromGGUF(struct gguf_context *gguf) {
    int tok_key = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    if (tok_key < 0) {
        RS_LOG_ERR("BERT GGUF: tokenizer.ggml.tokens not found");
        return false;
    }
    int n_tokens = (int)gguf_get_arr_n(gguf, tok_key);
    id_to_token_.resize(n_tokens);
    token_to_id_.reserve(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        std::string s = std::string(gguf_get_arr_str(gguf, tok_key, i));
        id_to_token_[i] = s;
        token_to_id_[s] = i;
    }

    auto try_i32 = [&](const char *k, int defv) {
        int idx = gguf_find_key(gguf, k);
        if (idx < 0) return defv;
        // Some writers store as INT32, others as UINT32; try both.
        auto t = gguf_get_kv_type(gguf, idx);
        if (t == GGUF_TYPE_UINT32) return (int)gguf_get_val_u32(gguf, idx);
        return (int)gguf_get_val_i32(gguf, idx);
    };
    auto try_bool = [&](const char *k, bool defv) {
        int idx = gguf_find_key(gguf, k);
        if (idx < 0) return defv;
        return (bool)gguf_get_val_bool(gguf, idx);
    };

    unk_id_  = try_i32("tokenizer.ggml.unk_id",  100);
    cls_id_  = try_i32("tokenizer.ggml.cls_id",  101);
    sep_id_  = try_i32("tokenizer.ggml.sep_id",  102);
    pad_id_  = try_i32("tokenizer.ggml.pad_id",  0);
    mask_id_ = try_i32("tokenizer.ggml.mask_id", 103);
    do_lower_case_ = try_bool("tokenizer.ggml.do_lower_case", true);

    RS_LOG_INFO("BERT tokenizer: vocab=%d unk=%d cls=%d sep=%d pad=%d mask=%d lower=%d",
                n_tokens, unk_id_, cls_id_, sep_id_, pad_id_, mask_id_, (int)do_lower_case_);
    return n_tokens > 0;
}

const std::string &WordPieceTokenizer::id_to_token(int id) const {
    static const std::string empty;
    if (id < 0 || id >= (int)id_to_token_.size()) return empty;
    return id_to_token_[id];
}

int WordPieceTokenizer::token_to_id(const std::string &t) const {
    auto it = token_to_id_.find(t);
    return it == token_to_id_.end() ? -1 : it->second;
}

std::vector<std::string> WordPieceTokenizer::BasicTokenize(const std::string &text) const {
    // Pass 1: per-codepoint normalize (lowercase + strip accents + strip control)
    // and force CJK / punctuation to their own pre-token by surrounding them
    // with spaces.
    std::string spaced;
    spaced.reserve(text.size() * 2);
    size_t pos = 0;
    while (pos < text.size()) {
        uint32_t cp = utf8_next(text, pos);
        if (cp == 0 || cp == 0xFFFD) continue;
        if (is_control_cp(cp)) continue;
        if (is_whitespace_cp(cp)) { spaced.push_back(' '); continue; }
        if (do_lower_case_) {
            if (is_combining_mark_cp(cp)) continue;
            cp = to_lower_cp(cp);
        }
        if (is_chinese_cp(cp) || is_punct_cp(cp)) {
            spaced.push_back(' ');
            utf8_append(spaced, cp);
            spaced.push_back(' ');
        } else {
            utf8_append(spaced, cp);
        }
    }

    // Pass 2: whitespace split.
    std::vector<std::string> out;
    std::string cur;
    for (char c : spaced) {
        if (c == ' ') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void WordPieceTokenizer::WordPiece(const std::string &pre_token,
                                   std::vector<int> &out) const {
    // Per HF BertWordPieceTokenizer: tokens longer than max_input_chars
    // (default 100) emit a single [UNK].
    const int max_input_chars = 100;
    // Count UTF-8 chars (codepoints) to compare with the HF limit.
    int n_cp = 0;
    for (size_t i = 0; i < pre_token.size(); ) {
        size_t prev = i;
        (void)utf8_next(pre_token, i);
        if (i == prev) { i++; }
        n_cp++;
        if (n_cp > max_input_chars) break;
    }
    if (n_cp > max_input_chars) { out.push_back(unk_id_); return; }

    // Build byte offsets for each codepoint so we can slice on char boundaries.
    std::vector<size_t> starts;
    starts.reserve(pre_token.size());
    for (size_t i = 0; i < pre_token.size(); ) {
        starts.push_back(i);
        size_t prev = i;
        (void)utf8_next(pre_token, i);
        if (i == prev) i++;
    }
    starts.push_back(pre_token.size());
    int n = (int)starts.size() - 1;  // number of codepoints

    std::vector<int> sub_ids;
    int start = 0;
    bool bad = false;
    while (start < n) {
        int end = n;
        int found = -1;
        size_t found_end = 0;
        while (start < end) {
            std::string sub = pre_token.substr(starts[start], starts[end] - starts[start]);
            if (start > 0) sub = "##" + sub;
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                found = it->second;
                found_end = end;
                break;
            }
            end--;
        }
        if (found < 0) { bad = true; break; }
        sub_ids.push_back(found);
        start = (int)found_end;
    }
    if (bad) { out.push_back(unk_id_); return; }
    for (int id : sub_ids) out.push_back(id);
}

std::vector<int> WordPieceTokenizer::Tokenize(
    const std::string &text, int max_subwords,
    std::vector<int> *out_word_boundaries) const
{
    if (max_subwords <= 0) max_subwords = 510;

    std::vector<std::string> pre_tokens = BasicTokenize(text);

    std::vector<int> ids;
    std::vector<int> word_starts;  // 1 for the first subword of a new word, 0 otherwise

    ids.push_back(cls_id_);
    word_starts.push_back(1);

    int budget = max_subwords;
    for (const auto &pt : pre_tokens) {
        if (budget <= 0) break;
        std::vector<int> sub;
        WordPiece(pt, sub);
        if ((int)sub.size() > budget) {
            // Truncate at subword boundary; matches HF "truncation=True" behaviour.
            sub.resize(budget);
        }
        for (size_t k = 0; k < sub.size(); ++k) {
            ids.push_back(sub[k]);
            word_starts.push_back(k == 0 ? 1 : 0);
        }
        budget -= (int)sub.size();
    }

    ids.push_back(sep_id_);
    word_starts.push_back(1);

    if (out_word_boundaries) *out_word_boundaries = std::move(word_starts);
    return ids;
}

// =====================================================================
// BertModel
// =====================================================================

BertModel::BertModel() = default;

BertModel::~BertModel() {
    if (sched_) ggml_backend_sched_free(sched_);
    if (weight_buf_) ggml_backend_buffer_free(weight_buf_);
    if (weight_ctx_) ggml_free(weight_ctx_);
    if (backend_) ggml_backend_free(backend_);
    if (gguf_ctx_) gguf_free(gguf_ctx_);
}

bool BertModel::MapTensors() {
    auto find_t = [&](const char *name) -> ggml_tensor * {
        return ggml_get_tensor(weight_ctx_, name);
    };
    auto require = [&](const char *name) -> ggml_tensor * {
        ggml_tensor *t = find_t(name);
        if (!t) RS_LOG_WARN("BERT: missing tensor '%s'", name);
        return t;
    };

    emb_.word_emb     = require("embeddings.word.weight");
    emb_.position_emb = require("embeddings.position.weight");
    emb_.token_type_emb = find_t("embeddings.token_type.weight");
    emb_.ln_w         = require("embeddings.LayerNorm.weight");
    emb_.ln_b         = require("embeddings.LayerNorm.bias");

    if (!emb_.word_emb || !emb_.position_emb || !emb_.ln_w || !emb_.ln_b) return false;

    layers_.assign(hp_.num_layers, BertLayerWeights{});
    char buf[256];
    for (int i = 0; i < hp_.num_layers; ++i) {
        BertLayerWeights &l = layers_[i];
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.self.query.weight", i); l.attn.q_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.self.query.bias",   i); l.attn.q_b = find_t(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.self.key.weight",   i); l.attn.k_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.self.key.bias",     i); l.attn.k_b = find_t(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.self.value.weight", i); l.attn.v_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.self.value.bias",   i); l.attn.v_b = find_t(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.output.dense.weight", i); l.attn.o_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.output.dense.bias",   i); l.attn.o_b = find_t(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.output.LayerNorm.weight", i); l.ln_attn_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.attention.output.LayerNorm.bias",   i); l.ln_attn_b = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.intermediate.dense.weight", i); l.ffn.w1_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.intermediate.dense.bias",   i); l.ffn.w1_b = find_t(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.output.dense.weight", i); l.ffn.w2_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.output.dense.bias",   i); l.ffn.w2_b = find_t(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.output.LayerNorm.weight", i); l.ln_ffn_w = require(buf);
        snprintf(buf, sizeof(buf), "encoder.layer.%d.output.LayerNorm.bias",   i); l.ln_ffn_b = require(buf);

        if (!l.attn.q_w || !l.attn.k_w || !l.attn.v_w || !l.attn.o_w ||
            !l.ln_attn_w || !l.ffn.w1_w || !l.ffn.w2_w || !l.ln_ffn_w) {
            RS_LOG_ERR("BERT: layer %d missing required weights", i);
            return false;
        }
    }
    return true;
}

bool BertModel::LoadFromGGUF(const char *path, bool use_gpu) {
    (void)use_gpu;  // CPU-only for now; BERT is small and runs once per utterance.

    struct gguf_init_params gp = { /*no_alloc=*/true, /*ctx=*/&weight_ctx_ };
    gguf_ctx_ = gguf_init_from_file(path, gp);
    if (!gguf_ctx_) {
        RS_LOG_ERR("BERT: failed to open GGUF: %s", path);
        return false;
    }

    auto get_i32 = [&](const char *k, int defv) {
        int idx = gguf_find_key(gguf_ctx_, k);
        if (idx < 0) return defv;
        auto t = gguf_get_kv_type(gguf_ctx_, idx);
        if (t == GGUF_TYPE_UINT32) return (int)gguf_get_val_u32(gguf_ctx_, idx);
        return (int)gguf_get_val_i32(gguf_ctx_, idx);
    };
    auto get_f32 = [&](const char *k, float defv) {
        int idx = gguf_find_key(gguf_ctx_, k);
        return idx < 0 ? defv : gguf_get_val_f32(gguf_ctx_, idx);
    };

    hp_.hidden_size       = get_i32("bert.hidden_size", 768);
    hp_.num_layers        = get_i32("bert.num_layers", 12);
    hp_.num_heads         = get_i32("bert.num_heads", 12);
    hp_.intermediate_size = get_i32("bert.intermediate_size", 3072);
    hp_.max_position      = get_i32("bert.max_position_embeddings", 512);
    hp_.vocab_size        = get_i32("bert.vocab_size", 30522);
    hp_.type_vocab_size   = get_i32("bert.type_vocab_size", 2);
    hp_.layer_norm_eps    = get_f32("bert.layer_norm_eps", 1e-12f);
    hp_.feature_layers    = get_i32("bert.feature_layers", 4);

    RS_LOG_INFO("BERT: hidden=%d layers=%d heads=%d ff=%d max_pos=%d vocab=%d eps=%g feat=%d",
                hp_.hidden_size, hp_.num_layers, hp_.num_heads,
                hp_.intermediate_size, hp_.max_position, hp_.vocab_size,
                hp_.layer_norm_eps, hp_.feature_layers);

    if (!tok_.LoadFromGGUF(gguf_ctx_)) return false;

    backend_ = ggml_backend_cpu_init();
    if (!backend_) { RS_LOG_ERR("BERT: failed to init CPU backend"); return false; }

    weight_buf_ = ggml_backend_alloc_ctx_tensors(weight_ctx_, backend_);
    if (!weight_buf_) { RS_LOG_ERR("BERT: failed to alloc weight buffer"); return false; }

    // Read tensor data from the file.
    FILE *f = fopen(path, "rb");
    if (!f) { RS_LOG_ERR("BERT: failed to open '%s' for tensor data", path); return false; }
    size_t data_offset = gguf_get_data_offset(gguf_ctx_);
    int n_tensors = (int)gguf_get_n_tensors(gguf_ctx_);
    std::vector<char> read_buf;
    for (int i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(gguf_ctx_, i);
        ggml_tensor *t = ggml_get_tensor(weight_ctx_, name);
        if (!t) continue;
        size_t t_offset = gguf_get_tensor_offset(gguf_ctx_, i);
        size_t t_size = ggml_nbytes(t);
        if (t_size == 0) continue;
        if (read_buf.size() < t_size) read_buf.resize(t_size);
        fseek(f, (long)(data_offset + t_offset), SEEK_SET);
        if (fread(read_buf.data(), 1, t_size, f) != t_size) {
            RS_LOG_ERR("BERT: failed to read tensor '%s'", name);
            fclose(f); return false;
        }
        ggml_backend_tensor_set(t, read_buf.data(), 0, t_size);
    }
    fclose(f);

    if (!MapTensors()) return false;

    ggml_backend_t b = backend_;
    sched_ = ggml_backend_sched_new(&b, nullptr, 1, 32768, false, false);
    if (!sched_) { RS_LOG_ERR("BERT: failed to create scheduler"); return false; }

    return true;
}

// LayerNorm on [H, T] tensor over the H axis. Returns a new tensor.
static ggml_tensor *bert_layer_norm(ggml_context *ctx, ggml_tensor *x,
                                    ggml_tensor *w, ggml_tensor *b, float eps) {
    ggml_tensor *y = ggml_norm(ctx, x, eps);
    if (w) y = ggml_mul(ctx, y, w);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// One BERT encoder layer (Post-LN, same as HuBERT but without layer dropout).
//   x = LN_attn(x + MHA(x))
//   x = LN_ffn (x + FFN(x))
// x layout: [H, T] (ne0=H, ne1=T).
static ggml_tensor *bert_layer_graph(ggml_context *ctx, ggml_tensor *x,
                                     const BertLayerWeights &l,
                                     int n_head, int head_dim, float eps) {
    int H = (int)x->ne[0];
    int T = (int)x->ne[1];

    ggml_tensor *q = ggml_mul_mat(ctx, l.attn.q_w, x); if (l.attn.q_b) q = ggml_add(ctx, q, l.attn.q_b);
    ggml_tensor *k = ggml_mul_mat(ctx, l.attn.k_w, x); if (l.attn.k_b) k = ggml_add(ctx, k, l.attn.k_b);
    ggml_tensor *v = ggml_mul_mat(ctx, l.attn.v_w, x); if (l.attn.v_b) v = ggml_add(ctx, v, l.attn.v_b);

    q = ggml_reshape_3d(ctx, q, head_dim, n_head, T);
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [head_dim, T, n_head]
    k = ggml_reshape_3d(ctx, k, head_dim, n_head, T);
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_reshape_3d(ctx, v, head_dim, n_head, T);
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [T, head_dim, n_head]

    ggml_tensor *scores = ggml_mul_mat(ctx, k, q);  // [T, T, n_head]
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)head_dim));
    scores = ggml_soft_max(ctx, scores);

    ggml_tensor *attn_out = ggml_mul_mat(ctx, v, scores);  // [head_dim, T, n_head]
    // Combine heads: ne=[HD, T_q, NH] -> [HD, NH, T_q] so reshape_2d gives PyTorch
    // memory order h = hd + nh*HD (HD innermost inside the combined H dim).
    attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));  // [head_dim, n_head, T]
    attn_out = ggml_reshape_2d(ctx, attn_out, H, T);

    attn_out = ggml_mul_mat(ctx, l.attn.o_w, attn_out);
    if (l.attn.o_b) attn_out = ggml_add(ctx, attn_out, l.attn.o_b);

    x = ggml_add(ctx, x, attn_out);
    x = bert_layer_norm(ctx, x, l.ln_attn_w, l.ln_attn_b, eps);

    ggml_tensor *ff = ggml_mul_mat(ctx, l.ffn.w1_w, x);
    if (l.ffn.w1_b) ff = ggml_add(ctx, ff, l.ffn.w1_b);
    ff = ggml_gelu_erf(ctx, ff);
    ff = ggml_mul_mat(ctx, l.ffn.w2_w, ff);
    if (l.ffn.w2_b) ff = ggml_add(ctx, ff, l.ffn.w2_b);

    x = ggml_add(ctx, x, ff);
    x = bert_layer_norm(ctx, x, l.ln_ffn_w, l.ln_ffn_b, eps);
    return x;
}

std::vector<float> BertModel::Encode(const std::string &text,
                                     std::vector<int> *out_subword_ids,
                                     std::vector<int> *out_word_boundaries) {
    std::vector<int> subword_ids = tok_.Tokenize(text, -1, out_word_boundaries);
    if (out_subword_ids) *out_subword_ids = subword_ids;

    int T = (int)subword_ids.size();
    int H = hp_.hidden_size;
    if (T <= 0 || H <= 0) return {};
    if (T > hp_.max_position) {
        RS_LOG_WARN("BERT: input length %d > max_position %d; truncating", T, hp_.max_position);
        T = hp_.max_position;
        subword_ids.resize(T);
        if (out_subword_ids) out_subword_ids->resize(T);
        if (out_word_boundaries) out_word_boundaries->resize(T);
    }

    // Build graph in a fresh context. Size grows with n_layers; 8 MiB is plenty
    // for typical short TTS sentences.
    size_t mem_size = 16 * 1024 * 1024;
    std::vector<uint8_t> mem_buf(mem_size);
    struct ggml_init_params gip = { mem_size, mem_buf.data(), true };
    ggml_context *ctx0 = ggml_init(gip);
    if (!ctx0) { RS_LOG_ERR("BERT: ggml_init failed"); return {}; }

    ggml_cgraph *gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor *ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(ids_t);
    ggml_set_name(ids_t, "input_ids");

    ggml_tensor *pos_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_input(pos_t);
    ggml_set_name(pos_t, "position_ids");

    ggml_tensor *tt_t = nullptr;
    if (emb_.token_type_emb) {
        tt_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_input(tt_t);
        ggml_set_name(tt_t, "token_type_ids");
    }

    // word + position + token_type embeddings -> [H, T]
    ggml_tensor *x = ggml_get_rows(ctx0, emb_.word_emb, ids_t);
    ggml_tensor *pe = ggml_get_rows(ctx0, emb_.position_emb, pos_t);
    x = ggml_add(ctx0, x, pe);
    if (tt_t) {
        ggml_tensor *te = ggml_get_rows(ctx0, emb_.token_type_emb, tt_t);
        x = ggml_add(ctx0, x, te);
    }
    // get_rows from a [H, N] table indexed by [T] gives [H, T] — good.
    x = bert_layer_norm(ctx0, x, emb_.ln_w, emb_.ln_b, hp_.layer_norm_eps);

    // Debug: optionally short-circuit and return the embedding output.
    ggml_tensor *debug_emb_out = x;
    ggml_set_name(debug_emb_out, "emb_after_ln");
    if (getenv("RS_BERT_DEBUG_EMB")) {
        ggml_set_output(debug_emb_out);
    }

    // Run encoder layers; collect last `feature_layers` outputs for mean pooling.
    int n_feat = std::max(1, hp_.feature_layers);
    std::vector<ggml_tensor *> hidden_states;
    hidden_states.reserve(hp_.num_layers);

    int head_dim = hp_.head_dim();
    float eps_override = hp_.layer_norm_eps;
    if (const char *e = getenv("RS_BERT_EPS")) eps_override = (float)atof(e);
    int dump_layer = -1;
    if (const char *e = getenv("RS_BERT_DEBUG_LAYER")) dump_layer = atoi(e);
    bool dump_q0 = getenv("RS_BERT_DEBUG_Q0") != nullptr;
    ggml_tensor *layer_dump = nullptr;
    ggml_tensor *q0_dump = nullptr;
    if (dump_q0) {
        // Compute Q for layer 0 only.
        const auto &l0 = layers_[0];
        ggml_tensor *q = ggml_mul_mat(ctx0, l0.attn.q_w, x);
        if (l0.attn.q_b) q = ggml_add(ctx0, q, l0.attn.q_b);
        q0_dump = q;
        ggml_set_name(q0_dump, "q0_dump");
    }
    for (int i = 0; i < hp_.num_layers; ++i) {
        x = bert_layer_graph(ctx0, x, layers_[i], hp_.num_heads, head_dim, eps_override);
        if (i == dump_layer) { layer_dump = x; ggml_set_name(x, "layer_dump"); }
        hidden_states.push_back(x);
    }

    // Mean of last N hidden states.
    int start = std::max(0, (int)hidden_states.size() - n_feat);
    ggml_tensor *acc = hidden_states[start];
    for (int i = start + 1; i < (int)hidden_states.size(); ++i) {
        acc = ggml_add(ctx0, acc, hidden_states[i]);
    }
    int used = (int)hidden_states.size() - start;
    ggml_tensor *out = ggml_scale(ctx0, acc, 1.0f / (float)used);
    ggml_set_output(out);
    ggml_set_name(out, "bert_features");

    if (getenv("RS_BERT_DEBUG_EMB")) {
        // Return the embedding-stage output instead of the encoder mean,
        // so we can diff against HF model.embeddings(input_ids).
        out = debug_emb_out;
    }
    if (layer_dump) {
        out = layer_dump;
    }
    if (q0_dump) {
        out = q0_dump;
    }

    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        RS_LOG_ERR("BERT: sched_alloc_graph failed");
        ggml_free(ctx0);
        return {};
    }

    // Upload inputs.
    std::vector<int32_t> ids32(T), pos32(T), tt32(T, 0);
    for (int i = 0; i < T; ++i) {
        ids32[i] = subword_ids[i];
        pos32[i] = i;
    }
    ggml_backend_tensor_set(ids_t, ids32.data(), 0, T * sizeof(int32_t));
    ggml_backend_tensor_set(pos_t, pos32.data(), 0, T * sizeof(int32_t));
    if (tt_t) ggml_backend_tensor_set(tt_t, tt32.data(), 0, T * sizeof(int32_t));

    ggml_backend_sched_graph_compute(sched_, gf);

    // out has ggml shape [H, T] (ne0=H, ne1=T). Its raw memory layout is
    // numpy[T, H].tobytes(), i.e. row-major [T, H] — exactly what the
    // OpenVoice2 graph wants as `bert_in[T, dim]`. Copy through directly.
    std::vector<float> result((size_t)T * H);
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_free(ctx0);
    return result;
}

}  // namespace rapidspeech
