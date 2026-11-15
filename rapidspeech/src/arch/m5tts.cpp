// M5-TTS model: loading, llama3 byte-level BPE, state management, LM frame
// generation orchestration, arch registration.
// LM graphs live in m5tts_lm.cpp; codec graphs in m5tts_codec.cpp.

#include "arch/m5tts.h"
#include "arch/m5tts_graph.h"

#include "frontend/wetext_normalizer.h"
#include "utils/rs_data_paths.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace m5tts {

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

static int kv_i32(gguf_context *g, const char *key, int def) {
  const int i = gguf_find_key(g, key);
  if (i < 0) return def;
  return gguf_get_val_i32(g, i);
}

static float kv_f32(gguf_context *g, const char *key, float def) {
  const int i = gguf_find_key(g, key);
  if (i < 0) return def;
  return (float)gguf_get_val_f32(g, i);
}

static std::vector<int> kv_i32_arr(gguf_context *g, const char *key) {
  std::vector<int> out;
  const int i = gguf_find_key(g, key);
  if (i < 0) return out;
  const int n = (int)gguf_get_arr_n(g, i);
  const int32_t *data = (const int32_t *)gguf_get_arr_data(g, i);
  out.assign(data, data + n);
  return out;
}

static ggml_tensor *graph_tensor(ggml_cgraph *gf, const char *name) {
  ggml_tensor *t = ggml_graph_get_tensor(gf, name);
  if (!t) RS_LOG_ERR("M5TTS: missing graph tensor %s", name);
  return t;
}

bool Model::SetInputF32(ggml_cgraph *gf, const char *name, const float *data,
                        size_t n_elem) {
  ggml_tensor *t = graph_tensor(gf, name);
  if (!t) return false;
  ggml_backend_tensor_set(t, data, 0, n_elem * sizeof(float));
  return true;
}

bool Model::SetInputI32(ggml_cgraph *gf, const char *name, const int32_t *data,
                        size_t n_elem) {
  ggml_tensor *t = graph_tensor(gf, name);
  if (!t) return false;
  ggml_backend_tensor_set(t, data, 0, n_elem * sizeof(int32_t));
  return true;
}

bool Model::AllocGraph(ggml_backend_sched_t sched, ggml_cgraph *gf,
                       const char *what) {
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("M5TTS: %s graph alloc failed", what);
    return false;
  }
  return true;
}

bool Model::ComputeGraph(ggml_backend_sched_t sched, ggml_cgraph *gf,
                         const char *what) {
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("M5TTS: %s graph compute failed", what);
    ggml_backend_sched_reset(sched);
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// llama3 byte-level BPE
// ---------------------------------------------------------------------------

void BPETokenizer::init_byte_encoder() {
  int n = 0;
  for (int b = 0; b < 256; b++) {
    bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                     (b >= 0xAE && b <= 0xFF);
    // GPT-2 byte encoder: every byte maps to a Unicode codepoint stored as
    // UTF-8 (printable bytes b >= 0x80 encode as 2 bytes; non-printable
    // bytes map to codepoint 256+n).
    uint32_t cp;
    if (printable) {
      cp = (uint32_t)b;
    } else {
      cp = 256 + (uint32_t)n;
      n++;
    }
    if (cp < 0x80) {
      byte2str[b] = std::string(1, (char)cp);
    } else {
      byte2str[b] = std::string(1, (char)(0xC0 | (cp >> 6))) +
                    std::string(1, (char)(0x80 | (cp & 0x3F)));
    }
  }
}

bool BPETokenizer::load_from_gguf(struct gguf_context *g) {
  const int tok_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
  if (tok_idx < 0) {
    RS_LOG_ERR("M5TTS: GGUF missing tokenizer.ggml.tokens");
    return false;
  }
  const int n_tokens = (int)gguf_get_arr_n(g, tok_idx);
  id_to_token.resize(n_tokens);
  for (int i = 0; i < n_tokens; i++) {
    id_to_token[i] = gguf_get_arr_str(g, tok_idx, i);
    token_to_id[id_to_token[i]] = i;
  }
  const int mrg_idx = gguf_find_key(g, "tokenizer.ggml.merges");
  if (mrg_idx < 0) {
    RS_LOG_ERR("M5TTS: GGUF missing tokenizer.ggml.merges");
    return false;
  }
  const int n_merges = (int)gguf_get_arr_n(g, mrg_idx);
  for (int i = 0; i < n_merges; i++) {
    // converter joins pairs with \x01 so tokens containing spaces stay intact
    merges[std::string(gguf_get_arr_str(g, mrg_idx, i))] = i;
  }
  init_byte_encoder();
  loaded = true;
  RS_LOG_INFO("M5TTS: tokenizer loaded (%d tokens, %d merges)", n_tokens,
              n_merges);
  if (const char *dbg = std::getenv("RS_M5TTS_DEBUG_BPE")) {
    // dump first 2 merges and probe a CJK pair
    for (int i = 0; i < 2 && i < n_merges; i++) {
      const std::string &m = id_to_token.empty() ? std::string()
                                                 : std::string();
      (void)m;
    }
    auto probe = [&](const char *a, const char *b) {
      std::string key = std::string(a) + "\x01" + b;
      auto it = merges.find(key);
      fprintf(stderr, "BPE probe '%s'+'%s': %s\n", a, b,
              it == merges.end() ? "MISS" : "HIT");
    };
    fprintf(stderr, "BPE merge[0] bytes: ");
    for (unsigned char ch : std::string(gguf_get_arr_str(g, mrg_idx, 0)))
      fprintf(stderr, "%02x ", ch);
    fprintf(stderr, "\nBPE merge[1] bytes: ");
    for (unsigned char ch : std::string(gguf_get_arr_str(g, mrg_idx, 1)))
      fprintf(stderr, "%02x ", ch);
    fprintf(stderr, "\n");
    // 猪 bytes: 0xe7 -> 'ç'(U+00E7), 0x8c -> U+013C?, 0xaa -> U+00AA
    // byte 0x8c=140 nonprintable n=67 -> cp=323 (U+0143 'Ń')? verify table
    {
      unsigned char b140 = 0x8c;
      fprintf(stderr, "BPE byte2str[0x8c]: ");
      for (unsigned char ch : byte2str[b140]) fprintf(stderr, "%02x ", ch);
      fprintf(stderr, "\nBPE byte2str[0xe7]: ");
      for (unsigned char ch : byte2str[0xe7]) fprintf(stderr, "%02x ", ch);
      fprintf(stderr, "\n");
    }
    probe("\xc3\xa7", "\xc4\xae");  // 'ç' 'Į'
  }
  return true;
}

// UTF-8 decode one codepoint, returns byte length.
static int utf8_dec(const char *s, int avail, uint32_t *cp) {
  if (avail <= 0) return 0;
  const unsigned char c = (unsigned char)s[0];
  if (c < 0x80) { *cp = c; return 1; }
  if ((c & 0xE0) == 0xC0 && avail >= 2) {
    *cp = ((uint32_t)(c & 0x1F) << 6) | ((uint32_t)s[1] & 0x3F);
    return 2;
  }
  if ((c & 0xF0) == 0xE0 && avail >= 3) {
    *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) |
          ((uint32_t)s[2] & 0x3F);
    return 3;
  }
  if ((c & 0xF8) == 0xF0 && avail >= 4) {
    *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
          ((uint32_t)(s[2] & 0x3F) << 6) | ((uint32_t)s[3] & 0x3F);
    return 4;
  }
  *cp = c;
  return 1;
}

// Approximate \p{L} (letters) and \p{N} (digits) classifiers covering the
// blocks relevant for zh/en/ja/ko/de/fr plus common punctuation.
static bool cp_is_letter(uint32_t cp) {
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
  if (cp < 0x7F) return false;
  if (cp >= 0xC0 && cp <= 0x24F && cp != 0xD7 && cp != 0xF7) return true;
  if (cp >= 0x370 && cp <= 0x3FF && cp != 0x37E) return true;   // Greek
  if (cp >= 0x400 && cp <= 0x52F) return true;                  // Cyrillic
  if (cp >= 0x590 && cp <= 0x6FF) return true;                  // Hebr/Arab
  if (cp >= 0x900 && cp <= 0x97F) return true;                  // Devanagari
  if (cp >= 0x3040 && cp <= 0x30FF) return true;                // kana
  if (cp >= 0x3400 && cp <= 0x4DBF) return true;                // CJK ext A
  if (cp >= 0x4E00 && cp <= 0x9FFF) return true;                // CJK
  if (cp >= 0xA960 && cp <= 0xA97F) return true;
  if (cp >= 0xAC00 && cp <= 0xD7AF) return true;                // Hangul
  if (cp >= 0xF900 && cp <= 0xFAFF) return true;
  if (cp >= 0x20000 && cp <= 0x2FA1F) return true;
  return false;
}

static bool cp_is_digit(uint32_t cp) {
  if (cp >= '0' && cp <= '9') return true;
  if (cp >= 0xFF10 && cp <= 0xFF19) return true;  // fullwidth digits
  return false;
}

static bool cp_is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }

static bool cp_is_space(uint32_t cp) {
  if (cp == ' ' || cp == '\t' || cp_is_newline(cp)) return true;
  if (cp == 0x3000) return true;  // ideographic space
  return false;
}

static void pretokenize(const std::string &text,
                         std::vector<std::string> &out) {
  const char *s = text.data();
  const int n = (int)text.size();
  int i = 0;
  while (i < n) {
    uint32_t cp;
    int cl = utf8_dec(s + i, n - i, &cp);
    if (cl == 0) break;

    // 1. contractions ('s|'t|'re|'ve|'m|'ll|'d), case-insensitive
    if (cp == '\'' && i + 1 < n) {
      static const char *cons[] = {"s",  "t", "re", "ve", "m", "ll", "d"};
      bool matched = false;
      for (const char *c : cons) {
        int len = (int)strlen(c);
        if (i + 1 + len > n) continue;
        bool ok = true;
        for (int k = 0; k < len; k++) {
          char a = s[i + 1 + k], b = c[k];
          if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
          if (a != b) { ok = false; break; }
        }
        if (ok) {
          out.emplace_back(s + i, 1 + len);
          i += 1 + len;
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }

    // 5. \s*[\r\n]+
    if (cp_is_space(cp)) {
      int j = i;
      bool saw_nl = false;
      while (j < n) {
        uint32_t c2;
        int l2 = utf8_dec(s + j, n - j, &c2);
        if (!cp_is_space(c2)) break;
        if (cp_is_newline(c2)) saw_nl = true;
        j += l2;
      }
      if (saw_nl) {
        out.emplace_back(s + i, j - i);
        i = j;
        continue;
      }
      // 6/7. pure whitespace run; one trailing space may glue to the next
      // token (llama3 " ?…" patterns)
      int end = j;
      int keep = 0;
      if (end < n && s[end - 1] == ' ' && i + 1 < end) keep = 1;
      if (end - keep > i) out.emplace_back(s + i, end - keep - i);
      i = end - keep;
      if (i < n) continue;
      break;
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    if (cp_is_letter(cp) || (!cp_is_digit(cp) && !cp_is_newline(cp))) {
      if (!cp_is_letter(cp)) {
        uint32_t nxt;
        int nl = utf8_dec(s + i + cl, n - i - cl, &nxt);
        if (nl > 0 && cp_is_letter(nxt)) {
          int j = i + cl;
          while (j < n) {
            uint32_t c2;
            int l2 = utf8_dec(s + j, n - j, &c2);
            if (!cp_is_letter(c2)) break;
            j += l2;
          }
          out.emplace_back(s + i, j - i);
          i = j;
          continue;
        }
      }
      if (cp_is_letter(cp)) {
        int j = i;
        while (j < n) {
          uint32_t c2;
          int l2 = utf8_dec(s + j, n - j, &c2);
          if (!cp_is_letter(c2)) break;
          j += l2;
        }
        out.emplace_back(s + i, j - i);
        i = j;
        continue;
      }
    }

    // 3. \p{N}{1,3}
    if (cp_is_digit(cp)) {
      int j = i, cnt = 0;
      while (j < n && cnt < 3) {
        uint32_t c2;
        int l2 = utf8_dec(s + j, n - j, &c2);
        if (!cp_is_digit(c2)) break;
        j += l2;
        cnt++;
      }
      out.emplace_back(s + i, j - i);
      i = j;
      continue;
    }

    // 4. ?[^\s\p{L}\p{N}]+[\r\n]*
    if (!cp_is_space(cp) && !cp_is_letter(cp) && !cp_is_digit(cp)) {
      int j = i;
      while (j < n) {
        uint32_t c2;
        int l2 = utf8_dec(s + j, n - j, &c2);
        if (cp_is_space(c2) || cp_is_letter(c2) || cp_is_digit(c2)) break;
        j += l2;
      }
      while (j < n) {
        uint32_t c2;
        int l2 = utf8_dec(s + j, n - j, &c2);
        if (!cp_is_newline(c2)) break;
        j += l2;
      }
      out.emplace_back(s + i, j - i);
      i = j;
      continue;
    }

    out.emplace_back(s + i, cl);
    i += cl;
  }
}

static std::vector<int32_t> bpe_merge_chunk(
    const BPETokenizer &t, const std::string &byte_encoded) {
  std::vector<std::string> parts;
  const char *s = byte_encoded.data();
  const int n = (int)byte_encoded.size();
  int i = 0;
  while (i < n) {
    int cl = 1;
    const unsigned char ch = (unsigned char)s[i];
    if (ch >= 0xC0 && ch < 0xE0) cl = 2;
    else if (ch >= 0xE0 && ch < 0xF0) cl = 3;
    else if (ch >= 0xF0) cl = 4;
    parts.emplace_back(s + i, std::min(cl, n - i));
    i += cl;
  }
  if (std::getenv("RS_M5TTS_DEBUG_BPE") && parts.size() >= 2) {
    fprintf(stderr, "BPE merge start: %zu parts:", parts.size());
    for (size_t i = 0; i < std::min(parts.size(), (size_t)6); i++)
      fprintf(stderr, " [%s]", parts[i].c_str());
    auto it0 = t.merges.find(parts[0] + "\x01" + parts[1]);
    fprintf(stderr, " | lookup[0+1]: %s\n",
            it0 == t.merges.end() ? "MISS" : "HIT");
  }
  while (parts.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best_i = SIZE_MAX;
    for (size_t k = 0; k + 1 < parts.size(); k++) {
      auto it = t.merges.find(parts[k] + "\x01" + parts[k + 1]);
      if (it != t.merges.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = k;
      }
    }
    if (best_i == SIZE_MAX) break;
    parts[best_i] = parts[best_i] + parts[best_i + 1];
    parts.erase(parts.begin() + best_i + 1);
  }
  if (std::getenv("RS_M5TTS_DEBUG_BPE") && parts.size() >= 2) {
    fprintf(stderr, "BPE merged: %zu parts:", parts.size());
    for (size_t i = 0; i < std::min(parts.size(), (size_t)6); i++)
      fprintf(stderr, " [%s]", parts[i].c_str());
    fprintf(stderr, "\n");
  }
  std::vector<int32_t> ids;
  for (auto &p : parts) {
    auto it = t.token_to_id.find(p);
    if (it != t.token_to_id.end()) ids.push_back(it->second);
    else if (std::getenv("RS_M5TTS_DEBUG_BPE"))
      fprintf(stderr, "BPE dropped piece: [%s]\n", p.c_str());
  }
  return ids;
}

static void byte_encode(const std::string &tk, const BPETokenizer &t,
                        std::string &enc) {
  for (unsigned char c : tk) enc += t.byte2str[c];
}

static void encode_piece(const std::string &chunk, const BPETokenizer &t,
                         std::vector<int32_t> &ids) {
  std::vector<std::string> toks;
  pretokenize(chunk, toks);
  if (std::getenv("RS_M5TTS_DEBUG_BPE")) {
    fprintf(stderr, "BPE piece '%.30s' -> %zu chunks:", chunk.c_str(),
            toks.size());
    for (size_t i = 0; i < std::min(toks.size(), (size_t)6); i++)
      fprintf(stderr, " [%s]", toks[i].c_str());
    fprintf(stderr, "\n");
  }
  for (auto &tk : toks) {
    std::string enc;
    byte_encode(tk, t, enc);
    auto part = bpe_merge_chunk(t, enc);
    ids.insert(ids.end(), part.begin(), part.end());
  }
}

std::vector<int32_t> BPETokenizer::encode_raw(const std::string &text) const {
  std::vector<int32_t> ids;
  static const char *specials[] = {
      "<|caption_begin|>", "<|caption_end|>", "<|lang_begin|>", "<|lang_end|>",
      "<|accent_begin|>", "<|accent_end|>", "<|speaker_start|>",
      "<|speaker_end|>", "<|begin_of_text|>", "<|end_of_text|>"};
  size_t pos = 0;
  while (pos < text.size()) {
    size_t best_pos = std::string::npos;
    size_t best_len = 0;
    for (const char *sp : specials) {
      size_t p = text.find(sp, pos);
      if (p != std::string::npos &&
          (best_pos == std::string::npos || p < best_pos)) {
        best_pos = p;
        best_len = strlen(sp);
      }
    }
    if (best_pos == std::string::npos) {
      encode_piece(text.substr(pos), *this, ids);
      break;
    }
    if (best_pos > pos) encode_piece(text.substr(pos, best_pos - pos), *this, ids);
    auto it = token_to_id.find(text.substr(best_pos, best_len));
    if (it != token_to_id.end()) ids.push_back(it->second);
    pos = best_pos + best_len;
  }
  return ids;
}

std::vector<int32_t> BPETokenizer::encode(const std::string &text) const {
  std::vector<int32_t> ids;
  ids.push_back(bos_id);
  auto raw = encode_raw(text);
  ids.insert(ids.end(), raw.begin(), raw.end());
  ids.push_back(eos_id);
  return ids;
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

Model::Model() = default;

Model::~Model() {
  if (buf_kv_) ggml_backend_buffer_free(buf_kv_);
  if (ctx_kv_) ggml_free(ctx_kv_);
}

bool Model::LoadHParams(struct gguf_context *g) {
  hp_.bb_layers = kv_i32(g, "m5tts.backbone_layers", 8);
  hp_.bb_heads = kv_i32(g, "m5tts.backbone_heads", 8);
  hp_.bb_kv_heads = kv_i32(g, "m5tts.backbone_kv_heads", 2);
  hp_.bb_head_dim = kv_i32(g, "m5tts.backbone_head_dim", 64);
  hp_.bb_ff = kv_i32(g, "m5tts.backbone_ff", 2048);
  hp_.dc_layers = kv_i32(g, "m5tts.decoder_layers", 5);
  hp_.dc_heads = kv_i32(g, "m5tts.decoder_heads", 8);
  hp_.dc_kv_heads = kv_i32(g, "m5tts.decoder_kv_heads", 2);
  hp_.dc_head_dim = kv_i32(g, "m5tts.decoder_head_dim", 64);
  hp_.dc_ff = kv_i32(g, "m5tts.decoder_ff", 2048);
  hp_.hidden = kv_i32(g, "m5tts.hidden_size", 512);
  hp_.text_vocab = kv_i32(g, "m5tts.text_vocab", 129992);
  hp_.audio_vocab = kv_i32(g, "m5tts.audio_vocab", 2051);
  hp_.codebooks = kv_i32(g, "m5tts.codebooks", 16);
  hp_.max_seq_len = kv_i32(g, "m5tts.max_seq_len", 2048);
  hp_.norm_eps = kv_f32(g, "m5tts.norm_eps", 1e-5f);
  hp_.temperature = kv_f32(g, "m5tts.temperature", 0.9f);
  hp_.topk = kv_i32(g, "m5tts.topk", 10);
  hp_.min_frames = kv_i32(g, "m5tts.min_frames", 6);
  hp_.max_frames = kv_i32(g, "m5tts.max_frames", 500);
  hp_.sample_rate = kv_i32(g, "m5tts.sample_rate", 24000);
  hp_.hop_length = kv_i32(g, "m5tts.hop_length", 1920);
  hp_.bos_id = kv_i32(g, "m5tts.bos_token_id", 128000);
  hp_.eos_id = kv_i32(g, "m5tts.eos_token_id", 128001);
  static const char *sp_keys[8] = {
      "m5tts.special.caption_begin", "m5tts.special.caption_end",
      "m5tts.special.lang_begin", "m5tts.special.lang_end",
      "m5tts.special.accent_begin", "m5tts.special.accent_end",
      "m5tts.special.speaker_start", "m5tts.special.speaker_end"};
  for (int i = 0; i < 8; i++)
    hp_.special_ids[i] = kv_i32(g, sp_keys[i], 129984 + i);

  hp_.cd_layers = kv_i32(g, "codec.layers", 8);
  hp_.cd_heads = kv_i32(g, "codec.heads", 16);
  hp_.cd_kv_heads = kv_i32(g, "codec.kv_heads", 16);
  hp_.cd_head_dim = kv_i32(g, "codec.head_dim", 64);
  hp_.cd_hidden = kv_i32(g, "codec.hidden_size", 512);
  hp_.cd_ff = kv_i32(g, "codec.ff_dim", 1024);
  hp_.cd_latent = kv_i32(g, "codec.latent_dim", 1024);
  hp_.cd_dec_dim = kv_i32(g, "codec.decoder_dim", 1536);
  hp_.cd_cb_dim = kv_i32(g, "codec.codebook_dim", 512);
  hp_.cd_cb_size = kv_i32(g, "codec.codebook_size", 2048);
  hp_.cd_sem_size = kv_i32(g, "codec.semantic_codebook_size", 4096);
  hp_.cd_window = kv_i32(g, "codec.window", 72);
  hp_.cd_eps = kv_f32(g, "codec.norm_eps", 1e-5f);
  hp_.cd_theta = kv_f32(g, "codec.rope_theta", 10000.0f);
  hp_.upsample_rates = kv_i32_arr(g, "codec.upsample_rates");
  hp_.upsampling_ratios = kv_i32_arr(g, "codec.upsampling_ratios");

  hp_.ce_layers = kv_i32(g, "codec.enc_layers", 8);
  hp_.ce_heads = kv_i32(g, "codec.enc_heads", 8);
  hp_.ce_kv_heads = kv_i32(g, "codec.enc_kv_heads", 8);
  hp_.ce_head_dim = kv_i32(g, "codec.enc_head_dim", 64);
  hp_.ce_hidden = kv_i32(g, "codec.enc_hidden_size", 512);
  hp_.ce_ff = kv_i32(g, "codec.enc_ff_dim", 2048);
  hp_.ce_window = kv_i32(g, "codec.enc_window", 250);
  hp_.ce_filters = kv_i32(g, "codec.enc_num_filters", 64);
  hp_.ce_compress = kv_i32(g, "codec.enc_compress", 2);
  hp_.ce_valid_q = kv_i32(g, "codec.enc_valid_quantizers", 16);
  hp_.ce_eps = kv_f32(g, "codec.enc_norm_eps", 1e-5f);
  hp_.ce_theta = kv_f32(g, "codec.enc_rope_theta", 10000.0f);
  hp_.enc_ratios = kv_i32_arr(g, "codec.enc_upsampling_ratios");
  if (hp_.enc_ratios.empty()) hp_.enc_ratios = {8, 6, 5, 4};
  // forward-order downsample strides are the reverse of the ratio list
  std::reverse(hp_.enc_ratios.begin(), hp_.enc_ratios.end());

  bpe_.bos_id = hp_.bos_id;
  bpe_.eos_id = hp_.eos_id;
  return true;
}

bool Model::Load(const std::unique_ptr<rs_context_t> &ctx,
                 ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("M5TTS: invalid context in Load");
    return false;
  }
  backend_ = backend;
  backends_ = ctx->backends;
  gguf_context *g = ctx->ctx_gguf;

  LoadHParams(g);
  meta_.arch_name = "m5-tts";
  meta_.audio_sample_rate = hp_.sample_rate;
  meta_.n_mels = 0;
  meta_.vocab_size = hp_.text_vocab;
  meta_.use_external_frontend = true;  // codec pipeline consumes raw PCM

  if (!bpe_.load_from_gguf(g)) return false;

  if (!MapTensors(ctx->gguf_data, g)) return false;
  if (!BuildCaches(backend)) return false;
  BuildNormalizedCodebooks(ctx->sched);

  // lazy WeText zh normalizer (best-effort; pass-through when data missing)
  RS_LOG_INFO("M5TTS: model loaded (backbone %dL/%dH, MTP %dL, codec %dL)",
              hp_.bb_layers, hp_.bb_heads, hp_.dc_layers, hp_.cd_layers);
  return true;
}

std::shared_ptr<RSState> Model::CreateState() {
  return std::make_shared<State>();
}

// ---------------------------------------------------------------------------
// tensor mapping
// ---------------------------------------------------------------------------

static ggml_tensor *get_t(std::map<std::string, ggml_tensor *> &tensors,
                          const std::string &name) {
  auto it = tensors.find(name);
  if (it == tensors.end())
    throw std::runtime_error("M5TTS: missing tensor " + name);
  return it->second;
}

bool Model::MapTensors(struct ggml_context *gguf_data, gguf_context *g) {
  (void)gguf_data;
  std::map<std::string, ggml_tensor *> tensors;
  const int n = (int)gguf_get_n_tensors(g);
  for (int i = 0; i < n; i++) {
    const char *name = gguf_get_tensor_name(g, i);
    ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }

  try {
    text_emb_ = get_t(tensors, "text_embeddings.weight");
    audio_emb_ = get_t(tensors, "audio_embeddings.weight");
    projection_w_ = get_t(tensors, "projection.weight");
    codebook0_head_ = get_t(tensors, "codebook0_head.weight");
    audio_head_ = get_t(tensors, "audio_head");
    bb_inv_freq_ = get_t(tensors, "backbone.inv_freq");
    dc_inv_freq_ = get_t(tensors, "decoder.inv_freq");
    cd_inv_freq_ = get_t(tensors, "codec.inv_freq");
    ce_inv_freq_ = get_t(tensors, "codec.encoder.inv_freq");

    auto load_llama = [&](const char *prefix, int n_layers,
                          std::vector<LlamaBlock> &layers, bool codec_scales) {
      layers.resize(n_layers);
      for (int l = 0; l < n_layers; l++) {
        auto &b = layers[l];
        const std::string p =
            std::string(prefix) + ".layers." + std::to_string(l);
        b.ln1_w = get_t(tensors, p + ".input_layernorm.weight");
        b.q_w = get_t(tensors, p + ".self_attn.q_proj.weight");
        b.k_w = get_t(tensors, p + ".self_attn.k_proj.weight");
        b.v_w = get_t(tensors, p + ".self_attn.v_proj.weight");
        b.o_w = get_t(tensors, p + ".self_attn.o_proj.weight");
        b.ln2_w =
            get_t(tensors, p + ".post_attention_layernorm.weight");
        b.gate_w = get_t(tensors, p + ".mlp.gate_proj.weight");
        b.up_w = get_t(tensors, p + ".mlp.up_proj.weight");
        b.down_w = get_t(tensors, p + ".mlp.down_proj.weight");
        if (codec_scales) {
          b.attn_gamma = get_t(tensors, p + ".self_attn_layer_scale.scale");
          b.mlp_gamma = get_t(tensors, p + ".mlp_layer_scale.scale");
        }
      }
    };
    load_llama("backbone", hp_.bb_layers, bb_layers_, false);
    bb_norm_w_ = get_t(tensors, "backbone.norm.weight");
    load_llama("decoder", hp_.dc_layers, dc_layers_, false);
    dc_norm_w_ = get_t(tensors, "decoder.norm.weight");

    // ---- codec decoder transformer ----
    load_llama("codec.decoder.pre_transformer", hp_.cd_layers, cdw_.tf_layers,
               true);
    cdw_.tf_norm_w =
        get_t(tensors, "codec.decoder.pre_transformer.norm.weight");
    cdw_.tf_input_proj_w =
        get_t(tensors, "codec.decoder.pre_transformer.input_proj.weight");
    cdw_.tf_input_proj_b =
        get_t(tensors, "codec.decoder.pre_transformer.input_proj.bias");
    cdw_.tf_output_proj_w =
        get_t(tensors, "codec.decoder.pre_transformer.output_proj.weight");
    cdw_.tf_output_proj_b =
        get_t(tensors, "codec.decoder.pre_transformer.output_proj.bias");

    // ---- codec decoder conv stack ----
    auto has = [&](const std::string &name) {
      return tensors.count(name) != 0;
    };
    auto conv = [&](const std::string &name) {
      Conv1D cd;
      cd.w = get_t(tensors, name + ".weight");
      cd.b = has(name + ".bias") ? get_t(tensors, name + ".bias") : nullptr;
      cd.k = (int)cd.w->ne[0];
      cd.in_c = (int)cd.w->ne[1];
      cd.out_c = (int)cd.w->ne[2];
      cd.dw = cd.in_c == 1 && cd.out_c > 1;
      return cd;
    };
    auto snake = [&](const std::string &name) {
      SnakeAct a;
      a.alpha = get_t(tensors, name + ".alpha");
      a.beta = get_t(tensors, name + ".beta");
      return a;
    };
    cdw_.pre_conv = conv("codec.decoder.pre_conv.conv");
    const int n_up = (int)hp_.upsampling_ratios.size();
    cdw_.upstage.resize(n_up);
    for (int u = 0; u < n_up; u++) {
      auto &up = cdw_.upstage[u];
      const std::string p =
          "codec.decoder.upsample." + std::to_string(u);
      up.transconv = conv(p + ".0.conv");
      up.transconv.stride = hp_.upsampling_ratios[u];
      up.dwconv = conv(p + ".1.dwconv.conv");
      up.norm_w = get_t(tensors, p + ".1.norm.weight");
      up.norm_b = get_t(tensors, p + ".1.norm.bias");
      up.pw1_w = get_t(tensors, p + ".1.pwconv1.weight");
      up.pw1_b = get_t(tensors, p + ".1.pwconv1.bias");
      up.pw2_w = get_t(tensors, p + ".1.pwconv2.weight");
      up.pw2_b = get_t(tensors, p + ".1.pwconv2.bias");
      up.gamma = get_t(tensors, p + ".1.gamma");
    }
    cdw_.decoder_in = conv("codec.decoder.decoder.0.conv");
    static const int ru_dils[3] = {1, 3, 9};
    const int n_blk = (int)hp_.upsample_rates.size();
    cdw_.blocks.resize(n_blk);
    for (int b = 0; b < n_blk; b++) {
      auto &blk = cdw_.blocks[b];
      const std::string p =
          "codec.decoder.decoder." + std::to_string(b + 1);
      blk.act0 = snake(p + ".block.0");
      blk.transconv = conv(p + ".block.1.conv");
      blk.transconv.stride = hp_.upsample_rates[b];
      for (int ru = 0; ru < 3; ru++) {
        const std::string rp = p + ".block." + std::to_string(2 + ru);
        blk.ru[ru].act1 = snake(rp + ".act1");
        blk.ru[ru].act2 = snake(rp + ".act2");
        blk.ru[ru].conv1 = conv(rp + ".conv1.conv");
        blk.ru[ru].conv2 = conv(rp + ".conv2.conv");
        blk.ru[ru].conv1.dilation = ru_dils[ru];
      }
    }
    cdw_.act_final = snake("codec.decoder.decoder.5");
    cdw_.decoder_out = conv("codec.decoder.decoder.6.conv");

    // ---- codec decoder RVQ ----
    const std::string rfp = "codec.decoder.quantizer.rvq_first.";
    cdw_.sem_embed = get_t(tensors, rfp + "vq.layers.0.codebook.embedding_sum");
    rvq_usage_.push_back(
        get_t(tensors, rfp + "vq.layers.0.codebook.cluster_usage"));
    rvq_tables_.push_back(cdw_.sem_embed);
    cdw_.sem_out_proj_w = get_t(tensors, rfp + "output_proj.weight");
    const std::string rsp = "codec.decoder.quantizer.rvq_rest.";
    cdw_.acou_embed.clear();
    for (int l = 0; l < hp_.codebooks - 1; l++) {
      const std::string p = rsp + "vq.layers." + std::to_string(l) + ".codebook.";
      cdw_.acou_embed.push_back(get_t(tensors, p + "embedding_sum"));
      rvq_usage_.push_back(get_t(tensors, p + "cluster_usage"));
      rvq_tables_.push_back(cdw_.acou_embed.back());
    }
    cdw_.acou_out_proj_w = get_t(tensors, rsp + "output_proj.weight");

    // ---- codec encoder (SEANet + transformer + quantizer) ----
    cew_.stem = conv("codec.encoder.encoder.layers.0.conv");
    static const int res_layer[4] = {1, 4, 7, 10};
    static const int down_layer[4] = {3, 6, 9, 12};
    const int n_st = (int)hp_.enc_ratios.size();
    cew_.stages.resize(n_st);
    for (int s = 0; s < n_st; s++) {
      auto &st = cew_.stages[s];
      st.res1 = conv("codec.encoder.encoder.layers." +
                     std::to_string(res_layer[s]) + ".block.1.conv");
      st.res2 = conv("codec.encoder.encoder.layers." +
                     std::to_string(res_layer[s]) + ".block.3.conv");
      st.down = conv("codec.encoder.encoder.layers." +
                     std::to_string(down_layer[s]) + ".conv");
      st.down.stride = hp_.enc_ratios[s];
    }
    cew_.final_conv = conv("codec.encoder.encoder.layers.14.conv");
    cew_.downsample = conv("codec.encoder.downsample.conv");
    cew_.downsample.stride = hp_.ce_compress;
    cew_.tf_layers.resize(hp_.ce_layers);
    for (int l = 0; l < hp_.ce_layers; l++) {
      auto &b = cew_.tf_layers[l];
      const std::string p =
          "codec.encoder.encoder_transformer.layers." + std::to_string(l);
      b.ln1_w = get_t(tensors, p + ".input_layernorm.weight");
      b.ln1_b = get_t(tensors, p + ".input_layernorm.bias");
      b.q_w = get_t(tensors, p + ".self_attn.q_proj.weight");
      b.k_w = get_t(tensors, p + ".self_attn.k_proj.weight");
      b.v_w = get_t(tensors, p + ".self_attn.v_proj.weight");
      b.o_w = get_t(tensors, p + ".self_attn.o_proj.weight");
      b.ln2_w = get_t(tensors, p + ".post_attention_layernorm.weight");
      b.ln2_b = get_t(tensors, p + ".post_attention_layernorm.bias");
      b.fc1_w = get_t(tensors, p + ".mlp.fc1.weight");
      b.fc2_w = get_t(tensors, p + ".mlp.fc2.weight");
      b.attn_gamma = get_t(tensors, p + ".self_attn_layer_scale.scale");
      b.mlp_gamma = get_t(tensors, p + ".mlp_layer_scale.scale");
    }
    const std::string esq =
        "codec.encoder.quantizer.semantic_residual_vector_quantizer.";
    const std::string eaq =
        "codec.encoder.quantizer.acoustic_residual_vector_quantizer.";
    cew_.sem_in_proj_w = get_t(tensors, esq + "input_proj.weight");
    cew_.sem_out_proj_w = get_t(tensors, esq + "output_proj.weight");
    cew_.sem_embed = get_t(tensors, esq + "layers.0.codebook.embedding_sum");
    rvq_usage_.push_back(get_t(tensors, esq + "layers.0.codebook.cluster_usage"));
    rvq_tables_.push_back(cew_.sem_embed);
    cew_.acou_in_proj_w = get_t(tensors, eaq + "input_proj.weight");
    cew_.acou_out_proj_w = get_t(tensors, eaq + "output_proj.weight");
    cew_.acou_embed.clear();
    for (int l = 0; l < hp_.codebooks - 1; l++) {
      const std::string p = eaq + "layers." + std::to_string(l) + ".codebook.";
      cew_.acou_embed.push_back(get_t(tensors, p + "embedding_sum"));
      rvq_usage_.push_back(get_t(tensors, p + "cluster_usage"));
      rvq_tables_.push_back(cew_.acou_embed.back());
    }
  } catch (const std::exception &e) {
    RS_LOG_ERR("M5TTS: %s", e.what());
    return false;
  }
  return true;
}

bool Model::BuildCaches(ggml_backend_t backend) {
  const int hd = hp_.bb_head_dim;
  const int max_bb = hp_.max_seq_len;
  const int max_cd = 4096;  // codec decoder transformer context (frames)

  std::vector<ggml_tensor *> cache_tensors;
  auto add = [&](const char *name, int d0, int d1, int d2, int d3) {
    ggml_tensor *t = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, d0, d1, d2, d3);
    ggml_set_name(t, name);
    cache_tensors.push_back(t);
    return t;
  };

  // init KV context
  {
    ggml_init_params ip = {ggml_tensor_overhead() * 192, nullptr, true};
    ctx_kv_ = ggml_init(ip);
    if (!ctx_kv_) return false;
  }
  auto add_scalar = [&](const char *name) {
    ggml_tensor *t = ggml_new_tensor_1d(ctx_kv_, GGML_TYPE_F32, 1);
    ggml_set_name(t, name);
    cache_tensors.push_back(t);
    return t;
  };
  sc_eps5_ = add_scalar("sc_eps5");
  sc_eps9_ = add_scalar("sc_eps9");
  bb_kv_k_ = add("bb_kv_k", hd, max_bb, hp_.bb_kv_heads, hp_.bb_layers);
  bb_kv_v_ = add("bb_kv_v", hd, max_bb, hp_.bb_kv_heads, hp_.bb_layers);
  dc_kv_k_ = add("dc_kv_k", hp_.dc_head_dim, 16, hp_.dc_kv_heads, hp_.dc_layers);
  dc_kv_v_ = add("dc_kv_v", hp_.dc_head_dim, 16, hp_.dc_kv_heads, hp_.dc_layers);
  cd_kv_k_ = add("cd_kv_k", hp_.cd_head_dim, max_cd, hp_.cd_kv_heads, hp_.cd_layers);
  cd_kv_v_ = add("cd_kv_v", hp_.cd_head_dim, max_cd, hp_.cd_kv_heads, hp_.cd_layers);

  // codec decoder conv states: 21 tensors per codec_cached.json shapes
  // (pre_conv k3 → 2; dwconv k7 → 6; RU k7 dil d → 6*d; transconv tails
  // rate; decoder.6 k7 → 6)
  auto add_state = [&](const char *name, int C, int S) {
    ConvState cs;
    cs.name = name;
    cs.C = C;
    cs.S = S;
    cs.t = ggml_new_tensor_2d(ctx_kv_, GGML_TYPE_F32, C, S);
    ggml_set_name(cs.t, name);
    cache_tensors.push_back(cs.t);
    conv_states_.push_back(cs);
  };
  add_state("pre_conv", hp_.cd_cb_dim /*512*/, 2);
  for (int u = 0; u < (int)hp_.upsampling_ratios.size(); u++)
    add_state(("up" + std::to_string(u) + ".dwconv").c_str(), hp_.cd_latent, 6);
  add_state("dec0", hp_.cd_latent, 6);
  {
    int ch = hp_.cd_dec_dim;
    for (int b = 0; b < (int)hp_.upsample_rates.size(); b++) {
      const int rate = hp_.upsample_rates[b];
      ch = ch / 2;
      add_state(("blk" + std::to_string(b) + ".tail").c_str(), ch, rate);
      add_state(("blk" + std::to_string(b) + ".ru1").c_str(), ch, 6);
      add_state(("blk" + std::to_string(b) + ".ru3").c_str(), ch, 18);
      add_state(("blk" + std::to_string(b) + ".ru9").c_str(), ch, 54);
    }
    add_state("final", ch, 6);
  }

  for (size_t i = 0; i < rvq_tables_.size(); i++) {
    ggml_tensor *E = rvq_tables_[i];
    ggml_tensor *t = ggml_new_tensor_2d(ctx_kv_, GGML_TYPE_F32, E->ne[0], E->ne[1]);
    ggml_set_name(t, ("rvq_norm_" + std::to_string(i)).c_str());
    rvq_norm_.push_back(t);
  }

  buf_kv_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend);
  if (!buf_kv_) {
    RS_LOG_ERR("M5TTS: failed to allocate KV cache buffers");
    return false;
  }
  // zero the caches
  for (auto *t : cache_tensors) {
    std::vector<uint8_t> zero(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, zero.data(), 0, ggml_nbytes(t));
  }
  cd_offset_ = 0;
  {
    const float e5 = 1e-5f, e9 = 1e-9f;
    ggml_backend_tensor_set(sc_eps5_, &e5, 0, sizeof(float));
    ggml_backend_tensor_set(sc_eps9_, &e9, 0, sizeof(float));
  }
  return true;
}

// ---------------------------------------------------------------------------
// Normalize RVQ codebooks: E = embedding_sum / max(cluster_usage, eps).
// One-time graph at Load; the normalized tables replace the raw pointers in
// the decoder/encoder weight structs.
// ---------------------------------------------------------------------------
void Model::BuildNormalizedCodebooks(ggml_backend_sched_t sched) {
  (void)sched;
  const size_t n = rvq_tables_.size();
  // host-side normalize: E = embedding_sum / max(cluster_usage, 1e-5)
  for (size_t i = 0; i < n; i++) {
    ggml_tensor *E = rvq_tables_[i];
    ggml_tensor *U = rvq_usage_[i];
    ggml_tensor *dst = rvq_norm_[i];
    const size_t ne = (size_t)E->ne[0] * (size_t)E->ne[1];
    std::vector<float> e(ne), u((size_t)U->ne[0]);
    ggml_backend_tensor_get(E, e.data(), 0, ne * sizeof(float));
    ggml_backend_tensor_get(U, u.data(), 0, u.size() * sizeof(float));
    for (int64_t v = 0; v < E->ne[1]; v++) {
      const float denom = std::max(u[(size_t)v], 1e-5f);
      for (int64_t d = 0; d < E->ne[0]; d++)
        e[(size_t)v * E->ne[0] + (size_t)d] /= denom;
    }
    ggml_backend_tensor_set(dst, e.data(), 0, ne * sizeof(float));
  }
  cdw_.sem_embed = rvq_norm_[0];
  for (int l = 0; l < hp_.codebooks - 1; l++)
    cdw_.acou_embed[l] = rvq_norm_[1 + l];
  const size_t enc_off = (size_t)hp_.codebooks;
  cew_.sem_embed = rvq_norm_[enc_off];
  for (int l = 0; l < hp_.codebooks - 1; l++)
    cew_.acou_embed[l] = rvq_norm_[enc_off + 1 + l];
  if (const char *dp = std::getenv("RS_M5TTS_DUMP_CODEBOOK")) {
    ggml_tensor *E = rvq_norm_[enc_off + 1];  // encoder acoustic layer 0
    size_t total = (size_t)E->ne[0] * E->ne[1];
    std::vector<float> e(total);
    ggml_backend_tensor_get(E, e.data(), 0, total * 4);
    FILE *f = fopen(dp, "wb");
    if (f) { fwrite(e.data(), 4, total, f); fclose(f); }
  }
  RS_LOG_INFO("M5TTS: normalized %zu RVQ codebooks", n);
}

// ---------------------------------------------------------------------------
// Reference audio preprocessing (mirrors infer_onnx.py read_reference).
// ---------------------------------------------------------------------------
std::vector<float> Model::PrepareReference(const std::vector<float> &pcm,
                                           int sr, int target_sr) {
  std::vector<float> x = pcm;
  const size_t cap = (size_t)20 * target_sr;  // bound encoder memory
  if (x.size() > cap) x.resize(cap);

  if (sr != target_sr && sr > 0 && !x.empty()) {
    const double ratio = (double)target_sr / (double)sr;
    const size_t n_out = (size_t)((double)(x.size() - 1) * ratio) + 1;
    std::vector<float> y;
    y.reserve(n_out);
    for (size_t i = 0; i < n_out; i++) {
      const double pos = (double)i / ratio;
      const size_t i0 = (size_t)pos;
      const double frac = pos - (double)i0;
      if (i0 + 1 < x.size())
        y.push_back((float)((double)x[i0] * (1.0 - frac) +
                            (double)x[i0 + 1] * frac));
      else
        y.push_back(x.back());
    }
    x = std::move(y);
  }

  float peak = 0.0f;
  for (float v : x) peak = std::max(peak, std::fabs(v));
  if (peak > 0.0f) {
    const float g = 1.0f / (peak + 1e-8f);
    for (float &v : x) v *= g;
  }

  // silence trim: keep speech segments split at >150 ms gaps, 50 ms edge pads
  const float thr = x.size() > (size_t)(10 * target_sr) ? 0.02f : 0.015f;
  std::vector<int64_t> speech;
  for (size_t i = 0; i < x.size(); i++)
    if (std::fabs(x[i]) > thr) speech.push_back((int64_t)i);
  if (speech.empty()) return x;
  const int64_t gap = (int64_t)(0.15 * target_sr);
  std::vector<std::pair<int64_t, int64_t>> segs;
  int64_t start = speech[0], prev = speech[0];
  for (size_t j = 1; j < speech.size(); j++) {
    if (speech[j] - prev > gap) {
      segs.push_back({start, prev});
      start = speech[j];
    }
    prev = speech[j];
  }
  const int64_t end_final =
      ((int64_t)x.size() - prev - 1 >= gap) ? prev + 1 : (int64_t)x.size();
  segs.push_back({start, end_final});
  const int64_t pad = 1200;
  std::vector<float> out;
  for (auto &seg : segs) {
    const int64_t a = std::max((int64_t)0, seg.first - pad);
    const int64_t b = std::min((int64_t)x.size(), seg.second + pad);
    if (b > a) out.insert(out.end(), x.begin() + a, x.begin() + b);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Conditioning prefix (matches infer_onnx.py conditioned_text).
// ---------------------------------------------------------------------------
std::string Model::ConditionText(const std::string &ref_text,
                                 const std::string &text,
                                 const std::string &lang) const {
  static const char *prefix =
      "<|caption_begin|><|caption_end|><|accent_begin|><|accent_end|>"
      "<|speaker_start|>None<|speaker_end|><|lang_begin|>";
  std::string out = prefix;
  out += lang;
  out += "<|lang_end|>";
  out += ref_text;
  out += text;
  return out;
}

// The wetext zh FST normalizes full-width punctuation to ASCII; the M5
// model (WeTextProcessing with full_to_half=False) expects full-width
// punctuation. Restore it for punctuation that follows a CJK byte.
static std::string RestoreZhPunctuation(const std::string &in) {
  static const std::unordered_map<char, std::string> map = {
      {',', "\xEF\xBC\x8C"}, {'.', "\xE3\x80\x82"},
      {'?', "\xEF\xBC\x9F"}, {'!', "\xEF\xBC\x81"},
      {';', "\xEF\xBC\x9B"}, {':', "\xEF\xBC\x9A"}};
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    const char c = in[i];
    auto it = map.find(c);
    if (it != map.end() && i > 0 && (unsigned char)in[i - 1] >= 0x80)
      out += it->second;
    else
      out += c;
  }
  return out;
}

static std::string NormalizeLanguage(const char *language) {
  std::string l = language ? language : "";
  for (auto &ch : l) ch = (char)std::tolower((unsigned char)ch);
  if (l.empty() || l == "zh" || l == "zh-cn" || l == "chinese") return "Chinese";
  if (l == "en" || l == "english") return "English";
  if (l == "de" || l == "german") return "German";
  if (l == "fr" || l == "french") return "French";
  if (l == "ja" || l == "japanese") return "Japanese";
  if (l == "ko" || l == "korean") return "Korean";
  return "Chinese";
}

// ---------------------------------------------------------------------------
// push APIs
// ---------------------------------------------------------------------------
bool Model::PushText(RSState &state_raw, const char *text, const char *language,
                     const char *instruct) {
  auto &s = static_cast<State &>(state_raw);
  s.text = text ? text : "";
  s.language = NormalizeLanguage(language);
  (void)instruct;
  s.prompt_codes.clear();
  s.T_prompt = 0;
  s.gen_codes.clear();
  s.T_gen = 0;
  s.generated = false;
  s.audio_output.clear();
  return true;
}

bool Model::PushReferenceAudio(RSState &state_raw, const float *samples,
                               int n_samples, int sample_rate,
                               ggml_backend_sched_t sched) {
  (void)sched;
  auto &s = static_cast<State &>(state_raw);
  if (!samples || n_samples <= 0) return false;
  s.ref_audio.assign(samples, samples + n_samples);
  s.ref_sr = sample_rate > 0 ? sample_rate : hp_.sample_rate;
  s.ref_ready = true;
  return true;
}

bool Model::PushReferenceText(RSState &state_raw, const char *ref_text) {
  auto &s = static_cast<State &>(state_raw);
  s.ref_text = ref_text ? ref_text : "";
  return true;
}

int Model::GetAudioOutput(RSState &state_raw, float **out_data) {
  auto &s = static_cast<State &>(state_raw);
  if (out_data)
    *out_data = s.audio_output.empty() ? nullptr : s.audio_output.data();
  return (int)s.audio_output.size();
}

// ---------------------------------------------------------------------------
// sampling (mirrors onnx/graphs.py DepthFrame.sample)
// ---------------------------------------------------------------------------
int Model::SampleLogits(const float *logits, int n, bool allow_eos,
                        std::mt19937 &rng) const {
  std::vector<float> x(logits, logits + n);
  if (!allow_eos && n > 0) x[0] = -1e5f;
  for (auto &v : x) v /= hp_.temperature;
  const int K = std::min(hp_.topk, n);
  std::vector<float> s = x;
  std::nth_element(s.begin(), s.begin() + (n - K), s.end());
  const float cutoff = s[n - K];
  for (auto &v : x)
    if (v < cutoff) v = -std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (float v : x) mx = std::max(mx, v);
  float sum = 0.0f;
  for (auto &v : x) {
    v = (v == -std::numeric_limits<float>::infinity())
            ? 0.0f
            : std::exp(v - mx);
    sum += v;
  }
  if (sum <= 0.0f) return 0;
  for (auto &v : x) v /= sum;
  // argmax(probs / noise), noise ~ Exponential(1) per vocab entry
  std::exponential_distribution<float> ed(1.0f);
  int best = 0;
  float best_v = -1.0f;
  for (int i = 0; i < n; i++) {
    const float noise = std::max(ed(rng), 1e-9f);
    const float p = x[i] / noise;
    if (p > best_v) {
      best_v = p;
      best = i;
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// Encode: reference encode + prefill + multi-token frame loop
// ---------------------------------------------------------------------------
bool Model::Encode(const std::vector<float> &input_frames, RSState &state_raw,
                   ggml_backend_sched_t sched) {
  (void)input_frames;
  auto &s = static_cast<State &>(state_raw);
  if (s.text.empty()) {
    RS_LOG_ERR("M5TTS: no text pushed");
    return false;
  }
  rng_.seed((uint32_t)seed_);

  // WeText zh normalization (pass-through when FST data is missing)
  const bool is_zh = s.language == "Chinese";
  if (is_zh && !tn_tried_) {
    tn_tried_ = true;
    std::string wt_dir = rs::find_data_dir("wetext", "RS_WETEXT_DATA_DIR");
    if (!wt_dir.empty()) {
      if (!tn_) tn_ = std::make_unique<rs::WeTextNormalizer>();
      tn_->Load(wt_dir + "/zh_tn_tagger.fst",
                wt_dir + "/zh_tn_verbalizer.fst");
    }
  }
  std::string text = s.text;
  if (is_zh && tn_ && tn_->IsLoaded()) {
    text = RestoreZhPunctuation(tn_->Normalize(text));
    if (std::getenv("RS_M5TTS_DEBUG_BPE"))
      fprintf(stderr, "M5TTS TN: '%s' -> '%s'\n", s.text.c_str(),
              text.c_str());
  }

  const std::string sentence = ConditionText(s.ref_text, text, s.language);
  s.text_ids = bpe_.encode(sentence);
  const int Tt = (int)s.text_ids.size();
  RS_LOG_INFO("M5TTS: %d text tokens (lang=%s%s)", Tt, s.language.c_str(),
              s.ref_ready ? ", voice clone" : "");
  if (const char *dp = std::getenv("RS_M5TTS_DUMP_IDS")) {
    FILE *f = fopen(dp, "w");
    if (f) {
      for (int id : s.text_ids) fprintf(f, "%d\n", id);
      fclose(f);
    }
  }

  s.prompt_codes.clear();
  s.T_prompt = 0;
  if (s.ref_ready && !s.ref_audio.empty()) {
    std::vector<float> pcm;
    if (const char *ap = std::getenv("RS_M5TTS_AUDIO_BIN")) {
      // Debug override: use the exact reference audio (int32 n, then n f32).
      FILE *af = fopen(ap, "rb");
      if (af) {
        int32_t n = 0;
        if (fread(&n, 4, 1, af) == 1 && n > 0) {
          pcm.resize((size_t)n);
          if (fread(pcm.data(), 4, (size_t)n, af) == (size_t)n)
            RS_LOG_INFO("M5TTS: using %d external audio samples", n);
        }
        fclose(af);
      }
    }
    if (pcm.empty())
      pcm = PrepareReference(s.ref_audio, s.ref_sr, hp_.sample_rate);
    if (const char *dp = std::getenv("RS_M5TTS_DUMP_AUDIO")) {
      FILE *f = fopen(dp, "wb");
      if (f) { fwrite(pcm.data(), 4, pcm.size(), f); fclose(f); }
    }
    if (!pcm.empty()) {
      if (const char *pp = std::getenv("RS_M5TTS_PROMPT_BIN")) {
        // Debug override: use reference prompt codes (int32 T, then T*16).
        FILE *pf = fopen(pp, "rb");
        if (pf) {
          int32_t T = 0;
          if (fread(&T, 4, 1, pf) == 1 && T > 0) {
            s.prompt_codes.resize((size_t)T * 16);
            if (fread(s.prompt_codes.data(), 4, (size_t)T * 16, pf) ==
                (size_t)T * 16) {
              s.T_prompt = T;
              RS_LOG_INFO("M5TTS: using %d external prompt codes", T);
            }
          }
          fclose(pf);
        }
      } else if (!EncodeReference(sched, pcm.data(), (int)pcm.size(),
                                  s.prompt_codes, s.T_prompt)) {
        return false;
      }
      RS_LOG_INFO("M5TTS: reference encoded to %d frames", s.T_prompt);
      if (const char *dp = std::getenv("RS_M5TTS_DUMP_PROMPT")) {
        FILE *f = fopen(dp, "wb");
        if (f) {
          fwrite(s.prompt_codes.data(), 4, s.prompt_codes.size(), f);
          fclose(f);
        }
      }
    }
  }
  if (Tt + s.T_prompt + hp_.max_frames > hp_.max_seq_len) {
    RS_LOG_ERR("M5TTS: sequence too long (text %d + prompt %d, budget %d)", Tt,
               s.T_prompt, hp_.max_seq_len);
    return false;
  }

  std::vector<float> lh, c0_logits;
  int n_past = 0;
  if (!Prefill(sched, s.text_ids.data(), Tt,
               s.T_prompt ? s.prompt_codes.data() : nullptr, s.T_prompt, lh,
               c0_logits, n_past))
    return false;
  if (const char *dp = std::getenv("RS_M5TTS_DUMP_PREFILL")) {
    FILE *f = fopen(dp, "wb");
    if (f) {
      fwrite(lh.data(), 4, lh.size(), f);
      fwrite(c0_logits.data(), 4, c0_logits.size(), f);
      fclose(f);
      RS_LOG_INFO("M5TTS: dumped prefill hidden+logits");
    }
  }

  s.gen_codes.clear();
  s.T_gen = 0;
  int32_t raw[16];
  std::vector<float> next_hidden = lh;
  bool eos = false;
  const auto t0 = std::chrono::steady_clock::now();
  for (int index = 0; index < hp_.max_frames; index++) {
    if (index > 0) {
      if (!BackboneStep(sched, next_hidden.data(), n_past++, lh, c0_logits))
        return false;
    }
    const bool allow_eos = index > hp_.min_frames;
    std::memset(raw, 0, sizeof(raw));
    raw[0] = SampleLogits(c0_logits.data(), hp_.audio_vocab, allow_eos, rng_);
    std::vector<float> logits;
    if (!DepthStep(sched, 1, 2, lh.data(), &raw[0], 0, logits)) return false;
    for (int k = 1; k < hp_.codebooks; k++) {
      const int32_t code =
          SampleLogits(logits.data(), hp_.audio_vocab, true, rng_);
      raw[k] = code;
      if (k + 1 < hp_.codebooks) {
        if (!DepthStep(sched, k + 1, 1, nullptr, &raw[k], k + 1, logits))
          return false;
      }
    }
    bool all_zero = true;
    for (int k = 0; k < hp_.codebooks; k++)
      if (raw[k] != 0) {
        all_zero = false;
        break;
      }
    if (all_zero) {
      eos = true;
      break;
    }
    for (int k = 0; k < hp_.codebooks; k++)
      s.gen_codes.push_back(std::max(raw[k] - 1, 0));
    s.T_gen++;
    if (!NextHidden(sched, raw, next_hidden)) return false;
  }
  s.generated = true;
  const double gen_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  RS_LOG_INFO("M5TTS: %d frames (%s) in %.2fs (LM)", s.T_gen,
              eos ? "eos" : "truncated", gen_s);
  return true;
}

// ---------------------------------------------------------------------------
// Decode: codec decode → PCM (prompt cut, 0.1 s tail trim)
// ---------------------------------------------------------------------------
bool Model::Decode(RSState &state_raw, ggml_backend_sched_t sched) {
  auto &s = static_cast<State &>(state_raw);
  if (!s.generated) {
    RS_LOG_ERR("M5TTS: Decode before Encode");
    return false;
  }
  if (s.T_gen == 0) {
    RS_LOG_ERR("M5TTS: no frames generated");
    return false;
  }
  const int T_all = s.T_prompt + s.T_gen;
  std::vector<int32_t> all((size_t)T_all * 16);
  if (s.T_prompt)
    std::copy(s.prompt_codes.begin(), s.prompt_codes.end(), all.begin());
  std::copy(s.gen_codes.begin(), s.gen_codes.end(),
            all.begin() + (size_t)s.T_prompt * 16);
  // Debug override: decode a fixed code sequence from a binary file
  // (int32 T, then T*16 int32 codes) to isolate the codec from the LM.
  if (const char *p = std::getenv("RS_M5TTS_CODECS_BIN")) {
    FILE *f = fopen(p, "rb");
    if (f) {
      int32_t T = 0;
      if (fread(&T, 4, 1, f) == 1 && T > 0) {
        all.resize((size_t)T * 16);
        if (fread(all.data(), 4, (size_t)T * 16, f) == (size_t)T * 16) {
          RS_LOG_INFO("M5TTS: using %d external codes", T);
          const int T_prompt_ext = s.T_prompt;
          ResetCodecState();
          std::vector<float> pcm;
          if (!DecodeCodes(sched, all.data(), T, pcm)) return false;
          const size_t cut = (size_t)T_prompt_ext * hp_.hop_length;
          if (pcm.size() > cut) pcm.erase(pcm.begin(), pcm.begin() + cut);
          const size_t trim = 2400;
          if (pcm.size() > trim) pcm.resize(pcm.size() - trim);
          if (const char *dp = std::getenv("RS_M5TTS_DUMP_PCM")) {
            FILE *df = fopen(dp, "wb");
            if (df) { fwrite(pcm.data(), 4, pcm.size(), df); fclose(df); }
          }
          s.audio_output = std::move(pcm);
          return true;
        }
      }
      fclose(f);
    }
  }

  ResetCodecState();
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> pcm;
  if (!DecodeCodes(sched, all.data(), T_all, pcm)) return false;
  const double dec_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();

  const size_t cut = (size_t)s.T_prompt * hp_.hop_length;
  if (pcm.size() > cut) pcm.erase(pcm.begin(), pcm.begin() + cut);
  const size_t trim = 2400;  // 0.1 s
  if (pcm.size() > trim) pcm.resize(pcm.size() - trim);
  s.audio_output = std::move(pcm);
  RS_LOG_INFO("M5TTS: codec decoded %.2fs audio in %.2fs",
              (double)s.audio_output.size() / hp_.sample_rate, dec_s);
  return true;
}

void Model::ResetCodecState() {
  for (auto &cs : conv_states_) {
    std::vector<uint8_t> zero(ggml_nbytes(cs.t), 0);
    ggml_backend_tensor_set(cs.t, zero.data(), 0, ggml_nbytes(cs.t));
  }
  cd_offset_ = 0;
}

}  // namespace m5tts

// ---------------------------------------------------------------------------
// arch registration
// ---------------------------------------------------------------------------
struct M5TTSRegistrar {
  M5TTSRegistrar() {
    rs_register_model_arch("m5-tts",
                           []() { return std::make_shared<m5tts::Model>(); });
  }
} global_m5tts_reg;
