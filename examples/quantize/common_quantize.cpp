#include "common_quantize.h"
#include "imatrix_collector.h"
#include <climits>
#include <cstring>
#include <inttypes.h>
#include <map>
#include <mutex>
#include <regex>
#include <stdarg.h>
#include <thread>

// ============================================
// Quantization type mapping
// ============================================

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0",       GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1",       GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0",       GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1",       GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0",       GGML_FTYPE_MOSTLY_Q8_0},
    {"q2_k",       GGML_FTYPE_MOSTLY_Q2_K},
    {"q2_k_m",     GGML_FTYPE_MOSTLY_Q2_K_M},
    {"q3_k",       GGML_FTYPE_MOSTLY_Q3_K},
    {"q3_k_m",     GGML_FTYPE_MOSTLY_Q3_K_M},
    {"q4_k",       GGML_FTYPE_MOSTLY_Q4_K},
    {"q4_k_m",     GGML_FTYPE_MOSTLY_Q4_K_M},
    {"q5_k",       GGML_FTYPE_MOSTLY_Q5_K},
    {"q5_k_m",     GGML_FTYPE_MOSTLY_Q5_K_M},
    {"q6_k",       GGML_FTYPE_MOSTLY_Q6_K},
    {"iq1_m",      GGML_FTYPE_MOSTLY_IQ1_M},
    {"iq1_s",      GGML_FTYPE_MOSTLY_IQ1_S},
    {"iq1_s_m",    GGML_FTYPE_MOSTLY_IQ1_S_M},
    {"iq2_s",      GGML_FTYPE_MOSTLY_IQ2_S},
    {"iq2_xs",     GGML_FTYPE_MOSTLY_IQ2_XS},
    {"iq2_xxs",    GGML_FTYPE_MOSTLY_IQ2_XXS},
    {"iq2_xxs_m",  GGML_FTYPE_MOSTLY_IQ2_XXS_M},
    {"iq3_xxs",    GGML_FTYPE_MOSTLY_IQ3_XXS},
    {"iq3_s",      GGML_FTYPE_MOSTLY_IQ3_S},
    {"iq4_nl",     GGML_FTYPE_MOSTLY_IQ4_NL},
    {"iq4_xs",     GGML_FTYPE_MOSTLY_IQ4_XS},
};

void ggml_print_ftypes(FILE *fp) {
  for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
    fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
  }
}

enum ggml_ftype ggml_parse_ftype(const char *str) {
  enum ggml_ftype ftype;
  const auto it = GGML_FTYPE_MAP.find(str);
  if (it != GGML_FTYPE_MAP.end()) {
    ftype = it->second;
  } else if (str[0] >= '0' && str[0] <= '9') {
    ftype = (enum ggml_ftype)atoi(str);
  } else {
    fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
    return GGML_FTYPE_UNKNOWN;
  }
  return ftype;
}

// ============================================
// Internal helpers
// ============================================

static void zeros(std::ofstream &file, size_t n) {
  char zero = 0;
  for (size_t i = 0; i < n; ++i) {
    file.write(&zero, 1);
  }
}

static std::string format_tensor_shape(const struct ggml_tensor *t) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
  for (int i = 1; i < GGML_MAX_DIMS; i++) {
    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64,
             t->ne[i]);
  }
  return buf;
}

static std::string format(const char *fmt, ...) {
  va_list ap;
  va_list ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  GGML_ASSERT(size >= 0 && size < INT_MAX);
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  GGML_ASSERT(size2 == size);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

// ============================================
// Multi-threaded quantization
// ============================================

static size_t rs_tensor_quantize_internal(enum ggml_type new_type,
                                          const float *f32_data, void *new_data,
                                          const int64_t chunk_size,
                                          int64_t nrows, int64_t n_per_row,
                                          const float *imatrix,
                                          std::vector<std::thread> &workers,
                                          const int nthread) {

  if (nthread < 2) {
    // single-thread
    size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0,
                                          nrows, n_per_row, imatrix);
    if (!ggml_validate_row_data(new_type, new_data, new_size)) {
      throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
  }

  std::mutex mutex;
  int64_t counter = 0;
  size_t new_size = 0;
  bool valid = true;

  auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data,
                  new_data, chunk_size, nrows, n_per_row, imatrix]() {
    const int64_t nrows_per_chunk = chunk_size / n_per_row;
    size_t local_size = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      int64_t first_row = counter;
      counter += nrows_per_chunk;
      if (first_row >= nrows) {
        if (local_size > 0) {
          new_size += local_size;
        }
        break;
      }
      lock.unlock();

      const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
      size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data,
                                             first_row * n_per_row, this_nrow,
                                             n_per_row, imatrix);
      local_size += this_size;

      // validate the quantized data
      const size_t row_size = ggml_row_size(new_type, n_per_row);
      void *this_data = (char *)new_data + first_row * row_size;
      if (!ggml_validate_row_data(new_type, this_data, this_size)) {
        std::unique_lock<std::mutex> lock(mutex);
        valid = false;
        break;
      }
    }
  };

  for (int it = 0; it < nthread - 1; ++it) {
    workers.emplace_back(compute);
  }
  compute();
  for (auto &w : workers) {
    w.join();
  }
  workers.clear();

  if (!valid) {
    throw std::runtime_error("quantized data validation failed");
  }
  return new_size;
}

// ============================================
// Tensor categorization for mixed-precision strategies
// ============================================

enum class tensor_category {
  TOKEN_EMBD,
  ATTENTION_V,
  ATTENTION_K,
  ATTENTION_Q,
  ATTENTION_OUTPUT,
  FFN_UP,
  FFN_GATE,
  FFN_DOWN,
  OUTPUT,
  OTHER,
};

// Extract (layer_index, n_total_layers_estimate) from tensor names like:
//   model.layers.{N}.*                     (Qwen3-style LLM, ~28 layers)
//   encoder.encoders.{N}.*                 (SenseVoice/FunASR main encoder)
//   encoder.tp_encoders.{N}.*              (SenseVoice/FunASR TP encoder)
//   ctc_decoder.blocks.{N}.*               (FunASR CTC decoder)
//   audio_adaptor.blocks.{N}.*             (FunASR audio adaptor)
//   semantic_model.encoder.layers.{N}.*    (HuBERT semantic encoder)
//   encoder.encoders0.*                    (single first block — layer 0)
// Returns {-1, 0} if no layer index is recognised.
// The n_total estimate is used by is_critical_layer / is_first_last_8th to
// derive proportional first/last-1/8 bounds. Estimates do not need to be
// exact — the heuristic is robust to small mismatches.
static std::pair<int, int> extract_layer_info(const std::string &name) {
  struct PrefixInfo {
    const char *prefix;
    int         n_total_hint;
  };
  static const PrefixInfo prefixes[] = {
      {"model.layers.",                  28},  // Qwen3-style LLM
      {"llm.layers.",                    28},  // Qwen3 (alt name)
      {"llm.model.layers.",              28},  // FunASR LLM
      {"encoder.encoders.",              50},  // SenseVoice/FunASR main
      {"encoder.tp_encoders.",           20},  // SenseVoice/FunASR TP
      {"semantic_model.encoder.layers.", 12},  // HuBERT
      {"ctc_decoder.blocks.",             5},  // FunASR CTC decoder
      {"audio_adaptor.blocks.",           2},  // FunASR audio adaptor
  };
  for (const auto &pi : prefixes) {
    size_t pos = name.find(pi.prefix);
    if (pos != std::string::npos) {
      pos += strlen(pi.prefix);
      size_t end = name.find('.', pos);
      if (end != std::string::npos) {
        try {
          return {std::stoi(name.substr(pos, end - pos)), pi.n_total_hint};
        } catch (...) {
          return {-1, 0};
        }
      }
    }
  }
  // encoders0 is a single first block — always layer 0 of a 1-layer module
  if (name.find("encoder.encoders0.") != std::string::npos) {
    return {0, 1};
  }
  return {-1, 0};
}

static bool name_ends_with(const std::string &name, const char *suffix) {
  size_t nlen = name.size();
  size_t slen = strlen(suffix);
  return nlen >= slen && name.compare(nlen - slen, slen, suffix) == 0;
}

static tensor_category categorize_tensor(const std::string &name) {
  // Token embeddings (may have prefixes like "llm.", "model.")
  if (name_ends_with(name, "embed_tokens.weight") ||
      name == "embed.weight") {
    return tensor_category::TOKEN_EMBD;
  }
  // Output projection
  if (name_ends_with(name, "lm_head.weight") ||
      name_ends_with(name, "ctc.ctc_lo.weight") ||
      name_ends_with(name, "ctc_out_linear.weight")) {
    return tensor_category::OUTPUT;
  }
  // Attention V (most sensitive to quantization)
  if (name.find("v_proj.weight") != std::string::npos ||
      name.find("linear_v.weight") != std::string::npos) {
    return tensor_category::ATTENTION_V;
  }
  // Attention K
  if (name.find("k_proj.weight") != std::string::npos ||
      name.find("linear_k.weight") != std::string::npos) {
    return tensor_category::ATTENTION_K;
  }
  // Attention Q
  if (name.find("q_proj.weight") != std::string::npos ||
      name.find("linear_q.weight") != std::string::npos) {
    return tensor_category::ATTENTION_Q;
  }
  // Attention Output
  if (name.find("o_proj.weight") != std::string::npos ||
      name.find("out_proj.weight") != std::string::npos ||
      name.find("linear_out.weight") != std::string::npos) {
    return tensor_category::ATTENTION_OUTPUT;
  }
  // FFN Up / Gate
  if (name.find("up_proj.weight") != std::string::npos ||
      name.find("w_1.weight") != std::string::npos) {
    return tensor_category::FFN_UP;
  }
  if (name.find("gate_proj.weight") != std::string::npos) {
    return tensor_category::FFN_GATE;
  }
  // FFN Down
  if (name.find("down_proj.weight") != std::string::npos ||
      name.find("w_2.weight") != std::string::npos) {
    return tensor_category::FFN_DOWN;
  }
  if (name.find("intermediate_dense.weight") != std::string::npos) {
    return tensor_category::FFN_UP;
  }
  if (name.find("output_dense.weight") != std::string::npos) {
    return tensor_category::FFN_DOWN;
  }
  return tensor_category::OTHER;
}

// ============================================
// Per-tensor type selection for mixed-precision strategies
// ============================================

static ggml_type
rs_get_qtype_for_tensor(const rs_quantize_options &opts, const std::string &name,
                        const struct ggml_tensor *tensor) {
  const ggml_ftype ftype = opts.ftype;
  const std::vector<std::string> &to_skip = opts.to_skip;

  auto cat = categorize_tensor(name);

  auto align_check = [tensor](ggml_type qt) -> bool {
    return tensor->ne[0] % ggml_blck_size(qt) == 0;
  };

  // Demote to an alignment-safe type when the preferred k-quant block (256)
  // does not divide ne[0]. Small leading dims (e.g. ctc_decoder w_2 with
  // ne[0]=128) cannot use Q4_K/Q5_K/Q6_K/IQ*; fall back to Q8_0 (block=32)
  // when possible, otherwise keep the original dtype.
  auto pick = [tensor, &align_check](ggml_type preferred) -> ggml_type {
    if (align_check(preferred)) return preferred;
    if (align_check(GGML_TYPE_Q8_0)) return GGML_TYPE_Q8_0;
    return tensor->type;
  };

  // --- Per-tensor type overrides (highest priority).
  // These win over the skip list, --pure, and K_M / IQ_M bumps so that the
  // caller can force a specific qtype on embed_tokens / lm_head / ctc_lo
  // regardless of the default policy.  Skip the dimensionality/weight check
  // for these — categorize_tensor already constrains to known weight names.
  if (cat == tensor_category::TOKEN_EMBD &&
      opts.token_embedding_type != GGML_TYPE_COUNT) {
    return pick(opts.token_embedding_type);
  }
  if (cat == tensor_category::OUTPUT &&
      opts.output_tensor_type != GGML_TYPE_COUNT) {
    return pick(opts.output_tensor_type);
  }

  // Check if this tensor should be quantized at all
  bool quantize = name.rfind("weight") == name.size() - 6;
  for (const auto &s : to_skip) {
    if (std::regex_match(name, std::regex(s))) {
      quantize = false;
      break;
    }
  }
  quantize &= (ggml_n_dims(tensor) >= 2);
  quantize &= name.find("_norm.weight") == std::string::npos;
  quantize &= name.find(".norm.weight") == std::string::npos;

  if (!quantize) {
    return tensor->type;
  }

  auto [layer, n_total] = extract_layer_info(name);

  // --- --pure: bypass all mixed-precision bumps; use the ftype's underlying
  // qtype uniformly. Routes through the `default` branch below.
  ggml_ftype eff_ftype = ftype;
  if (opts.pure) {
    eff_ftype = (ggml_ftype)-128;  // unreachable, forces default branch
  }

  // Proportional first/last heuristics: bump precision on roughly the first
  // and last 1/8 of layers. Scales correctly across 5-layer CTC decoder,
  // 20-layer TP encoder, 28-layer Qwen3 LLM, and 50-layer SenseVoice main
  // encoder. For very small modules (<8 layers) we treat the first and last
  // layer as critical.
  auto is_critical_layer = [](int layer, int n_total) -> bool {
    if (layer < 0 || n_total <= 0) return false;
    if (n_total < 8) {
      return layer == 0 || layer == n_total - 1;
    }
    int eighth = n_total / 8;
    return layer < eighth || layer >= n_total - eighth;
  };

  auto is_first_last_8th = is_critical_layer;

  // Select per-tensor type based on the overall strategy
  switch (eff_ftype) {
  // ================================================================
  // Q2_K_M: ~3.0 bpw mixed. Baseline = Q2_K with aggressive bumps for
  // tensors that determine CTC peak sharpness (the failure mode of pure
  // Q2_K on small encoders like SenseVoice for KWS).
  //   embed       → Q4_K   (token table)
  //   output / lm_head / ctc_lo → Q4_K   (CTC head is the KWS-killer)
  //   attn_v      → Q4_K critical layers, Q3_K elsewhere
  //   attn_k      → Q3_K   (less critical than V per llama.cpp lore)
  //   attn_output → Q3_K critical, Q2_K elsewhere
  //   ffn_down    → Q3_K first/last 1/8 layers, Q2_K elsewhere
  //   default     → Q2_K
  // Sits between pure Q2_K (which fails on KWS) and Q3_K (which works).
  // ================================================================
  case GGML_FTYPE_MOSTLY_Q2_K_M: {
    if (cat == tensor_category::TOKEN_EMBD) {
      return align_check(GGML_TYPE_Q4_K) ? GGML_TYPE_Q4_K : pick(GGML_TYPE_Q3_K);
    }
    if (cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q4_K) ? GGML_TYPE_Q4_K : pick(GGML_TYPE_Q3_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K);
    }
    if (cat == tensor_category::ATTENTION_K) {
      return pick(GGML_TYPE_Q3_K);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q3_K : GGML_TYPE_Q2_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q3_K : GGML_TYPE_Q2_K);
    }
    return pick(GGML_TYPE_Q2_K);
  }

  // ================================================================
  // Q3_K_M: ~3.9 bpw mixed. Baseline = Q3_K with critical-tensor bumps.
  // embed → Q4_K, lm_head/ctc → Q5_K, attn_v → Q4_K (Q5_K critical),
  // attn_k → Q4_K, attn_output → Q3_K (Q4_K critical),
  // ffn_down → Q4_K on first/last 1/8 layers else Q3_K.
  // Designed to avoid the LLM repetition / token drift seen with pure Q2_K
  // on small (<1B) LLMs while staying ~15% smaller than Q4_K_M.
  // ================================================================
  case GGML_FTYPE_MOSTLY_Q3_K_M: {
    if (cat == tensor_category::TOKEN_EMBD) {
      return align_check(GGML_TYPE_Q4_K) ? GGML_TYPE_Q4_K : pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q5_K) ? GGML_TYPE_Q5_K : pick(GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::ATTENTION_K) {
      return pick(GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K);
    }
    return pick(GGML_TYPE_Q3_K);
  }

  // ================================================================
  // Q4_K_M: ~4.5 bpw mixed. Baseline = Q4_K, embed/lm_head = Q6_K
  // ================================================================
  case GGML_FTYPE_MOSTLY_Q4_K_M: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q6_K) ? GGML_TYPE_Q6_K : pick(GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K);
    }
    return pick(GGML_TYPE_Q4_K);
  }

  // ================================================================
  // Q5_K_M: ~5.5 bpw mixed. Baseline = Q5_K, embed/lm_head = Q6_K
  // ================================================================
  case GGML_FTYPE_MOSTLY_Q5_K_M: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q6_K) ? GGML_TYPE_Q6_K : pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q6_K : GGML_TYPE_Q6_K);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT ||
        cat == tensor_category::ATTENTION_Q ||
        cat == tensor_category::ATTENTION_K) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K);
    }
    return pick(GGML_TYPE_Q5_K);
  }

  // ================================================================
  // IQ3_XXS: ~3.06 bpw. embed/output → IQ3_S, attn_v → Q4_K
  // ================================================================
  case GGML_FTYPE_MOSTLY_IQ3_XXS: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_IQ3_S) ? GGML_TYPE_IQ3_S : pick(GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ3_S);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ3_XXS);
    }
    return pick(GGML_TYPE_IQ3_XXS);
  }

  // ================================================================
  // IQ3_S: ~3.44 bpw. embed/output → Q4_K, attn_v → Q4_K
  // ================================================================
  case GGML_FTYPE_MOSTLY_IQ3_S: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q4_K) ? GGML_TYPE_Q4_K : pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ3_S);
    }
    return pick(GGML_TYPE_IQ3_S);
  }

  // ================================================================
  // IQ4_NL: ~4.44 bpw. embed/output → Q5_K, attn_v → Q5_K
  // ================================================================
  case GGML_FTYPE_MOSTLY_IQ4_NL: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q5_K) ? GGML_TYPE_Q5_K : pick(GGML_TYPE_Q6_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q5_K : GGML_TYPE_IQ4_NL);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q5_K : GGML_TYPE_IQ4_NL);
    }
    return pick(GGML_TYPE_IQ4_NL);
  }

  // ================================================================
  // IQ4_XS: ~4.19 bpw. embed/output → Q5_K, attn_v → Q5_K
  // ================================================================
  case GGML_FTYPE_MOSTLY_IQ4_XS: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q5_K) ? GGML_TYPE_Q5_K : pick(GGML_TYPE_Q6_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ4_XS);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ4_XS);
    }
    return pick(GGML_TYPE_IQ4_XS);
  }

  // ================================================================
  // IQ2_XXS_M: ~2.4 bpw mixed. Baseline = IQ2_XXS with critical bumps so
  // embed/lm_head don't tank a sub-1B LLM. Needs imatrix; tensors without
  // an imatrix entry fall back to Q4_K (see ggml_quantize_requires_imatrix
  // handling in rapid_speech_ggml_quantize).
  //   embed       → Q4_K
  //   lm_head     → Q4_K
  //   attn_v      → Q4_K critical, IQ3_XXS else
  //   attn_k      → IQ3_XXS
  //   attn_output → IQ3_XXS critical, IQ2_XXS else
  //   ffn_down    → IQ3_XXS first/last 1/8, IQ2_XXS else
  //   default     → IQ2_XXS
  // ================================================================
  case GGML_FTYPE_MOSTLY_IQ2_XXS_M: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q4_K) ? GGML_TYPE_Q4_K : pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ3_XXS);
    }
    if (cat == tensor_category::ATTENTION_K) {
      return pick(GGML_TYPE_IQ3_XXS);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XXS);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XXS);
    }
    return pick(GGML_TYPE_IQ2_XXS);
  }

  // ================================================================
  // IQ1_S_M: ~1.9 bpw mixed.  Extreme low bit; only useful with imatrix
  // and an LLM tolerant of heavy degradation.  Bumps the same critical
  // tensors as IQ2_XXS_M but with even smaller body fallback.
  //   embed       → Q4_K
  //   lm_head     → Q4_K
  //   attn_v      → Q4_K critical, IQ2_XXS else
  //   attn_k      → IQ2_XXS
  //   attn_output → IQ2_XXS critical, IQ1_S else
  //   ffn_down    → IQ2_XXS first/last 1/8, IQ1_S else
  //   default     → IQ1_S
  // ================================================================
  case GGML_FTYPE_MOSTLY_IQ1_S_M: {
    if (cat == tensor_category::TOKEN_EMBD ||
        cat == tensor_category::OUTPUT) {
      return align_check(GGML_TYPE_Q4_K) ? GGML_TYPE_Q4_K : pick(GGML_TYPE_Q5_K);
    }
    if (cat == tensor_category::ATTENTION_V) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_Q4_K : GGML_TYPE_IQ2_XXS);
    }
    if (cat == tensor_category::ATTENTION_K) {
      return pick(GGML_TYPE_IQ2_XXS);
    }
    if (cat == tensor_category::ATTENTION_OUTPUT) {
      return pick(is_critical_layer(layer, n_total) ? GGML_TYPE_IQ2_XXS : GGML_TYPE_IQ1_S);
    }
    if (cat == tensor_category::FFN_DOWN) {
      return pick(is_first_last_8th(layer, n_total) ? GGML_TYPE_IQ2_XXS : GGML_TYPE_IQ1_S);
    }
    return pick(GGML_TYPE_IQ1_S);
  }

  default: {
    // Uniform quantization — resolve ftype → qtype.  Handle RS-custom M
    // variants here too: --pure routes through this branch with the user's
    // chosen ftype intact, so we need a mapping that knows about Q*_K_M /
    // IQ*_M (ggml_ftype_to_ggml_type does not).
    auto resolve_base_qtype = [](ggml_ftype f) -> ggml_type {
      switch (f) {
      case GGML_FTYPE_MOSTLY_Q2_K_M:    return GGML_TYPE_Q2_K;
      case GGML_FTYPE_MOSTLY_Q3_K_M:    return GGML_TYPE_Q3_K;
      case GGML_FTYPE_MOSTLY_Q4_K_M:    return GGML_TYPE_Q4_K;
      case GGML_FTYPE_MOSTLY_Q5_K_M:    return GGML_TYPE_Q5_K;
      case GGML_FTYPE_MOSTLY_IQ1_S_M:   return GGML_TYPE_IQ1_S;
      case GGML_FTYPE_MOSTLY_IQ2_XXS_M: return GGML_TYPE_IQ2_XXS;
      default:                          return ggml_ftype_to_ggml_type(f);
      }
    };
    ggml_type qtype = resolve_base_qtype(ftype);
    if (qtype == GGML_TYPE_COUNT) return tensor->type;
    return pick(qtype);
  }
  }
}

// ============================================
// Main quantize function
// ============================================

bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const rs_quantize_options &opts) {

  const ggml_ftype ftype = opts.ftype;
  const int nthread = opts.nthread;
  const std::vector<std::string> &to_skip = opts.to_skip;
  const std::string &imatrix_file = opts.imatrix_file;

  // Validate that the ftype is supported
  {
    bool supported = false;
    switch (ftype) {
    case GGML_FTYPE_MOSTLY_Q4_0:
    case GGML_FTYPE_MOSTLY_Q4_1:
    case GGML_FTYPE_MOSTLY_Q5_0:
    case GGML_FTYPE_MOSTLY_Q5_1:
    case GGML_FTYPE_MOSTLY_Q8_0:
    case GGML_FTYPE_MOSTLY_Q2_K:
    case GGML_FTYPE_MOSTLY_Q3_K:
    case GGML_FTYPE_MOSTLY_Q4_K:
    case GGML_FTYPE_MOSTLY_Q5_K:
    case GGML_FTYPE_MOSTLY_Q6_K:
    case GGML_FTYPE_MOSTLY_Q2_K_M:
    case GGML_FTYPE_MOSTLY_Q3_K_M:
    case GGML_FTYPE_MOSTLY_Q4_K_M:
    case GGML_FTYPE_MOSTLY_Q5_K_M:
    case GGML_FTYPE_MOSTLY_IQ1_M:
    case GGML_FTYPE_MOSTLY_IQ1_S:
    case GGML_FTYPE_MOSTLY_IQ1_S_M:
    case GGML_FTYPE_MOSTLY_IQ2_S:
    case GGML_FTYPE_MOSTLY_IQ2_XS:
    case GGML_FTYPE_MOSTLY_IQ2_XXS:
    case GGML_FTYPE_MOSTLY_IQ2_XXS_M:
    case GGML_FTYPE_MOSTLY_IQ3_XXS:
    case GGML_FTYPE_MOSTLY_IQ3_S:
    case GGML_FTYPE_MOSTLY_IQ4_NL:
    case GGML_FTYPE_MOSTLY_IQ4_XS:
      supported = true;
      break;
    default:
      break;
    }
    if (!supported) {
      fprintf(stderr, "%s: unsupported model type %d\n", __func__, ftype);
      return false;
    }
  }

  printf("%s: quantizing with strategy ftype=%d%s ..\n", __func__, ftype,
         opts.pure ? " (--pure)" : "");
  if (opts.token_embedding_type != GGML_TYPE_COUNT) {
    printf("%s: token-embedding-type override: %s\n", __func__,
           ggml_type_name(opts.token_embedding_type));
  }
  if (opts.output_tensor_type != GGML_TYPE_COUNT) {
    printf("%s: output-tensor-type override:   %s\n", __func__,
           ggml_type_name(opts.output_tensor_type));
  }

  // Load importance matrix (activation-aware quantization)
  IMatrixCollector imatrix_collector;
  if (!imatrix_file.empty()) {
      if (!imatrix_collector.load(imatrix_file)) {
          fprintf(stderr, "%s: failed to load imatrix from %s\n", __func__,
                  imatrix_file.c_str());
          return false;
      }
      printf("%s: loaded imatrix with %zu entries from %s\n", __func__,
             imatrix_collector.stats.size(), imatrix_file.c_str());
  }

  size_t total_size_org = 0;
  size_t total_size_new = 0;

  std::vector<std::thread> workers;
  workers.reserve(nthread);

  int idx = 0;
  std::vector<uint8_t> work;

  struct gguf_context *ctx_out = gguf_init_empty();

  // copy the KV pairs from the input file
  gguf_set_kv(ctx_out, gguf_input);
  gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
  gguf_set_val_u32(ctx_out, "general.file_type", ftype);

  const int n_tensors = gguf_get_n_tensors(gguf_input);

  // open model gguf file for reading tensor data
  auto fin = std::ifstream(fname_inp, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: cannot open model file for loading tensors\n",
            __func__);
    return false;
  }

  // Add tensor descriptors to output context (types will be updated during
  // quantization)
  for (int i = 0; i < n_tensors; ++i) {
    const std::string name = gguf_get_tensor_name(gguf_input, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx, name.c_str());
    gguf_add_tensor(ctx_out, tensor);
  }

  std::ofstream fout;

  auto close_ofstream = [&]() {
    if (fout.is_open()) {
      fout.seekp(0);
      std::vector<uint8_t> data(gguf_get_meta_size(ctx_out));
      gguf_get_meta_data(ctx_out, data.data());
      fout.write((const char *)data.data(), data.size());
      fout.close();
    }
  };

  auto new_ofstream = [&]() {
    GGML_ASSERT(ctx_out && "Found uninitialized gguf_context");
    fout = std::ofstream(fname_out, std::ios::binary);
    fout.exceptions(std::ofstream::failbit);
    const size_t meta_size = gguf_get_meta_size(ctx_out);
    ::zeros(fout, meta_size);
  };

  new_ofstream();

  for (int i = 0; i < n_tensors; ++i) {
    const std::string name = gguf_get_tensor_name(gguf_input, i);
    const size_t offset = gguf_get_tensor_offset(gguf_input, i) +
                          gguf_get_data_offset(gguf_input);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx, name.c_str());

    if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) {
      fprintf(
          stderr,
          "%s: unsupported tensor type %d (%s) for quantization (tensor: %s)\n",
          __func__, tensor->type, ggml_type_name((ggml_type)tensor->type),
          name.c_str());
      close_ofstream();
      return false;
    }

    fin.seekg(offset, std::ios::beg);
    if (!fin) {
      fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__,
              name.c_str());
      close_ofstream();
      return false;
    }

    // read tensor data
    int num_bytes = ggml_nbytes(tensor);
    fin.read(reinterpret_cast<char *>(tensor->data), num_bytes);

    printf("[%4d/%4d] %48s - [%s], type = %6s, ", ++idx, n_tensors,
           name.c_str(), format_tensor_shape(tensor).c_str(),
           ggml_type_name(tensor->type));

    // Determine per-tensor target type (handles mixed-precision strategies)
    enum ggml_type new_type =
        rs_get_qtype_for_tensor(opts, name, tensor);
    bool quantize = (new_type != tensor->type) && ggml_is_quantized(new_type);

    if (!quantize && new_type == tensor->type) {
      // Check if rs_get_qtype_for_tensor indicated a skip due to alignment
      // by returning the original type for a weight tensor that would normally
      // be quantized
      bool is_weight = name.rfind("weight") == name.size() - 6;
      if (is_weight && ggml_n_dims(tensor) >= 2 &&
          name.find("_norm.weight") == std::string::npos &&
          name.find(".norm.weight") == std::string::npos) {
        bool in_skip = false;
        for (const auto &s : to_skip) {
          if (std::regex_match(name, std::regex(s))) {
            in_skip = true;
            break;
          }
        }
        if (!in_skip) {
          // Skipped due to alignment — note it
          printf("(skipped: alignment) ");
        }
      }
    }

    void *new_data;
    size_t new_size;

    // Imatrix lookup for activation-aware quantization
    const float *imatrix = nullptr;
    if (quantize && !imatrix_file.empty()) {
        imatrix = imatrix_collector.get_imatrix(name, tensor->ne[0]);
    }

    // IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S require imatrix data.
    // Fall back to Q4_K for tensors missing imatrix (e.g. embeddings that
    // use GET_ROWS instead of MUL_MAT, so have no activation statistics).
    if (quantize && !imatrix && ggml_quantize_requires_imatrix(new_type)) {
        if (tensor->ne[0] % ggml_blck_size(GGML_TYPE_Q4_K) == 0) {
            new_type = GGML_TYPE_Q4_K;
            printf("(no imatrix, fallback Q4_K) ");
        } else {
            new_type = tensor->type;
            quantize = false;
            printf("(no imatrix, keeping %s) ", ggml_type_name(tensor->type));
        }
    }

    if (!quantize) {
      new_type = tensor->type;
      new_data = tensor->data;
      new_size = ggml_nbytes(tensor);
      printf("size = %8.3f MB\n", ggml_nbytes(tensor) / 1024.0 / 1024.0);
    } else {
      const int64_t nelements = ggml_nelements(tensor);

      float *f32_data;
      std::vector<float> f32_converted; // only used when input is F16

      if (tensor->type == GGML_TYPE_F32) {
        f32_data = (float *)tensor->data;
      } else {
        // F16 → F32: need separate buffer since sizes differ
        f32_converted.resize(nelements);
        auto data_f16 = (const ggml_fp16_t *)tensor->data;
        for (int64_t j = 0; j < nelements; ++j) {
          f32_converted[j] = ggml_fp16_to_fp32(data_f16[j]);
        }
        f32_data = f32_converted.data();
      }

      printf("converting to %s .. ", ggml_type_name(new_type));
      fflush(stdout);

      if (work.size() < (size_t)nelements * 4) {
        work.resize(nelements * 4); // upper bound on size
      }
      new_data = work.data();

      const int64_t n_per_row = tensor->ne[0];
      const int64_t nrows = tensor->ne[1];

      static const int64_t min_chunk_size = 32 * 512;
      const int64_t chunk_size =
          (n_per_row >= min_chunk_size
               ? n_per_row
               : n_per_row * ((min_chunk_size + n_per_row - 1) / n_per_row));

      const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
      const int64_t nchunk = (nelements_matrix + chunk_size - 1) / chunk_size;
      const int64_t nthread_use =
          nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk))
                      : 1;

      // quantize each expert/slice separately
      new_size = 0;
      for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
        const float *f32_data_03 = f32_data + i03 * nelements_matrix;
        void *new_data_03 =
            (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
        const float *imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

        new_size += rs_tensor_quantize_internal(
            new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row,
            imatrix_03, workers, nthread_use);
      }
      printf("size = %8.2f MiB -> %8.2f MiB\n",
             ggml_nbytes(tensor) / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    total_size_org += ggml_nbytes(tensor);
    total_size_new += new_size;

    // update the gguf meta data
    gguf_set_tensor_type(ctx_out, name.c_str(), new_type);
    gguf_set_tensor_data(ctx_out, name.c_str(), new_data);

    // write tensor data + padding
    fout.write((const char *)new_data, new_size);
    zeros(fout, GGML_PAD(new_size, GGUF_DEFAULT_ALIGNMENT) - new_size);
  }

  close_ofstream();
  gguf_free(ctx_out);

  auto ftype_label = [](ggml_ftype f) -> const char * {
    switch (f) {
    case GGML_FTYPE_MOSTLY_Q2_K_M:    return "Q2_K_M";
    case GGML_FTYPE_MOSTLY_Q3_K_M:    return "Q3_K_M";
    case GGML_FTYPE_MOSTLY_Q4_K_M:    return "Q4_K_M";
    case GGML_FTYPE_MOSTLY_Q5_K_M:    return "Q5_K_M";
    case GGML_FTYPE_MOSTLY_IQ1_S_M:   return "IQ1_S_M";
    case GGML_FTYPE_MOSTLY_IQ2_XXS_M: return "IQ2_XXS_M";
    default:                          return ggml_type_name(ggml_ftype_to_ggml_type(f));
    }
  };

  printf("%s: model size  = %8.2f MB\n", __func__,
         total_size_org / 1024.0 / 1024.0);
  printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__,
         total_size_new / 1024.0 / 1024.0, ftype, ftype_label(ftype));

  return true;
}

bool rapid_speech_ggml_quantize(ggml_context *ctx, gguf_context *gguf_input,
                                const std::string &fname_inp,
                                const std::string &fname_out,
                                const ggml_ftype ftype, const int nthread,
                                const std::vector<std::string> &to_quant,
                                const std::vector<std::string> &to_skip,
                                const std::string &imatrix_file) {
  rs_quantize_options opts;
  opts.ftype = ftype;
  opts.nthread = nthread;
  opts.to_quant = to_quant;
  opts.to_skip = to_skip;
  opts.imatrix_file = imatrix_file;
  return rapid_speech_ggml_quantize(ctx, gguf_input, fname_inp, fname_out, opts);
}
