#include "llm_model.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

// ============================================
// Logging Helpers
// ============================================

#define LLM_LOG_INFO(fmt, ...) std::printf("[LLM] " fmt "\n", ##__VA_ARGS__)
#define LLM_LOG_ERROR(fmt, ...)                                                \
  std::fprintf(stderr, "[LLM] Error: " fmt "\n", ##__VA_ARGS__)
#define LLM_LOG_WARN(fmt, ...)                                                 \
  std::fprintf(stderr, "[LLM] Warning: " fmt "\n", ##__VA_ARGS__)

// ============================================
// Model Registry
// ============================================

static std::map<std::string, model_loader_fn> &get_model_registry() {
  static std::map<std::string, model_loader_fn> registry;
  return registry;
}


// GPT-2 style byte-to-unicode mapping table
// Maps each byte (0x00-0xFF) to its Unicode representation
// This ensures all tokens are valid printable Unicode strings
static const char *BYTE_ENCODER[256] = {
    "\xc4\x80", "\xc4\x81", "\xc4\x82", "\xc4\x83", "\xc4\x84", "\xc4\x85",
    "\xc4\x86", "\xc4\x87", "\xc4\x88", "\xc4\x89", "\xc4\x8a", "\xc4\x8b",
    "\xc4\x8c", "\xc4\x8d", "\xc4\x8e", "\xc4\x8f", "\xc4\x90", "\xc4\x91",
    "\xc4\x92", "\xc4\x93", "\xc4\x94", "\xc4\x95", "\xc4\x96", "\xc4\x97",
    "\xc4\x98", "\xc4\x99", "\xc4\x9a", "\xc4\x9b", "\xc4\x9c", "\xc4\x9d",
    "\xc4\x9e", "\xc4\x9f", "\xc4\xa0", "\x21",     "\x22",     "\x23",
    "\x24",     "\x25",     "\x26",     "\x27",     "\x28",     "\x29",
    "\x2a",     "\x2b",     "\x2c",     "\x2d",     "\x2e",     "\x2f",
    "\x30",     "\x31",     "\x32",     "\x33",     "\x34",     "\x35",
    "\x36",     "\x37",     "\x38",     "\x39",     "\x3a",     "\x3b",
    "\x3c",     "\x3d",     "\x3e",     "\x3f",     "\x40",     "\x41",
    "\x42",     "\x43",     "\x44",     "\x45",     "\x46",     "\x47",
    "\x48",     "\x49",     "\x4a",     "\x4b",     "\x4c",     "\x4d",
    "\x4e",     "\x4f",     "\x50",     "\x51",     "\x52",     "\x53",
    "\x54",     "\x55",     "\x56",     "\x57",     "\x58",     "\x59",
    "\x5a",     "\x5b",     "\x5c",     "\x5d",     "\x5e",     "\x5f",
    "\x60",     "\x61",     "\x62",     "\x63",     "\x64",     "\x65",
    "\x66",     "\x67",     "\x68",     "\x69",     "\x6a",     "\x6b",
    "\x6c",     "\x6d",     "\x6e",     "\x6f",     "\x70",     "\x71",
    "\x72",     "\x73",     "\x74",     "\x75",     "\x76",     "\x77",
    "\x78",     "\x79",     "\x7a",     "\x7b",     "\x7c",     "\x7d",
    "\x7e",     "\xc4\xa1", "\xc4\xa2", "\xc4\xa3", "\xc4\xa4", "\xc4\xa5",
    "\xc4\xa6", "\xc4\xa7", "\xc4\xa8", "\xc4\xa9", "\xc4\xaa", "\xc4\xab",
    "\xc4\xac", "\xc4\xad", "\xc4\xae", "\xc4\xaf", "\xc4\xb0", "\xc4\xb1",
    "\xc4\xb2", "\xc4\xb3", "\xc4\xb4", "\xc4\xb5", "\xc4\xb6", "\xc4\xb7",
    "\xc4\xb8", "\xc4\xb9", "\xc4\xba", "\xc4\xbb", "\xc4\xbc", "\xc4\xbd",
    "\xc4\xbe", "\xc4\xbf", "\xc5\x80", "\xc5\x81", "\xc5\x82", "\xc2\xa1",
    "\xc2\xa2", "\xc2\xa3", "\xc2\xa4", "\xc2\xa5", "\xc2\xa6", "\xc2\xa7",
    "\xc2\xa8", "\xc2\xa9", "\xc2\xaa", "\xc2\xab", "\xc2\xac", "\xc5\x83",
    "\xc2\xae", "\xc2\xaf", "\xc2\xb0", "\xc2\xb1", "\xc2\xb2", "\xc2\xb3",
    "\xc2\xb4", "\xc2\xb5", "\xc2\xb6", "\xc2\xb7", "\xc2\xb8", "\xc2\xb9",
    "\xc2\xba", "\xc2\xbb", "\xc2\xbc", "\xc2\xbd", "\xc2\xbe", "\xc2\xbf",
    "\xc3\x80", "\xc3\x81", "\xc3\x82", "\xc3\x83", "\xc3\x84", "\xc3\x85",
    "\xc3\x86", "\xc3\x87", "\xc3\x88", "\xc3\x89", "\xc3\x8a", "\xc3\x8b",
    "\xc3\x8c", "\xc3\x8d", "\xc3\x8e", "\xc3\x8f", "\xc3\x90", "\xc3\x91",
    "\xc3\x92", "\xc3\x93", "\xc3\x94", "\xc3\x95", "\xc3\x96", "\xc3\x97",
    "\xc3\x98", "\xc3\x99", "\xc3\x9a", "\xc3\x9b", "\xc3\x9c", "\xc3\x9d",
    "\xc3\x9e", "\xc3\x9f", "\xc3\xa0", "\xc3\xa1", "\xc3\xa2", "\xc3\xa3",
    "\xc3\xa4", "\xc3\xa5", "\xc3\xa6", "\xc3\xa7", "\xc3\xa8", "\xc3\xa9",
    "\xc3\xaa", "\xc3\xab", "\xc3\xac", "\xc3\xad", "\xc3\xae", "\xc3\xaf",
    "\xc3\xb0", "\xc3\xb1", "\xc3\xb2", "\xc3\xb3", "\xc3\xb4", "\xc3\xb5",
    "\xc3\xb6", "\xc3\xb7", "\xc3\xb8", "\xc3\xb9", "\xc3\xba", "\xc3\xbb",
    "\xc3\xbc", "\xc3\xbd", "\xc3\xbe", "\xc3\xbf",
};

/**
 * Pre-process input bytes using GPT-2 byte-to-unicode mapping
 * This converts raw bytes to the format used in the vocabulary
 */
static std::string encode_bytes(const std::string &text) {
  std::string encoded;
  for (unsigned char c : text) {
    encoded += BYTE_ENCODER[c];
  }
  return encoded;
}

/**
 * Decode byte-level BPE tokens back to original bytes
 * Inverse of encode_bytes()
 */
static std::string decode_bytes(const std::string &text) {
  std::string decoded;
  size_t pos = 0;
  while (pos < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[pos]);
    size_t char_len = 1;

    // Determine UTF-8 character length
    if ((c & 0x80) == 0x00) {
      char_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      char_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      char_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      char_len = 4;
    }

    if (pos + char_len > text.size()) {
      // Invalid UTF-8, skip
      pos++;
      continue;
    }

    // Look up the byte value
    bool found = false;
    for (int byte_val = 0; byte_val < 256; byte_val++) {
      const char *encoded = BYTE_ENCODER[byte_val];
      size_t enc_len = strlen(encoded);
      if (enc_len == char_len &&
          text.compare(pos, char_len, encoded, enc_len) == 0) {
        decoded += static_cast<char>(byte_val);
        pos += char_len;
        found = true;
        break;
      }
    }

    if (!found) {
      // Not a byte encoding, keep as-is (for normal ASCII/Unicode)
      decoded += text.substr(pos, char_len);
      pos += char_len;
    }
  }
  return decoded;
}

void llm_register_model(const std::string &arch_name, model_loader_fn loader) {
  get_model_registry()[arch_name] = loader;
}

llm_model_ptr llm_load_model(struct gguf_context *ctx_gguf,
                             ggml_backend_t backend) {
  int64_t arch_key = gguf_find_key(ctx_gguf, "general.architecture");
  if (arch_key == -1) {
    LLM_LOG_ERROR("GGUF missing 'general.architecture' key");
    return nullptr;
  }

  std::string arch_name = gguf_get_val_str(ctx_gguf, arch_key);
  LLM_LOG_INFO("Loading model architecture: %s", arch_name.c_str());

  auto it = get_model_registry().find(arch_name);
  if (it == get_model_registry().end()) {
    LLM_LOG_ERROR("Unknown architecture: %s", arch_name.c_str());
    return nullptr;
  }

  return it->second(ctx_gguf, backend);
}

// ============================================
// llm_vocab Implementation
// ============================================

const llm_vocab::token &llm_vocab::get_token(int32_t id) const {
  static token empty_token;
  auto it = tokens_.find(id);
  if (it != tokens_.end()) {
    return it->second;
  }
  return empty_token;
}

std::string llm_vocab::decode(int32_t id) const {
  std::string token = get_token(id).text;
  return decode_bytes(token);
}

void llm_vocab::add_special_token(const std::string &text, int32_t id) {
  special_tokens_[text] = id;
  if (id >= 0) {
    tokens_[id].text = text;
  }
  token_to_id_[text] = id;
}

int32_t llm_vocab::get_special_token(const std::string &name) const {
  auto it = special_tokens_.find(name);
  if (it != special_tokens_.end()) {
    return it->second;
  }
  return -1;
}

void llm_vocab::load_qwen3_special_tokens() {
  // Qwen3 special tokens - escape characters preserved as provided
  add_special_token("<tool_call>", 151643);
  add_special_token("<|endoftext|>", 151643);
  add_special_token("<|im_start|>", 151644);
  add_special_token("<|im_end|>", 151645);
  add_special_token("<|object_ref_start|>", 151646);
  add_special_token("<|object_ref_end|>", 151647);
  add_special_token("<|box_start|>", 151648);
  add_special_token("<|box_end|>", 151649);
  add_special_token("<|quad_start|>", 151650);
  add_special_token("<|quad_end|>", 151651);
  add_special_token("<|vision_start|>", 151652);
  add_special_token("<|vision_end|>", 151653);
  add_special_token("<|vision_pad|>", 151654);
  add_special_token("<|image_pad|>", 151655);
  add_special_token("<|video_pad|>", 151656);
  add_special_token("<tool_call>", 151657);
  add_special_token("</tool_call>", 151658);
  add_special_token("<|fim_prefix|>", 151659);
  add_special_token("<|fim_middle|>", 151660);
  add_special_token("<|fim_suffix|>", 151661);
  add_special_token("<|fim_pad|>", 151662);
  add_special_token("<|repo_name|>", 151663);
  add_special_token("<|file_sep|>", 151664);
  add_special_token("<tool_response>", 151665);
  add_special_token("</tool_response>", 151666);
  add_special_token("<think>", 151667);
  add_special_token("</think>", 151668);
  // Speech tokens for FunASR Nano
  add_special_token("<|startofspeech|>", 151669);
  add_special_token("<|endofspeech|>", 151670);
}

bool llm_vocab::load_from_gguf(struct gguf_context *ctx_gguf) {
  int token_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
  if (token_idx == -1) {
    LLM_LOG_WARN("No tokenizer tokens found");
    return false;
  }

  int n_tokens = gguf_get_arr_n(ctx_gguf, token_idx);

  tokens_.clear();
  token_to_id_.clear();
  special_tokens_.clear();

  for (int i = 0; i < n_tokens; ++i) {
    token tok;
    tok.text = gguf_get_arr_str(ctx_gguf, token_idx, i);
    tok.score = 0.0f;
    tok.attr = 0;
    tokens_[i] = tok;
    token_to_id_[tok.text] = i;
  }

  int score_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.scores");
  if (score_idx != -1) {
    const float *scores = (const float *)gguf_get_arr_data(ctx_gguf, score_idx);
    for (int i = 0; i < n_tokens && i < gguf_get_arr_n(ctx_gguf, score_idx);
         ++i) {
      tokens_[i].score = scores[i];
    }
  }

  int type_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.token_type");
  if (type_idx != -1) {
    const int32_t *types =
        (const int32_t *)gguf_get_arr_data(ctx_gguf, type_idx);
    for (int i = 0; i < n_tokens && i < gguf_get_arr_n(ctx_gguf, type_idx);
         ++i) {
      tokens_[i].attr = static_cast<uint8_t>(types[i]);
    }
  }

  // Load BPE merges
  int merges_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
  if (merges_idx != -1) {
    bpe_merges_.clear();
    bpe_ranks_.clear();
    int n_merges = gguf_get_arr_n(ctx_gguf, merges_idx);
    for (int i = 0; i < n_merges; ++i) {
      std::string merge = gguf_get_arr_str(ctx_gguf, merges_idx, i);
      // Merge format: "left right" (space-separated)
      size_t space_pos = merge.find(' ');
      if (space_pos != std::string::npos) {
        std::string left = merge.substr(0, space_pos);
        std::string right = merge.substr(space_pos + 1);
        bpe_merges_.emplace_back(left, right);
        bpe_ranks_[{left, right}] = static_cast<int32_t>(i);
      }
    }
    LLM_LOG_INFO("Loaded %d BPE merges", n_merges);
  }

  auto find_special_token = [&](const char *name) -> int32_t {
    auto it = token_to_id_.find(name);
    if (it != token_to_id_.end()) {
      special_tokens_[name] = it->second;
      return it->second;
    }
    return -1;
  };

  token_bos_ = find_special_token("<s>");
  if (token_bos_ == -1)
    token_bos_ = find_special_token("<bos>");
  if (token_bos_ == -1)
    token_bos_ = 1;

  token_eos_ = find_special_token("<|im_end|>");
  if (token_eos_ == -1)
    token_eos_ = find_special_token("</s>");
  if (token_eos_ == -1)
    token_eos_ = find_special_token("<eos>");
  if (token_eos_ == -1)
    token_eos_ = 2;

  token_unk_ = find_special_token("<|unk|>");
  if (token_unk_ == -1)
    token_unk_ = find_special_token("<unk>");
  if (token_unk_ == -1)
    token_unk_ = 3;

  token_pad_ = find_special_token("<pad>");
  if (token_pad_ == -1)
    token_pad_ = 0;

  LLM_LOG_INFO("Vocabulary loaded: %d tokens, BOS=%d, EOS=%d, UNK=%d, PAD=%d",
               n_tokens, token_bos_, token_eos_, token_unk_, token_pad_);
  return true;
}

void llm_vocab::build_token_map() const {
  if (token_map_built_)
    return;
  token_to_id_.clear();
  for (const auto &kv : tokens_) {
    token_to_id_[kv.second.text] = kv.first;
  }
  // Re-apply special tokens to override any conflicts
  for (const auto &kv : special_tokens_) {
    token_to_id_[kv.first] = kv.second;
  }
  token_map_built_ = true;
}


std::vector<std::string>
llm_vocab::tokenize_to_chars(const std::string &text) const {
  // Split text into unicode characters (not used for byte-level BPE)
  std::vector<std::string> tokens;
  size_t pos = 0;
  while (pos < text.size()) {
    unsigned char c = static_cast<unsigned char>(text[pos]);
    size_t char_len = 1;
    if ((c & 0x80) == 0x00) {
      char_len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      char_len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      char_len = 3;
    } else if ((c & 0xF8) == 0xF0) {
      char_len = 4;
    }
    char_len = std::min(char_len, text.size() - pos);
    tokens.push_back(text.substr(pos, char_len));
    pos += char_len;
  }
  return tokens;
}

std::vector<int32_t> llm_vocab::tokenize(const std::string &text,
                                         bool add_bos) const {
  if (!token_map_built_) {
    build_token_map();
  }

  std::vector<int32_t> tokens;
  if (add_bos && token_bos_ >= 0) {
    tokens.push_back(token_bos_);
  }

  // Preprocess text using GPT-2 byte-to-unicode mapping
  std::string encoded_text = encode_bytes(text);

  // Step 1: Extract special tokens first (they should not be split or merged)
  std::vector<std::pair<std::string, bool>> segments; // (text, is_special)
  size_t pos = 0;

  while (pos < encoded_text.size()) {
    // Try to find the longest matching special token starting at current
    // position
    std::string best_special;
    size_t best_len = 0;

    for (const auto &kv : special_tokens_) {
      const std::string &special = kv.first;
      if (encoded_text.substr(
              pos, std::min(special.size(), encoded_text.size() - pos)) ==
          special) {
        if (special.size() > best_len) {
          best_len = special.size();
          best_special = special;
        }
      }
    }

    if (best_len > 0) {
      // Found a special token
      if (!segments.empty() && segments.back().second) {
        // Merge with previous special segment (shouldn't happen, but safe)
        segments.back().first += best_special;
      } else {
        segments.emplace_back(best_special, true);
      }
      pos += best_len;
    } else {
      // Not a special token - collect as normal text
      size_t normal_start = pos;
      while (pos < encoded_text.size()) {
        // Check if next position starts a special token
        bool found_special = false;
        for (const auto &kv : special_tokens_) {
          const std::string &special = kv.first;
          if (!special.empty() && pos < encoded_text.size() &&
              encoded_text.substr(
                  pos, std::min(special.size(), encoded_text.size() - pos)) ==
                  special) {
            found_special = true;
            break;
          }
        }
        if (found_special)
          break;
        pos++;
      }
      if (pos > normal_start) {
        segments.emplace_back(
            encoded_text.substr(normal_start, pos - normal_start), false);
      }
    }
  }

  // Step 2: Process each segment
  for (const auto &segment : segments) {
    if (segment.second) {
      // Special token - directly look up ID
      int32_t token_id = find_token_id(segment.first);
      if (token_id >= 0) {
        tokens.push_back(token_id);
      } else {
        tokens.push_back(token_unk_);
      }
    } else {
      // Normal text - greedy longest-match tokenization on encoded text
      // The text has already been converted to vocab format via encode_bytes()

      size_t seg_pos = 0;
      while (seg_pos < segment.first.size()) {
        int32_t best_id = -1;
        size_t best_len = 0;

        // Try to find the longest matching token (up to 12 bytes for
        // efficiency)
        size_t max_len = std::min(segment.first.size() - seg_pos, size_t(12));

        for (size_t len = max_len; len > 0; --len) {
          std::string candidate = segment.first.substr(seg_pos, len);
          int32_t token_id = find_token_id(candidate);

          if (token_id >= 0) {
            best_id = token_id;
            best_len = len;
            break;
          }
        }

        if (best_id >= 0) {
          tokens.push_back(best_id);
          seg_pos += best_len;
        } else {
          // Unknown token - skip one character and continue
          tokens.push_back(token_unk_);
          seg_pos += 1;
        }
      }
    }
  }

  return tokens;
}

std::string llm_vocab::detokenize(const std::vector<int32_t> &tokens) const {
  std::string result;
  for (int32_t id : tokens) {
    if (id >= 0) {
      auto it = tokens_.find(id);
      if (it != tokens_.end()) {
        // Decode byte-level BPE tokens back to original bytes
        result += decode_bytes(it->second.text);
      }
    }
  }
  return result;
}

int32_t llm_vocab::find_token_id(const std::string &text) const {
  if (!token_map_built_) {
    build_token_map();
  }

  auto it = token_to_id_.find(text);
  if (it != token_to_id_.end()) {
    return it->second;
  }
  return -1;
}

bool llm_model::load_from_file(const std::string &file_path,
                               ggml_backend_t backend) {
  LLM_LOG_INFO("Loading model from: %s", file_path.c_str());

  struct gguf_init_params params = {/*.no_alloc = */ true,
                                    /*.ctx      = */ &ctx_weights_};

  struct gguf_context *ctx_gguf =
      gguf_init_from_file(file_path.c_str(), params);
  if (!ctx_gguf) {
    LLM_LOG_ERROR("Failed to load GGUF: %s", file_path.c_str());
    return false;
  }

  bool success = load_from_gguf(ctx_gguf, nullptr, backend, file_path);
  gguf_free(ctx_gguf);
  return success;
}

bool llm_model::load_from_gguf(struct gguf_context *ctx_gguf,
                               struct ggml_context *gguf_data,
                               ggml_backend_t backend,
                               const std::string &file_path) {
  auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count();

  if (!load_hparams(ctx_gguf)) {
    return false;
  }

  load_vocab(ctx_gguf);

  if (gguf_data) {
    if (!load_tensors_from_gguf_data(ctx_gguf, gguf_data, backend)) {
      ggml_backend_buffer_free(buffer_weights_);
      return false;
    }
  } else {
    if (!ctx_weights_) {
      struct ggml_init_params params = {
          /*.mem_size   =*/ggml_tensor_overhead() *
                  gguf_get_n_tensors(ctx_gguf) +
              (1 << 20),
          /*.mem_buffer =*/nullptr,
          /*.no_alloc   =*/true};
      ctx_weights_ = ggml_init(params);
      if (!ctx_weights_) {
        LLM_LOG_ERROR("Failed to initialize weights context");
        return false;
      }
    }

    buffer_weights_ = ggml_backend_alloc_ctx_tensors(ctx_weights_, backend);
    if (!buffer_weights_) {
      LLM_LOG_ERROR("Failed to allocate weight buffers");
      return false;
    }

    if (!load_tensors_from_file(ctx_gguf, file_path, backend)) {
      ggml_backend_buffer_free(buffer_weights_);
      return false;
    }
  }

  auto end_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();

  LLM_LOG_INFO("Model loaded in %.2f ms", (end_us - start_us) / 1000.0);
  return true;
}

bool llm_model::load_hparams(struct gguf_context *ctx_gguf) {
  std::string arch_key = "llm";
  int arch_idx = gguf_find_key(ctx_gguf, "general.architecture");
  std::string arch_str =
      (arch_idx != -1) ? gguf_get_val_str(ctx_gguf, arch_idx) : "unknown";

  if (arch_str == "qwen3" || arch_str == "FunASRNano") {
    hparams_.arch = LLM_ARCH_QWEN3;
    arch_key = arch_str == "FunASRNano" ? "qwen3" : arch_str;
  } else if (arch_str == "llama") {
    hparams_.arch = LLM_ARCH_LLAMA;
    arch_key = "llama";
  } else {
    hparams_.arch = LLM_ARCH_UNKNOWN;
  }

  auto get_i32 = [&](const char *key, int32_t default_val) -> int32_t {
    std::string full_key;
    const char *percent = std::strstr(key, "%s");
    if (percent) {
      full_key =
          std::string(key, percent - key) + arch_key + std::string(percent + 2);
      key = full_key.c_str();
    }
    int idx = gguf_find_key(ctx_gguf, key);
    return (idx != -1) ? gguf_get_val_i32(ctx_gguf, idx) : default_val;
  };

  auto get_f32 = [&](const char *key, float default_val) -> float {
    std::string full_key;
    const char *percent = std::strstr(key, "%s");
    if (percent) {
      full_key =
          std::string(key, percent - key) + arch_key + std::string(percent + 2);
      key = full_key.c_str();
    }
    int idx = gguf_find_key(ctx_gguf, key);
    return (idx != -1) ? gguf_get_val_f32(ctx_gguf, idx) : default_val;
  };

  hparams_.n_vocab = get_i32("tokenizer.vocab_size", 0);
  hparams_.n_embd = get_i32("%s.embedding_length", 0);
  hparams_.n_layer = get_i32("%s.block_count", 0);
  hparams_.n_head = get_i32("%s.attention.head_count", 0);
  hparams_.n_head_kv = get_i32("%s.attention.head_count_kv", hparams_.n_head);
  hparams_.n_ff = get_i32("%s.feed_forward_length", 0);
  hparams_.n_ctx_train = get_i32("%s.context_length", 4096);
  hparams_.head_dim = get_i32("%s.attention.key_length", 1024);
  hparams_.n_rot = get_i32("%s.rope.dimension_count", hparams_.head_dim);
  hparams_.f_norm_eps = get_f32("%s.attention.layer_norm_epsilon", 1e-5f);
  hparams_.f_norm_rms_eps =
      get_f32("%s.attention.layer_norm_rms_epsilon", hparams_.f_norm_eps);

  hparams_.rope_freq_base = get_i32("%s.rope.freq_base", 10000);
  hparams_.rope_freq_scale = get_f32("%s.rope.freq_scale", 1.0f);

  if (!hparams_.is_valid()) {
    LLM_LOG_ERROR("Invalid hyperparameters");
    return false;
  }

  LLM_LOG_INFO("Hyperparameters: n_embd=%d, n_layer=%d, n_head=%d, n_vocab=%d",
               hparams_.n_embd, hparams_.n_layer, hparams_.n_head,
               hparams_.n_vocab);
  return true;
}

bool llm_model::load_vocab(struct gguf_context *ctx_gguf) {
  return vocab_.load_from_gguf(ctx_gguf);
}

bool llm_model::load_metadata_from_gguf(struct gguf_context *ctx_gguf) {
  if (!load_hparams(ctx_gguf)) {
    return false;
  }
  load_vocab(ctx_gguf);
  return true;
}

bool llm_model::load_tensors_from_gguf_data(struct gguf_context *ctx_gguf,
                                            struct ggml_context *gguf_data,
                                            ggml_backend_t backend) {
  (void)backend;
  int n_tensors = gguf_get_n_tensors(ctx_gguf);

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    ggml_tensor *src = ggml_get_tensor(gguf_data, name);
    if (!src || !src->data)
      continue;

    ggml_tensor *dst = ggml_get_tensor(ctx_weights_, name);
    if (!dst)
      continue;

    size_t t_size = ggml_nbytes(dst);
    if (t_size == 0)
      continue;

    ggml_backend_tensor_set(dst, src->data, 0, t_size);
  }

  LLM_LOG_INFO("Loaded %d tensors from gguf_data", n_tensors);
  return true;
}

bool llm_model::load_tensors_from_file(struct gguf_context *ctx_gguf,
                                       const std::string &file_path,
                                       ggml_backend_t backend) {
  (void)backend;
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    LLM_LOG_ERROR("Failed to open model file: %s", file_path.c_str());
    return false;
  }

  size_t data_offset = gguf_get_data_offset(ctx_gguf);
  int n_tensors = gguf_get_n_tensors(ctx_gguf);
  std::vector<char> read_buf;

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    ggml_tensor *dst = ggml_get_tensor(ctx_weights_, name);
    if (!dst)
      continue;

    size_t t_offset = gguf_get_tensor_offset(ctx_gguf, i);
    size_t t_size = ggml_nbytes(dst);

    if (t_size == 0)
      continue;

    if (read_buf.size() < t_size) {
      read_buf.resize(t_size);
    }

    file.seekg(data_offset + t_offset, std::ios::beg);
    file.read(read_buf.data(), t_size);

    if (file.gcount() != static_cast<std::streamsize>(t_size)) {
      LLM_LOG_ERROR("Failed to read tensor: %s", name);
      return false;
    }

    ggml_backend_tensor_set(dst, read_buf.data(), 0, t_size);
  }

  LLM_LOG_INFO("Loaded %d tensors from file", n_tensors);
  return true;
}

uint64_t llm_model::n_elements() const {
  uint64_t total = 0;
  total += tok_embd_ ? ggml_nelements(tok_embd_) : 0;
  total += output_norm_ ? ggml_nelements(output_norm_) : 0;
  total += output_norm_b_ ? ggml_nelements(output_norm_b_) : 0;
  total += output_ ? ggml_nelements(output_) : 0;

  for (const auto &layer : layers_) {
    if (layer.attn_norm)
      total += ggml_nelements(layer.attn_norm);
    if (layer.wq)
      total += ggml_nelements(layer.wq);
    if (layer.wk)
      total += ggml_nelements(layer.wk);
    if (layer.wv)
      total += ggml_nelements(layer.wv);
    if (layer.wo)
      total += ggml_nelements(layer.wo);
    if (layer.ffn_gate)
      total += ggml_nelements(layer.ffn_gate);
    if (layer.ffn_up)
      total += ggml_nelements(layer.ffn_up);
    if (layer.ffn_down)
      total += ggml_nelements(layer.ffn_down);
  }
  return total;
}

size_t llm_model::memory_size() const {
  return buffer_weights_ ? ggml_backend_buffer_get_size(buffer_weights_) : 0;
}

bool llm_model::map_tensors_qwen3(
    std::map<std::string, ggml_tensor *> &tensors) {
  const std::string arch_prefix = "llm";

  auto find_tensor = [&](const std::string &suffix) -> ggml_tensor * {
    std::string full_name = arch_prefix + "." + suffix;
    auto it = tensors.find(full_name);
    if (it != tensors.end())
      return it->second;
    full_name = "qwen3." + suffix;
    it = tensors.find(full_name);
    return (it != tensors.end()) ? it->second : nullptr;
  };

  tok_embd_ = find_tensor("model.embed_tokens.weight");
  if (!tok_embd_)
    tok_embd_ = find_tensor("model.embed_tokens");

  output_norm_ = find_tensor("model.norm.weight");
  output_norm_b_ = find_tensor("model.norm.bias");
  output_ = find_tensor("lm_head.weight");

  layers_.resize(hparams_.n_layer);
  for (uint32_t il = 0; il < hparams_.n_layer; ++il) {
    llm_layer &layer = layers_[il];
    std::string layer_prefix =
        arch_prefix + ".model.layers." + std::to_string(il) + ".";
    std::string layer_prefix_alt = "qwen3.blk." + std::to_string(il) + ".";

    auto find_layer_tensor = [&](const std::string &suffix) -> ggml_tensor * {
      std::string name = layer_prefix + suffix;
      auto it = tensors.find(name);
      if (it != tensors.end())
        return it->second;
      name = layer_prefix_alt + suffix;
      it = tensors.find(name);
      return (it != tensors.end()) ? it->second : nullptr;
    };

    layer.attn_norm = find_layer_tensor("input_layernorm.weight");
    layer.attn_norm_b = find_layer_tensor("input_layernorm.bias");
    layer.wq = find_layer_tensor("self_attn.q_proj.weight");
    layer.wk = find_layer_tensor("self_attn.k_proj.weight");
    layer.wv = find_layer_tensor("self_attn.v_proj.weight");
    layer.wo = find_layer_tensor("self_attn.o_proj.weight");
    layer.attn_q_norm = find_layer_tensor("self_attn.q_norm.weight");
    layer.attn_q_norm_b = find_layer_tensor("self_attn.q_norm.bias");
    layer.attn_k_norm = find_layer_tensor("self_attn.k_norm.weight");
    layer.attn_k_norm_b = find_layer_tensor("self_attn.k_norm.bias");
    layer.ffn_norm = find_layer_tensor("post_attention_layernorm.weight");
    layer.ffn_norm_b = find_layer_tensor("post_attention_layernorm.bias");
    layer.ffn_gate = find_layer_tensor("mlp.gate_proj.weight");
    layer.ffn_up = find_layer_tensor("mlp.up_proj.weight");
    layer.ffn_down = find_layer_tensor("mlp.down_proj.weight");
  }

  if (!tok_embd_) {
    LLM_LOG_ERROR("Missing token embeddings");
    return false;
  }

  LLM_LOG_INFO("Mapped %zu Qwen3 layers", layers_.size());
  return true;
}

bool llm_model::map_tensors_llama(
    std::map<std::string, ggml_tensor *> &tensors) {
  const std::string arch_prefix = "llama";

  auto find_tensor = [&](const std::string &suffix) -> ggml_tensor * {
    std::string full_name = arch_prefix + "." + suffix;
    auto it = tensors.find(full_name);
    return (it != tensors.end()) ? it->second : nullptr;
  };

  tok_embd_ = find_tensor("token_embd.weight");
  output_norm_ = find_tensor("output_norm.weight");
  output_ = find_tensor("output.weight");

  layers_.resize(hparams_.n_layer);
  for (uint32_t il = 0; il < hparams_.n_layer; ++il) {
    llm_layer &layer = layers_[il];
    std::string layer_prefix = arch_prefix + ".blk." + std::to_string(il) + ".";

    auto find_layer_tensor = [&](const std::string &suffix) -> ggml_tensor * {
      std::string name = layer_prefix + suffix;
      auto it = tensors.find(name);
      return (it != tensors.end()) ? it->second : nullptr;
    };

    layer.attn_norm = find_layer_tensor("attn_norm.weight");
    layer.wq = find_layer_tensor("attn_q.weight");
    layer.wk = find_layer_tensor("attn_k.weight");
    layer.wv = find_layer_tensor("attn_v.weight");
    layer.wo = find_layer_tensor("attn_output.weight");
    layer.ffn_norm = find_layer_tensor("ffn_norm.weight");
    layer.ffn_gate = find_layer_tensor("ffn_gate.weight");
    layer.ffn_up = find_layer_tensor("ffn_up.weight");
    layer.ffn_down = find_layer_tensor("ffn_down.weight");
  }

  LLM_LOG_INFO("Mapped %zu Llama layers", layers_.size());
  return true;
}

void llm_model::print_info() const {
  LLM_LOG_INFO("=== Model Info ===");
  LLM_LOG_INFO("  Name       : %s", name_.c_str());
  LLM_LOG_INFO("  Architecture: %d", static_cast<int>(hparams_.arch));
  LLM_LOG_INFO("  Parameters : %.2f B", n_elements() / 1e9);
  LLM_LOG_INFO("  Memory     : %.2f GB", memory_size() / 1e9);
  LLM_LOG_INFO("  Vocab Size : %d", vocab_.size());
  LLM_LOG_INFO("==================");
}