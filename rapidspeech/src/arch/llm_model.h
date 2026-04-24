#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "llm_graph.h"
#include "llm_types.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

// ============================================
// Vocabulary
// ============================================

/**
 * Simplified vocabulary for tokenization
 */
class llm_vocab {
public:
  struct token {
    std::string text;
    float score;
    uint8_t attr; // Token attributes (normal, control, etc.)
  };

  llm_vocab() = default;
  ~llm_vocab() = default;

  // Load vocabulary from GGUF
  bool load_from_gguf(struct gguf_context *ctx_gguf);

  // Get token count
  uint32_t size() const { return static_cast<uint32_t>(tokens_.size()); }

  // Get token by ID
  const token &get_token(int32_t id) const;

  std::string decode(int32_t id) const;

  // Special tokens
  int32_t token_bos() const { return token_bos_; }
  int32_t token_eos() const { return token_eos_; }
  int32_t token_unk() const { return token_unk_; }
  int32_t token_pad() const { return token_pad_; }

  // Check token type
  bool is_eos(int32_t id) const { return id == token_eos_; }
  bool is_bos(int32_t id) const { return id == token_bos_; }

  // Tokenization (simple BPE)
  std::vector<int32_t> tokenize(const std::string &text,
                                bool add_bos = true) const;

  // Detokenization
  std::string detokenize(const std::vector<int32_t> &tokens) const;

  // Find token ID by text (for prompt embedding)
  int32_t find_token_id(const std::string &text) const;

  // Add special tokens
  void add_special_token(const std::string &text, int32_t id);

  // Get special token by name
  int32_t get_special_token(const std::string &name) const;

  // Load Qwen3 special tokens
  void load_qwen3_special_tokens();

private:
  // Token storage: ID -> token
  std::map<int32_t, token> tokens_;
  // Reverse lookup: text -> token ID (mutable for lazy building)
  mutable std::map<std::string, int32_t> token_to_id_;

  // Special token IDs
  int32_t token_bos_ = 1;
  int32_t token_eos_ = 2;
  int32_t token_unk_ = 3;
  int32_t token_pad_ = 0;

  // Special tokens map: name -> token ID
  std::map<std::string, int32_t> special_tokens_;

  llm_vocab_type type_ = LLM_VOCAB_TYPE_BPE;
  llm_vocab_pre_type pre_type_ = LLM_VOCAB_PRE_TYPE_DEFAULT;

  // BPE merges (optional) - loaded from tokenizer.ggml.merges
  std::vector<std::pair<std::string, std::string>> bpe_merges_;
  // BPE rank lookup: (left, right) -> rank
  mutable std::map<std::pair<std::string, std::string>, int32_t> bpe_ranks_;

  // Next available token ID
  int32_t next_token_id_ = 4;

  // Build token map for fast lookup
  mutable bool token_map_built_ = false;
  void build_token_map() const;

  // BPE tokenization helpers
  std::vector<std::string> tokenize_to_chars(const std::string &text) const;
  std::vector<std::string>
  bpe_merge(const std::vector<std::string> &tokens) const;
};

// ============================================
// Model Layer
// ============================================

/**
 * Transformer layer weights
 */
struct llm_layer {
  // Attention normalization
  ggml_tensor *attn_norm = nullptr;
  ggml_tensor *attn_norm_b = nullptr;

  // QKV projections
  ggml_tensor *wq = nullptr;
  ggml_tensor *wk = nullptr;
  ggml_tensor *wv = nullptr;
  ggml_tensor *wo = nullptr;

  // QK normalization (optional, for Qwen)
  ggml_tensor *attn_q_norm = nullptr;
  ggml_tensor *attn_q_norm_b = nullptr;
  ggml_tensor *attn_k_norm = nullptr;
  ggml_tensor *attn_k_norm_b = nullptr;

  // FFN normalization
  ggml_tensor *ffn_norm = nullptr;
  ggml_tensor *ffn_norm_b = nullptr;

  // FFN (SwiGLU for Qwen)
  ggml_tensor *ffn_gate = nullptr; // Gate projection
  ggml_tensor *ffn_up = nullptr;   // Up projection
  ggml_tensor *ffn_down = nullptr; // Down projection
};

// ============================================
// Model
// ============================================

/**
 * LLM Model
 *
 * Usage:
 * 1. Load model from GGUF: llm_model::load_from_file()
 * 2. Create graph builder for architecture
 * 3. Generate tokens
 */
class llm_model {
public:
  llm_model() = default;
  ~llm_model() = default;

  /**
   * Load model from GGUF file
   *
   * @param file_path Path to GGUF file
   * @param backend Backend for weight allocation
   * @return true if successful
   */
  bool load_from_file(const std::string &file_path, ggml_backend_t backend);

  /**
   * Load model from GGUF context (already loaded)
   */
  bool load_from_gguf(struct gguf_context *ctx_gguf,
                      struct ggml_context *gguf_data, ggml_backend_t backend,
                      const std::string &file_path = "");

  /**
   * Load hyperparameters and vocabulary only (for shared tensor loading)
   */
  bool load_metadata_from_gguf(struct gguf_context *ctx_gguf);

  // Get model info
  const std::string &name() const { return name_; }
  llm_arch arch() const { return hparams_.arch; }
  const llm_hparams &hparams() const { return hparams_; }
  const llm_vocab &vocab() const { return vocab_; }

  // Get internal context for tensor loading
  struct ggml_context *get_weight_ctx() const { return ctx_weights_; }

  // Get layer weights
  const llm_layer &layer(int32_t il) const { return layers_.at(il); }
  std::vector<llm_layer> &layers() { return layers_; }
  const std::vector<llm_layer> &layers() const { return layers_; }

  // Get output weights
  ggml_tensor *output_norm() const { return output_norm_; }
  ggml_tensor *output_norm_b() const { return output_norm_b_; }
  ggml_tensor *output() const { return output_; }
  ggml_tensor *tok_embd() const { return tok_embd_; }

  // Get model size
  uint64_t n_elements() const;
  size_t memory_size() const;

  // Print model info
  void print_info() const;

  // Architecture-specific tensor mapping (public for combined model loading)
  bool map_tensors_qwen3(std::map<std::string, ggml_tensor *> &tensors);
  bool map_tensors_llama(std::map<std::string, ggml_tensor *> &tensors);

private:
  std::string name_;
  llm_hparams hparams_;
  llm_vocab vocab_;

  // Weights
  ggml_tensor *tok_embd_ = nullptr;
  ggml_tensor *output_norm_ = nullptr;
  ggml_tensor *output_norm_b_ = nullptr;
  ggml_tensor *output_ = nullptr;

  std::vector<llm_layer> layers_;

  // GGUF resources
  struct ggml_context *ctx_weights_ = nullptr;
  ggml_backend_buffer *buffer_weights_ = nullptr;

  // Load helpers
  bool load_hparams(struct gguf_context *ctx_gguf);
  bool load_vocab(struct gguf_context *ctx_gguf);
  bool load_tensors_from_gguf_data(struct gguf_context *ctx_gguf,
                                   struct ggml_context *gguf_data,
                                   ggml_backend_t backend);
  bool load_tensors_from_file(struct gguf_context *ctx_gguf,
                              const std::string &file_path,
                              ggml_backend_t backend);
};

using llm_model_ptr = std::shared_ptr<llm_model>;

// ============================================
// Model Parameters
// ============================================

struct llm_model_params {
  ggml_type type_w = GGML_TYPE_F16; // Weight type
  bool use_mmap = true;             // Memory-map model file
  bool use_mlock = false;           // Lock model in memory
  int32_t n_gpu = 0;                // Number of GPUs to use
};

// ============================================
// Model Registry
// ============================================

/**
 * Register model architecture for auto-detection
 */
using model_loader_fn =
    std::function<llm_model_ptr(struct gguf_context *, ggml_backend_t)>;

void llm_register_model(const std::string &arch_name, model_loader_fn loader);
llm_model_ptr llm_load_model(struct gguf_context *ctx_gguf,
                             ggml_backend_t backend);
