#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "llm_kv_cache.h"
#include "llm_types.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

// ============================================
// Model Hyperparameters
// ============================================

/**
 * LLM hyperparameters
 *
 * Note: Only includes parameters needed for inference.
 * Training-specific parameters are omitted.
 */
struct llm_hparams {
  // Architecture
  llm_arch arch = LLM_ARCH_UNKNOWN;

  // Basic dimensions
  uint32_t n_vocab = 0;   // Vocabulary size
  uint32_t n_embd = 0;    // Hidden dimension
  uint32_t n_layer = 0;   // Number of layers
  uint32_t n_head = 0;    // Number of attention heads
  uint32_t n_head_kv = 0; // Number of KV heads (for GQA/MQA)
  uint32_t n_rot = 0;     // RoPE dimension
  uint32_t head_dim = 0;
  // FFN dimensions
  uint32_t n_ff = 0; // FFN intermediate dimension

  // Normalization
  float f_norm_eps = 1e-5f;
  float f_norm_rms_eps = 1e-6f;

  // RoPE
  uint32_t rope_freq_base = 10000;
  float rope_freq_scale = 1.0f;
  float rope_attn_factor = 1.0f;

  // Attention
  float f_attn_logit_softcapping = 0.0f;
  bool causal_attn = true;
  bool use_kq_norm = true;

  // Expert/MoE (optional)
  uint32_t n_expert = 0;
  uint32_t n_expert_used = 0;

  // Context
  uint32_t n_ctx_train = 0; // Context size model was trained on

  // Quantization (optional)
  bool quantized = false;

  // Check if parameters are valid
  bool is_valid() const { return n_vocab > 0 && n_embd > 0 && n_layer > 0; }
};

// ============================================
// Context Parameters
// ============================================

/**
 * Runtime context parameters
 */
struct llm_cparams {
  uint32_t n_ctx = 0;     // Context size for inference
  uint32_t n_batch = 32;  // Batch size for prompt processing
  uint32_t n_ubatch = 32; // Micro-batch size
  uint32_t n_seq_max = 1; // Max concurrent sequences

  int32_t n_threads = 4;       // CPU threads for generation
  int32_t n_threads_batch = 4; // CPU threads for batch processing

  // RoPE scaling
  float rope_freq_base = 10000.0f;
  float rope_freq_scale = 1.0f;
  uint32_t n_ctx_orig_yarn = 0;

  // Attention
  bool causal_attn = true;
  bool flash_attn = false;
  bool offload_kqv = true;

  // Embeddings mode
  bool embeddings = false;

  // Performance
  bool no_perf = false;
  bool warmup = true;
};

// ============================================
// Graph Parameters
// ============================================

struct llm_graph_params {
  llm_arch arch = LLM_ARCH_UNKNOWN;
  llm_hparams hparams;
  llm_cparams cparams;

  ggml_backend_sched_t sched = nullptr;
  ggml_backend_t backend_cpu = nullptr;

  uint32_t n_outputs = 0;  // Number of output tokens
  bool embeddings = false; // Output embeddings instead of logits
};

// ============================================
// Output Mode
// ============================================

/**
 * Output mode for graph building
 *
 * Different use cases require different outputs:
 * - LOGITS: Standard language modeling (next token prediction)
 * - EMBEDDINGS: Extract hidden states for downstream tasks
 * - FEATURES: Extract intermediate features for TTS/ASR
 * - ENCODER: Encoder-only output (for encoder-decoder models)
 */
enum class llm_output_mode {
  OUTPUT_LOGITS,     // Output logits for token prediction
  OUTPUT_EMBEDDINGS, // Output final hidden states
  OUTPUT_FEATURES,   // Output intermediate features (for TTS/ASR)
  OUTPUT_ENCODER,    // Encoder output (for seq2seq)
};

// ============================================
// Graph Build Options
// ============================================

/**
 * Options for controlling graph building
 *
 * This allows flexible configuration for different use cases:
 * - TTS: May need intermediate layer outputs
 * - ASR: May need encoder features
 * - LLM: Standard logits output
 */
struct llm_build_opts {
  llm_output_mode output_mode = llm_output_mode::OUTPUT_LOGITS;

  // Layer control
  int32_t n_layers_start = 0;    // Start layer index
  int32_t n_layers_end = -1;     // End layer index (-1 = all layers)
  bool skip_embeddings = false;  // Skip token embedding lookup
  bool skip_output_norm = false; // Skip output normalization
  bool skip_lm_head = false;     // Skip LM head projection

  // Feature extraction
  bool extract_intermediate = false;   // Extract intermediate layer outputs
  std::vector<int32_t> extract_layers; // Specific layers to extract

  // Attention control
  bool use_kv_cache = true;    // Use KV cache for decoding
  bool update_kv_cache = true; // Update KV cache with new keys/values
  bool causal_mask = true;     // Use causal attention mask

  // Input control
  bool input_is_embeds = false; // Input is embeddings, not token IDs
  bool add_position_ids = true; // Add position IDs to input

  // Performance
  bool use_flash_attn = false; // Use flash attention (if available)
  bool offload_kqv = true;     // Offload K/Q/V to CPU

  // Debug
  bool debug_dump_graph = false; // Dump graph for debugging
};

// ============================================
// Graph Element Types
// ============================================

/**
 * Normalization types
 */
enum class llm_norm_type {
  LAYER_NORM, // LayerNorm
  RMS_NORM,   // RMSNorm
  GROUP_NORM, // GroupNorm
};

/**
 * FFN activation types
 */
enum class llm_ffn_type {
  FFN_GELU,   // GELU activation
  FFN_SILU,   // SiLU/Swish activation
  FFN_RELU,   // ReLU activation
  FFN_SWIGLU, // SwiGLU (parallel gate)
  FFN_GEGLU,  // GeGLU (parallel gate)
};

// Forward declaration
class llm_model;

// ============================================
// Graph Result
// ============================================

/**
 * Manages computation graph and its resources
 */
class llm_graph_result {
public:
  explicit llm_graph_result(int64_t max_nodes = 8192);
  ~llm_graph_result();

  // Get output tensors
  ggml_tensor *get_logits() const { return t_logits_; }
  ggml_tensor *get_embd() const { return t_embd_; }
  ggml_tensor *get_input_tokens() const { return t_inp_tokens_; }

  // Get intermediate outputs (for TTS/ASR feature extraction)
  ggml_tensor *get_intermediate_output(size_t idx) const;
  void add_intermediate_output(ggml_tensor *tensor);
  size_t get_intermediate_count() const { return intermediate_outputs_.size(); }

  // Get graph
  ggml_cgraph *get_graph() const { return gf_; }
  ggml_context *get_ctx() const { return ctx_; }

  // Get build options used
  const llm_build_opts &get_opts() const { return opts_; }

  // Reset graph for new batch
  void reset();

  // Check if graph can be reused with new parameters
  bool can_reuse(const llm_graph_params &params,
                 const llm_build_opts &opts) const;

  // Update parameters for graph reuse
  void update_params(const llm_graph_params &params,
                     const llm_build_opts &opts);

  // Set output tensors (for builder classes)
  void set_logits(ggml_tensor *logits) { t_logits_ = logits; }
  void set_embd(ggml_tensor *embd) { t_embd_ = embd; }
  void set_graph(ggml_cgraph *gf) { gf_ = gf; }
  void set_ctx(ggml_context *ctx) { ctx_ = ctx; }

  // Set input data after graph allocation
  void set_input_data(ggml_tensor *tensor, const void *data, size_t size);
  void set_position_ids(ggml_tensor *positions, const llm_pos *pos,
                        uint32_t n_tokens);
  void set_input_tokens(ggml_tensor *inp_tokens, const int32_t *tokens,
                        uint32_t n_tokens);
  void set_causal_mask(ggml_tensor *mask, uint32_t n_tokens,
                       uint32_t n_kv_cache);

  // Get input tensor for setting data
  ggml_tensor *get_input_tensor(const char *name) const;

  // Set all input data for LLM graph (convenience method)
  void set_llm_inputs(const int32_t *tokens, const llm_pos *pos,
                      uint32_t n_tokens, uint32_t n_kv_cache = 0);

  // Friend declaration for builder
  friend class llm_graph_builder;

private:
  ggml_context *ctx_ = nullptr;
  ggml_cgraph *gf_ = nullptr;

  // Output tensors
  ggml_tensor *t_logits_ = nullptr;
  ggml_tensor *t_embd_ = nullptr;
  ggml_tensor *t_inp_tokens_ = nullptr;

  // Intermediate outputs (for TTS/ASR)
  std::vector<ggml_tensor *> intermediate_outputs_;

  // Parameters for reuse check
  llm_graph_params params_;
  llm_build_opts opts_;
  int64_t max_nodes_;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

// ============================================
// Graph Builder Context
// ============================================

/**
 * Base class for architecture-specific graph builders
 *
 * Usage:
 * 1. Create architecture-specific builder (e.g., llm_build_qwen3)
 * 2. Call build_graph() to create computation graph
 * 3. Execute graph with ggml_backend_sched_graph_compute()
 *
 * Design Pattern: Template Method
 * - Base class provides common graph building blocks
 * - Subclasses implement architecture-specific details
 */
class llm_graph_builder {
public:
  llm_graph_builder(const llm_hparams &hparams, const llm_cparams &cparams,
                    ggml_backend_sched_t sched);

  virtual ~llm_graph_builder() = default;

  /**
   * Build computation graph for given input tokens
   *
   * @param tokens Input token IDs [n_tokens]
   * @param n_tokens Number of tokens
   * @param kv_cache KV cache (optional, nullptr for prompt-only)
   * @param pos Positions for each token (optional, defaults to 0..n_tokens-1)
   * @param opts Build options (optional, defaults to standard LLM mode)
   * @return Graph result containing logits or embeddings
   */
  virtual llm_graph_result_ptr
  build_graph(const int32_t *tokens, uint32_t n_tokens,
              llm_kv_cache *kv_cache = nullptr, const llm_pos *pos = nullptr,
              const llm_build_opts *opts = nullptr) = 0;

  /**
   * Build computation graph for given input embeddings
   *
   * This is useful for TTS/ASR where input is already embeddings
   *
   * @param embeds Input embeddings [n_embd, n_tokens]
   * @param n_tokens Number of tokens
   * @param kv_cache KV cache (optional)
   * @param pos Positions for each token (optional)
   * @param opts Build options (optional)
   * @return Graph result
   */
  virtual llm_graph_result_ptr build_graph_from_embeds(
      ggml_tensor *embeds, uint32_t n_tokens, llm_kv_cache *kv_cache = nullptr,
      const llm_pos *pos = nullptr, const llm_build_opts *opts = nullptr);

protected:
  const llm_hparams &hparams_;
  const llm_cparams &cparams_;
  ggml_backend_sched_t sched_;

  // Common graph building utilities
  ggml_context *ctx_ = nullptr;
  ggml_cgraph *gf_ = nullptr;

  // Current build options
  llm_build_opts current_opts_;

  // Helper: Create new graph context
  void init_graph(int64_t max_nodes = 8192);

  // Helper: Free graph context
  void free_graph();

  // ========================================
  // Modular Graph Building Blocks
  // These can be reused by subclasses
  // ========================================

  /**
   * Build token embedding lookup
   *
   * @param ctx Graph context
   * @param tokens Input token IDs
   * @param n_tokens Number of tokens
   * @param tok_embd Token embedding weights
   * @return Embedding tensor [n_embd, n_tokens]
   */
  ggml_tensor *build_token_embeds(ggml_context *ctx, const int32_t *tokens,
                                  uint32_t n_tokens, ggml_tensor *tok_embd);

  /**
   * Build position embeddings/IDs
   *
   * @param ctx Graph context
   * @param pos Position array (nullptr for sequential)
   * @param n_tokens Number of tokens
   * @return Position tensor [n_tokens]
   */
  ggml_tensor *build_position_ids(ggml_context *ctx, const llm_pos *pos,
                                  uint32_t n_tokens);

  /**
   * Build causal attention mask
   *
   * @param ctx Graph context
   * @param n_tokens Number of tokens
   * @param n_kv_cache KV cache size (0 for no cache)
   * @return Mask tensor [n_tokens + n_kv_cache, n_tokens]
   */
  ggml_tensor *build_causal_mask(ggml_context *ctx, uint32_t n_tokens,
                                 uint32_t n_kv_cache = 0);

  /**
   * Build attention bias for QK
   *
   * @param ctx Graph context
   * @param n_tokens Number of tokens
   * @param scale Scale factor
   * @return Bias tensor
   */
  ggml_tensor *build_attn_bias(ggml_context *ctx, uint32_t n_tokens,
                               float scale = 1.0f);

  /**
   * Build RoPE embedding
   *
   * @param ctx Graph context
   * @param cur Input tensor
   * @param pos Position tensor
   * @param n_rot RoPE dimension
   * @param n_head Number of heads
   * @param rope_type RoPE type
   * @return RoPE-embedded tensor
   */
  ggml_tensor *build_rope_embeds(ggml_context *ctx, ggml_tensor *cur,
                                 ggml_tensor *pos, int32_t n_rot,
                                 int32_t n_head,
                                 int32_t rope_type = GGML_ROPE_TYPE_NEOX);

  /**
   * Build KV cache lookup
   *
   * @param ctx Graph context
   * @param k_cur Current K tensor
   * @param v_cur Current V tensor
   * @param kv_cache KV cache
   * @param pos Positions
   * @param n_tokens Number of tokens
   * @param il Layer index
   * @return Pair of {K, V} tensors with cache
   */
  std::pair<ggml_tensor *, ggml_tensor *>
  build_kv_cache_lookup(ggml_context *ctx, ggml_tensor *k_cur,
                        ggml_tensor *v_cur, llm_kv_cache *kv_cache,
                        uint32_t n_tokens, int32_t il);

  /**
   * Build multi-head attention
   *
   * @param ctx Graph context
   * @param q Query tensor
   * @param k Key tensor
   * @param v Value tensor
   * @param mask Attention mask
   * @param scale Scale factor
   * @param n_head Number of heads
   * @param n_head_kv Number of KV heads (for GQA)
   * @return Attention output
   */
  ggml_tensor *build_multi_head_attn(ggml_context *ctx, ggml_tensor *q,
                                     ggml_tensor *k, ggml_tensor *v,
                                     ggml_tensor *mask, float scale,
                                     int32_t n_head, int32_t n_head_kv = 0);

  /**
   * Build flash attention
   *
   * @param ctx Graph context
   * @param q Query tensor
   * @param k Key tensor
   * @param v Value tensor
   * @param mask Attention mask
   * @param scale Scale factor
   * @return Attention output
   */
  ggml_tensor *build_flash_attn(ggml_context *ctx, ggml_tensor *q,
                                ggml_tensor *k, ggml_tensor *v,
                                ggml_tensor *mask, float scale);

  /**
   * Build normalization layer
   *
   * @param ctx Graph context
   * @param cur Input tensor
   * @param weight Normalization weight
   * @param bias Normalization bias (optional)
   * @param type Normalization type
   * @param eps Epsilon for numerical stability
   * @return Normalized tensor
   */
  ggml_tensor *build_norm(ggml_context *ctx, ggml_tensor *cur,
                          ggml_tensor *weight, ggml_tensor *bias,
                          llm_norm_type type, float eps = 1e-6f);

  /**
   * Build FFN layer with various activation types
   *
   * @param ctx Graph context
   * @param cur Input tensor
   * @param w_up Up projection weight
   * @param w_gate Gate projection weight (optional, for gated FFN)
   * @param w_down Down projection weight
   * @param ffn_type FFN type
   * @return FFN output
   */
  ggml_tensor *build_ffn(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *w_up,
                         ggml_tensor *w_gate, ggml_tensor *w_down,
                         llm_ffn_type ffn_type);

  /**
   * Build residual connection
   *
   * @param ctx Graph context
   * @param cur Current tensor
   * @param residual Residual tensor
   * @return cur + residual
   */
  ggml_tensor *build_residual(ggml_context *ctx, ggml_tensor *cur,
                              ggml_tensor *residual);

  /**
   * Build LM head projection
   *
   * @param ctx Graph context
   * @param cur Input tensor
   * @param output Output projection weight
   * @return Logits tensor [n_vocab, n_tokens]
   */
  ggml_tensor *build_lm_head(ggml_context *ctx, ggml_tensor *cur,
                             ggml_tensor *output);

  /**
   * Build output normalization
   *
   * @param ctx Graph context
   * @param cur Input tensor
   * @param weight Normalization weight
   * @param bias Normalization bias (optional)
   * @return Normalized tensor
   */
  ggml_tensor *build_output_norm(ggml_context *ctx, ggml_tensor *cur,
                                 ggml_tensor *weight, ggml_tensor *bias);
};

// ============================================
// Graph Builder Factory
// ============================================

/**
 * Factory function type for creating graph builders
 */
using graph_builder_factory = std::function<std::unique_ptr<llm_graph_builder>(
    const llm_model &model, const llm_cparams &cparams,
    ggml_backend_sched_t sched)>;

/**
 * Register graph builder for architecture
 */
void llm_register_graph_builder(llm_arch arch, graph_builder_factory factory);

/**
 * Create graph builder for model
 *
 * @param model Model to create builder for
 * @param cparams Context parameters
 * @param sched Scheduler
 * @return Graph builder instance
 */
std::unique_ptr<llm_graph_builder>
llm_create_graph_builder(const llm_model &model, const llm_cparams &cparams,
                         ggml_backend_sched_t sched);

// ============================================
// Common Graph Operations (Legacy API)
// ============================================
// Note: These are kept for backward compatibility.
// New code should use the methods in llm_graph_builder.

/**
 * Build normalization layer (legacy API)
 */
ggml_tensor *llm_build_norm(ggml_context *ctx, ggml_tensor *cur,
                            const llm_hparams &hparams, ggml_tensor *weight,
                            ggml_tensor *bias, llm_norm_type type,
                            int32_t il = -1);

/**
 * Build FFN layer (legacy API)
 */
ggml_tensor *llm_build_ffn(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *up,
                           ggml_tensor *gate, ggml_tensor *down,
                           llm_ffn_type ffn_type, int32_t il = -1);

/**
 * Build RoPE embedding (legacy API)
 */
ggml_tensor *llm_build_rope(ggml_context *ctx, ggml_tensor *cur,
                            ggml_tensor *pos, const llm_hparams &hparams,
                            int32_t n_rot, int32_t il = -1);

/**
 * Build multi-head attention (legacy API)
 */
ggml_tensor *llm_build_attn(ggml_context *ctx, ggml_tensor *q, ggml_tensor *k,
                            ggml_tensor *v, ggml_tensor *kq_mask, float scale,
                            int32_t il = -1);
