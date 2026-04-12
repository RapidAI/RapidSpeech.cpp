#include "llm_graph.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llm_model.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// ============================================
// llm_graph_result Implementation
// ============================================

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes_(max_nodes) {

  // Initialize context and graph
  size_t mem_size =
      max_nodes * ggml_tensor_overhead() + (1 << 20); // 1MB buffer
  struct ggml_init_params params = {
      mem_size, nullptr,
      true // No allocation
  };

  ctx_ = ggml_init(params);
  if (!ctx_) {
    throw std::runtime_error("Failed to initialize graph context");
  }

  gf_ = ggml_new_graph_custom(ctx_, max_nodes, false);
  if (!gf_) {
    ggml_free(ctx_);
    throw std::runtime_error("Failed to create graph");
  }
}

llm_graph_result::~llm_graph_result() {
  if (ctx_) {
    ggml_free(ctx_);
  }
}

void llm_graph_result::reset() {
  if (gf_) {
    ggml_graph_clear(gf_);
  }
  t_logits_ = nullptr;
  t_embd_ = nullptr;
  t_inp_tokens_ = nullptr;
  intermediate_outputs_.clear();
}

bool llm_graph_result::can_reuse(const llm_graph_params &params,
                                 const llm_build_opts &opts) const {
  // Check if parameters match for graph reuse
  return params.arch == params_.arch &&
         params.hparams.n_embd == params_.hparams.n_embd &&
         params.hparams.n_layer == params_.hparams.n_layer &&
         params.hparams.n_head == params_.hparams.n_head &&
         params.cparams.n_batch == params_.cparams.n_batch &&
         opts.output_mode == opts_.output_mode &&
         opts.causal_mask == opts_.causal_mask;
}

void llm_graph_result::update_params(const llm_graph_params &params,
                                     const llm_build_opts &opts) {
  params_ = params;
  opts_ = opts;
}

ggml_tensor *llm_graph_result::get_intermediate_output(size_t idx) const {
  if (idx >= intermediate_outputs_.size()) {
    return nullptr;
  }
  return intermediate_outputs_[idx];
}

void llm_graph_result::add_intermediate_output(ggml_tensor *tensor) {
  intermediate_outputs_.push_back(tensor);
}

void llm_graph_result::set_input_data(ggml_tensor *tensor, const void *data,
                                      size_t size) {
  if (tensor && tensor->data && data) {
    memcpy(tensor->data, data, size);
  }
}

void llm_graph_result::set_position_ids(ggml_tensor *positions,
                                        const llm_pos *pos, uint32_t n_tokens) {
  if (!positions || !positions->data) {
    return;
  }

  if (pos) {
    memcpy(positions->data, pos, n_tokens * sizeof(llm_pos));
  } else {
    // Default: sequential positions 0, 1, 2, ..., n_tokens-1
    int32_t *pos_data = static_cast<int32_t *>(positions->data);
    for (uint32_t i = 0; i < n_tokens; ++i) {
      pos_data[i] = static_cast<int32_t>(i);
    }
  }
}

ggml_tensor *llm_graph_result::get_input_tensor(const char *name) const {
  if (!gf_ || !name) {
    return nullptr;
  }
  return ggml_get_tensor(ctx_, name);
}

void llm_graph_result::set_input_tokens(ggml_tensor *inp_tokens,
                                        const int32_t *tokens,
                                        uint32_t n_tokens) {
  if (!inp_tokens || !inp_tokens->data || !tokens) {
    return;
  }
  ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));
}

void llm_graph_result::set_causal_mask(ggml_tensor *mask, uint32_t n_tokens,
                                       uint32_t n_kv_cache) {
  if (!mask || !mask->data) {
    return;
  }

  const uint32_t n_kv = n_kv_cache + n_tokens;
  std::vector<float> mask_data(n_tokens * n_kv);
  for (uint32_t j = 0; j < n_kv; ++j) {
    for (uint32_t i = 0; i < n_tokens; ++i) {
      const bool is_causal = (n_kv_cache + i) < j;
      mask_data[j * n_tokens + i] = is_causal ? -INFINITY : 0.0f;
    }
  }
  ggml_backend_tensor_set(mask, mask_data.data(), 0,
                          mask_data.size() * sizeof(float));
}

void llm_graph_result::set_llm_inputs(const int32_t *tokens, const llm_pos *pos,
                                      uint32_t n_tokens, uint32_t n_kv_cache) {
  // Set input tokens if provided
  if (tokens) {
    ggml_tensor *inp_tokens = get_input_tensor("inp_tokens");
    if (inp_tokens) {
      set_input_tokens(inp_tokens, tokens, n_tokens);
    }
  }

  // Set position ids
  ggml_tensor *positions = get_input_tensor("position_ids");
  if (!positions) {
    positions = get_input_tensor("position_ids_seq");
  }
  if (positions) {
    set_position_ids(positions, pos, n_tokens);
  }

  // Set causal mask if needed
  ggml_tensor *mask = get_input_tensor("causal_mask");
  if (mask && n_kv_cache > 0) {
    set_causal_mask(mask, n_tokens, n_kv_cache);
  }
}

// ============================================
// llm_graph_builder Implementation
// ============================================

llm_graph_builder::llm_graph_builder(const llm_hparams &hparams,
                                     const llm_cparams &cparams,
                                     ggml_backend_sched_t sched)
    : hparams_(hparams), cparams_(cparams), sched_(sched) {}

void llm_graph_builder::init_graph(int64_t max_nodes) {
  size_t mem_size = max_nodes * ggml_tensor_overhead() + (1 << 20);
  struct ggml_init_params params = {mem_size, nullptr, true};

  ctx_ = ggml_init(params);
  gf_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

void llm_graph_builder::free_graph() {
  if (ctx_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
    gf_ = nullptr;
  }
}

llm_graph_result_ptr llm_graph_builder::build_graph_from_embeds(
    ggml_tensor *embeds, uint32_t n_tokens, llm_kv_cache *kv_cache,
    const llm_pos *pos, const llm_build_opts *opts) {
  // Default implementation: not supported
  // Subclasses can override this for TTS/ASR use cases
  (void)embeds;
  (void)n_tokens;
  (void)kv_cache;
  (void)pos;
  (void)opts;
  return nullptr;
}

// ============================================
// Modular Graph Building Blocks
// ============================================

ggml_tensor *llm_graph_builder::build_token_embeds(ggml_context *ctx,
                                                   const int32_t *tokens,
                                                   uint32_t n_tokens,
                                                   ggml_tensor *tok_embd) {
  // Create input tensor
  ggml_tensor *inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(inp_tokens);

  // Note: Data is set after graph allocation via ggml_backend_tensor_set
  // Store tokens pointer for later use
  ggml_set_name(inp_tokens, "inp_tokens");

  // Lookup embeddings: [n_embd, n_tokens]
  ggml_tensor *cur = ggml_get_rows(ctx, tok_embd, inp_tokens);

  // Scale embeddings by sqrt(n_embd) if needed
  // cur = ggml_scale_inplace(ctx, cur,
  // sqrtf(static_cast<float>(hparams_.n_embd)));

  return cur;
}

ggml_tensor *llm_graph_builder::build_position_ids(ggml_context *ctx,
                                                   const llm_pos *pos,
                                                   uint32_t n_tokens) {
  ggml_tensor *positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
  ggml_set_input(positions);

  // Note: Data is set after graph allocation via ggml_backend_tensor_set
  // Position data is stored for later use
  if (pos) {
    ggml_set_name(positions, "position_ids");
  } else {
    ggml_set_name(positions, "position_ids_seq");
  }

  return positions;
}

ggml_tensor *llm_graph_builder::build_causal_mask(ggml_context *ctx,
                                                  uint32_t n_tokens,
                                                  uint32_t n_kv_cache) {
  const uint32_t n_kv = n_kv_cache + n_tokens;

  // Create causal mask: -inf for future positions
  ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_kv);
  ggml_set_input(mask);

  // Note: Mask data is set after graph allocation via ggml_backend_tensor_set
  // Mask pattern is static and can be computed once
  ggml_set_name(mask, "causal_mask");

  return mask;
}

ggml_tensor *llm_graph_builder::build_attn_bias(ggml_context *ctx,
                                                uint32_t n_tokens,
                                                float scale) {
  // Create attention bias tensor
  ggml_tensor *bias =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_tokens, n_tokens);
  ggml_set_input(bias);

  // Note: Bias data is set after graph allocation
  ggml_set_name(bias, "attn_bias");

  return bias;
}

ggml_tensor *llm_graph_builder::build_rope_embeds(ggml_context *ctx,
                                                  ggml_tensor *cur,
                                                  ggml_tensor *pos,
                                                  int32_t n_rot, int32_t n_head,
                                                  int32_t rope_type) {
  (void)n_head; // May be used for extended RoPE

  return ggml_rope_ext(ctx, cur, pos, nullptr, n_rot, rope_type,
                       hparams_.n_ctx_train, hparams_.rope_freq_base,
                       hparams_.rope_freq_scale,
                       0.0f, // ext_factor
                       hparams_.rope_attn_factor,
                       0.0f, // beta_fast
                       0.0f  // beta_slow
  );
}

std::pair<ggml_tensor *, ggml_tensor *>
llm_graph_builder::build_kv_cache_lookup(ggml_context *ctx, ggml_tensor *k_cur,
                                         ggml_tensor *v_cur,
                                         llm_kv_cache *kv_cache,
                                         uint32_t n_tokens, int32_t il) {
  if (!kv_cache || !current_opts_.use_kv_cache) {
    return {k_cur, v_cur};
  }

  // Note: Full KV cache integration requires slot management
  // This is a simplified version for basic cache usage

  // For now, just return current K/V without cache integration
  // Full implementation would:
  // 1. Call kv_cache->prepare() to get slot info
  // 2. Use kv_cache->cpy_k/cpy_v to store K/V
  // 3. Use kv_cache->get_k/get_v to retrieve cached K/V
  // 4. Call kv_cache->commit() after graph execution

  (void)n_tokens;
  (void)il;

  return {k_cur, v_cur};
}

ggml_tensor *llm_graph_builder::build_multi_head_attn(
    ggml_context *ctx,
    ggml_tensor *q,     // [d_k, n_head, n_tokens]
    ggml_tensor *k,     // [d_k, n_head_kv, n_tokens_kv]
    ggml_tensor *v,     // [d_k, n_head_kv, n_tokens_kv]
    ggml_tensor *mask,
    float scale,
    int32_t n_head,
    int32_t n_head_kv) {

    const int32_t d_k = q->ne[0];
    const int32_t n_tokens = q->ne[2];
    const int32_t n_tokens_kv = k->ne[2];

    // 1. Handle GQA: Repeat K and V to match Q's head count (e.g., 8 -> 16)
    ggml_tensor * k_repeated = k;
    ggml_tensor * v_repeated = v;
    if (n_head != n_head_kv) {
        k_repeated = ggml_repeat(ctx, k, ggml_new_tensor_3d(ctx, k->type, d_k, n_head, n_tokens_kv));
        v_repeated = ggml_repeat(ctx, v, ggml_new_tensor_3d(ctx, v->type, d_k, n_head, n_tokens_kv));
    }

    // 2. Permute for Attention calculation [d_k, n_tokens, n_head]
    ggml_tensor * q_perm = ggml_permute(ctx, q,          0, 2, 1, 3);
    ggml_tensor * k_perm = ggml_permute(ctx, k_repeated, 0, 2, 1, 3);
    ggml_tensor * v_perm = ggml_permute(ctx, v_repeated, 0, 2, 1, 3);

    // 3. QK^T: Result kq shape [n_tokens_kv, n_tokens, n_head]
    // A=k_perm (ne0=d_k), B=q_perm (ne0=d_k). ne0 match!
    ggml_tensor * kq = ggml_mul_mat(ctx, k_perm, q_perm);

    kq = ggml_scale_inplace(ctx, kq, scale);
    if (mask) {
        kq = ggml_add_inplace(ctx, kq, mask);
    }
    ggml_tensor * probs = ggml_soft_max_inplace(ctx, kq);

    // 4. Important: Transpose V to align for (Probs * V)
    // v_perm is [d_k, n_kv, n_head] -> [128, 52, 16]
    // v_tr is [n_kv, d_k, n_head]   -> [52, 128, 16]
    ggml_tensor * v_tr = ggml_cont(ctx, ggml_transpose(ctx, v_perm));

    // 5. Final Matrix Multiply: Softmax * V
    // ggml_mul_mat(A, B) calculates B * A^T
    // A = v_tr (ne0=52, ne1=128, ne2=16)
    // B = probs (ne0=52, ne1=52, ne2=16)
    // ne0(A) == ne0(B) (52 == 52) -> Match!
    // Result kqv shape: [ne1(A), ne1(B), ne2] -> [128, 52, 16]
    ggml_tensor * kqv = ggml_mul_mat(ctx, v_tr, probs);

    // 6. Return to original layout [d_k, n_head, n_tokens]
    return ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
}

ggml_tensor *llm_graph_builder::build_flash_attn(ggml_context *ctx,
                                                 ggml_tensor *q, ggml_tensor *k,
                                                 ggml_tensor *v,
                                                 ggml_tensor *mask,
                                                 float scale) {
  // Flash attention implementation
  // Note: This requires ggml flash attention support
  (void)mask; // Flash attention handles masking internally

  return ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
}

ggml_tensor *llm_graph_builder::build_norm(ggml_context *ctx, ggml_tensor *cur,
                                           ggml_tensor *weight,
                                           ggml_tensor *bias,
                                           llm_norm_type type, float eps) {
  switch (type) {
  case llm_norm_type::RMS_NORM:
    cur = ggml_rms_norm(ctx, cur, eps);
    break;
  case llm_norm_type::LAYER_NORM:
    cur = ggml_norm(ctx, cur, eps);
    break;
  case llm_norm_type::GROUP_NORM:
    // TODO: Implement group norm
    cur = ggml_norm(ctx, cur, eps);
    break;
  }

  cur = ggml_mul(ctx, cur, weight);
  if (bias) {
    cur = ggml_add(ctx, cur, bias);
  }

  return cur;
}

ggml_tensor *llm_graph_builder::build_ffn(ggml_context *ctx, ggml_tensor *cur,
                                          ggml_tensor *w_up,
                                          ggml_tensor *w_gate,
                                          ggml_tensor *w_down,
                                          llm_ffn_type ffn_type) {
  ggml_tensor *result;

  switch (ffn_type) {
  case llm_ffn_type::FFN_SWIGLU: {
    // SwiGLU: gate * sigmoid(gate) * up
    ggml_tensor *gate_out = ggml_mul_mat(ctx, w_gate, cur);
    gate_out = ggml_silu(ctx, gate_out);

    ggml_tensor *up_out = ggml_mul_mat(ctx, w_up, cur);

    result = ggml_mul(ctx, gate_out, up_out);
    result = ggml_mul_mat(ctx, w_down, result);
    break;
  }

  case llm_ffn_type::FFN_GEGLU: {
    // GeGLU: GELU(gate) * up
    ggml_tensor *gate_out = ggml_mul_mat(ctx, w_gate, cur);
    gate_out = ggml_gelu(ctx, gate_out);

    ggml_tensor *up_out = ggml_mul_mat(ctx, w_up, cur);

    result = ggml_mul(ctx, gate_out, up_out);
    result = ggml_mul_mat(ctx, w_down, result);
    break;
  }

  case llm_ffn_type::FFN_GELU: {
    result = ggml_mul_mat(ctx, w_up, cur);
    result = ggml_gelu(ctx, result);
    result = ggml_mul_mat(ctx, w_down, result);
    break;
  }

  case llm_ffn_type::FFN_SILU: {
    result = ggml_mul_mat(ctx, w_up, cur);
    result = ggml_silu(ctx, result);
    result = ggml_mul_mat(ctx, w_down, result);
    break;
  }

  case llm_ffn_type::FFN_RELU: {
    result = ggml_mul_mat(ctx, w_up, cur);
    result = ggml_relu(ctx, result);
    result = ggml_mul_mat(ctx, w_down, result);
    break;
  }

  default:
    // Fallback to simple linear
    result = ggml_mul_mat(ctx, w_up, cur);
    result = ggml_mul_mat(ctx, w_down, result);
    break;
  }

  return result;
}

ggml_tensor *llm_graph_builder::build_residual(ggml_context *ctx,
                                               ggml_tensor *cur,
                                               ggml_tensor *residual) {
  return ggml_add(ctx, cur, residual);
}

ggml_tensor *llm_graph_builder::build_lm_head(ggml_context *ctx,
                                              ggml_tensor *cur,
                                              ggml_tensor *output) {
  return ggml_mul_mat(ctx, output, cur);
}

ggml_tensor *llm_graph_builder::build_output_norm(ggml_context *ctx,
                                                  ggml_tensor *cur,
                                                  ggml_tensor *weight,
                                                  ggml_tensor *bias) {
  return build_norm(ctx, cur, weight, bias, llm_norm_type::RMS_NORM,
                    hparams_.f_norm_rms_eps);
}

// ============================================
// Common Graph Operations
// ============================================

ggml_tensor *llm_build_norm(ggml_context *ctx, ggml_tensor *cur,
                            const llm_hparams &hparams, ggml_tensor *weight,
                            ggml_tensor *bias, llm_norm_type type, int32_t il) {
  switch (type) {
  case llm_norm_type::RMS_NORM:
    cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    break;
  case llm_norm_type::LAYER_NORM:
    cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
    break;
  case llm_norm_type::GROUP_NORM:
    // TODO: Implement group norm
    cur = ggml_norm(ctx, cur, hparams.f_norm_eps);
    break;
  }

  cur = ggml_mul(ctx, cur, weight);
  if (bias) {
    cur = ggml_add(ctx, cur, bias);
  }

  return cur;
}

ggml_tensor *llm_build_ffn(ggml_context *ctx, ggml_tensor *cur, ggml_tensor *up,
                           ggml_tensor *gate, ggml_tensor *down,
                           llm_ffn_type ffn_type, int32_t il) {
  ggml_tensor *result;

  switch (ffn_type) {
  case llm_ffn_type::FFN_SWIGLU: {
    // SwiGLU: gate * sigmoid(gate) * up
    ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
    gate_out = ggml_silu(ctx, gate_out);

    ggml_tensor *up_out = ggml_mul_mat(ctx, up, cur);

    result = ggml_mul(ctx, gate_out, up_out);
    result = ggml_mul_mat(ctx, down, result);
    break;
  }

  case llm_ffn_type::FFN_GELU: {
    result = ggml_mul_mat(ctx, up, cur);
    result = ggml_gelu(ctx, result);
    result = ggml_mul_mat(ctx, down, result);
    break;
  }

  case llm_ffn_type::FFN_SILU: {
    result = ggml_mul_mat(ctx, up, cur);
    result = ggml_silu(ctx, result);
    result = ggml_mul_mat(ctx, down, result);
    break;
  }

  default:
    // Fallback to simple linear
    result = ggml_mul_mat(ctx, up, cur);
    result = ggml_mul_mat(ctx, down, result);
    break;
  }

  return result;
}

ggml_tensor *llm_build_rope(ggml_context *ctx, ggml_tensor *cur,
                            ggml_tensor *pos, const llm_hparams &hparams,
                            int32_t n_rot, int32_t il) {
  return ggml_rope_ext(ctx, cur, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX,
                       hparams.n_ctx_train, hparams.rope_freq_base,
                       hparams.rope_freq_scale,
                       0.0f, // ext_factor
                       hparams.rope_attn_factor,
                       0.0f, // beta_fast
                       0.0f  // beta_slow
  );
}

ggml_tensor *llm_build_attn(ggml_context *ctx, ggml_tensor *q, ggml_tensor *k,
                            ggml_tensor *v, ggml_tensor *kq_mask, float scale,
                            int32_t il) {
  (void)il; // Layer index for debugging

  const int32_t d_k = q->ne[0]; // Head dimension
  const int32_t n_tokens = q->ne[2];

  // Permute q/k/v: [d_k, n_head, n_tokens] -> [d_k, n_tokens, n_head, 1]
  ggml_tensor *q_perm = ggml_permute(ctx, q, 0, 2, 1, 3);
  ggml_tensor *k_perm = ggml_permute(ctx, k, 0, 2, 1, 3);
  ggml_tensor *v_perm = ggml_permute(ctx, v, 0, 2, 1, 3);

  // Q * K^T: Result [n_tokens, n_tokens, n_head, 1]
  ggml_tensor *kq = ggml_mul_mat(ctx, k_perm, q_perm);

  // Scale
  kq = ggml_scale_inplace(ctx, kq, scale);

  // Mask (optional)
  if (kq_mask) {
    kq = ggml_add(ctx, kq, kq_mask);
  }

  // Softmax
  ggml_tensor *probs = ggml_soft_max_inplace(ctx, kq);

  // V^T * probs
  ggml_tensor *v_transposed = ggml_cont(ctx, ggml_transpose(ctx, v_perm));
  ggml_tensor *kqv = ggml_mul_mat(ctx, v_transposed, probs);

  // Transpose back to [d_k, n_head, n_tokens, 1]
  kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 1, 2, 0, 3));

  return kqv;
}

// ============================================
// Graph Builder Factory Implementation
// ============================================

namespace {
// Registry for graph builder factories
struct GraphBuilderRegistry {
  std::unordered_map<llm_arch, graph_builder_factory> factories;

  void register_builder(llm_arch arch, graph_builder_factory factory) {
    factories[arch] = std::move(factory);
  }

  graph_builder_factory *get_builder(llm_arch arch) {
    auto it = factories.find(arch);
    if (it == factories.end()) {
      return nullptr;
    }
    return &it->second;
  }
};

GraphBuilderRegistry &get_registry() {
  static GraphBuilderRegistry registry;
  return registry;
}
} // namespace

void llm_register_graph_builder(llm_arch arch, graph_builder_factory factory) {
  get_registry().register_builder(arch, std::move(factory));
}

std::unique_ptr<llm_graph_builder>
llm_create_graph_builder(const llm_model &model, const llm_cparams &cparams,
                         ggml_backend_sched_t sched) {
  llm_arch arch = model.arch();

  auto *factory = get_registry().get_builder(arch);
  if (!factory) {
    // Fallback: try to create a generic builder
    // This should rarely happen if architectures are properly registered
    return nullptr;
  }

  return (*factory)(model, cparams, sched);
}
