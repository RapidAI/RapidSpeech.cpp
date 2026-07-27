#include "llm_graph.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llm_model.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {

ggml_tensor *llm_view_root(ggml_tensor *tensor) {
  while (tensor && tensor->view_src) {
    tensor = tensor->view_src;
  }
  return tensor;
}

} // namespace

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
  if (tensor && data && size > 0) {
    ggml_backend_tensor_set(tensor, data, 0, size);
  }
}

void llm_graph_result::set_position_ids(ggml_tensor *positions,
                                        const llm_pos *pos, uint32_t n_tokens) {
  if (!positions) {
    return;
  }

  if (pos) {
    ggml_backend_tensor_set(positions, pos, 0, n_tokens * sizeof(llm_pos));
  } else {
    // Default: sequential positions 0, 1, 2, ..., n_tokens-1
    std::vector<llm_pos> default_pos(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
      default_pos[i] = static_cast<llm_pos>(i);
    }
    ggml_backend_tensor_set(positions, default_pos.data(), 0,
                            n_tokens * sizeof(llm_pos));
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

  // Mask shape: [n_kv, n_tokens], ne[0]=n_kv, ne[1]=n_tokens
  // F32 mask tensor for use with ggml_add(kq_f32, mask_f32).
  const uint32_t n_kv = static_cast<uint32_t>(mask->ne[0]);
  const size_t n_elem = (size_t)n_tokens * n_kv;
  std::vector<float> mask_data(n_elem, 0.0f);

  for (uint32_t i = 0; i < n_tokens; ++i) {
    const uint32_t cur_pos = n_kv_cache + i;
    for (uint32_t j = 0; j < n_kv; ++j) {
      const size_t index = (size_t)i * n_kv + j;
      // -1e4: exp(-1e4) underflows to 0 in f32 (true zero leakage) and stays
      // well above F16 saturation (-65504), so the flash-attn F16 cast at
      // build_flash_attn() is also safe. Avoids the CPU-softmax NaN edge case
      // that motivated commit 0908d89's earlier -7.2f workaround.
      mask_data[index] = (j > cur_pos) ? -1e4f : 0.0f;
    }
  }

  ggml_backend_tensor_set(mask, mask_data.data(), 0,
                          mask_data.size() * sizeof(float));
}

void llm_graph_result::set_kv_write_indices(uint32_t index,
                                            uint32_t n_layers) {
  for (uint32_t il = 0; il < n_layers; ++il) {
    char name[64];
    snprintf(name, sizeof(name), "kv_write_idx_layer_%u", il);
    ggml_tensor *idx = get_input_tensor(name);
    if (!idx || !idx->data) {
      continue;
    }

    const int64_t n_idx = ggml_nelements(idx);
    std::vector<int32_t> rows(n_idx);
    for (int64_t i = 0; i < n_idx; ++i) {
      rows[i] = static_cast<int32_t>(index + static_cast<uint32_t>(i));
    }
    ggml_backend_tensor_set(idx, rows.data(), 0,
                            rows.size() * sizeof(int32_t));
  }
}

void llm_graph_result::set_kv_write_indices_batched(const uint32_t *indices,
                                                    uint32_t n_seq,
                                                    uint32_t n_layers) {
  if (!indices) return;
  std::vector<int32_t> rows(indices, indices + n_seq);
  for (uint32_t il = 0; il < n_layers; ++il) {
    char name[64];
    snprintf(name, sizeof(name), "kv_write_idx_layer_%u", il);
    ggml_tensor *idx = get_input_tensor(name);
    if (!idx || !idx->data) continue;
    const int64_t n_idx = ggml_nelements(idx);
    const int64_t n = std::min<int64_t>((int64_t)n_seq, n_idx);
    ggml_backend_tensor_set(idx, rows.data(), 0, (size_t)n * sizeof(int32_t));
  }
}

void llm_graph_result::set_batched_causal_mask(ggml_tensor *mask,
                                               uint32_t n_kv_max,
                                               const uint32_t *n_kv_cache_per_seq,
                                               uint32_t n_seq) {
  if (!mask || !mask->data || !n_kv_cache_per_seq) return;
  // mask shape [n_kv_max, 1, 1, n_seq]; row-major, ne[0]=n_kv_max contiguous,
  // ne[3]=n_seq is the slowest axis. Slice s occupies
  // [s * n_kv_max, (s+1) * n_kv_max) floats.
  const size_t slice = (size_t)n_kv_max;
  std::vector<float> slice_data(slice, 0.0f);
  for (uint32_t s = 0; s < n_seq; ++s) {
    const uint32_t cur = n_kv_cache_per_seq[s];
    for (uint32_t j = 0; j < n_kv_max; ++j) {
      slice_data[j] = (j > cur) ? -1e4f : 0.0f;
    }
    const size_t off = (size_t)s * slice * sizeof(float);
    ggml_backend_tensor_set(mask, slice_data.data(), off,
                            slice_data.size() * sizeof(float));
  }
}

void llm_graph_result::assign_kv_backend(ggml_backend_sched_t sched,
                                         ggml_backend_t backend,
                                         uint32_t n_layers) {
  if (!sched || !backend) {
    return;
  }

  for (uint32_t il = 0; il < n_layers; ++il) {
    if (opts_.gpu_kv_k && opts_.gpu_kv_k[il]) {
      ggml_backend_sched_set_tensor_backend(sched, opts_.gpu_kv_k[il],
                                            backend);
    }
    if (opts_.gpu_kv_v && opts_.gpu_kv_v[il]) {
      ggml_backend_sched_set_tensor_backend(sched, opts_.gpu_kv_v[il],
                                            backend);
    }

    char name[64];
    snprintf(name, sizeof(name), "kv_write_idx_layer_%u", il);
    if (ggml_tensor *idx = get_input_tensor(name)) {
      ggml_backend_sched_set_tensor_backend(sched, idx, backend);
    }
  }

  if (!gf_) {
    return;
  }
  const int n_nodes = ggml_graph_n_nodes(gf_);
  for (int i = 0; i < n_nodes; ++i) {
    ggml_tensor *node = ggml_graph_node(gf_, i);
    if (!node) {
      continue;
    }
    if (node->op != GGML_OP_SET && node->op != GGML_OP_SET_ROWS) {
      continue;
    }

    ggml_tensor *view_src = llm_view_root(node->view_src);
    if (!view_src && node->src[2]) {
      view_src = llm_view_root(node->src[2]);
    }
    if (!view_src) {
      continue;
    }

    bool touches_kv = false;
    for (uint32_t il = 0; il < n_layers && !touches_kv; ++il) {
      touches_kv =
          (opts_.gpu_kv_k && view_src == opts_.gpu_kv_k[il]) ||
          (opts_.gpu_kv_v && view_src == opts_.gpu_kv_v[il]);
    }
    if (touches_kv) {
      ggml_backend_sched_set_tensor_backend(sched, node, backend);
    }
  }
}

// Set position ids
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

  // Clear temporary KV output tracking from previous build
  tmp_kv_outputs_k_.clear();
  tmp_kv_outputs_v_.clear();
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

  // Create a 2D input tensor for the causal mask.
  // Shape: [n_kv, n_tokens], where n_kv = n_kv_cache + n_tokens.
  // Uses F32 type for compatibility with ggml_add(kq_f32, mask_f32).
  ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
  ggml_set_input(mask);
  ggml_set_name(mask, "causal_mask");

  return mask;
}

ggml_tensor *llm_graph_builder::build_causal_mask_batched(ggml_context *ctx,
                                                          uint32_t n_kv_max,
                                                          uint32_t n_seq) {
  // 4D input [n_kv_max, 1, 1, n_seq]. ne3=n_seq isolates sequences in
  // ggml_flash_attn_ext; ne1=1 (n_batch, one query per seq); ne2=1 broadcasts
  // over n_head_kv. Runtime fill via set_batched_causal_mask.
  ggml_tensor *mask =
      ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv_max, 1, 1, n_seq);
  ggml_set_input(mask);
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
  (void)kv_cache;

  const bool has_gpu_kv = current_opts_.use_kv_cache &&
                          current_opts_.gpu_kv_k && current_opts_.gpu_kv_v &&
                          current_opts_.n_kv_max > 0;
  if (has_gpu_kv) {
    ggml_tensor *k_gpu = current_opts_.gpu_kv_k[il];
    ggml_tensor *v_gpu = current_opts_.gpu_kv_v[il];
    if (k_gpu && v_gpu) {
      const int64_t head_dim = k_cur->ne[0];
      const int64_t n_head_kv = k_cur->ne[1];
      const int64_t n_cur = k_cur->ne[2];
      const int64_t kv_dim = head_dim * n_head_kv;

      ggml_tensor *k_cur_2d =
          ggml_reshape_2d(ctx, ggml_cont(ctx, k_cur), kv_dim, n_cur);
      ggml_tensor *v_cur_2d =
          ggml_reshape_2d(ctx, ggml_cont(ctx, v_cur), kv_dim, n_cur);

      if (!current_opts_.is_decode_step) {
        const size_t slot_off =
            (size_t)current_opts_.kv_slot_row_offset * (size_t)kv_dim *
            sizeof(float);
        ggml_tensor *k_after =
            ggml_set_2d_inplace(ctx, k_gpu, k_cur_2d, k_gpu->nb[1], slot_off);
        ggml_tensor *v_after =
            ggml_set_2d_inplace(ctx, v_gpu, v_cur_2d, v_gpu->nb[1], slot_off);
        ggml_tensor *k_view =
            ggml_view_2d(ctx, k_after, kv_dim, n_cur, k_after->nb[1], slot_off);
        ggml_tensor *v_view =
            ggml_view_2d(ctx, v_after, kv_dim, n_cur, v_after->nb[1], slot_off);
        return {ggml_reshape_3d(ctx, k_view, head_dim, n_head_kv, n_cur),
                ggml_reshape_3d(ctx, v_view, head_dim, n_head_kv, n_cur)};
      }

      // ── Batched decode (n_seq sequences, one token each) ─────────────────
      // k_gpu is the shared [kv_dim, n_kv_max*n_seq] buffer; slot s owns rows
      // [s*n_kv_max, (s+1)*n_kv_max). ONE set_rows writes every slot's new
      // column at absolute row s*n_kv_max + n_cached[s] (write_idx supplied at
      // runtime). The 2D buffer's memory is identical to the 4D flash-attn
      // layout [head_dim, n_head_kv, n_kv_max, n_seq], so attention views it
      // zero-copy. No chained writes, no shared-buffer aliasing.
      if (current_opts_.n_seq > 1) {
        const uint32_t n_seq = current_opts_.n_seq;
        ggml_tensor *k_cur_2d_batch =
            ggml_reshape_2d(ctx, ggml_cont(ctx, k_cur), kv_dim, n_seq);
        ggml_tensor *v_cur_2d_batch =
            ggml_reshape_2d(ctx, ggml_cont(ctx, v_cur), kv_dim, n_seq);

        ggml_tensor *write_idx =
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seq);
        ggml_set_input(write_idx);
        char idx_name[64];
        snprintf(idx_name, sizeof(idx_name), "kv_write_idx_layer_%d", il);
        ggml_set_name(write_idx, idx_name);

        ggml_tensor *k_after =
            ggml_set_rows(ctx, k_gpu, k_cur_2d_batch, write_idx);
        ggml_tensor *v_after =
            ggml_set_rows(ctx, v_gpu, v_cur_2d_batch, write_idx);

        // View the first n_seq slices of the shared [kv_dim, n_kv_max*n_slots]
        // buffer (the scheduler keeps DECODING slots contiguous at physical
        // 0..n_seq-1). Memory == 4D flash-attn layout
        // [head_dim, n_head_kv, n_kv_max, n_seq] -> zero-copy reshape.
        const int64_t n_cols = (int64_t)current_opts_.n_kv_max * n_seq;
        ggml_tensor *k_view =
            ggml_view_2d(ctx, k_after, kv_dim, n_cols, k_after->nb[1], 0);
        ggml_tensor *v_view =
            ggml_view_2d(ctx, v_after, kv_dim, n_cols, v_after->nb[1], 0);
        return {ggml_reshape_4d(ctx, k_view, head_dim, n_head_kv,
                                current_opts_.n_kv_max, n_seq),
                ggml_reshape_4d(ctx, v_view, head_dim, n_head_kv,
                                current_opts_.n_kv_max, n_seq)};
      }

      if (current_opts_.fixed_kv_cache_shape) {
        ggml_tensor *write_idx =
            ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cur);
        ggml_set_input(write_idx);
        char idx_name[64];
        snprintf(idx_name, sizeof(idx_name), "kv_write_idx_layer_%d", il);
        ggml_set_name(write_idx, idx_name);

        ggml_tensor *k_after = ggml_set_rows(ctx, k_gpu, k_cur_2d, write_idx);
        ggml_tensor *v_after = ggml_set_rows(ctx, v_gpu, v_cur_2d, write_idx);

        // View this slot's slice of the shared [kv_dim, n_kv_max*n_seq] buffer
        // (slot s = rows [s*n_kv_max, (s+1)*n_kv_max)). kv_slot_row_offset is 0
        // for the legacy single-buffer path.
        const int64_t n_total = current_opts_.n_kv_max;
        const size_t slot_off =
            (size_t)current_opts_.kv_slot_row_offset * (size_t)kv_dim *
            sizeof(float);
        ggml_tensor *k_view =
            ggml_view_2d(ctx, k_after, kv_dim, n_total, k_after->nb[1], slot_off);
        ggml_tensor *v_view =
            ggml_view_2d(ctx, v_after, kv_dim, n_total, v_after->nb[1], slot_off);
        return {ggml_reshape_3d(ctx, k_view, head_dim, n_head_kv, n_total),
                ggml_reshape_3d(ctx, v_view, head_dim, n_head_kv, n_total)};
      }
    }
  }

  if (current_opts_.is_decode_step && current_opts_.n_kv_cache > 0) {
    // Decode step: concatenate cached K/V with current K/V
    return build_kv_cache_concat(ctx, k_cur, v_cur, current_opts_.n_kv_cache,
                                 il);
  }

  // Prefill step: optionally mark K/V as output for host-side extraction
  (void)n_tokens;
  (void)il;

  if (current_opts_.use_kv_cache) {
    // Create contiguous 3D copies for both attention computation and output
    // extraction For contiguous tensors, 3D [head_dim, n_head_kv, n_tokens] and
    // 2D [kv_dim, n_tokens] have identical memory layout, so we can extract
    // data directly from the 3D tensor without creating a separate 2D output.
    ggml_tensor *k_3d = ggml_cont(ctx, k_cur);
    ggml_tensor *v_3d = ggml_cont(ctx, v_cur);

    char k_name[64], v_name[64];
    snprintf(k_name, sizeof(k_name), "kv_k_layer_%d", il);
    snprintf(v_name, sizeof(v_name), "kv_v_layer_%d", il);
    ggml_set_name(k_3d, k_name);
    ggml_set_name(v_3d, v_name);

    // Mark the 3D contiguous tensors as output for host-side extraction
    ggml_set_output(k_3d);
    ggml_set_output(v_3d);
    tmp_kv_outputs_k_.push_back(k_3d);
    tmp_kv_outputs_v_.push_back(v_3d);

    // Return 3D tensors for attention computation
    return {k_3d, v_3d};
  }

  return {k_cur, v_cur};
}

std::pair<ggml_tensor *, ggml_tensor *>
llm_graph_builder::build_kv_cache_concat(ggml_context *ctx, ggml_tensor *k_cur,
                                         ggml_tensor *v_cur, uint32_t n_cached,
                                         int32_t il) {
  const int64_t head_dim = k_cur->ne[0];
  const int64_t n_head_kv = k_cur->ne[1];

  // O3 optimisation: use GPU-persistent KV buffer if provided.
  // The GPU KV buffer already contains the first n_cached columns from
  // previous steps. We create a view of [kv_dim, n_cached+1] for attention,
  // but the NEW column at position n_cached is NOT yet written — it will be
  // written by the host after compute via ggml_backend_tensor_set.
  //
  // For correctness, we still need the current step's K/V to be part of the
  // attention input. We use the original concat path but REPLACE the
  // kv_cached_k/v input tensors with views into the GPU KV buffer.
  if (current_opts_.gpu_kv_k && current_opts_.gpu_kv_v &&
      current_opts_.n_kv_max > 0) {
    ggml_tensor *k_gpu = current_opts_.gpu_kv_k[il];
    ggml_tensor *v_gpu = current_opts_.gpu_kv_v[il];

    if (k_gpu && v_gpu && n_cached > 0) {
      // Create a view of the first n_cached columns in the GPU buffer
      // This replaces the kv_cached_k/v_layer_N input tensors
      const int64_t kv_dim_2d = head_dim * n_head_kv;
      ggml_tensor *k_cached_view =
          ggml_view_2d(ctx, k_gpu, kv_dim_2d, n_cached, k_gpu->nb[1], 0);
      ggml_tensor *v_cached_view =
          ggml_view_2d(ctx, v_gpu, kv_dim_2d, n_cached, v_gpu->nb[1], 0);
      // No ggml_set_input needed — the data is already in GPU memory

      // Reshape current K/V to 2D: [kv_dim, 1]
      ggml_tensor *k_cur_2d =
          ggml_reshape_2d(ctx, ggml_cont(ctx, k_cur), kv_dim_2d, k_cur->ne[2]);
      ggml_tensor *v_cur_2d =
          ggml_reshape_2d(ctx, ggml_cont(ctx, v_cur), kv_dim_2d, v_cur->ne[2]);

      // Concat GPU cached view + current K/V
      ggml_tensor *k_final_2d = ggml_concat(ctx, k_cached_view, k_cur_2d, 1);
      ggml_tensor *v_final_2d = ggml_concat(ctx, v_cached_view, v_cur_2d, 1);

      ggml_tensor *k_cont_2d = ggml_cont(ctx, k_final_2d);
      ggml_tensor *v_cont_2d = ggml_cont(ctx, v_final_2d);

      // Reshape to 3D for attention: [head_dim, n_head_kv, n_cached + 1]
      const int64_t n_total = n_cached + (int64_t)k_cur->ne[2];
      ggml_tensor *k_final =
          ggml_reshape_3d(ctx, k_cont_2d, head_dim, n_head_kv, n_total);
      ggml_tensor *v_final =
          ggml_reshape_3d(ctx, v_cont_2d, head_dim, n_head_kv, n_total);

      // Mark outputs for host-side extraction of the NEW column
      if (current_opts_.use_kv_cache) {
        char k_name[64], v_name[64];
        snprintf(k_name, sizeof(k_name), "kv_k_layer_%d", il);
        snprintf(v_name, sizeof(v_name), "kv_v_layer_%d", il);
        ggml_set_name(k_cont_2d, k_name);
        ggml_set_name(v_cont_2d, v_name);
        ggml_set_output(k_cont_2d);
        ggml_set_output(v_cont_2d);
        tmp_kv_outputs_k_.push_back(k_cont_2d);
        tmp_kv_outputs_v_.push_back(v_cont_2d);
      }

      return {k_final, v_final};
    }
  }

  // Original path: create input tensors for cached K/V (uploaded from host)
  ggml_tensor *k_cached =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim * n_head_kv, n_cached);
  ggml_tensor *v_cached =
      ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim * n_head_kv, n_cached);
  ggml_set_input(k_cached);
  ggml_set_input(v_cached);

  char k_cache_name[64], v_cache_name[64];
  snprintf(k_cache_name, sizeof(k_cache_name), "kv_cached_k_layer_%d", il);
  snprintf(v_cache_name, sizeof(v_cache_name), "kv_cached_v_layer_%d", il);
  ggml_set_name(k_cached, k_cache_name);
  ggml_set_name(v_cached, v_cache_name);

  // Reshape current K/V to 2D: [head_dim * n_head_kv, n_tokens]
  ggml_tensor *k_cur_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, k_cur),
                                          head_dim * n_head_kv, k_cur->ne[2]);
  ggml_tensor *v_cur_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, v_cur),
                                          head_dim * n_head_kv, v_cur->ne[2]);

  // Concat along dim[1]: [head_dim*n_head_kv, n_cached + n_tokens]
  ggml_tensor *k_final_2d = ggml_concat(ctx, k_cached, k_cur_2d, 1);
  ggml_tensor *v_final_2d = ggml_concat(ctx, v_cached, v_cur_2d, 1);

  // Make contiguous copies of 2D concat results
  ggml_tensor *k_cont_2d = ggml_cont(ctx, k_final_2d);
  ggml_tensor *v_cont_2d = ggml_cont(ctx, v_final_2d);

  // Reshape to 3D for attention: [head_dim, n_head_kv, n_cached + n_tokens]
  const int64_t n_total = n_cached + (int64_t)k_cur->ne[2];
  ggml_tensor *k_final =
      ggml_reshape_3d(ctx, k_cont_2d, head_dim, n_head_kv, n_total);
  ggml_tensor *v_final =
      ggml_reshape_3d(ctx, v_cont_2d, head_dim, n_head_kv, n_total);

  // Mark 2D outputs for consistent host-side extraction
  if (current_opts_.use_kv_cache) {
    char k_name[64], v_name[64];
    snprintf(k_name, sizeof(k_name), "kv_k_layer_%d", il);
    snprintf(v_name, sizeof(v_name), "kv_v_layer_%d", il);
    ggml_set_name(k_cont_2d, k_name);
    ggml_set_name(v_cont_2d, v_name);

    ggml_set_output(k_cont_2d);
    ggml_set_output(v_cont_2d);
    tmp_kv_outputs_k_.push_back(k_cont_2d);
    tmp_kv_outputs_v_.push_back(v_cont_2d);
  }

  return {k_final, v_final};
}

ggml_tensor *llm_graph_builder::build_multi_head_attn(
    ggml_context *ctx,
    ggml_tensor *q, // Expected: [d_k, n_head, n_tokens]
    ggml_tensor *k, // Expected: [d_k, n_head_kv, n_tokens_kv]
    ggml_tensor *v, // Expected: [d_k, n_head_kv, n_tokens_kv]
    ggml_tensor *mask, float scale, int32_t n_head, int32_t n_head_kv) {

  const int32_t d_k = q->ne[0];
  const int32_t n_tokens = q->ne[2];
  const int32_t n_tokens_kv = k->ne[2];

  // 1. Grouped-Query Attention (GQA) handling
  ggml_tensor *k_repeated = k;
  ggml_tensor *v_repeated = v;

  if (n_head != n_head_kv) {
    const int n_rep = n_head / n_head_kv;

    // --- Process Key (k) ---
    // Original k is [d_k, n_head_kv, n_tokens_kv]
    // Step 1: Reshape to [d_k, 1, n_head_kv, n_tokens_kv]
    struct ggml_tensor *k_reshaped =
        ggml_reshape_4d(ctx, k, d_k, 1, n_head_kv, n_tokens_kv);
    // Step 2: Repeat the 2nd dimension (ne[1]) to get [d_k, n_rep, n_head_kv,
    // n_tokens_kv]
    struct ggml_tensor *k_expanded = ggml_repeat(
        ctx, k_reshaped,
        ggml_new_tensor_4d(ctx, k->type, d_k, n_rep, n_head_kv, n_tokens_kv));
    // Step 3: Reshape to [d_k, n_head, n_tokens_kv]
    k_repeated = ggml_reshape_3d(ctx, k_expanded, d_k, n_head, n_tokens_kv);

    // --- Process Value (v) ---
    struct ggml_tensor *v_reshaped =
        ggml_reshape_4d(ctx, v, d_k, 1, n_head_kv, n_tokens_kv);
    struct ggml_tensor *v_expanded = ggml_repeat(
        ctx, v_reshaped,
        ggml_new_tensor_4d(ctx, v->type, d_k, n_rep, n_head_kv, n_tokens_kv));
    v_repeated = ggml_reshape_3d(ctx, v_expanded, d_k, n_head, n_tokens_kv);
  }

  // 2. Align layouts for Attention calculation
  // Q is [d_k, n_head, n_tokens]. We want [d_k, n_tokens, n_head]
  ggml_tensor *q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

  // K and V are [d_k, n_head, n_tokens_kv]. We want [d_k, n_tokens_kv, n_head]
  // This makes the 3rd dimension (n_head) the batch dimension for mul_mat
  ggml_tensor *k_perm =
      ggml_cont(ctx, ggml_permute(ctx, k_repeated, 0, 2, 1, 3));
  ggml_tensor *v_perm =
      ggml_cont(ctx, ggml_permute(ctx, v_repeated, 0, 2, 1, 3));

  // 3. Compute Attention Scores: kq = Q @ K^T
  // A: k_perm [d_k, n_tokens_kv, n_head], B: q_perm [d_k, n_tokens, n_head]
  // ggml_mul_mat: result is [n_tokens_kv, n_tokens, n_head]
  ggml_tensor *kq = ggml_mul_mat(ctx, k_perm, q_perm);
  // Force F32 accumulator: long-context prefill (reduction over d_k=128) on
  // CUDA Ampere/Turing/Ada uses F16 accumulators by default, which collapses
  // attention scores and produces degenerate softmax. See FunASR Nano F16
  // regression where lm_head logits saturate to token 0.
  ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

  kq = ggml_scale(ctx, kq, scale);

  if (mask) {
    // mask is [n_kv, n_tokens], kq is [n_tokens_kv, n_tokens, n_head]
    // ggml_add broadcasts src1 (mask) across higher dimensions of src0 (kq)
    kq = ggml_add(ctx, kq, mask);
  }

  ggml_tensor *probs = ggml_soft_max_inplace(ctx, kq);

  // 4. Compute Weighted Sum: kqv = Probs @ V
  // V_perm is [d_k, n_tokens_kv, n_head]
  // To use ggml_mul_mat(A, B) -> B * A^T:
  // A must be [n_tokens_kv, d_k, n_head], B is [n_tokens_kv, n_tokens, n_head]
  ggml_tensor *v_tr = ggml_cont(ctx, ggml_transpose(ctx, v_perm));

  // Result: [d_k, n_tokens, n_head]
  ggml_tensor *kqv = ggml_mul_mat(ctx, v_tr, probs);

  // 5. Final permutation back to [d_k, n_head, n_tokens]
  return ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
}

ggml_tensor *llm_graph_builder::build_flash_attn(ggml_context *ctx,
                                                 ggml_tensor *q, ggml_tensor *k,
                                                 ggml_tensor *v,
                                                 ggml_tensor *mask,
                                                 float scale) {
  // Flash attention with optional causal mask.
  //
  // Single-sequence (3D inputs): q/k/v are [d_k, n_head*, n_tokens/n_kv].
  //   permute (0,2,1,3) -> [d_k, n_tokens, n_head] etc.; ne3=1.
  //
  // Batched (4D inputs, n_seq>1): q=[d_k, 1, n_head, n_seq] already in flash
  //   layout (n_batch=1, ne3=n_seq) so q needs no permute; k/v are
  //   [d_k, n_head_kv, n_kv, n_seq] -> permute (0,2,1,3) -> [d_k, n_kv,
  //   n_head_kv, n_seq]. Each sequence attends only over its own KV slice
  //   (isolated by ne3) - no N^2 cross-sequence attention.
  const bool batched = ggml_n_dims(q) >= 4;

  if (batched) {
    q = ggml_cont(ctx, q); // [d_k, 1, n_head, n_seq] (already flash layout)
    k = ggml_cont(ctx,
                  ggml_permute(ctx, k, 0, 2, 1, 3)); // -> [d_k, n_kv, n_head_kv, n_seq]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
  } else {
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
  }

  // ggml_flash_attn_ext requires mask to be F16, convert if necessary.
  if (mask && mask->type != GGML_TYPE_F16) {
    mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
  }

  ggml_tensor *result =
      ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);

  if (batched) {
    // result [d_k, n_head, 1, n_seq] -> [d_k, n_head, n_seq]
    return ggml_reshape_3d(ctx, result, result->ne[0], result->ne[1],
                           result->ne[3]);
  }
  // Result is [d_k, n_head, n_batch, 1], reshape to [d_k, n_head, n_batch]
  return ggml_reshape_3d(ctx, result, result->ne[0], result->ne[1],
                         result->ne[2]);
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
    ggml_tensor *gate_out = ggml_mul_mat(ctx, w_gate, cur);
    ggml_tensor *up_out   = ggml_mul_mat(ctx, w_up,   cur);
    result = ggml_swiglu_split(ctx, gate_out, up_out);
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
  ggml_tensor *logits = ggml_mul_mat(ctx, output, cur);
  // Force F32 accumulator: large-vocab GEMM (e.g. Qwen3 151936) overflows
  // F16 accumulators on CUDA Ampere/Turing/Ada, collapsing logits to token 0.
  ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
  return logits;
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
    ggml_tensor *gate_out = ggml_mul_mat(ctx, gate, cur);
    ggml_tensor *up_out   = ggml_mul_mat(ctx, up,   cur);
    result = ggml_swiglu_split(ctx, gate_out, up_out);
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

  // Qwen2 vs Qwen3 share the LLM_ARCH_QWEN* enums but differ in attention:
  // Qwen2 has Q/K/V bias and no Q/K-norm; Qwen3 is the opposite. Route by
  // hparams flag rather than relying on registration order, so OmniVoice
  // (Qwen3 weights tagged "qwen2"/"omnivoice-lm") and CosyVoice3-LLM (true
  // Qwen2) end up on the right builder.
  if (arch == LLM_ARCH_QWEN2 || arch == LLM_ARCH_QWEN3) {
    llm_arch desired = model.hparams().use_qkv_bias ? LLM_ARCH_QWEN2
                                                    : LLM_ARCH_QWEN3;
    auto *factory = get_registry().get_builder(desired);
    if (factory) return (*factory)(model, cparams, sched);
  }

  auto *factory = get_registry().get_builder(arch);
  if (!factory) {
    return nullptr;
  }

  return (*factory)(model, cparams, sched);
}
