#include "qwen3.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

/**
 * Mimics torch.repeat_interleave(repeats, dim=-3) specifically for GQA.
 * Goal: Transform [H0, H1] into [H0, H0, H1, H1]
 */
struct ggml_tensor * ggml_repeat_interleave_gqa(
    struct ggml_context * ctx,
    struct ggml_tensor * kv_tensor,
    int repeats
) {
  if (repeats <= 1) {
    return kv_tensor;
  }

  // Original tensor dims: {d_k, n_h, n_s, n_b} -> e.g., {128, 8, 52, 1}
  const int64_t d_k = kv_tensor->ne[0];
  const int64_t n_h = kv_tensor->ne[1];
  const int64_t n_s = kv_tensor->ne[2];
  const int64_t n_b = kv_tensor->ne[3];

  // 1. Reshape to split the head dimension into its own repeating axis.
  // Move heads (n_h) and sequences (n_s) to higher dims to isolate the repeating axis.
  // Logic: {d_k, 1, n_h, n_s * n_b}
  struct ggml_tensor * view_src = ggml_reshape_4d(ctx, kv_tensor,
      d_k,
      1,
      n_h,
      n_s * n_b
  );

  // 2. Set target shape to expand the '1' dimension.
  // This forces an interleaved repeat pattern in memory.
  struct ggml_tensor * target_shape = ggml_new_tensor_4d(ctx, kv_tensor->type,
      d_k,
      repeats,
      n_h,
      n_s * n_b
  );

  // 3. Repeat operation.
  // This creates the [H0_data, H0_copy, H1_data, H1_copy...] pattern.
  struct ggml_tensor * repeated = ggml_repeat(ctx, view_src, target_shape);

  // 4. Reshape back to standard multi-head attention format.
  // Final shape: {128, 16, 52, 1}
  struct ggml_tensor * result = ggml_reshape_4d(ctx, repeated,
      d_k,
      n_h * repeats,
      n_s,
      n_b
  );

  return result;
}

// ============================================
// Qwen3 Special Tokens
// ============================================

static llm_model_ptr load_qwen3_model(struct gguf_context *ctx_gguf,
                                      ggml_backend_t backend) {
  auto model = std::make_shared<llm_model>();

  if (!model->load_from_gguf(ctx_gguf, nullptr, backend, "")) {
    return nullptr;
  }

  if (model->arch() != LLM_ARCH_QWEN3) {
    return nullptr;
  }

  // Load Qwen3 special tokens
  auto &vocab = const_cast<llm_vocab &>(model->vocab());
  // Note: Special tokens are loaded from GGUF vocabulary automatically
  // Qwen3 specific tokens will be loaded by name matching
  vocab.load_qwen3_special_tokens();

  return model;
}

static std::unique_ptr<llm_graph_builder>
create_qwen3_builder(const llm_model &model, const llm_cparams &cparams,
                     ggml_backend_sched_t sched) {
  return std::make_unique<llm_build_qwen3>(model, cparams, sched);
}

void llm_register_qwen3() {
  llm_register_model("qwen3", load_qwen3_model);
  llm_register_model("qwen2", load_qwen3_model);

  llm_register_graph_builder(LLM_ARCH_QWEN3, create_qwen3_builder);
  llm_register_graph_builder(LLM_ARCH_QWEN2, create_qwen3_builder);
}

static struct Qwen3Registrar {
  Qwen3Registrar() { llm_register_qwen3(); }
} g_qwen3_registrar;

// ============================================
// llm_build_qwen3 Implementation
// ============================================

llm_build_qwen3::llm_build_qwen3(const llm_model &model,
                                 const llm_cparams &cparams,
                                 ggml_backend_sched_t sched)
    : llm_graph_builder(model.hparams(), cparams, sched), model_(model) {}

bool llm_build_qwen3::should_extract_layer(int32_t il) const {
  if (!current_opts_.extract_intermediate) {
    return false;
  }
  if (current_opts_.extract_layers.empty()) {
    return false;
  }
  return std::find(current_opts_.extract_layers.begin(),
                   current_opts_.extract_layers.end(),
                   il) != current_opts_.extract_layers.end();
}

llm_graph_result_ptr llm_build_qwen3::build_graph(const int32_t *tokens,
                                                  uint32_t n_tokens,
                                                  llm_kv_cache *kv_cache,
                                                  const llm_pos *pos,
                                                  const llm_build_opts *opts) {
  const auto &hparams = model_.hparams();
  current_opts_ = opts ? *opts : llm_build_opts();
  init_graph(16384);

  auto result = std::make_unique<llm_graph_result>(16384);

  ggml_tensor *cur = nullptr;
  if (current_opts_.skip_embeddings) {
    cur = ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(cur);
  } else {
    cur = build_embedding_layer(ctx_, tokens, n_tokens);
  }

  const int32_t n_layers_start = current_opts_.n_layers_start;
  const int32_t n_layers_end = current_opts_.n_layers_end < 0
                                   ? static_cast<int32_t>(hparams.n_layer)
                                   : current_opts_.n_layers_end;

  // Build position tensor once for all layers
  ggml_tensor *positions = build_position_tensor(ctx_, pos, n_tokens);

  // Build causal mask once for all layers
  ggml_tensor *causal_mask = nullptr;
  if (current_opts_.causal_mask) {
    uint32_t n_kv_cache = kv_cache ? kv_cache->size() : current_opts_.n_kv_cache;
    causal_mask = build_causal_mask_tensor(ctx_, n_tokens, n_kv_cache);
  }

  for (int32_t il = n_layers_start; il < n_layers_end; ++il) {
    cur = build_transformer_layer(ctx_, cur, il, kv_cache, positions, causal_mask, n_tokens);
    if (should_extract_layer(il)) {
      result->add_intermediate_output(cur);
    }
  }

  switch (current_opts_.output_mode) {
  case llm_output_mode::OUTPUT_LOGITS: {
    if (!current_opts_.skip_output_norm) {
      cur = build_output_norm(ctx_, cur, model_.output_norm(),
                              model_.output_norm_b());
    }
    if (!current_opts_.skip_lm_head) {
      cur = build_lm_head(ctx_, cur, model_.output());
    }
    result->set_logits(cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    break;
  }
  case llm_output_mode::OUTPUT_EMBEDDINGS:
  case llm_output_mode::OUTPUT_FEATURES: {
    result->set_embd(cur);
    ggml_set_name(cur, "embeddings");
    ggml_set_output(cur);
    break;
  }
  case llm_output_mode::OUTPUT_ENCODER: {
    result->set_embd(cur);
    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    break;
  }
  }

  ggml_build_forward_expand(gf_, cur);
  result->set_graph(gf_);
  result->set_ctx(ctx_);

  // Transfer KV output tensors to result for host-side extraction
  for (size_t i = 0; i < tmp_kv_outputs_k_.size(); ++i) {
    result->add_kv_output(tmp_kv_outputs_k_[i], tmp_kv_outputs_v_[i]);
  }
  result->update_params(
      {hparams.arch, hparams, cparams_, sched_, 0,
       current_opts_.output_mode == llm_output_mode::OUTPUT_LOGITS},
      current_opts_);

  ctx_ = nullptr;
  gf_ = nullptr;

  return result;
}

llm_graph_result_ptr llm_build_qwen3::build_graph_from_embeds(
    ggml_tensor *embeds, uint32_t n_tokens, llm_kv_cache *kv_cache,
    const llm_pos *pos, const llm_build_opts *opts) {
  const auto &hparams = model_.hparams();
  current_opts_ = opts ? *opts : llm_build_opts();
  current_opts_.skip_embeddings = true;
  init_graph(16384);

  auto result = std::make_unique<llm_graph_result>(16384);
  ggml_tensor *cur = embeds;

  const int32_t n_layers_start = current_opts_.n_layers_start;
  const int32_t n_layers_end = current_opts_.n_layers_end < 0
                                   ? static_cast<int32_t>(hparams.n_layer)
                                   : current_opts_.n_layers_end;

  // Build position tensor once for all layers
  ggml_tensor *positions = build_position_tensor(ctx_, pos, n_tokens);

  // Build causal mask once for all layers
  ggml_tensor *causal_mask = nullptr;
  if (current_opts_.causal_mask) {
    uint32_t n_kv_cache = kv_cache ? kv_cache->size() : current_opts_.n_kv_cache;
    causal_mask = build_causal_mask_tensor(ctx_, n_tokens, n_kv_cache);
  }

  for (int32_t il = n_layers_start; il < n_layers_end; ++il) {
    cur = build_transformer_layer(ctx_, cur, il, kv_cache, positions, causal_mask, n_tokens);
    if (should_extract_layer(il)) {
      result->add_intermediate_output(cur);
    }
  }

  switch (current_opts_.output_mode) {
  case llm_output_mode::OUTPUT_LOGITS: {
    if (!current_opts_.skip_output_norm) {
      cur = build_output_norm(ctx_, cur, model_.output_norm(),
                              model_.output_norm_b());
    }
    if (!current_opts_.skip_lm_head) {
      cur = build_lm_head(ctx_, cur, model_.output());
    }
    result->set_logits(cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    break;
  }
  case llm_output_mode::OUTPUT_EMBEDDINGS:
  case llm_output_mode::OUTPUT_FEATURES: {
    result->set_embd(cur);
    ggml_set_name(cur, "embeddings");
    ggml_set_output(cur);
    break;
  }
  case llm_output_mode::OUTPUT_ENCODER: {
    result->set_embd(cur);
    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    break;
  }
  }

  ggml_build_forward_expand(gf_, cur);
  result->set_graph(gf_);
  result->set_ctx(ctx_);

  // Transfer KV output tensors to result for host-side extraction
  for (size_t i = 0; i < tmp_kv_outputs_k_.size(); ++i) {
    result->add_kv_output(tmp_kv_outputs_k_[i], tmp_kv_outputs_v_[i]);
  }
  result->update_params(
      {hparams.arch, hparams, cparams_, sched_, 0,
       current_opts_.output_mode == llm_output_mode::OUTPUT_LOGITS},
      current_opts_);

  ctx_ = nullptr;
  gf_ = nullptr;

  return result;
}

ggml_tensor *llm_build_qwen3::build_embedding_layer(ggml_context *ctx,
                                                    const int32_t *tokens,
                                                    uint32_t n_tokens) {
  return build_token_embeds(ctx, tokens, n_tokens, model_.tok_embd());
}

ggml_tensor *llm_build_qwen3::build_transformer_layer(
    ggml_context *ctx, ggml_tensor *cur, int32_t il, llm_kv_cache *kv_cache,
    ggml_tensor *positions, ggml_tensor *causal_mask, uint32_t n_tokens) {
  const auto &layer = model_.layer(il);
  const auto &hparams = model_.hparams();

  ggml_tensor *residual = cur;
  cur = build_norm(ctx, cur, layer.attn_norm, nullptr, llm_norm_type::RMS_NORM,
                   hparams.f_norm_rms_eps);
  cur = build_attention_layer(ctx, cur, il, kv_cache, positions, causal_mask, n_tokens);
  cur = build_residual(ctx, cur, residual);

  residual = cur;
  cur = build_norm(ctx, cur, layer.ffn_norm, nullptr, llm_norm_type::RMS_NORM,
                   hparams.f_norm_rms_eps);
  cur = build_ffn_layer(ctx, cur, il);
  cur = build_residual(ctx, cur, residual);

  return cur;
}

ggml_tensor *
llm_build_qwen3::build_attention_layer(ggml_context *ctx, ggml_tensor *cur,
                                       int32_t il, llm_kv_cache *kv_cache,
                                       ggml_tensor *positions,
                                       ggml_tensor *causal_mask,
                                       uint32_t n_tokens) {
  const auto &layer = model_.layer(il);
  const auto &hparams = model_.hparams();

  const int32_t n_head = hparams.n_head;
  const int32_t n_head_kv = hparams.n_head_kv;
  const int32_t n_embd_head = hparams.head_dim;
  const int32_t n_rot = hparams.n_rot;

  ggml_tensor *q = ggml_mul_mat(ctx, layer.wq, cur);
  ggml_tensor *k = ggml_mul_mat(ctx, layer.wk, cur);
  ggml_tensor *v = ggml_mul_mat(ctx, layer.wv, cur);

  q = ggml_reshape_3d(ctx, q, n_embd_head, n_head, n_tokens);
  k = ggml_reshape_3d(ctx, k, n_embd_head, n_head_kv, n_tokens);
  v = ggml_reshape_3d(ctx, v, n_embd_head, n_head_kv, n_tokens);

  if (layer.attn_q_norm && layer.attn_k_norm && hparams.use_kq_norm) {
    q = ggml_rms_norm(ctx, q, hparams.f_norm_rms_eps);
    q = ggml_mul(ctx, q, layer.attn_q_norm);
    k = ggml_rms_norm(ctx, k, hparams.f_norm_rms_eps);
    k = ggml_mul(ctx, k, layer.attn_k_norm);
  }

  q = build_rope_embeds(ctx, q, positions, n_rot, n_head, GGML_ROPE_TYPE_NEOX);
  k = build_rope_embeds(ctx, k, positions, n_rot, n_head, GGML_ROPE_TYPE_NEOX);

  auto [k_final, v_final] =
      build_kv_cache_lookup(ctx, k, v, kv_cache, n_tokens, il);

  const float scale = 1.0f / sqrtf(static_cast<float>(n_embd_head));

  // DEBUG: Print Q, K, V shapes
  printf("[DEBUG Layer %d] Q shape: [%lld, %lld, %lld]\n", il, (long long)q->ne[0], (long long)q->ne[1], (long long)q->ne[2]);
  printf("[DEBUG Layer %d] K shape: [%lld, %lld, %lld]\n", il, (long long)k->ne[0], (long long)k->ne[1], (long long)k->ne[2]);
  printf("[DEBUG Layer %d] V shape: [%lld, %lld, %lld]\n", il, (long long)v->ne[0], (long long)v->ne[1], (long long)v->ne[2]);
  printf("[DEBUG Layer %d] K_final shape: [%lld, %lld, %lld]\n", il, (long long)k_final->ne[0], (long long)k_final->ne[1], (long long)k_final->ne[2]);
  printf("[DEBUG Layer %d] V_final shape: [%lld, %lld, %lld]\n", il, (long long)v_final->ne[0], (long long)v_final->ne[1], (long long)v_final->ne[2]);
  printf("[DEBUG Layer %d] scale: %f\n", il, scale);
  if (causal_mask) {
    printf("[DEBUG Layer %d] causal_mask shape: [%lld, %lld]\n", il, (long long)causal_mask->ne[0], (long long)causal_mask->ne[1]);
  }

  int n_heads_q = q->ne[2]; // 16
  int n_heads_kv = k->ne[2]; // 8
  int group_size = n_heads_q / n_heads_kv; // 2

  // if (group_size > 1) {
  //   // 1. Generate interleaved view
  //   struct ggml_tensor * k_inter = ggml_repeat_interleave_gqa(ctx, k_final, group_size);
  //
  //   // 2. CRITICAL: Force memory synchronization.
  //   // After this line, your 'sum' should double (e.g., ~ -50.91).
  //   k_final = ggml_cont(ctx, k_inter);
  //
  //   // 3. Permute to get {d_k, n_s, n_h_total, n_b}
  //   k_final = ggml_permute(ctx, k_final, 0, 2, 1, 3);
  //   ggml_set_name(k_final, "k_final");
  //
  //   // Repeat for Value tensor
  //   struct ggml_tensor * v_inter = ggml_repeat_interleave_gqa(ctx, v_final, group_size);
  //   v_final = ggml_cont(ctx, v_inter);
  //   v_final = ggml_permute(ctx, v_final, 0, 2, 1, 3);
  //   ggml_set_name(v_final, "v_final");
  // }

  if (current_opts_.use_flash_attn) {
    cur = build_flash_attn(ctx, q, k_final, v_final, causal_mask, scale);
  } else {
    cur = build_multi_head_attn(ctx, q, k_final, v_final, causal_mask, scale,
                                n_head, n_head_kv);
  }

  // DEBUG: Print attention output shape
  printf("[DEBUG Layer %d] Attention output shape (before reshape): [%lld, %lld, %lld]\n", il, (long long)cur->ne[0], (long long)cur->ne[1], (long long)cur->ne[2]);

  // Reshape from [head_dim, n_head, n_tokens] to [head_dim * n_head, n_tokens]
  // Note: head_dim * n_head may differ from n_embd in some architectures
  cur =
      ggml_reshape_2d(ctx, ggml_cont(ctx, cur), n_embd_head * n_head, n_tokens);
  cur = ggml_mul_mat(ctx, layer.wo, cur);

  // Set name for debugging so we can find it later
  ggml_set_name(cur, "attention_output_wo");

  return cur;
}

ggml_tensor *llm_build_qwen3::build_ffn_layer(ggml_context *ctx,
                                              ggml_tensor *cur, int32_t il) {
  const auto &layer = model_.layer(il);
  return build_ffn(ctx, cur, layer.ffn_up, layer.ffn_gate, layer.ffn_down,
                   llm_ffn_type::FFN_SWIGLU);
}

ggml_tensor *llm_build_qwen3::build_position_tensor(ggml_context *ctx,
                                                    const llm_pos *pos,
                                                    uint32_t n_tokens) {
  return build_position_ids(ctx, pos, n_tokens);
}

ggml_tensor *llm_build_qwen3::build_causal_mask_tensor(ggml_context *ctx,
                                                       uint32_t n_tokens,
                                                       uint32_t n_kv_cache) {
  return build_causal_mask(ctx, n_tokens, n_kv_cache);
}
