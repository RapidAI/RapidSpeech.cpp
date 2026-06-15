#include "qwen2.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <vector>

// ============================================
// Qwen2 model loader (registers GGUF arch string "cosyvoice3-llm")
// ============================================

static llm_model_ptr load_qwen2_model(struct gguf_context *ctx_gguf,
                                      ggml_backend_t backend) {
  auto model = std::make_shared<llm_model>();
  if (!model->load_from_gguf(ctx_gguf, nullptr, backend, "")) {
    return nullptr;
  }
  if (model->arch() != LLM_ARCH_QWEN2) {
    return nullptr;
  }
  return model;
}

static std::unique_ptr<llm_graph_builder>
create_qwen2_builder(const llm_model &model, const llm_cparams &cparams,
                     ggml_backend_sched_t sched) {
  return std::make_unique<llm_build_qwen2>(model, cparams, sched);
}

void llm_register_qwen2() {
  // Register loader for the cosyvoice3-llm GGUF arch string. The body is
  // Qwen2 + speech heads; downstream `cosyvoice3` arch-class will call
  // model->map_tensors_qwen2() during its own Load().
  llm_register_model("cosyvoice3-llm", load_qwen2_model);

  // Register a graph builder under LLM_ARCH_QWEN2. Whichever of qwen2/qwen3
  // registers last wins for that arch enum, but we always reach this builder
  // through `llm_create_graph_builder` which checks `use_qkv_bias` first
  // (see llm_graph.cpp) — so this registration is just defensive.
  llm_register_graph_builder(LLM_ARCH_QWEN2, create_qwen2_builder);
}

static struct Qwen2Registrar {
  Qwen2Registrar() { llm_register_qwen2(); }
} g_qwen2_registrar;

// ============================================
// llm_build_qwen2
// ============================================

llm_build_qwen2::llm_build_qwen2(const llm_model &model,
                                 const llm_cparams &cparams,
                                 ggml_backend_sched_t sched)
    : llm_graph_builder(model.hparams(), cparams, sched), model_(model) {}

bool llm_build_qwen2::should_extract_layer(int32_t il) const {
  if (!current_opts_.extract_intermediate) return false;
  if (current_opts_.extract_layers.empty()) return false;
  return std::find(current_opts_.extract_layers.begin(),
                   current_opts_.extract_layers.end(),
                   il) != current_opts_.extract_layers.end();
}

llm_graph_result_ptr llm_build_qwen2::build_graph(const int32_t *tokens,
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

  ggml_tensor *positions = build_position_tensor(ctx_, pos, n_tokens);
  ggml_tensor *causal_mask = nullptr;
  if (current_opts_.causal_mask) {
    uint32_t n_kv_cache =
        kv_cache ? kv_cache->size() : current_opts_.n_kv_cache;
    if (current_opts_.is_decode_step && current_opts_.fixed_kv_cache_shape &&
        current_opts_.n_kv_max > n_tokens) {
      n_kv_cache = current_opts_.n_kv_max - n_tokens;
    }
    causal_mask = build_causal_mask_tensor(ctx_, n_tokens, n_kv_cache);
  }

  for (int32_t il = n_layers_start; il < n_layers_end; ++il) {
    cur = build_transformer_layer(ctx_, cur, il, kv_cache, positions,
                                  causal_mask, n_tokens);
    if (should_extract_layer(il)) {
      ggml_set_name(cur, ("layer_" + std::to_string(il) + "_hidden").c_str());
      ggml_set_output(cur);
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

llm_graph_result_ptr llm_build_qwen2::build_graph_from_embeds(
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

  ggml_tensor *positions = build_position_tensor(ctx_, pos, n_tokens);
  ggml_tensor *causal_mask = nullptr;
  if (current_opts_.causal_mask) {
    uint32_t n_kv_cache =
        kv_cache ? kv_cache->size() : current_opts_.n_kv_cache;
    if (current_opts_.is_decode_step && current_opts_.fixed_kv_cache_shape &&
        current_opts_.n_kv_max > n_tokens) {
      n_kv_cache = current_opts_.n_kv_max - n_tokens;
    }
    causal_mask = build_causal_mask_tensor(ctx_, n_tokens, n_kv_cache);
  }

  for (int32_t il = n_layers_start; il < n_layers_end; ++il) {
    cur = build_transformer_layer(ctx_, cur, il, kv_cache, positions,
                                  causal_mask, n_tokens);
    if (should_extract_layer(il)) {
      ggml_set_name(cur, ("layer_" + std::to_string(il) + "_hidden").c_str());
      ggml_set_output(cur);
      result->add_intermediate_output(cur);
    }
  }

  switch (current_opts_.output_mode) {
  case llm_output_mode::OUTPUT_LOGITS: {
    if (n_tokens > 1 && !current_opts_.is_decode_step) {
      cur = ggml_view_2d(ctx_, cur,
                         cur->ne[0], 1, cur->nb[1],
                         (size_t)(n_tokens - 1) * cur->nb[1]);
      ggml_set_name(cur, "last_hidden");
    }
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

ggml_tensor *llm_build_qwen2::build_embedding_layer(ggml_context *ctx,
                                                    const int32_t *tokens,
                                                    uint32_t n_tokens) {
  return build_token_embeds(ctx, tokens, n_tokens, model_.tok_embd());
}

ggml_tensor *llm_build_qwen2::build_transformer_layer(
    ggml_context *ctx, ggml_tensor *cur, int32_t il, llm_kv_cache *kv_cache,
    ggml_tensor *positions, ggml_tensor *causal_mask, uint32_t n_tokens) {
  const auto &layer = model_.layer(il);
  const auto &hparams = model_.hparams();

  ggml_tensor *residual = cur;
  cur = build_norm(ctx, cur, layer.attn_norm, nullptr, llm_norm_type::RMS_NORM,
                   hparams.f_norm_rms_eps);
  cur = build_attention_layer(ctx, cur, il, kv_cache, positions, causal_mask,
                              n_tokens);
  cur = build_residual(ctx, cur, residual);

  residual = cur;
  cur = build_norm(ctx, cur, layer.ffn_norm, nullptr, llm_norm_type::RMS_NORM,
                   hparams.f_norm_rms_eps);
  cur = build_ffn_layer(ctx, cur, il);
  cur = build_residual(ctx, cur, residual);
  return cur;
}

ggml_tensor *llm_build_qwen2::build_attention_layer(
    ggml_context *ctx, ggml_tensor *cur, int32_t il, llm_kv_cache *kv_cache,
    ggml_tensor *positions, ggml_tensor *causal_mask, uint32_t n_tokens) {
  const auto &layer = model_.layer(il);
  const auto &hparams = model_.hparams();

  const int32_t n_head     = hparams.n_head;
  const int32_t n_head_kv  = hparams.n_head_kv;
  const int32_t n_embd_head = hparams.head_dim;
  const int32_t n_rot      = hparams.n_rot;

  ggml_tensor *q = nullptr;
  ggml_tensor *k = nullptr;
  ggml_tensor *v = nullptr;

  if (layer.wqkv) {
    // Fused QKV path: one mul_mat + one add, then three zero-cost views.
    // Layout per token in qkv: [Q (n_q_out) | K (n_kv_out) | V (n_kv_out)].
    const int64_t n_q_out  = (int64_t)n_head    * n_embd_head;
    const int64_t n_kv_out = (int64_t)n_head_kv * n_embd_head;
    const int64_t n_total  = n_q_out + 2 * n_kv_out;
    const size_t  es       = sizeof(float); // mul_mat output is F32

    ggml_tensor *qkv = ggml_mul_mat(ctx, layer.wqkv, cur); // [n_total, n_tokens]
    if (layer.wqkv_b) qkv = ggml_add(ctx, qkv, layer.wqkv_b);

    q = ggml_view_3d(ctx, qkv, n_embd_head, n_head,    n_tokens,
                     n_embd_head * es, n_total * es, 0);
    k = ggml_view_3d(ctx, qkv, n_embd_head, n_head_kv, n_tokens,
                     n_embd_head * es, n_total * es, n_q_out * es);
    v = ggml_view_3d(ctx, qkv, n_embd_head, n_head_kv, n_tokens,
                     n_embd_head * es, n_total * es,
                     (n_q_out + n_kv_out) * es);
  } else {
    q = ggml_mul_mat(ctx, layer.wq, cur);
    k = ggml_mul_mat(ctx, layer.wk, cur);
    v = ggml_mul_mat(ctx, layer.wv, cur);

    // Qwen2 difference vs Qwen3: add Q/K/V bias before head split.
    // q/k/v shape here is [n_embd_q_or_kv, n_tokens]; bias [n_embd_q_or_kv,]
    // broadcasts along ne[1] automatically.
    if (layer.wq_b) q = ggml_add(ctx, q, layer.wq_b);
    if (layer.wk_b) k = ggml_add(ctx, k, layer.wk_b);
    if (layer.wv_b) v = ggml_add(ctx, v, layer.wv_b);

    q = ggml_reshape_3d(ctx, q, n_embd_head, n_head, n_tokens);
    k = ggml_reshape_3d(ctx, k, n_embd_head, n_head_kv, n_tokens);
    v = ggml_reshape_3d(ctx, v, n_embd_head, n_head_kv, n_tokens);
  }

  // Qwen2 has no Q/K RMS-norm — skip the qwen3 block here.

  q = build_rope_embeds(ctx, q, positions, n_rot, n_head, GGML_ROPE_TYPE_NEOX);
  k = build_rope_embeds(ctx, k, positions, n_rot, n_head, GGML_ROPE_TYPE_NEOX);

  auto [k_final, v_final] =
      build_kv_cache_lookup(ctx, k, v, kv_cache, n_tokens, il);

  const float scale = 1.0f / sqrtf(static_cast<float>(n_embd_head));

  if (cparams_.flash_attn) {
    cur = build_flash_attn(ctx, q, k_final, v_final, causal_mask, scale);
  } else {
    cur = build_multi_head_attn(ctx, q, k_final, v_final, causal_mask, scale,
                                n_head, n_head_kv);
  }

  cur = ggml_reshape_2d(ctx, cur, n_embd_head * n_head, n_tokens);
  cur = ggml_mul_mat(ctx, layer.wo, cur);
  ggml_set_name(cur, "attention_output_wo");
  return cur;
}

ggml_tensor *llm_build_qwen2::build_ffn_layer(ggml_context *ctx,
                                              ggml_tensor *cur, int32_t il) {
  const auto &layer = model_.layer(il);
  return build_ffn(ctx, cur, layer.ffn_up, layer.ffn_gate, layer.ffn_down,
                   llm_ffn_type::FFN_SWIGLU);
}

ggml_tensor *llm_build_qwen2::build_position_tensor(ggml_context *ctx,
                                                    const llm_pos *pos,
                                                    uint32_t n_tokens) {
  return build_position_ids(ctx, pos, n_tokens);
}

ggml_tensor *llm_build_qwen2::build_causal_mask_tensor(ggml_context *ctx,
                                                       uint32_t n_tokens,
                                                       uint32_t n_kv_cache) {
  return build_causal_mask(ctx, n_tokens, n_kv_cache);
}
