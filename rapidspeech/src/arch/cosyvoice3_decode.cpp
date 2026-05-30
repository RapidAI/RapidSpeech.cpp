#include "cosyvoice3.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// =====================================================================
// Helpers — locating a non-CPU backend handle owned by the scheduler.
// =====================================================================

ggml_backend_t pick_gpu_backend(ggml_backend_sched_t sched,
                                ggml_backend_t fallback) {
  const int n_backends = ggml_backend_sched_get_n_backends(sched);
  for (int i = 0; i < n_backends; ++i) {
    ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
      return b;
    }
  }
  return fallback ? fallback : ggml_backend_sched_get_backend(sched, n_backends - 1);
}

// =====================================================================
// Build a tiny CPU-side projection graph that maps a single hidden state
// vector (1, d_model) through speech_lm_head → logits (1, speech_vocab).
//
// We feed `hidden` as an external buffer so callers can produce it from
// arbitrary upstream graphs (prefill or AR step).
// =====================================================================

std::vector<float> project_speech_logits(ggml_backend_sched_t sched,
                                         ggml_tensor *output_norm,
                                         ggml_tensor *speech_lm_head,
                                         float rms_eps,
                                         const float *hidden, int d_model,
                                         int speech_vocab) {
  ggml_init_params ip = {ggml_graph_overhead() + 8 * ggml_tensor_overhead(), nullptr,
                          true};
  ggml_context *ctx = ggml_init(ip);
  ggml_cgraph *gf = ggml_new_graph(ctx);

  ggml_tensor *h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, 1);
  ggml_set_name(h, "cv3_hidden_in");
  ggml_set_input(h);

  // Apply final RMSNorm (Qwen2 model.norm) before projection — qwen2 builder's
  // OUTPUT_EMBEDDINGS branch skips this, so we do it here on the CPU side.
  ggml_tensor *normed = h;
  if (output_norm) {
    normed = ggml_rms_norm(ctx, h, rms_eps);
    normed = ggml_mul(ctx, normed, output_norm);
  }

  // speech_lm_head has shape [d_model, speech_vocab] (llama convention).
  // ggml_mul_mat(W, x): W is [K, M], x is [K, N] → [M, N].
  ggml_tensor *logits = ggml_mul_mat(ctx, speech_lm_head, normed);
  ggml_set_name(logits, "cv3_speech_logits");
  ggml_set_output(logits);

  ggml_build_forward_expand(gf, logits);

  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("CosyVoice3-LLM: alloc speech_lm_head graph failed");
    ggml_free(ctx);
    return {};
  }
  ggml_backend_tensor_set(h, hidden, 0, (size_t)d_model * sizeof(float));
  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("CosyVoice3-LLM: speech_lm_head compute failed");
    ggml_free(ctx);
    return {};
  }

  std::vector<float> out((size_t)speech_vocab);
  ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));

  ggml_free(ctx);
  ggml_backend_sched_reset(sched);
  return out;
}

} // namespace

// =====================================================================
// Decode: prefill text → first speech token, then AR loop.
// =====================================================================

bool CosyVoice3LMModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  auto &s = static_cast<CosyVoice3State &>(state);
  if (!llm_model_) return false;
  if (s.text_token_ids.empty()) {
    RS_LOG_ERR("CosyVoice3-LLM: empty text — call PushText first");
    return false;
  }

  const auto &hp = llm_model_->hparams();
  const int n_layer = (int)hp.n_layer;
  const int d_model = (int)hp.n_embd;
  const int n_head_kv = (int)(hp.n_head_kv > 0 ? hp.n_head_kv : hp.n_head);
  const int head_dim = (int)hp.head_dim;
  const int kv_dim = head_dim * n_head_kv;

  s.speech_token_ids.clear();
  s.host_kv_k.assign(n_layer, std::vector<float>());
  s.host_kv_v.assign(n_layer, std::vector<float>());
  s.n_cached = 0;

  // CosyVoice3 LLM input format (from upstream CosyVoice3LM.forward):
  //   speech_embd[sos=codebook+0] | token_embd[text_ids] | speech_embd[task_id=codebook+2]
  // Both sos and task_id live in the SPEECH embedding table, not the text table.
  const int32_t sos_id    = speech_codebook_;      // 6561
  const int32_t task_id   = speech_codebook_ + 2;  // 6563
  const int n_text = (int)s.text_token_ids.size();
  const int total_T = 1 + n_text + 1;  // sos + text + task_id

  // Build prefill embed buffer: [total_T, d_model] F32.
  // Use ggml_get_rows to dequantize (tensors may be Q8_0/Q4_K).
  std::vector<float> prefill_emb((size_t)total_T * d_model);
  {
    // ids: [sos_id, text_ids..., task_id]
    std::vector<int32_t> speech_ids = {sos_id, task_id};
    std::vector<int32_t> text_ids_copy = s.text_token_ids;

    auto get_rows_f32 = [&](ggml_tensor *embd_w,
                            const std::vector<int32_t> &ids) -> std::vector<float> {
      ggml_init_params ip = {ggml_graph_overhead() + 4 * ggml_tensor_overhead(),
                              nullptr, true};
      ggml_context *c = ggml_init(ip);
      ggml_tensor *id_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, (int64_t)ids.size());
      ggml_set_input(id_t);
      ggml_tensor *out = ggml_get_rows(c, embd_w, id_t);
      ggml_set_output(out);
      ggml_cgraph *gf = ggml_new_graph(c);
      ggml_build_forward_expand(gf, out);
      if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        ggml_free(c); return {};
      }
      ggml_backend_tensor_set(id_t, ids.data(), 0, ids.size() * sizeof(int32_t));
      ggml_backend_sched_graph_compute(sched, gf);
      std::vector<float> buf(ids.size() * (size_t)d_model);
      ggml_backend_tensor_get(out, buf.data(), 0, buf.size() * sizeof(float));
      ggml_free(c);
      ggml_backend_sched_reset(sched);
      return buf;
    };

    // sos row
    auto sos_emb = get_rows_f32(llm_model_->speech_embd(), {sos_id});
    if (sos_emb.empty()) { RS_LOG_ERR("CosyVoice3-LLM: sos embed failed"); return false; }
    std::copy(sos_emb.begin(), sos_emb.end(), prefill_emb.begin());

    // text rows
    auto text_emb = get_rows_f32(llm_model_->tok_embd(), text_ids_copy);
    if (text_emb.empty()) { RS_LOG_ERR("CosyVoice3-LLM: text embed failed"); return false; }
    std::copy(text_emb.begin(), text_emb.end(),
              prefill_emb.begin() + d_model);

    // task_id row
    auto task_emb = get_rows_f32(llm_model_->speech_embd(), {task_id});
    if (task_emb.empty()) { RS_LOG_ERR("CosyVoice3-LLM: task embed failed"); return false; }
    std::copy(task_emb.begin(), task_emb.end(),
              prefill_emb.begin() + (size_t)(1 + n_text) * d_model);
  }

  // ----- Build the LLM graph builder for this segment.
  llm_cparams cparams;
  cparams.n_ctx = (uint32_t)(total_T + s.max_speech_tokens + 32);
  cparams.n_batch = (uint32_t)total_T;
  cparams.n_ubatch = cparams.n_batch;
  cparams.flash_attn = false;

  auto builder = std::make_unique<llm_build_qwen2>(*llm_model_, cparams, sched);

  // ----- Prefill positions: 0..total_T-1.
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) positions[i] = i;

  // Build a CPU-side embed tensor and run prefill via build_graph_from_embeds.
  ggml_init_params ep = {ggml_graph_overhead() + 4 * ggml_tensor_overhead(), nullptr, true};
  ggml_context *ctx_emb = ggml_init(ep);
  ggml_tensor *emb_in = ggml_new_tensor_2d(ctx_emb, GGML_TYPE_F32, d_model, total_T);
  ggml_set_name(emb_in, "cv3_prefill_emb");
  ggml_set_input(emb_in);

  llm_build_opts pf_opts;
  pf_opts.output_mode = llm_output_mode::OUTPUT_EMBEDDINGS;
  pf_opts.skip_embeddings = true;   // we supply embeds directly
  pf_opts.skip_output_norm = false;
  pf_opts.skip_lm_head = true;
  pf_opts.use_kv_cache = true;
  pf_opts.update_kv_cache = true;
  pf_opts.causal_mask = true;

  auto pf = builder->build_graph_from_embeds(emb_in, (uint32_t)total_T, nullptr,
                                              positions.data(), &pf_opts);
  if (!pf) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill graph build failed");
    ggml_free(ctx_emb);
    return false;
  }

  if (!ggml_backend_sched_alloc_graph(sched, pf->get_graph())) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill alloc failed");
    ggml_free(ctx_emb);
    return false;
  }

  ggml_backend_tensor_set(emb_in, prefill_emb.data(), 0,
                          prefill_emb.size() * sizeof(float));
  if (auto *pt = pf->get_input_tensor("position_ids")) {
    pf->set_position_ids(pt, positions.data(), (uint32_t)total_T);
  }
  if (auto *m = pf->get_input_tensor("causal_mask")) {
    pf->set_causal_mask(m, (uint32_t)total_T, 0);
  }

  if (ggml_backend_sched_graph_compute(sched, pf->get_graph()) !=
      GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill compute failed");
    ggml_free(ctx_emb);
    return false;
  }

  // Pull the last hidden row → speech_lm_head → logits.
  ggml_tensor *embd = pf->get_embd();
  if (!embd) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill produced no embeddings");
    ggml_free(ctx_emb);
    return false;
  }
  std::vector<float> last_hidden((size_t)d_model);
  {
    const size_t row_bytes = (size_t)d_model * sizeof(float);
    const size_t off = (size_t)(total_T - 1) * row_bytes;
    ggml_backend_tensor_get(embd, last_hidden.data(), off, row_bytes);
  }

  // Extract per-layer KV for the prefill into host buffers.
  s.n_cached = (uint32_t)total_T;
  for (int il = 0; il < n_layer; ++il) {
    ggml_tensor *k_out = pf->get_kv_output_k(il);
    ggml_tensor *v_out = pf->get_kv_output_v(il);
    if (!k_out || !v_out) {
      RS_LOG_ERR("CosyVoice3-LLM: prefill KV missing for layer %d", il);
      ggml_free(ctx_emb);
      return false;
    }
    const size_t bytes = ggml_nbytes(k_out);
    s.host_kv_k[il].resize(bytes / sizeof(float));
    s.host_kv_v[il].resize(bytes / sizeof(float));
    ggml_backend_tensor_get(k_out, s.host_kv_k[il].data(), 0, bytes);
    ggml_backend_tensor_get(v_out, s.host_kv_v[il].data(), 0, bytes);
  }

  ggml_free(ctx_emb);
  ggml_backend_sched_reset(sched);

  // ----- Project last hidden → speech logits.
  std::vector<float> logits = project_speech_logits(
      sched, llm_model_->output_norm(), llm_model_->speech_lm_head(),
      hp.f_norm_rms_eps, last_hidden.data(), d_model, speech_vocab_);
  if ((int)logits.size() != speech_vocab_) {
    RS_LOG_ERR("CosyVoice3-LLM: empty step-0 logits");
    return false;
  }
  if (s.dump_step0_logits) {
    *s.dump_step0_logits = logits;
  }
  // Environment-driven debug dump — lets the validation harness diff against
  // the PyTorch reference without adding C-API surface for a Phase-2-only path.
  if (const char *path = std::getenv("RS_CV3_DUMP_STEP0_LOGITS")) {
    if (path && *path) {
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(logits.data(), sizeof(float), logits.size(), fp);
        std::fclose(fp);
        RS_LOG_INFO("CosyVoice3-LLM: dumped step-0 logits (%d floats) → %s",
                    (int)logits.size(), path);
      } else {
        RS_LOG_ERR("CosyVoice3-LLM: cannot open RS_CV3_DUMP_STEP0_LOGITS=%s",
                   path);
      }
    }
  }

  // First speech token.
  int32_t tok = cosyvoice3_sample_ras(logits.data(), speech_vocab_,
                                      s.speech_token_ids, s.sampler, s.rng);
  if (tok >= speech_codebook_) {
    RS_LOG_INFO("CosyVoice3-LLM: stop on first sample (id=%d)", tok);
    return true;
  }
  s.speech_token_ids.push_back(tok);

  // ============================================================
  // AR loop: feed speech_embd[tok] back through the LLM each step.
  // We use the GPU-persistent KV path identical to funasr-nano so
  // gallocr doesn't reallocate the GPU compute buffer per step.
  // ============================================================
  const int32_t n_kv_max = (int32_t)s.n_cached + s.max_speech_tokens + 4;
  std::vector<ggml_tensor *> gpu_kv_k(n_layer, nullptr);
  std::vector<ggml_tensor *> gpu_kv_v(n_layer, nullptr);

  ggml_init_params kv_params = {
      (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16), nullptr,
      true};
  ggml_context *ctx_kv = ggml_init(kv_params);
  for (int il = 0; il < n_layer; ++il) {
    gpu_kv_k[il] = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
    gpu_kv_v[il] = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
    ggml_set_name(gpu_kv_k[il], ("cv3_kv_k_" + std::to_string(il)).c_str());
    ggml_set_name(gpu_kv_v[il], ("cv3_kv_v_" + std::to_string(il)).c_str());
  }
  ggml_backend_t gpu_b = pick_gpu_backend(sched, backend_);
  ggml_backend_buffer_t kv_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, gpu_b);
  if (!kv_buf) {
    RS_LOG_ERR("CosyVoice3-LLM: failed to alloc GPU KV cache");
    ggml_free(ctx_kv);
    return false;
  }

  for (int il = 0; il < n_layer; ++il) {
    const size_t bytes = (size_t)kv_dim * s.n_cached * sizeof(float);
    ggml_backend_tensor_set(gpu_kv_k[il], s.host_kv_k[il].data(), 0, bytes);
    ggml_backend_tensor_set(gpu_kv_v[il], s.host_kv_v[il].data(), 0, bytes);
  }

  // Warmup: reserve the gallocr at the LARGEST decode shape so the GPU
  // compute buffer is never reallocated mid-loop.
  // DISABLED for debugging
  /*
  {
    llm_build_opts wopts;
    wopts.output_mode = llm_output_mode::OUTPUT_EMBEDDINGS;
    wopts.skip_embeddings = true;
    wopts.skip_lm_head = true;
    wopts.use_kv_cache = true;
    wopts.is_decode_step = true;
    wopts.causal_mask = true;
    wopts.n_kv_cache = (uint32_t)(n_kv_max - 1);
    wopts.n_kv_max = (uint32_t)n_kv_max;
    wopts.gpu_kv_k = gpu_kv_k.data();
    wopts.gpu_kv_v = gpu_kv_v.data();

    ggml_init_params dp = {32 * ggml_tensor_overhead() + (1 << 16), nullptr,
                            true};
    ggml_context *ctx_warm = ggml_init(dp);
    ggml_tensor *dummy = ggml_new_tensor_2d(ctx_warm, GGML_TYPE_F32, d_model, 1);
    ggml_set_input(dummy);
    llm_pos wpos = (llm_pos)(n_kv_max - 1);
    auto warm = builder->build_graph_from_embeds(dummy, 1, nullptr, &wpos,
                                                 &wopts);
    if (warm) ggml_backend_sched_reserve(sched, warm->get_graph());
    ggml_free(ctx_warm);
    ggml_backend_sched_reset(sched);
  }
  */

  std::vector<float> kv_stage_k((size_t)kv_dim);
  std::vector<float> kv_stage_v((size_t)kv_dim);
  std::vector<float> step_hidden((size_t)d_model);

  for (int step = 0; step < s.max_speech_tokens && tok != stop_token_id_;
       ++step) {
    // Build a fresh per-step graph that:
    //   1. loads speech_embd[tok] into a [d_model, 1] tensor
    //   2. runs Qwen2 from_embeds with is_decode_step=true
    // We do this in a single graph so the embedding row lookup runs on the
    // GPU alongside the transformer body.
    ggml_init_params sp = {64 * ggml_tensor_overhead() + (1 << 16), nullptr,
                            true};
    ggml_context *ctx_step = ggml_init(sp);

    ggml_tensor *id_in = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, 1);
    ggml_set_name(id_in, "cv3_speech_id");
    ggml_set_input(id_in);
    ggml_tensor *embed = ggml_get_rows(ctx_step, llm_model_->speech_embd(),
                                       id_in);
    // ggml_get_rows returns [d_model, 1] when input is 1D[1].

    llm_build_opts dec_opts;
    dec_opts.output_mode = llm_output_mode::OUTPUT_EMBEDDINGS;
    dec_opts.skip_embeddings = true;
    dec_opts.skip_lm_head = true;
    dec_opts.use_kv_cache = true;
    dec_opts.is_decode_step = true;
    dec_opts.causal_mask = true;
    dec_opts.n_kv_cache = s.n_cached;
    dec_opts.n_kv_max = (uint32_t)n_kv_max;
    dec_opts.gpu_kv_k = gpu_kv_k.data();
    dec_opts.gpu_kv_v = gpu_kv_v.data();

    llm_pos pos_step = (llm_pos)s.n_cached;
    auto dec = builder->build_graph_from_embeds(embed, 1, nullptr, &pos_step,
                                                &dec_opts);
    if (!dec) {
      RS_LOG_ERR("CosyVoice3-LLM: AR step %d build failed", step);
      ggml_free(ctx_step);
      break;
    }

    if (!ggml_backend_sched_alloc_graph(sched, dec->get_graph())) {
      RS_LOG_ERR("CosyVoice3-LLM: AR step %d alloc failed", step);
      ggml_free(ctx_step);
      break;
    }

    ggml_backend_tensor_set(id_in, &tok, 0, sizeof(int32_t));

    if (auto *pt = dec->get_input_tensor("position_ids")) {
      dec->set_position_ids(pt, &pos_step, 1);
    }
    if (auto *m = dec->get_input_tensor("causal_mask")) {
      dec->set_causal_mask(m, 1, s.n_cached);
    }

    if (ggml_backend_sched_graph_compute(sched, dec->get_graph()) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("CosyVoice3-LLM: AR step %d compute failed", step);
      ggml_free(ctx_step);
      break;
    }

    // Pull hidden state and append the new KV column.
    if (ggml_tensor *h = dec->get_embd()) {
      ggml_backend_tensor_get(h, step_hidden.data(), 0,
                              (size_t)d_model * sizeof(float));
    }
    for (int il = 0; il < n_layer; ++il) {
      ggml_tensor *k_out = dec->get_kv_output_k(il);
      ggml_tensor *v_out = dec->get_kv_output_v(il);
      if (!k_out || !v_out) continue;

      // k_out/v_out are the FULL concatenated cache [kv_dim, n_cached+1] from
      // build_kv_cache_concat. Extract the NEW column (last column) and write
      // it to the GPU buffer at position n_cached.
      const size_t col_bytes = (size_t)kv_dim * sizeof(float);
      const size_t read_off = (size_t)s.n_cached * kv_dim * sizeof(float);
      const size_t write_off = (size_t)s.n_cached * kv_dim * sizeof(float);
      ggml_backend_tensor_get(k_out, kv_stage_k.data(), read_off, col_bytes);
      ggml_backend_tensor_get(v_out, kv_stage_v.data(), read_off, col_bytes);
      ggml_backend_tensor_set(gpu_kv_k[il], kv_stage_k.data(), write_off,
                              col_bytes);
      ggml_backend_tensor_set(gpu_kv_v[il], kv_stage_v.data(), write_off,
                              col_bytes);
    }
    s.n_cached++;

    ggml_free(ctx_step);
    ggml_backend_sched_reset(sched);

    // Project hidden → logits via speech_lm_head.
    std::vector<float> step_logits = project_speech_logits(
        sched, llm_model_->output_norm(), llm_model_->speech_lm_head(),
        hp.f_norm_rms_eps, step_hidden.data(), d_model, speech_vocab_);
    if ((int)step_logits.size() != speech_vocab_) {
      RS_LOG_ERR("CosyVoice3-LLM: AR step %d empty logits", step);
      break;
    }

    int32_t next = cosyvoice3_sample_ras(step_logits.data(), speech_vocab_,
                                         s.speech_token_ids, s.sampler, s.rng);
    if (next >= speech_codebook_) {
      RS_LOG_INFO("CosyVoice3-LLM: stop at step %d (id=%d)", step + 1, next);
      tok = next;
      break;
    }
    s.speech_token_ids.push_back(next);
    tok = next;
  }

  ggml_backend_buffer_free(kv_buf);
  ggml_free(ctx_kv);

  RS_LOG_INFO("CosyVoice3-LLM: generated %zu speech tokens",
              s.speech_token_ids.size());
  return true;
}
