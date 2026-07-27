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
#include <limits>
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
                                         int speech_vocab, int n,
                                         std::function<void(struct ggml_tensor *)> *imatrix_cb) {
  ggml_init_params ip = {ggml_graph_overhead() + 8 * ggml_tensor_overhead(), nullptr,
                          true};
  ggml_context *ctx = ggml_init(ip);
  ggml_cgraph *gf = ggml_new_graph(ctx);

  ggml_tensor *h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n);
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
  ggml_backend_tensor_set(h, hidden, 0, (size_t)d_model * n * sizeof(float));
  if (cv3_sched_compute(sched, gf, imatrix_cb) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("CosyVoice3-LLM: speech_lm_head compute failed");
    ggml_free(ctx);
    return {};
  }

  std::vector<float> out((size_t)speech_vocab * n);
  ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));

  ggml_free(ctx);
  ggml_backend_sched_reset(sched);
  return out;
}

} // namespace

// =====================================================================
// Decode: prefill text → first speech token, then AR loop.
// =====================================================================

// Originally `Decode` — the LLM AR loop that emits speech tokens. After the
// Flow+HiFT wiring landed, the public Decode lives in cosyvoice3.cpp and
// chains RunLM → Flow → HiFT. This function is now the LM-only stage.
bool CosyVoice3LMModel::RunLM(CosyVoice3State &s, ggml_backend_sched_t sched) {
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
  s.host_kv_k.clear();
  s.host_kv_v.clear();
  s.n_cached = 0;

  // CosyVoice3 LLM input format (matches upstream `CosyVoice3LM.inference`):
  //   [sos] [prompt_text_ids + tts_text_ids] [task_id] [prompt_speech_token_embds]
  //
  // Both sos and task_id live in the SPEECH embedding table. prompt_text and
  // prompt_speech_token are present only when --ref was provided. The AR loop
  // continues from the end and emits the actual gen speech tokens.
  //
  // `<|endofprompt|>` (id 151646) is a hard requirement: upstream
  // CosyVoice3LM.inference asserts `151646 in text`. From example.py the
  // upstream caller-side convention for CosyVoice3 zero_shot is:
  //   prompt_text = "You are a helpful assistant.<|endofprompt|>" + ref_text
  // i.e. the system prefix and 151646 sit BEFORE the ref transcript, and the
  // whole bundle is concatenated with tts_text before the LM forward.
  //
  // Instruct2 mode (`s.instruct_text_token_ids` non-empty): upstream
  // `frontend_instruct2` replaces `prompt_text` with `instruct_text` and
  // deletes `llm_prompt_speech_token`. We mirror that here: instruct ids
  // take the ref-transcript slot AND `n_pt` is forced to 0 for the LM prefix.
  // The Flow path still uses `s.prompt_token` / `s.prompt_feat` / `s.embedding`
  // so timbre cloning continues to work — only LM-side prosody is steered by
  // instruct text.
  const int32_t sos_id      = speech_codebook_;      // 6561
  const int32_t task_id     = speech_codebook_ + 2;  // 6563
  const int32_t eop_id      = 151646;                // <|endofprompt|>

  const bool instruct_mode = !s.instruct_text_token_ids.empty();
  const std::vector<int32_t> &lm_prompt_text_ids =
      instruct_mode ? s.instruct_text_token_ids : s.prompt_text_token_ids;

  std::vector<int32_t> sys_prefix_ids = llm_model_->vocab().tokenize(
      std::string("You are a helpful assistant."), false);

  std::vector<int32_t> prompt_text_with_eop;
  prompt_text_with_eop.reserve(sys_prefix_ids.size() + 1 +
                               lm_prompt_text_ids.size());
  prompt_text_with_eop.insert(prompt_text_with_eop.end(),
                              sys_prefix_ids.begin(), sys_prefix_ids.end());
  prompt_text_with_eop.push_back(eop_id);
  prompt_text_with_eop.insert(prompt_text_with_eop.end(),
                              lm_prompt_text_ids.begin(),
                              lm_prompt_text_ids.end());

  std::vector<int32_t> combined_text;
  combined_text.reserve(prompt_text_with_eop.size() +
                        s.text_token_ids.size());
  combined_text.insert(combined_text.end(),
                       prompt_text_with_eop.begin(),
                       prompt_text_with_eop.end());
  combined_text.insert(combined_text.end(),
                       s.text_token_ids.begin(),
                       s.text_token_ids.end());
  const int n_text  = (int)combined_text.size();
  const int n_pt    = instruct_mode ? 0 : (int)s.prompt_token.size();
  const int total_T = 1 + n_text + 1 + n_pt;  // sos + text + task + prompt_speech

  // RS_CV3_DUMP_PROMPT_IDS=/tmp/ids -> dump bpe text ids, prompt speech tokens
  // and segment lengths so the python harness can verify input construction.
  // Header layout: [n_prompt_text_eop, n_tts, n_prompt_speech,
  //                 sos_id, task_id, speech_codebook]
  // Body: prompt_text_with_eop_ids, tts_text_ids, prompt_speech_ids.
  if (const char *p = std::getenv("RS_CV3_DUMP_PROMPT_IDS")) {
    if (FILE *fp = std::fopen(p, "wb")) {
      const int32_t header[6] = {
          (int32_t)prompt_text_with_eop.size(),
          (int32_t)s.text_token_ids.size(),
          (int32_t)s.prompt_token.size(),
          sos_id, task_id, speech_codebook_};
      std::fwrite(header, sizeof(int32_t), 6, fp);
      if (!prompt_text_with_eop.empty())
        std::fwrite(prompt_text_with_eop.data(), sizeof(int32_t),
                    prompt_text_with_eop.size(), fp);
      if (!s.text_token_ids.empty())
        std::fwrite(s.text_token_ids.data(), sizeof(int32_t),
                    s.text_token_ids.size(), fp);
      if (!s.prompt_token.empty())
        std::fwrite(s.prompt_token.data(), sizeof(int32_t),
                    s.prompt_token.size(), fp);
      std::fclose(fp);
      RS_LOG_INFO("CosyVoice3-LLM: dumped prompt ids "
                  "(prompt_text_eop=%zu tts=%zu prompt_speech=%zu sos=%d task=%d) -> %s",
                  prompt_text_with_eop.size(),
                  s.text_token_ids.size(), s.prompt_token.size(), sos_id,
                  task_id, p);
    }
  }

  // Build prefill embed buffer: [total_T, d_model] F32.
  // Use ggml_get_rows to dequantize (tensors may be Q8_0/Q4_K).
  std::vector<float> prefill_emb((size_t)total_T * d_model);
  {
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

    // text rows (prompt_text + tts_text)
    if (!combined_text.empty()) {
      auto text_emb = get_rows_f32(llm_model_->tok_embd(), combined_text);
      if (text_emb.empty()) { RS_LOG_ERR("CosyVoice3-LLM: text embed failed"); return false; }
      std::copy(text_emb.begin(), text_emb.end(),
                prefill_emb.begin() + d_model);
    }

    // task_id row
    auto task_emb = get_rows_f32(llm_model_->speech_embd(), {task_id});
    if (task_emb.empty()) { RS_LOG_ERR("CosyVoice3-LLM: task embed failed"); return false; }
    std::copy(task_emb.begin(), task_emb.end(),
              prefill_emb.begin() + (size_t)(1 + n_text) * d_model);

    // prompt_speech_token rows — only when a reference voice was supplied.
    if (n_pt > 0) {
      auto pt_emb = get_rows_f32(llm_model_->speech_embd(), s.prompt_token);
      if (pt_emb.empty()) { RS_LOG_ERR("CosyVoice3-LLM: prompt_token embed failed"); return false; }
      std::copy(pt_emb.begin(), pt_emb.end(),
                prefill_emb.begin() + (size_t)(1 + n_text + 1) * d_model);
    }
  }

  // RS_CV3_DUMP_PREFILL_EMB=/tmp/prefill_emb.bin -> dump the [total_T, d_model]
  // prefill embedding buffer so PyTorch can inject the same input and skip its
  // own embedding-table lookups when verifying transformer numerics.
  if (const char *p = std::getenv("RS_CV3_DUMP_PREFILL_EMB")) {
    if (FILE *fp = std::fopen(p, "wb")) {
      int32_t header[2] = {total_T, d_model};
      std::fwrite(header, sizeof(int32_t), 2, fp);
      std::fwrite(prefill_emb.data(), sizeof(float), prefill_emb.size(), fp);
      std::fclose(fp);
      RS_LOG_INFO("CosyVoice3-LLM: dumped prefill embeds [%d, %d] -> %s",
                  total_T, d_model, p);
    }
  }

  // ----- Build the LLM graph builder for this segment.
  llm_cparams cparams;
  cparams.n_ctx = (uint32_t)(total_T + s.max_speech_tokens + 32);
  cparams.n_batch = (uint32_t)total_T;
  cparams.n_ubatch = cparams.n_batch;
  cparams.flash_attn = true;

  auto builder = std::make_unique<llm_build_qwen2>(*llm_model_, cparams, sched);

  const int32_t n_kv_max = (int32_t)total_T + s.max_speech_tokens + 4;
  std::vector<ggml_tensor *> gpu_kv_k(n_layer, nullptr);
  std::vector<ggml_tensor *> gpu_kv_v(n_layer, nullptr);
  ggml_backend_buffer_t kv_buf = nullptr;
  ggml_context *ctx_kv = nullptr;
  auto free_kv_cache = [&]() {
    if (kv_buf) {
      ggml_backend_buffer_free(kv_buf);
      kv_buf = nullptr;
    }
    if (ctx_kv) {
      ggml_free(ctx_kv);
      ctx_kv = nullptr;
    }
  };

  ggml_init_params kv_params = {
      (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16), nullptr,
      true};
  ctx_kv = ggml_init(kv_params);
  for (int il = 0; il < n_layer; ++il) {
    gpu_kv_k[il] = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
    gpu_kv_v[il] = ggml_new_tensor_2d(ctx_kv, GGML_TYPE_F32, kv_dim, n_kv_max);
    ggml_set_name(gpu_kv_k[il], ("cv3_kv_k_" + std::to_string(il)).c_str());
    ggml_set_name(gpu_kv_v[il], ("cv3_kv_v_" + std::to_string(il)).c_str());
  }
  ggml_backend_t gpu_b = pick_gpu_backend(sched, backend_);
  kv_buf = ggml_backend_alloc_ctx_tensors(ctx_kv, gpu_b);
  if (!kv_buf) {
    RS_LOG_ERR("CosyVoice3-LLM: failed to alloc GPU KV cache");
    ggml_free(ctx_kv);
    return false;
  }

  std::vector<float> zero_kv((size_t)kv_dim * n_kv_max, 0.0f);
  for (int il = 0; il < n_layer; ++il) {
    ggml_backend_tensor_set(gpu_kv_k[il], zero_kv.data(), 0,
                            zero_kv.size() * sizeof(float));
    ggml_backend_tensor_set(gpu_kv_v[il], zero_kv.data(), 0,
                            zero_kv.size() * sizeof(float));
  }

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
  pf_opts.n_kv_max = (uint32_t)n_kv_max;
  pf_opts.gpu_kv_k = gpu_kv_k.data();
  pf_opts.gpu_kv_v = gpu_kv_v.data();
  pf_opts.causal_mask = true;

  // RS_CV3_DUMP_LAYER_HIDDEN=/tmp/layer -> dump per-layer last-row hidden
  const char *layer_dump_dir = std::getenv("RS_CV3_DUMP_LAYER_HIDDEN");
  if (layer_dump_dir && *layer_dump_dir) {
    pf_opts.extract_intermediate = true;
    pf_opts.extract_layers.clear();
    for (int il = 0; il < n_layer; ++il)
      pf_opts.extract_layers.push_back(il);
  }

  auto pf = builder->build_graph_from_embeds(emb_in, (uint32_t)total_T, nullptr,
                                              positions.data(), &pf_opts);
  if (!pf) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill graph build failed");
    ggml_free(ctx_emb);
    free_kv_cache();
    return false;
  }

  pf->assign_kv_backend(sched, gpu_b, n_layer);
  if (!ggml_backend_sched_alloc_graph(sched, pf->get_graph())) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill alloc failed");
    ggml_free(ctx_emb);
    free_kv_cache();
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

  if (cv3_sched_compute(sched, pf->get_graph(), &imatrix_cb_) !=
      GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill compute failed");
    ggml_free(ctx_emb);
    free_kv_cache();
    return false;
  }

  // Pull the last hidden row → speech_lm_head → logits.
  ggml_tensor *embd = pf->get_embd();
  if (!embd) {
    RS_LOG_ERR("CosyVoice3-LLM: prefill produced no embeddings");
    ggml_free(ctx_emb);
    free_kv_cache();
    return false;
  }
  std::vector<float> last_hidden((size_t)d_model);
  {
    const size_t row_bytes = (size_t)d_model * sizeof(float);
    const size_t off = (size_t)(total_T - 1) * row_bytes;
    ggml_backend_tensor_get(embd, last_hidden.data(), off, row_bytes);
  }
  if (const char *p = std::getenv("RS_CV3_DUMP_HIDDEN")) {
    if (FILE *fp = std::fopen(p, "wb")) {
      std::fwrite(last_hidden.data(), sizeof(float), last_hidden.size(), fp);
      std::fclose(fp);
    }
  }

  // Dump per-layer last-row hidden states for debugging
  if (const char *layer_dir = std::getenv("RS_CV3_DUMP_LAYER_HIDDEN")) {
    if (layer_dir && *layer_dir) {
      for (int il = 0; il < n_layer; ++il) {
        ggml_tensor *layer_out = pf->get_intermediate_output((size_t)il);
        if (!layer_out) continue;
        std::vector<float> row((size_t)d_model);
        const size_t row_bytes = (size_t)d_model * sizeof(float);
        const size_t off = (size_t)(total_T - 1) * row_bytes;
        ggml_backend_tensor_get(layer_out, row.data(), off, row_bytes);
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%02d.bin", layer_dir, il);
        if (FILE *fp = std::fopen(path, "wb")) {
          std::fwrite(row.data(), sizeof(float), row.size(), fp);
          std::fclose(fp);
        }
      }
      RS_LOG_INFO("CosyVoice3-LLM: dumped %d layer hiddens → %s", n_layer,
                  layer_dir);
    }
  }

  s.n_cached = (uint32_t)total_T;

  ggml_free(ctx_emb);
  ggml_backend_sched_reset(sched);

  // ----- Project last hidden → speech logits.
  std::vector<float> logits = project_speech_logits(
      sched, llm_model_->output_norm(), llm_model_->speech_lm_head(),
      hp.f_norm_rms_eps, last_hidden.data(), d_model, speech_vocab_, 1,
      &imatrix_cb_);
  if ((int)logits.size() != speech_vocab_) {
    RS_LOG_ERR("CosyVoice3-LLM: empty step-0 logits");
    free_kv_cache();
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

  // Upstream defaults: min_token_text_ratio=2, max_token_text_ratio=20.
  // While step < min_len, PyTorch passes ignore_eos=True to its sampler, which
  // suppresses any token id >= speech_token_size (i.e. the 200 stop ids). We
  // mirror that by masking those ids to -inf before sampling.
  const int n_tts_text = (int)s.text_token_ids.size();
  const int min_len    = n_tts_text * 2;
  const int max_len    = std::min(s.max_speech_tokens, n_tts_text * 20);

  auto mask_stop_ids = [&](std::vector<float> &lg) {
    for (int i = speech_codebook_; i < (int)lg.size(); ++i) {
      lg[i] = -std::numeric_limits<float>::infinity();
    }
  };
  if (0 < min_len) mask_stop_ids(logits);

  // First speech token.
  int32_t tok = cosyvoice3_sample_ras(logits.data(), speech_vocab_,
                                      s.speech_token_ids, s.sampler, s.rng);
  if (tok >= speech_codebook_) {
    RS_LOG_INFO("CosyVoice3-LLM: stop on first sample (id=%d)", tok);
    free_kv_cache();
    return true;
  }
  s.speech_token_ids.push_back(tok);

  // ============================================================
  // AR loop: feed speech_embd[tok] back through the LLM each step.
  // KV stays device-resident: each graph scatters the new column with
  // ggml_set_rows and attends over a fixed n_kv_max view.
  // ============================================================

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
    wopts.fixed_kv_cache_shape = true;
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
    if (warm) {
      warm->assign_kv_backend(sched, gpu_b, n_layer);
      ggml_backend_sched_reserve(sched, warm->get_graph());
    }
    ggml_free(ctx_warm);
    ggml_backend_sched_reset(sched);
  }

  std::vector<float> step_hidden((size_t)d_model);

  // RS_CV3_DUMP_AR_DIR=/tmp/ar -> dump per-step AR hidden, logits, token,
  // and per-layer KV new column for the first RS_CV3_DUMP_AR_STEPS steps.
  // Also writes prefill summary so the PyTorch reference can prime its
  // past_key_values and replay AR from the same starting point.
  const char *ar_dir = std::getenv("RS_CV3_DUMP_AR_DIR");
  int ar_max_steps = 0;
  if (ar_dir && *ar_dir) {
    const char *n_env = std::getenv("RS_CV3_DUMP_AR_STEPS");
    ar_max_steps = n_env ? std::atoi(n_env) : 8;
    if (ar_max_steps <= 0) ar_max_steps = 8;
    // Write prefill KV: one file per layer, [kv_dim, n_cached] F32.
    for (int il = 0; il < n_layer; ++il) {
      std::vector<float> kv_dump((size_t)kv_dim * s.n_cached);
      const size_t bytes = kv_dump.size() * sizeof(float);
      char path[512];
      snprintf(path, sizeof(path), "%s/prefill_kv_k_%02d.bin", ar_dir, il);
      if (FILE *fp = std::fopen(path, "wb")) {
        ggml_backend_tensor_get(gpu_kv_k[il], kv_dump.data(), 0, bytes);
        std::fwrite(kv_dump.data(), sizeof(float), kv_dump.size(), fp);
        std::fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/prefill_kv_v_%02d.bin", ar_dir, il);
      if (FILE *fp = std::fopen(path, "wb")) {
        ggml_backend_tensor_get(gpu_kv_v[il], kv_dump.data(), 0, bytes);
        std::fwrite(kv_dump.data(), sizeof(float), kv_dump.size(), fp);
        std::fclose(fp);
      }
    }
    // Write the first sampled token (step -1 / prefill output).
    {
      char path[512];
      snprintf(path, sizeof(path), "%s/step_-1_token.bin", ar_dir);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(&tok, sizeof(int32_t), 1, fp);
        std::fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/step_-1_hidden.bin", ar_dir);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(last_hidden.data(), sizeof(float), last_hidden.size(), fp);
        std::fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/step_-1_logits.bin", ar_dir);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(logits.data(), sizeof(float), logits.size(), fp);
        std::fclose(fp);
      }
    }
    // Meta: n_cached (prefill length), n_layer, kv_dim, d_model, speech_vocab.
    {
      char path[512];
      snprintf(path, sizeof(path), "%s/meta.bin", ar_dir);
      if (FILE *fp = std::fopen(path, "wb")) {
        int32_t meta[5] = {(int32_t)s.n_cached, n_layer, kv_dim, d_model,
                           speech_vocab_};
        std::fwrite(meta, sizeof(int32_t), 5, fp);
        std::fclose(fp);
      }
    }
    RS_LOG_INFO("CosyVoice3-LLM: AR dump dir=%s max_steps=%d", ar_dir,
                ar_max_steps);
  }

  for (int step = 0; step < max_len && tok < speech_codebook_; ++step) {
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
    dec_opts.fixed_kv_cache_shape = true;
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

    dec->assign_kv_backend(sched, gpu_b, n_layer);
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
    dec->set_kv_write_indices(s.n_cached, n_layer);

    if (cv3_sched_compute(sched, dec->get_graph(), &imatrix_cb_) !=
        GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("CosyVoice3-LLM: AR step %d compute failed", step);
      ggml_free(ctx_step);
      break;
    }

    // Pull hidden state. KV was appended device-side by ggml_set_rows.
    if (ggml_tensor *h = dec->get_embd()) {
      ggml_backend_tensor_get(h, step_hidden.data(), 0,
                              (size_t)d_model * sizeof(float));
    }
    s.n_cached++;

    ggml_free(ctx_step);
    ggml_backend_sched_reset(sched);

    // Project hidden → logits via speech_lm_head.
    std::vector<float> step_logits = project_speech_logits(
        sched, llm_model_->output_norm(), llm_model_->speech_lm_head(),
        hp.f_norm_rms_eps, step_hidden.data(), d_model, speech_vocab_, 1,
        &imatrix_cb_);
    if ((int)step_logits.size() != speech_vocab_) {
      RS_LOG_ERR("CosyVoice3-LLM: AR step %d empty logits", step);
      break;
    }

    // step+1 is the index of the NEXT token about to be sampled (step 0 of
    // the AR loop produces speech_token #1 since the prefill emitted #0).
    if (step + 1 < min_len) mask_stop_ids(step_logits);

    int32_t next = cosyvoice3_sample_ras(step_logits.data(), speech_vocab_,
                                         s.speech_token_ids, s.sampler, s.rng);

    // RS_CV3_DUMP_AR_DIR: dump THIS step's hidden, logits, KV new column,
    // input token (tok = current step's input), and sampled token.
    if (ar_dir && step < ar_max_steps) {
      char path[512];
      snprintf(path, sizeof(path), "%s/step_%02d_input_token.bin", ar_dir,
               step);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(&tok, sizeof(int32_t), 1, fp);
        std::fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/step_%02d_hidden.bin", ar_dir, step);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(step_hidden.data(), sizeof(float), step_hidden.size(), fp);
        std::fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/step_%02d_logits.bin", ar_dir, step);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(step_logits.data(), sizeof(float), step_logits.size(), fp);
        std::fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/step_%02d_sampled.bin", ar_dir, step);
      if (FILE *fp = std::fopen(path, "wb")) {
        std::fwrite(&next, sizeof(int32_t), 1, fp);
        std::fclose(fp);
      }
      // Dump the NEW K/V column we just appended at position s.n_cached-1
      // (n_cached was incremented above).
      for (int il = 0; il < n_layer; ++il) {
        const size_t off = (size_t)(s.n_cached - 1) * kv_dim;
        std::vector<float> k_col((size_t)kv_dim);
        std::vector<float> v_col((size_t)kv_dim);
        ggml_backend_tensor_get(gpu_kv_k[il], k_col.data(),
                                off * sizeof(float),
                                (size_t)kv_dim * sizeof(float));
        ggml_backend_tensor_get(gpu_kv_v[il], v_col.data(),
                                off * sizeof(float),
                                (size_t)kv_dim * sizeof(float));
        snprintf(path, sizeof(path), "%s/step_%02d_kv_k_%02d.bin", ar_dir,
                 step, il);
        if (FILE *fp = std::fopen(path, "wb")) {
          std::fwrite(k_col.data(), sizeof(float), k_col.size(), fp);
          std::fclose(fp);
        }
        snprintf(path, sizeof(path), "%s/step_%02d_kv_v_%02d.bin", ar_dir,
                 step, il);
        if (FILE *fp = std::fopen(path, "wb")) {
          std::fwrite(v_col.data(), sizeof(float), v_col.size(), fp);
          std::fclose(fp);
        }
      }
    }

    if (next >= speech_codebook_) {
      RS_LOG_INFO("CosyVoice3-LLM: stop at step %d (id=%d)", step + 1, next);
      tok = next;
      break;
    }
    s.speech_token_ids.push_back(next);
    tok = next;

    // Streaming hook (rs-tts-online). Returning false from the callback
    // means the orchestrator wants to stop early; we synthesize an EOS so
    // the rest of the AR-loop cleanup runs normally.
    if (lm_token_cb_ && !lm_token_cb_(next)) {
      RS_LOG_INFO("CosyVoice3-LLM: stream callback requested abort at step %d",
                  step + 1);
      tok = speech_codebook_;
      break;
    }
  }

  ggml_backend_buffer_free(kv_buf);
  ggml_free(ctx_kv);

  RS_LOG_INFO("CosyVoice3-LLM: generated %zu speech tokens",
              s.speech_token_ids.size());
  if (const char *p = std::getenv("RS_CV3_DUMP_SPEECH_TOKENS")) {
    if (FILE *fp = std::fopen(p, "wb")) {
      std::fwrite(s.speech_token_ids.data(), sizeof(int32_t),
                  s.speech_token_ids.size(), fp);
      std::fclose(fp);
      RS_LOG_INFO("CosyVoice3-LLM: dumped %zu speech tokens -> %s",
                  s.speech_token_ids.size(), p);
    }
  }
  return true;
}

// =====================================================================
// Continuous-batching interface — RunLM split into Prefill + Step so a
// scheduler can interleave AR steps of several requests (see
// cosyvoice3_batch.h). Mirrors RunLM exactly minus the debug dumps.
// =====================================================================

bool CosyVoice3LMModel::RunLM_Prefill(CosyVoice3State &s,
                                      CosyVoice3LMSlot &slot,
                                      ggml_backend_sched_t sched) {
  if (!llm_model_) return false;
  if (s.text_token_ids.empty()) {
    RS_LOG_ERR("CosyVoice3-batch: empty text — call PushText first");
    return false;
  }

  // Voice prep (same requirement as Decode(): the LM is OOD without a
  // voice prior — see the comment there).
  if (flow_ready_ && hift_ready_) {
    if (!PrepareVoice(s, sched)) return false;
  }

  const auto &hp = llm_model_->hparams();
  const int n_layer = (int)hp.n_layer;
  const int d_model = (int)hp.n_embd;
  const int n_head_kv = (int)(hp.n_head_kv > 0 ? hp.n_head_kv : hp.n_head);
  const int head_dim = (int)hp.head_dim;
  const int kv_dim = head_dim * n_head_kv;

  s.speech_token_ids.clear();
  s.n_cached = 0;

  const int32_t sos_id  = speech_codebook_;
  const int32_t task_id = speech_codebook_ + 2;
  const int32_t eop_id  = 151646;

  const bool instruct_mode = !s.instruct_text_token_ids.empty();
  const std::vector<int32_t> &lm_prompt_text_ids =
      instruct_mode ? s.instruct_text_token_ids : s.prompt_text_token_ids;

  std::vector<int32_t> sys_prefix_ids = llm_model_->vocab().tokenize(
      std::string("You are a helpful assistant."), false);

  std::vector<int32_t> combined_text;
  combined_text.reserve(sys_prefix_ids.size() + 1 + lm_prompt_text_ids.size() +
                        s.text_token_ids.size());
  combined_text.insert(combined_text.end(), sys_prefix_ids.begin(),
                       sys_prefix_ids.end());
  combined_text.push_back(eop_id);
  combined_text.insert(combined_text.end(), lm_prompt_text_ids.begin(),
                       lm_prompt_text_ids.end());
  combined_text.insert(combined_text.end(), s.text_token_ids.begin(),
                       s.text_token_ids.end());

  const int n_text  = (int)combined_text.size();
  const int n_pt    = instruct_mode ? 0 : (int)s.prompt_token.size();
  const int total_T = 1 + n_text + 1 + n_pt;

  // ── prefill embed buffer [total_T, d_model] ──────────────────────────────
  std::vector<float> prefill_emb((size_t)total_T * d_model);
  {
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
      if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_free(c); return {}; }
      ggml_backend_tensor_set(id_t, ids.data(), 0, ids.size() * sizeof(int32_t));
      ggml_backend_sched_graph_compute(sched, gf);
      std::vector<float> buf(ids.size() * (size_t)d_model);
      ggml_backend_tensor_get(out, buf.data(), 0, buf.size() * sizeof(float));
      ggml_free(c);
      ggml_backend_sched_reset(sched);
      return buf;
    };

    auto sos_emb = get_rows_f32(llm_model_->speech_embd(), {sos_id});
    if (sos_emb.empty()) return false;
    std::copy(sos_emb.begin(), sos_emb.end(), prefill_emb.begin());

    if (!combined_text.empty()) {
      auto text_emb = get_rows_f32(llm_model_->tok_embd(), combined_text);
      if (text_emb.empty()) return false;
      std::copy(text_emb.begin(), text_emb.end(), prefill_emb.begin() + d_model);
    }

    auto task_emb = get_rows_f32(llm_model_->speech_embd(), {task_id});
    if (task_emb.empty()) return false;
    std::copy(task_emb.begin(), task_emb.end(),
              prefill_emb.begin() + (size_t)(1 + n_text) * d_model);

    if (n_pt > 0) {
      auto pt_emb = get_rows_f32(llm_model_->speech_embd(), s.prompt_token);
      if (pt_emb.empty()) return false;
      std::copy(pt_emb.begin(), pt_emb.end(),
                prefill_emb.begin() + (size_t)(1 + n_text + 1) * d_model);
    }
  }

  // ── builder + KV cache (shared batched buffer, slot owns a slice) ────────
  // The batched KV buffer [kv_dim, n_kv_max * n_slots] per layer is allocated
  // once by InitBatchKV; this prefill writes slot.slot_index's slice. n_kv_max
  // is the shared batch capacity (same for all slots).
  if (batch_n_slots_ <= 0 || !batch_kv_buf_) {
    RS_LOG_ERR("CosyVoice3-batch: InitBatchKV not called before prefill");
    return false;
  }
  if (slot.n_kv_max <= 0) slot.n_kv_max = batch_n_kv_max_;

  llm_cparams cparams;
  cparams.n_ctx = (uint32_t)(batch_n_kv_max_ + 32);
  cparams.n_batch = (uint32_t)total_T;
  cparams.n_ubatch = cparams.n_batch;
  cparams.flash_attn = true;
  slot.cparams = cparams;  // builder holds a ref to this - must outlive it
  slot.builder = std::make_unique<llm_build_qwen2>(*llm_model_, slot.cparams, sched);

  // Zero this slot's slice so unused KV columns don't leak across requests.
  ggml_backend_t gpu_b = pick_gpu_backend(sched, backend_);
  {
    std::vector<float> zero_kv((size_t)kv_dim * slot.n_kv_max, 0.0f);
    const size_t slot_off =
        (size_t)slot.slot_index * (size_t)slot.n_kv_max * (size_t)kv_dim *
        sizeof(float);
    for (int il = 0; il < n_layer; ++il) {
      ggml_backend_tensor_set(batch_kv_k_[(size_t)il], zero_kv.data(),
                              slot_off, zero_kv.size() * sizeof(float));
      ggml_backend_tensor_set(batch_kv_v_[(size_t)il], zero_kv.data(),
                              slot_off, zero_kv.size() * sizeof(float));
    }
  }

  // ── prefill forward ─────────────────────────────────────────────────────
  std::vector<llm_pos> positions(total_T);
  for (int i = 0; i < total_T; ++i) positions[i] = i;

  ggml_init_params ep = {ggml_graph_overhead() + 4 * ggml_tensor_overhead(),
                          nullptr, true};
  ggml_context *ctx_emb = ggml_init(ep);
  ggml_tensor *emb_in =
      ggml_new_tensor_2d(ctx_emb, GGML_TYPE_F32, d_model, total_T);
  ggml_set_name(emb_in, "cv3_prefill_emb");
  ggml_set_input(emb_in);

  llm_build_opts pf_opts;
  pf_opts.output_mode = llm_output_mode::OUTPUT_EMBEDDINGS;
  pf_opts.skip_embeddings = true;
  pf_opts.skip_output_norm = false;
  pf_opts.skip_lm_head = true;
  pf_opts.use_kv_cache = true;
  pf_opts.update_kv_cache = true;
  pf_opts.n_kv_max = (uint32_t)slot.n_kv_max;
  pf_opts.kv_slot_row_offset = (uint32_t)slot.slot_index * (uint32_t)slot.n_kv_max;
  pf_opts.gpu_kv_k = batch_kv_k_.data();
  pf_opts.gpu_kv_v = batch_kv_v_.data();
  pf_opts.causal_mask = true;

  auto pf = slot.builder->build_graph_from_embeds(
      emb_in, (uint32_t)total_T, nullptr, positions.data(), &pf_opts);
  if (!pf) {
    ggml_free(ctx_emb);
    return false;
  }

  pf->assign_kv_backend(sched, gpu_b, n_layer);
  if (!ggml_backend_sched_alloc_graph(sched, pf->get_graph())) {
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

  if (cv3_sched_compute(sched, pf->get_graph(), &imatrix_cb_) !=
      GGML_STATUS_SUCCESS) {
    ggml_free(ctx_emb);
    return false;
  }

  ggml_tensor *embd = pf->get_embd();
  if (!embd) {
    ggml_free(ctx_emb);
    return false;
  }
  std::vector<float> last_hidden((size_t)d_model);
  {
    const size_t row_bytes = (size_t)d_model * sizeof(float);
    ggml_backend_tensor_get(embd, last_hidden.data(),
                            (size_t)(total_T - 1) * row_bytes, row_bytes);
  }
  s.n_cached = (uint32_t)total_T;
  ggml_free(ctx_emb);
  ggml_backend_sched_reset(sched);

  // ── first token ─────────────────────────────────────────────────────────
  std::vector<float> logits = project_speech_logits(
      sched, llm_model_->output_norm(), llm_model_->speech_lm_head(),
      hp.f_norm_rms_eps, last_hidden.data(), d_model, speech_vocab_, 1,
      &imatrix_cb_);
  if ((int)logits.size() != speech_vocab_) {
    return false;
  }

  const int n_tts_text = (int)s.text_token_ids.size();
  slot.min_len = n_tts_text * 2;
  slot.max_len = std::min(s.max_speech_tokens, n_tts_text * 20);
  slot.step = 0;
  slot.done = false;

  if (0 < slot.min_len) {
    for (int i = speech_codebook_; i < (int)logits.size(); ++i)
      logits[i] = -std::numeric_limits<float>::infinity();
  }

  int32_t tok = cosyvoice3_sample_ras(logits.data(), speech_vocab_,
                                      s.speech_token_ids, s.sampler, s.rng);
  if (tok >= speech_codebook_) {
    RS_LOG_INFO("CosyVoice3-batch: stop on first sample (id=%d)", tok);
    slot.done = true;
    return true;
  }
  s.speech_token_ids.push_back(tok);
  slot.cur_tok = tok;
  return true;
}

// ── Shared batched KV allocation ───────────────────────────────────────────
bool CosyVoice3LMModel::InitBatchKV(int n_slots, int n_kv_max,
                                    ggml_backend_sched_t sched) {
  FreeBatchKV();
  if (n_slots <= 0 || n_kv_max <= 0 || !llm_model_) return false;

  const auto &hp = llm_model_->hparams();
  const int n_layer = (int)hp.n_layer;
  const int n_head_kv = (int)(hp.n_head_kv > 0 ? hp.n_head_kv : hp.n_head);
  const int head_dim = (int)hp.head_dim;
  const int kv_dim = head_dim * n_head_kv;

  batch_n_slots_ = n_slots;
  batch_n_kv_max_ = n_kv_max;
  batch_kv_k_.assign((size_t)n_layer, nullptr);
  batch_kv_v_.assign((size_t)n_layer, nullptr);

  ggml_init_params kv_params = {
      (size_t)(n_layer * 2 + 4) * ggml_tensor_overhead() + (1 << 16), nullptr,
      true};
  batch_kv_ctx_ = ggml_init(kv_params);
  for (int il = 0; il < n_layer; ++il) {
    // [kv_dim, n_kv_max * n_slots]: slot s owns rows [s*n_kv_max, ...].
    batch_kv_k_[(size_t)il] = ggml_new_tensor_2d(batch_kv_ctx_, GGML_TYPE_F32,
                                                  kv_dim, (int64_t)n_kv_max * n_slots);
    batch_kv_v_[(size_t)il] = ggml_new_tensor_2d(batch_kv_ctx_, GGML_TYPE_F32,
                                                  kv_dim, (int64_t)n_kv_max * n_slots);
    ggml_set_name(batch_kv_k_[(size_t)il],
                  ("cv3_batch_kv_k_" + std::to_string(il)).c_str());
    ggml_set_name(batch_kv_v_[(size_t)il],
                  ("cv3_batch_kv_v_" + std::to_string(il)).c_str());
  }
  ggml_backend_t gpu_b = pick_gpu_backend(sched, backend_);
  batch_kv_buf_ = ggml_backend_alloc_ctx_tensors(batch_kv_ctx_, gpu_b);
  if (!batch_kv_buf_) {
    RS_LOG_ERR("CosyVoice3-batch: failed to alloc batched GPU KV cache");
    FreeBatchKV();
    return false;
  }
  std::vector<float> zero((size_t)kv_dim * n_kv_max * n_slots, 0.0f);
  for (int il = 0; il < n_layer; ++il) {
    ggml_backend_tensor_set(batch_kv_k_[(size_t)il], zero.data(), 0,
                            zero.size() * sizeof(float));
    ggml_backend_tensor_set(batch_kv_v_[(size_t)il], zero.data(), 0,
                            zero.size() * sizeof(float));
  }
  RS_LOG_INFO("CosyVoice3-batch: alloc batched KV n_slots=%d n_kv_max=%d "
              "(%.1f MiB)", n_slots, n_kv_max,
              (double)(kv_dim * n_kv_max * n_slots * 2 * n_layer * 4) / 1048576.0);
  return true;
}

void CosyVoice3LMModel::FreeBatchKV() {
  if (batch_kv_buf_) { ggml_backend_buffer_free(batch_kv_buf_); batch_kv_buf_ = nullptr; }
  if (batch_kv_ctx_) { ggml_free(batch_kv_ctx_); batch_kv_ctx_ = nullptr; }
  batch_kv_k_.clear();
  batch_kv_v_.clear();
  batch_n_slots_ = 0;
  batch_n_kv_max_ = 0;
}

void CosyVoice3LMModel::CopyBatchKVSlice(int src_slot, int dst_slot) {
  if (src_slot == dst_slot || !batch_kv_buf_ || !llm_model_) return;
  const auto &hp = llm_model_->hparams();
  const int n_layer = (int)hp.n_layer;
  const int n_head_kv = (int)(hp.n_head_kv > 0 ? hp.n_head_kv : hp.n_head);
  const int kv_dim = (int)hp.head_dim * n_head_kv;
  const size_t slice_f = (size_t)kv_dim * batch_n_kv_max_;
  const size_t slice_bytes = slice_f * sizeof(float);
  std::vector<float> buf(slice_f);
  for (int il = 0; il < n_layer; ++il) {
    const size_t so = (size_t)src_slot * slice_bytes;
    const size_t doff = (size_t)dst_slot * slice_bytes;
    ggml_backend_tensor_get(batch_kv_k_[(size_t)il], buf.data(), so, slice_bytes);
    ggml_backend_tensor_set(batch_kv_k_[(size_t)il], buf.data(), doff, slice_bytes);
    ggml_backend_tensor_get(batch_kv_v_[(size_t)il], buf.data(), so, slice_bytes);
    ggml_backend_tensor_set(batch_kv_v_[(size_t)il], buf.data(), doff, slice_bytes);
  }
}

// Single-slot step: delegate to the batched path with n=1.
bool CosyVoice3LMModel::RunLM_Step(CosyVoice3State &s,
                                   CosyVoice3LMSlot &slot,
                                   ggml_backend_sched_t sched) {
  CosyVoice3State *sp = &s;
  CosyVoice3LMSlot *slp = &slot;
  return RunLM_Step_Batch(&sp, &slp, 1, sched);
}

// ── Batched AR decode step (vLLM-style continuous batching) ────────────────
// Runs n slots' current tokens through ONE forward pass. n=1 takes the
// ordinary single-seq decode branch (n_seq==1); n>1 takes the ne3-batched
// flash-attn path where each sequence attends only over its own KV slice.
bool CosyVoice3LMModel::RunLM_Step_Batch(CosyVoice3State **states,
                                         CosyVoice3LMSlot **slots, int n,
                                         ggml_backend_sched_t sched) {
  if (!llm_model_ || n <= 0) return false;
  for (int i = 0; i < n; ++i) {
    if (!slots[i]->builder || slots[i]->done) return false;
  }
  if (batch_n_slots_ <= 0 || !batch_kv_buf_) {
    RS_LOG_ERR("CosyVoice3-batch: InitBatchKV not called before step");
    return false;
  }

  const auto &hp = llm_model_->hparams();
  const int n_layer = (int)hp.n_layer;
  const int d_model = (int)hp.n_embd;
  ggml_backend_t gpu_b = pick_gpu_backend(sched, backend_);

  ggml_init_params sp = {128 * ggml_tensor_overhead() + (1 << 18), nullptr,
                          true};
  ggml_context *ctx_step = ggml_init(sp);

  ggml_tensor *id_in = ggml_new_tensor_1d(ctx_step, GGML_TYPE_I32, n);
  ggml_set_name(id_in, "cv3_speech_id");
  ggml_set_input(id_in);
  ggml_tensor *embed =
      ggml_get_rows(ctx_step, llm_model_->speech_embd(), id_in); // [d_model, n]

  llm_build_opts dec_opts;
  dec_opts.output_mode = llm_output_mode::OUTPUT_EMBEDDINGS;
  dec_opts.skip_embeddings = true;
  dec_opts.skip_lm_head = true;
  dec_opts.use_kv_cache = true;
  dec_opts.is_decode_step = true;
  dec_opts.causal_mask = true;
  dec_opts.n_kv_max = (uint32_t)batch_n_kv_max_;
  dec_opts.fixed_kv_cache_shape = true;
  dec_opts.n_seq = (uint32_t)n;
  dec_opts.gpu_kv_k = batch_kv_k_.data();
  dec_opts.gpu_kv_v = batch_kv_v_.data();

  std::vector<llm_pos> pos(n);
  std::vector<uint32_t> write_idx(n);
  std::vector<uint32_t> n_cached_per_seq(n);
  for (int i = 0; i < n; ++i) {
    pos[i] = (llm_pos)states[i]->n_cached;
    n_cached_per_seq[i] = states[i]->n_cached;
    write_idx[i] = (uint32_t)slots[i]->slot_index * (uint32_t)batch_n_kv_max_ +
                   states[i]->n_cached;
  }
  if (n == 1) {
    dec_opts.kv_slot_row_offset =
        (uint32_t)slots[0]->slot_index * (uint32_t)batch_n_kv_max_;
  }

  auto dec = slots[0]->builder->build_graph_from_embeds(embed, (uint32_t)n,
                                                        nullptr, pos.data(),
                                                        &dec_opts);
  if (!dec) { ggml_free(ctx_step); return false; }

  dec->assign_kv_backend(sched, gpu_b, n_layer);
  if (!ggml_backend_sched_alloc_graph(sched, dec->get_graph())) {
    ggml_free(ctx_step);
    return false;
  }

  {
    std::vector<int32_t> toks(n);
    for (int i = 0; i < n; ++i) toks[i] = slots[i]->cur_tok;
    ggml_backend_tensor_set(id_in, toks.data(), 0, n * sizeof(int32_t));
  }
  if (auto *pt = dec->get_input_tensor("position_ids")) {
    dec->set_position_ids(pt, pos.data(), (uint32_t)n);
  }
  if (auto *m = dec->get_input_tensor("causal_mask")) {
    if (n > 1) {
      dec->set_batched_causal_mask(m, (uint32_t)batch_n_kv_max_,
                                   n_cached_per_seq.data(), (uint32_t)n);
    } else {
      dec->set_causal_mask(m, 1, states[0]->n_cached);
    }
  }
  if (n > 1) {
    dec->set_kv_write_indices_batched(write_idx.data(), (uint32_t)n, n_layer);
  } else {
    dec->set_kv_write_indices(write_idx[0], n_layer);
  }

  if (cv3_sched_compute(sched, dec->get_graph(), &imatrix_cb_) !=
      GGML_STATUS_SUCCESS) {
    ggml_free(ctx_step);
    return false;
  }

  std::vector<float> step_hidden((size_t)d_model * n);
  if (ggml_tensor *h = dec->get_embd()) {
    ggml_backend_tensor_get(h, step_hidden.data(), 0,
                            (size_t)d_model * n * sizeof(float));
  }
  for (int i = 0; i < n; ++i) states[i]->n_cached++;
  ggml_free(ctx_step);
  ggml_backend_sched_reset(sched);

  std::vector<float> all_logits = project_speech_logits(
      sched, llm_model_->output_norm(), llm_model_->speech_lm_head(),
      hp.f_norm_rms_eps, step_hidden.data(), d_model, speech_vocab_, n,
      &imatrix_cb_);
  if ((int)all_logits.size() != speech_vocab_ * n) return false;

  for (int i = 0; i < n; ++i) {
    const float *lg = all_logits.data() + (size_t)i * speech_vocab_;
    std::vector<float> lg_vec(lg, lg + speech_vocab_);
    if (slots[i]->step + 1 < slots[i]->min_len) {
      for (int t = speech_codebook_; t < speech_vocab_; ++t)
        lg_vec[(size_t)t] = -std::numeric_limits<float>::infinity();
    }
    int32_t next = cosyvoice3_sample_ras(lg_vec.data(), speech_vocab_,
                                         states[i]->speech_token_ids,
                                         states[i]->sampler, states[i]->rng);
    slots[i]->step++;

    if (next >= speech_codebook_) {
      RS_LOG_INFO("CosyVoice3-batch: slot %d EOS at step %d (%zu tokens)",
                  slots[i]->slot_index, slots[i]->step,
                  states[i]->speech_token_ids.size());
      slots[i]->done = true;
      continue;
    }
    states[i]->speech_token_ids.push_back(next);
    slots[i]->cur_tok = next;
    if (slots[i]->step >= slots[i]->max_len) {
      RS_LOG_INFO("CosyVoice3-batch: slot %d max_len %d reached",
                  slots[i]->slot_index, slots[i]->max_len);
      slots[i]->done = true;
    }
  }
  return true;
}
