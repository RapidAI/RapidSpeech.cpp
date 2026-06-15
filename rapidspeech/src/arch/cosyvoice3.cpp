#include "cosyvoice3.h"

#include "campplus.h"
#include "cosyvoice3_flow.h"
#include "cosyvoice3_hift.h"
#include "cosyvoice3_speech_tokenizer.h"
#include "frontend/cosyvoice3_mel.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llm_kv_cache.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

// =====================================================================
// Construction / metadata
// =====================================================================

CosyVoice3LMModel::CosyVoice3LMModel() {
  // The arch name is finalized in Load() based on whether the unified GGUF
  // contains Flow + HiFT weights. Until then, default to the legacy LLM-only
  // label so that GGUFs produced by `convert_cosyvoice3_llm_to_gguf.py` still
  // surface as `cosyvoice3-llm`.
  meta_.arch_name = "cosyvoice3-llm";
  meta_.audio_sample_rate = 24000;
  meta_.n_mels = 0;
  meta_.use_external_frontend = true;
}

CosyVoice3LMModel::~CosyVoice3LMModel() {
  // Order matters: drop the model wrappers BEFORE freeing the GGUF resources
  // they reference, otherwise tensor pointers dangle.
  speech_tokenizer_.reset();
  campplus_model_.reset();
  flow_.reset();
  hift_.reset();

  for (ExternalGguf *eg : {&tokenizer_gguf_, &campplus_gguf_}) {
    if (eg->buf) ggml_backend_buffer_free(eg->buf);
    if (eg->ctx_gguf) gguf_free(eg->ctx_gguf);
    if (eg->ctx_data) ggml_free(eg->ctx_data);
    eg->buf = nullptr; eg->ctx_gguf = nullptr; eg->ctx_data = nullptr;
  }
}

void CosyVoice3LMModel::set_imatrix_callback(
    std::function<void(struct ggml_tensor *)> cb) {
  // Propagate to Flow first (DiT is the main calibration target). HiFT is
  // skipped intentionally — its conv kernels (ne[0]=3..16) fail K-quant
  // alignment so imatrix data on them would never be consumed.
  if (flow_) flow_->set_imatrix_callback(cb);
  imatrix_cb_ = std::move(cb);
}

// =====================================================================
// External-GGUF loader. Mirrors rs_context_init_internal's load loop but
// targets a pre-existing backend handle (the main model's).
// =====================================================================

namespace {

bool load_external_gguf(const char *path, ggml_backend_t backend,
                        CosyVoice3LMModel::ExternalGguf *out) {
  if (!path || !*path || !backend || !out) return false;

  struct gguf_init_params gp = { /*.no_alloc=*/true, /*.ctx=*/&out->ctx_data };
  out->ctx_gguf = gguf_init_from_file(path, gp);
  if (!out->ctx_gguf) {
    RS_LOG_ERR("CosyVoice3: failed to load external GGUF: %s", path);
    return false;
  }
  out->buf = ggml_backend_alloc_ctx_tensors(out->ctx_data, backend);
  if (!out->buf) {
    RS_LOG_ERR("CosyVoice3: failed to allocate external GGUF tensors: %s",
               path);
    return false;
  }
  FILE *f = fopen(path, "rb");
  if (!f) {
    RS_LOG_ERR("CosyVoice3: fopen %s failed", path);
    return false;
  }
  const size_t data_off = gguf_get_data_offset(out->ctx_gguf);
  const int64_t n_tensors = gguf_get_n_tensors(out->ctx_gguf);
  std::vector<char> buf;
  for (int64_t i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(out->ctx_gguf, i);
    ggml_tensor *t = ggml_get_tensor(out->ctx_data, name);
    if (!t) continue;
    size_t t_off = gguf_get_tensor_offset(out->ctx_gguf, i);
    size_t t_sz  = ggml_nbytes(t);
    if (t_sz == 0) continue;
    if (buf.size() < t_sz) buf.resize(t_sz);
    fseek(f, data_off + t_off, SEEK_SET);
    if (fread(buf.data(), 1, t_sz, f) != t_sz) {
      RS_LOG_ERR("CosyVoice3: failed to read %s from %s", name, path);
      fclose(f);
      return false;
    }
    ggml_backend_tensor_set(t, buf.data(), 0, t_sz);
  }
  fclose(f);
  return true;
}

} // namespace

// =====================================================================
// Load
// =====================================================================

bool CosyVoice3LMModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                             ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("CosyVoice3: invalid context");
    return false;
  }
  backend_ = backend;

  llm_model_ = std::make_shared<llm_model>();
  if (!llm_model_->load_metadata_from_gguf(ctx->ctx_gguf)) {
    RS_LOG_ERR("CosyVoice3: load_metadata_from_gguf failed");
    return false;
  }

  std::map<std::string, ggml_tensor *> tensors;
  const int n = gguf_get_n_tensors(ctx->ctx_gguf);
  for (int i = 0; i < n; ++i) {
    const char *name = gguf_get_tensor_name(ctx->ctx_gguf, i);
    ggml_tensor *t = ggml_get_tensor(ctx->gguf_data, name);
    if (t) tensors[name] = t;
  }
  if (!llm_model_->map_tensors_qwen2(tensors)) {
    RS_LOG_ERR("CosyVoice3: map_tensors_qwen2 failed");
    return false;
  }

  // Build fused QKV weights (best-effort optimization). Skip on failure —
  // qwen2.cpp falls back to the 3× mul_mat path when layer.wqkv is null.
  if (!llm_model_->fuse_qkv_weights(backend_)) {
    RS_LOG_WARN("CosyVoice3: fuse_qkv_weights failed; using unfused QKV path");
  }

  if (llm_model_->speech_embd()) {
    speech_vocab_ = (int32_t)llm_model_->speech_embd()->ne[1];
  }
  if (llm_model_->speech_lm_head()) {
    const int32_t v_head = (int32_t)llm_model_->speech_lm_head()->ne[1];
    if (v_head != speech_vocab_) {
      RS_LOG_WARN("CosyVoice3: speech_embd=%d, lm_head=%d (using head)",
                  speech_vocab_, v_head);
      speech_vocab_ = v_head;
    }
  }
  {
    int key = gguf_find_key(ctx->ctx_gguf, "cosyvoice3.llm.speech_token_codebook");
    if (key >= 0) {
      speech_codebook_ = (int32_t)gguf_get_val_u32(ctx->ctx_gguf, key);
    } else {
      speech_codebook_ = speech_vocab_ - 1;
    }
  }
  stop_token_id_ = speech_codebook_;

  // CosyVoice3 tokenizer special tokens (see CosyVoice3Tokenizer in upstream
  // `cosyvoice/tokenizer/tokenizer.py`). `<|endofprompt|>` is the load-bearing
  // one: CosyVoice3LM.inference asserts that token id 151646 is present in
  // the concatenated text stream — it separates the assistant-prefix from the
  // ref transcript and trains the LLM where the "real" content begins.
  {
    auto &vocab = const_cast<llm_vocab &>(llm_model_->vocab());
    vocab.add_special_token("<|endoftext|>",   151643);
    vocab.add_special_token("<|im_start|>",    151644);
    vocab.add_special_token("<|im_end|>",      151645);
    vocab.add_special_token("<|endofprompt|>", 151646);
  }

  meta_.vocab_size = (int)llm_model_->hparams().n_vocab;

  // --------------------------------------------------------------------
  // Flow + HiFT (optional — present only in unified GGUFs).
  // --------------------------------------------------------------------
  flow_ = std::make_unique<CosyVoice3FlowModel>();
  if (flow_->Load(ctx->ctx_gguf, ctx->gguf_data, backend_)) {
    flow_ready_ = true;
  } else {
    RS_LOG_WARN("CosyVoice3: Flow weights not found — LLM-only mode");
    flow_.reset();
  }

  hift_ = std::make_unique<CosyVoice3HiFTModel>();
  if (hift_->Load(ctx->ctx_gguf, ctx->gguf_data, backend_)) {
    hift_ready_ = true;
  } else {
    RS_LOG_WARN("CosyVoice3: HiFT weights not found — LLM-only mode");
    hift_.reset();
  }

  // If both audio stages loaded, advertise as the unified arch (so the CLI's
  // PCM path is taken in examples/tts/rs-tts-offline.cpp).
  if (flow_ready_ && hift_ready_) {
    meta_.arch_name = "cosyvoice3";
  }

  // Default voice tuple (baked at conversion time). All three are optional.
  TryLoadDefaultVoiceFromGGUF(ctx->ctx_gguf, ctx->gguf_data);

  // Pre-baked voice GGUF (overrides the unified GGUF's default voice). Lets
  // production runs reuse a cached speech-token/feat/embedding tuple without
  // loading the speech tokenizer or CAMPPlus at all.
  if (const char *p = std::getenv("RS_CV3_VOICE_PATH")) {
    TryLoadVoiceFile(p);
  }

  // Optional external GGUFs for runtime voice cloning via --ref.
  if (const char *p = std::getenv("RS_CV3_CAMPPLUS_PATH")) {
    TryLoadExternalCampplus(p);
  }
  if (const char *p = std::getenv("RS_CV3_SPEECH_TOKENIZER_PATH")) {
    TryLoadExternalTokenizer(p);
  }

  RS_LOG_INFO("CosyVoice3 loaded (arch=%s flow=%d hift=%d tok=%d cam=%d "
              "def_voice=%d): layers=%u d_model=%u text_vocab=%u "
              "speech_vocab=%d stop=%d",
              meta_.arch_name.c_str(), (int)flow_ready_, (int)hift_ready_,
              (int)speech_tokenizer_ready_, (int)campplus_ready_,
              (int)!default_embedding_.empty(),
              llm_model_->hparams().n_layer, llm_model_->hparams().n_embd,
              llm_model_->hparams().n_vocab, speech_vocab_, stop_token_id_);
  return true;
}

// =====================================================================
// Default voice extraction from the unified GGUF.
// =====================================================================

bool CosyVoice3LMModel::TryLoadDefaultVoiceFromGGUF(gguf_context *ctx_gguf,
                                                    ggml_context *gguf_data) {
  auto pull = [&](const char *name, std::vector<float> &dst) {
    ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (!t) return false;
    dst.assign(ggml_nelements(t), 0.f);
    ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    return true;
  };
  auto pull_i32 = [&](const char *name, std::vector<int32_t> &dst) {
    ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (!t) return false;
    dst.assign(ggml_nelements(t), 0);
    ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(int32_t));
    return true;
  };
  (void)ctx_gguf;
  const bool has_text = pull_i32("cv3.default_voice.prompt_text_ids",
                                 default_prompt_text_ids_);
  const bool has_tok  = pull_i32("cv3.default_voice.prompt_token",
                                 default_prompt_token_);
  const bool has_feat = pull("cv3.default_voice.prompt_feat",
                             default_prompt_feat_);
  const bool has_emb  = pull("cv3.default_voice.embedding",
                             default_embedding_);
  if (!has_emb) {
    default_embedding_.clear();
    default_prompt_text_ids_.clear();
    default_prompt_token_.clear();
    default_prompt_feat_.clear();
    return false;
  }
  if (!has_text) default_prompt_text_ids_.clear();
  if (!has_tok)  default_prompt_token_.clear();
  if (!has_feat) default_prompt_feat_.clear();
  RS_LOG_INFO("CosyVoice3: default voice loaded (prompt_text_ids=%zu "
              "prompt_token=%zu prompt_feat=%zu embedding=%zu)",
              default_prompt_text_ids_.size(),
              default_prompt_token_.size(), default_prompt_feat_.size(),
              default_embedding_.size());
  return true;
}

// =====================================================================
// Voice bake/reuse — standalone voice GGUF (cv3.default_voice.* tensors).
// Baking caches the speech-tokenizer + CAMPPlus outputs so production
// synthesis with a fixed voice skips both forward passes (and both model
// files).
// =====================================================================

bool CosyVoice3LMModel::TryLoadVoiceFile(const char *path) {
  if (!path || !*path) return false;
  ggml_context *ctx_data = nullptr;
  struct gguf_init_params gp = { /*.no_alloc=*/false, /*.ctx=*/&ctx_data };
  gguf_context *ctx_gguf = gguf_init_from_file(path, gp);
  if (!ctx_gguf) {
    RS_LOG_ERR("CosyVoice3: failed to open voice GGUF: %s", path);
    return false;
  }
  // no_alloc=false → tensor data lives in host memory inside ctx_data.
  auto pull_f32 = [&](const char *name, std::vector<float> &dst) {
    ggml_tensor *t = ggml_get_tensor(ctx_data, name);
    if (!t || !t->data || t->type != GGML_TYPE_F32) return false;
    dst.assign((size_t)ggml_nelements(t), 0.f);
    std::memcpy(dst.data(), t->data, dst.size() * sizeof(float));
    return true;
  };
  auto pull_i32 = [&](const char *name, std::vector<int32_t> &dst) {
    ggml_tensor *t = ggml_get_tensor(ctx_data, name);
    if (!t || !t->data || t->type != GGML_TYPE_I32) return false;
    dst.assign((size_t)ggml_nelements(t), 0);
    std::memcpy(dst.data(), t->data, dst.size() * sizeof(int32_t));
    return true;
  };

  std::vector<int32_t> text_ids, tok;
  std::vector<float> feat, emb;
  pull_i32("cv3.default_voice.prompt_text_ids", text_ids);
  pull_i32("cv3.default_voice.prompt_token", tok);
  pull_f32("cv3.default_voice.prompt_feat", feat);
  const bool has_emb = pull_f32("cv3.default_voice.embedding", emb);

  std::string ref_text;
  {
    const int key = gguf_find_key(ctx_gguf, "cv3.voice.ref_text");
    if (key >= 0) ref_text = gguf_get_val_str(ctx_gguf, key);
  }
  gguf_free(ctx_gguf);
  ggml_free(ctx_data);

  if (!has_emb) {
    RS_LOG_ERR("CosyVoice3: voice GGUF %s lacks cv3.default_voice.embedding",
               path);
    return false;
  }
  default_prompt_text_ids_ = std::move(text_ids);
  default_prompt_token_    = std::move(tok);
  default_prompt_feat_     = std::move(feat);
  default_embedding_       = std::move(emb);
  RS_LOG_INFO("CosyVoice3: voice loaded from %s (text_ids=%zu tok=%zu "
              "feat=%zux80 emb=%zu ref_text='%s')",
              path, default_prompt_text_ids_.size(),
              default_prompt_token_.size(), default_prompt_feat_.size() / 80,
              default_embedding_.size(), ref_text.c_str());
  return true;
}

bool CosyVoice3LMModel::SaveVoiceFile(const CosyVoice3State &state,
                                      const char *path) {
  if (!path || !*path) return false;
  if (state.embedding.empty()) {
    RS_LOG_ERR("CosyVoice3: cannot save voice — state has no embedding");
    return false;
  }
  const size_t data_bytes =
      (state.prompt_text_token_ids.size() + state.prompt_token.size()) *
          sizeof(int32_t) +
      (state.prompt_feat.size() + state.embedding.size()) * sizeof(float);
  ggml_init_params ip = {
      /*.mem_size=*/data_bytes + 4 * ggml_tensor_overhead() + 1024,
      /*.mem_buffer=*/nullptr,
      /*.no_alloc=*/false,
  };
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) return false;
  gguf_context *g = gguf_init_empty();
  gguf_set_val_str(g, "general.architecture", "cosyvoice3-voice");
  if (!ref_text_.empty()) {
    gguf_set_val_str(g, "cv3.voice.ref_text", ref_text_.c_str());
  }

  auto add_i32 = [&](const char *name, const std::vector<int32_t> &v) {
    if (v.empty()) return;
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)v.size());
    ggml_set_name(t, name);
    std::memcpy(t->data, v.data(), v.size() * sizeof(int32_t));
    gguf_add_tensor(g, t);
  };
  add_i32("cv3.default_voice.prompt_text_ids", state.prompt_text_token_ids);
  add_i32("cv3.default_voice.prompt_token", state.prompt_token);
  if (!state.prompt_feat.empty()) {
    // Host layout is row-major [T, 80] (80 mel bins contiguous per frame) →
    // ggml ne0=80 (fastest), ne1=T.
    const int64_t T = (int64_t)(state.prompt_feat.size() / 80);
    ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 80, T);
    ggml_set_name(t, "cv3.default_voice.prompt_feat");
    std::memcpy(t->data, state.prompt_feat.data(),
                state.prompt_feat.size() * sizeof(float));
    gguf_add_tensor(g, t);
  }
  {
    ggml_tensor *t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                        (int64_t)state.embedding.size());
    ggml_set_name(t, "cv3.default_voice.embedding");
    std::memcpy(t->data, state.embedding.data(),
                state.embedding.size() * sizeof(float));
    gguf_add_tensor(g, t);
  }

  const bool ok = gguf_write_to_file(g, path, /*only_meta=*/false);
  gguf_free(g);
  ggml_free(ctx);
  if (ok) {
    RS_LOG_INFO("CosyVoice3: voice baked to %s (text_ids=%zu tok=%zu "
                "feat=%zux80 emb=%zu)",
                path, state.prompt_text_token_ids.size(),
                state.prompt_token.size(), state.prompt_feat.size() / 80,
                state.embedding.size());
  } else {
    RS_LOG_ERR("CosyVoice3: failed to write voice GGUF: %s", path);
  }
  return ok;
}

bool CosyVoice3LMModel::TryLoadExternalCampplus(const char *path) {
  if (!load_external_gguf(path, backend_, &campplus_gguf_)) return false;
  campplus_model_ = std::make_shared<CAMPPlusModel>();
  if (!campplus_model_->LoadDirect(campplus_gguf_.ctx_data,
                                   campplus_gguf_.ctx_gguf, backend_)) {
    RS_LOG_ERR("CosyVoice3: CAMPPlusModel::LoadDirect failed (%s)", path);
    campplus_model_.reset();
    return false;
  }
  campplus_ready_ = true;
  RS_LOG_INFO("CosyVoice3: CAMPPlus loaded from %s", path);
  return true;
}

bool CosyVoice3LMModel::TryLoadExternalTokenizer(const char *path) {
  if (!load_external_gguf(path, backend_, &tokenizer_gguf_)) return false;
  speech_tokenizer_ = std::make_shared<CosyVoice3SpeechTokenizer>();
  if (!speech_tokenizer_->Load(tokenizer_gguf_.ctx_gguf,
                               tokenizer_gguf_.ctx_data, backend_)) {
    RS_LOG_ERR("CosyVoice3: SpeechTokenizer::Load failed (%s)", path);
    speech_tokenizer_.reset();
    return false;
  }
  speech_tokenizer_ready_ = true;
  RS_LOG_INFO("CosyVoice3: speech_tokenizer_v3 loaded from %s", path);
  return true;
}

// =====================================================================
// State factory
// =====================================================================

std::shared_ptr<RSState> CosyVoice3LMModel::CreateState() {
  auto s = std::make_shared<CosyVoice3State>();
  s->sampler = default_sampler_;
  // Debug knob: when RS_CV3_GREEDY=1 is set, force temperature=0 (greedy
  // argmax). Lets us cleanly separate sampling-noise issues from prompt /
  // forward-pass issues.
  if (const char *p = std::getenv("RS_CV3_GREEDY")) {
    if (*p && *p != '0') s->sampler.temperature = 0.0f;
  }
  s->max_speech_tokens = max_tokens_;
  s->rng.seed(seed_);
  return s;
}

// =====================================================================
// PushText / PushReferenceText / PushReferenceAudio
// =====================================================================

bool CosyVoice3LMModel::PushText(RSState &state, const char *text,
                                 const char *language, const char *instruct) {
  (void)language;
  auto &s = static_cast<CosyVoice3State &>(state);
  if (!llm_model_) return false;

  // Instruct2 mode (upstream `inference_instruct2`): `instruct_text` replaces
  // the ref transcript in the LM prefix and llm_prompt_speech_token gets
  // dropped. The Flow path (prompt_token / prompt_feat / spk_emb) is
  // untouched, so timbre cloning still works — only prosody is steered by
  // the instruct text. We treat the OmniVoice gender placeholders
  // ("male"/"female") as "no instruct" so the shared --instruct CLI flag
  // keeps doing the right thing for both archs.
  s.instruct_text_token_ids.clear();
  if (instruct && *instruct) {
    std::string iv = instruct;
    const bool is_ov_gender_placeholder = (iv == "male" || iv == "female" ||
                                           iv == "Male" || iv == "Female");
    if (!is_ov_gender_placeholder) {
      s.instruct_text_token_ids =
          llm_model_->vocab().tokenize(iv, false);
      RS_LOG_INFO("CosyVoice3: instruct2 mode (%zu BPE ids from '%s')",
                  s.instruct_text_token_ids.size(), iv.c_str());
    }
  }

  if (!text || !*text) {
    s.text_token_ids.clear();
    return true;
  }
  s.text_token_ids = llm_model_->vocab().tokenize(std::string(text), false);
  {
    std::string head; char buf[16];
    for (size_t i = 0; i < std::min<size_t>(20, s.text_token_ids.size()); ++i) {
      snprintf(buf, sizeof(buf), "%d,", s.text_token_ids[i]);
      head += buf;
    }
    RS_LOG_INFO("CosyVoice3: tokenized %zu text ids from '%s' first=[%s]",
                s.text_token_ids.size(), text, head.c_str());
  }
  return !s.text_token_ids.empty();
}

bool CosyVoice3LMModel::PushReferenceText(RSState &state, const char *ref_text) {
  auto &s = static_cast<CosyVoice3State &>(state);
  ref_text_ = ref_text ? ref_text : "";
  if (!ref_text_.empty() && llm_model_) {
    s.prompt_text_token_ids = llm_model_->vocab().tokenize(ref_text_, false);
    RS_LOG_INFO("CosyVoice3: ref text → %zu BPE ids",
                s.prompt_text_token_ids.size());
  } else {
    s.prompt_text_token_ids.clear();
  }
  return true;
}

bool CosyVoice3LMModel::PushReferenceAudio(RSState &state, const float *samples,
                                           int n_samples, int sample_rate,
                                           ggml_backend_sched_t sched) {
  auto &s = static_cast<CosyVoice3State &>(state);
  if (!samples || n_samples <= 0) {
    RS_LOG_ERR("CosyVoice3: PushReferenceAudio called with empty pcm");
    return false;
  }
  if (!speech_tokenizer_ready_ || !campplus_ready_) {
    RS_LOG_ERR("CosyVoice3: --ref requires RS_CV3_SPEECH_TOKENIZER_PATH and "
               "RS_CV3_CAMPPLUS_PATH to be set");
    return false;
  }

  // 1) Resample to 16 kHz for the V3 tokenizer + CAMPPlus.
  std::vector<float> pcm_in(samples, samples + n_samples);
  std::vector<float> pcm_16k;
  if (sample_rate == 16000) {
    pcm_16k = pcm_in;
  } else {
    if (!resample_pcm(pcm_in, sample_rate, pcm_16k, 16000)) {
      RS_LOG_ERR("CosyVoice3: resample %d → 16000 failed", sample_rate);
      return false;
    }
  }

  {
    float amax = 0.f; double esq = 0.0;
    for (float v : pcm_16k) {
      if (std::fabs(v) > amax) amax = std::fabs(v);
      esq += (double)v * v;
    }
    double rms = std::sqrt(esq / std::max<size_t>(1, pcm_16k.size()));
    RS_LOG_INFO("CosyVoice3: pcm_16k stats — n=%zu sr=%d amax=%.4f rms=%.4f "
                "first5=[%.4f %.4f %.4f %.4f %.4f]",
                pcm_16k.size(), sample_rate, (double)amax, rms,
                (double)pcm_16k[0], (double)pcm_16k[1], (double)pcm_16k[2],
                (double)pcm_16k[3], (double)pcm_16k[4]);
    if (const char *p = std::getenv("RS_CV3_DUMP_PCM16K")) {
      if (FILE *f = std::fopen(p, "wb")) {
        std::fwrite(pcm_16k.data(), sizeof(float), pcm_16k.size(), f);
        std::fclose(f);
        RS_LOG_INFO("CosyVoice3: dumped pcm_16k (%zu floats) -> %s",
                    pcm_16k.size(), p);
      }
    }
  }

  // 2) Tokenize via V3 speech tokenizer (or load from an external dump for
  //    parity-debugging).
  std::vector<int32_t> prompt_token;
  if (const char *p = std::getenv("RS_CV3_PROMPT_TOKENS_BIN")) {
    FILE *f = std::fopen(p, "rb");
    if (f) {
      std::fseek(f, 0, SEEK_END);
      long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      const int n = (int)(sz / sizeof(int32_t));
      prompt_token.assign((size_t)n, 0);
      std::fread(prompt_token.data(), sizeof(int32_t), n, f);
      std::fclose(f);
      RS_LOG_INFO("CosyVoice3: loaded %d prompt tokens from %s (debug dump)",
                  n, p);
    }
  }
  if (prompt_token.empty() &&
      speech_tokenizer_->Tokenize(pcm_16k.data(), (int)pcm_16k.size(),
                                  prompt_token, sched) != 0) {
    RS_LOG_ERR("CosyVoice3: V3 tokenizer failed");
    return false;
  }
  {
    std::string head, tail;
    char buf[32];
    for (size_t i = 0; i < std::min<size_t>(15, prompt_token.size()); ++i) {
      snprintf(buf, sizeof(buf), "%d,", prompt_token[i]);
      head += buf;
    }
    for (size_t i = (prompt_token.size() > 5 ? prompt_token.size() - 5 : 0);
         i < prompt_token.size(); ++i) {
      snprintf(buf, sizeof(buf), "%d,", prompt_token[i]);
      tail += buf;
    }
    RS_LOG_INFO("CosyVoice3: ref → %zu speech tokens. first15=[%s] last5=[%s]",
                prompt_token.size(), head.c_str(), tail.c_str());
  }

  // 3) CAMPPlus spk embedding from the 16 kHz audio.
  CAMPPlusState cs;
  if (!campplus_model_->Embed(pcm_16k.data(), (int)pcm_16k.size(), cs, sched)) {
    RS_LOG_ERR("CosyVoice3: CAMPPlus::Embed failed");
    return false;
  }
  if (cs.embedding.empty()) {
    RS_LOG_ERR("CosyVoice3: CAMPPlus produced empty embedding");
    return false;
  }

  // 4) 80-mel @ 24 kHz for `prompt_feat` (Flow input). Resample to 24 kHz.
  std::vector<float> pcm_24k;
  if (sample_rate == 24000) {
    pcm_24k = pcm_in;
  } else {
    if (!resample_pcm(pcm_in, sample_rate, pcm_24k, 24000)) {
      RS_LOG_ERR("CosyVoice3: resample %d → 24000 failed", sample_rate);
      return false;
    }
  }
  std::vector<float> prompt_feat;
  const int T_pt_mel = compute_log_mel_80_24k(pcm_24k.data(),
                                              (int)pcm_24k.size(), prompt_feat);
  if (T_pt_mel <= 0) {
    RS_LOG_ERR("CosyVoice3: 80-mel @ 24 kHz failed");
    return false;
  }

  // 5) Align prompt_feat and prompt_token. Mirrors PT
  //    CosyVoiceFrontEnd.frontend_zero_shot (resample_rate==24000):
  //        token_len = min(speech_feat.shape[1] // 2, speech_token.shape[1])
  //        speech_feat  = speech_feat[:, :2*token_len]
  //        speech_token = speech_token[:, :token_len]
  //    Both are clipped — neither side is ever extended. Previously RS would
  //    pad prompt_feat with a duplicated last frame to match an oversized
  //    token count, which inflated prompt_feat by 1 frame and silently fed
  //    an extra token into Flow.
  constexpr int kTokenMelRatio = 2;
  const int mel_T   = (int)(prompt_feat.size() / 80);
  const int n_tok   = (int)prompt_token.size();
  const int token_len = std::min(mel_T / kTokenMelRatio, n_tok);
  prompt_feat.resize((size_t)token_len * kTokenMelRatio * 80);
  prompt_token.resize((size_t)token_len);

  // 6) Write everything into the state.
  s.prompt_token = std::move(prompt_token);
  s.prompt_feat  = std::move(prompt_feat);
  s.embedding    = std::move(cs.embedding);
  // Debug: dump prompt_feat (post-align) for PT cross-check.
  // Format: int32 T_pt_mel, int32 mel_dim(=80), then f32 feat[T*mel_dim].
  // Memory order matches RS_CV3_DUMP_FLOW_ENC: mel-fast (each frame's 80
  // mel-bins contiguous), i.e. row-major [T, mel_dim].
  if (const char *p = std::getenv("RS_CV3_DUMP_PROMPT_FEAT")) {
    if (FILE *f = std::fopen(p, "wb")) {
      int32_t hdr[2] = {(int32_t)(s.prompt_feat.size() / 80), 80};
      std::fwrite(hdr, sizeof(int32_t), 2, f);
      std::fwrite(s.prompt_feat.data(), sizeof(float), s.prompt_feat.size(), f);
      std::fclose(f);
      RS_LOG_INFO("CosyVoice3: dumped prompt_feat [%d, 80] -> %s",
                  hdr[0], p);
    }
  }
  RS_LOG_INFO("CosyVoice3: ref voice ready (tok=%zu, feat=%dx80, emb=%zu)",
              s.prompt_token.size(),
              (int)(s.prompt_feat.size() / 80),
              s.embedding.size());
  return true;
}

// =====================================================================
// PrepareVoice — populate state.{prompt_token, prompt_feat, embedding} from
// the GGUF-baked default voice when --ref was not provided (or extraction
// failed silently). Without an embedding the Flow can't produce intelligible
// audio, but a zeroed embedding keeps the pipeline functional for smoke
// testing.
// =====================================================================

bool CosyVoice3LMModel::PrepareVoice(CosyVoice3State &state,
                                     ggml_backend_sched_t sched) {
  (void)sched;
  // PushReferenceText/PushReferenceAudio already populated the state.
  if (!state.embedding.empty()) {
    if (state.prompt_text_token_ids.empty() &&
        !default_prompt_text_ids_.empty()) {
      // User supplied --ref audio but no --ref-text and the GGUF has a baked
      // default transcript: fall back to it so the LM still gets the
      // expected `[sys_prefix + eop + ref_transcript]` prefix.
      state.prompt_text_token_ids = default_prompt_text_ids_;
    }
    return true;
  }

  if (!default_embedding_.empty()) {
    state.embedding   = default_embedding_;
    state.prompt_token = default_prompt_token_;
    state.prompt_feat  = default_prompt_feat_;
    if (state.prompt_text_token_ids.empty()) {
      state.prompt_text_token_ids = default_prompt_text_ids_;
    }
    return true;
  }
  // No voice anywhere — fall back to a zero embedding so the pipeline still
  // runs end-to-end. Audio quality will be poor; this is a smoke-test mode.
  RS_LOG_WARN("CosyVoice3: no reference or default voice — synthesis will "
              "use a zero-embedding fallback");
  state.embedding.assign(192, 0.f);
  state.prompt_token.clear();
  state.prompt_feat.clear();
  return true;
}

// =====================================================================
// Decode = RunLM → RunFlow → RunHiFT
// =====================================================================

bool CosyVoice3LMModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  auto &s = static_cast<CosyVoice3State &>(state);
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
  };
  // Voice prep MUST run before RunLM — CosyVoice3 was trained always with a
  // voice prior, so the LM expects `[sos][sys_prefix + eop + ref_text +
  // tts_text][task][prompt_speech_token]` as input. With no prompt speech
  // tokens the LM goes OOD and produces babble (looping 5-token cycles in
  // greedy mode). PrepareVoice fills state.{prompt_text_token_ids,
  // prompt_token, prompt_feat, embedding} from the baked default when the
  // user didn't supply --ref.
  auto t_prep0 = clk::now();
  if (flow_ready_ && hift_ready_) {
    if (!PrepareVoice(s, sched)) return false;
    // Bake the resolved voice tuple once so later runs can reuse it via
    // RS_CV3_VOICE_PATH (skipping the tokenizer + CAMPPlus forwards).
    if (const char *p = std::getenv("RS_CV3_SAVE_VOICE_PATH")) {
      if (!voice_saved_ && SaveVoiceFile(s, p)) voice_saved_ = true;
    }
  }
  auto t_lm0 = clk::now();
  if (!RunLM(s, sched)) return false;
  auto t_lm1 = clk::now();
  // Debug knob: override the LM-sampled tokens with a binary dump of int32
  // ids. Lets us localize Flow/HiFT bugs by feeding PT-generated tokens
  // into the C++ post-LLM pipeline.
  if (const char *p = std::getenv("RS_CV3_LOAD_SPEECH_TOKENS_BIN")) {
    FILE *f = std::fopen(p, "rb");
    if (!f) {
      RS_LOG_ERR("CosyVoice3: cannot open RS_CV3_LOAD_SPEECH_TOKENS_BIN=%s", p);
      return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    const int n = (int)(sz / (long)sizeof(int32_t));
    s.speech_token_ids.assign((size_t)n, 0);
    std::fread(s.speech_token_ids.data(), sizeof(int32_t), (size_t)n, f);
    std::fclose(f);
    RS_LOG_INFO("CosyVoice3: overrode %d speech tokens from %s (debug)", n, p);
  }
  {
    std::string head; char buf[16];
    for (size_t i = 0; i < std::min<size_t>(30, s.speech_token_ids.size()); ++i) {
      snprintf(buf, sizeof(buf), "%d,", s.speech_token_ids[i]);
      head += buf;
    }
    RS_LOG_INFO("CosyVoice3-LLM: first30 speech tokens=[%s]", head.c_str());
  }
  if (!flow_ready_ || !hift_ready_) {
    // Legacy `cosyvoice3-llm` path: token-only output via GetTranscription.
    return true;
  }
  auto t_flow0 = clk::now();
  if (!flow_->RunFlow(s, sched)) {
    RS_LOG_ERR("CosyVoice3: RunFlow failed");
    return false;
  }
  auto t_flow1 = clk::now();
  // Debug knob: override Flow's mel output with a binary dump
  // (header: int32 T, int32 mel_dim; payload: f32 row-major [T, mel_dim]).
  // Lets us test HiFT in isolation with a known-good mel.
  if (const char *p = std::getenv("RS_CV3_LOAD_FLOW_MEL")) {
    FILE *f = std::fopen(p, "rb");
    if (!f) {
      RS_LOG_ERR("CosyVoice3: cannot open RS_CV3_LOAD_FLOW_MEL=%s", p);
      return false;
    }
    int32_t hdr[2] = {0, 0};
    std::fread(hdr, sizeof(int32_t), 2, f);
    s.mel_output.assign((size_t)hdr[0] * (size_t)hdr[1], 0.f);
    std::fread(s.mel_output.data(), sizeof(float), s.mel_output.size(), f);
    std::fclose(f);
    RS_LOG_INFO("CosyVoice3: overrode Flow mel [%d, %d] from %s (debug)",
                hdr[0], hdr[1], p);
  }
  if (!hift_->RunHiFT(s, sched)) {
    RS_LOG_ERR("CosyVoice3: RunHiFT failed");
    return false;
  }
  auto t_hift1 = clk::now();
  RS_LOG_INFO("CosyVoice3 timings: PrepareVoice=%.1fms LM=%.1fms (%zu tok) "
              "Flow=%.1fms HiFT=%.1fms total=%.1fms",
              ms(t_prep0, t_lm0), ms(t_lm0, t_lm1),
              s.speech_token_ids.size(),
              ms(t_flow0, t_flow1), ms(t_flow1, t_hift1),
              ms(t_prep0, t_hift1));
  return true;
}

// =====================================================================
// GetAudioOutput — returns a pointer to state.audio_output (zero-copy).
// Audio length in samples = return value.
// =====================================================================

int CosyVoice3LMModel::GetAudioOutput(RSState &state, float **out_data) {
  auto &s = static_cast<CosyVoice3State &>(state);
  if (!out_data || s.audio_output.empty()) {
    if (out_data) *out_data = nullptr;
    return 0;
  }
  *out_data = s.audio_output.data();
  return (int)s.audio_output.size();
}

// =====================================================================
// DecodeStream — RunLM (with per-token chunking) → RunFlowStreaming →
// RunHiFTStreaming → emit. Mirrors upstream `CosyVoice2Model.tts` with
// `token_hop_len=25` doubling to `token_max_hop_len=100` and the
// 3-token Flow `pre_lookahead_len`.
//
// Single-threaded: each chunk runs synchronously inside the LM token
// callback. The caller (RSProcessor) typically drives this from a
// background worker so `rs_process` can block on chunk arrival.
// =====================================================================

bool CosyVoice3LMModel::DecodeStream(RSState &state,
                                    ggml_backend_sched_t sched,
                                    AudioChunkCallback emit) {
  auto &s = static_cast<CosyVoice3State &>(state);
  if (!flow_ready_ || !hift_ready_) {
    RS_LOG_ERR("CosyVoice3: DecodeStream requires flow_ + hift_ loaded");
    return false;
  }
  if (!emit) {
    RS_LOG_ERR("CosyVoice3: DecodeStream needs a non-null emit callback");
    return false;
  }
  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
  };

  auto t_prep0 = clk::now();
  if (!PrepareVoice(s, sched)) return false;
  if (const char *p = std::getenv("RS_CV3_SAVE_VOICE_PATH")) {
    if (!voice_saved_ && SaveVoiceFile(s, p)) voice_saved_ = true;
  }

  // Stream tunables (match upstream CosyVoice2Model defaults).
  constexpr int kTokenHopInit = 25;
  constexpr int kTokenHopMax  = 100;
  constexpr int kStreamScale  = 2;
  // Non-finalize chunks run a reduced CFM trajectory: the tail of each
  // intermediate chunk is recomputed (with more context) on the next chunk,
  // so paying for the full 5 Euler steps there is wasted work. The final
  // (finalize=true) chunk keeps the default to preserve quality of the tail
  // that actually ships. Net: ~40 % Flow time off non-final chunks.
  constexpr int kFlowStepsNonFinal = 3;
  const int kPreLookahead = flow_->pre_lookahead_len();   // 3
  const int kTokenMelRatio = flow_->token_mel_ratio();    // 2
  const int kMelDim        = flow_->mel_dim();            // 80

  // Reset streaming state.
  s.stream_token_offset = 0;
  s.hift_stream = CosyVoice3State::HiFTStreamCache{};
  s.flow_done = false;
  s.hift_done = false;
  s.audio_output.clear();

  // Prompt-token pad: round prompt_speech_token length up to a multiple
  // of token_hop_len so the very first chunk's token grid aligns with
  // the eventual stable hop. Only consumed on the first chunk.
  int prompt_pad = 0;
  if (!s.prompt_token.empty()) {
    const int n_pt = (int)s.prompt_token.size();
    prompt_pad =
        ((n_pt + kTokenHopInit - 1) / kTokenHopInit) * kTokenHopInit - n_pt;
  }
  int cur_hop = kTokenHopInit;

  const int T_pt_mel = (int)(s.prompt_feat.size() / (size_t)kMelDim);

  // RNG isolation: in the offline path the sequence is
  //   RunLM (consumes state.rng for sampling)
  //   → Flow (consumes 1 state.rng then uses local)
  //   → HiFT (consumes state.rng heavily for NSF source noise)
  // In streaming the LM and the post-LM pipeline interleave per chunk, so
  // unless we keep their RNG streams separated the LM token output drifts
  // away from the offline reference (and from any fixed --seed). Snapshot
  // here and swap around every Flow+HiFT call.
  std::mt19937 pipeline_rng = s.rng;   // start from the same seed
  // Per-stage accumulators so we can attribute the streaming overhead.
  double total_flow_ms = 0.0, total_hift_ms = 0.0;
  int    chunks_emitted = 0;
  auto run_one_chunk = [&](int slice_end, bool finalize) -> bool {
    std::mt19937 lm_rng_saved = s.rng;
    s.rng = pipeline_rng;
    // RAII-ish restore: capture both flow/hift error paths.
    auto restore_rng = [&]() {
      pipeline_rng = s.rng;
      s.rng = lm_rng_saved;
    };
    auto t_flow0 = clk::now();
    // 1) Flow over speech_token_ids[0..slice_end]. Non-finalize keeps
    //    the prompt mel attached so we can slice by stream_token_offset.
    const int n_steps_override = finalize ? -1 : kFlowStepsNonFinal;
    if (!flow_->RunFlowStreaming(s, sched, slice_end, finalize,
                                 n_steps_override)) {
      RS_LOG_ERR("CosyVoice3-stream: RunFlowStreaming failed at slice=%d",
                 slice_end);
      restore_rng();
      return false;
    }
    auto t_flow1 = clk::now();
    total_flow_ms += ms(t_flow0, t_flow1);
    // 2) Locate the "new" mel slice for this chunk inside s.mel_output.
    //    Layouts:
    //      finalize=false: [T_pt_mel + slice_end*ratio, mel_dim]
    //      finalize=true : [(slice_end - 0)*ratio, mel_dim]   (prompt already trimmed)
    int mel_start, mel_end;
    if (finalize) {
      mel_start = s.stream_token_offset * kTokenMelRatio;
      mel_end   = slice_end * kTokenMelRatio;
    } else {
      mel_start = T_pt_mel + s.stream_token_offset * kTokenMelRatio;
      mel_end   = T_pt_mel + slice_end * kTokenMelRatio;
      // Upstream drops the lookahead-affected tail before HiFT so the
      // chunk is "stable" — those mel frames will be recomputed (with
      // more context) on the next chunk's Flow pass.
      mel_end  -= kPreLookahead * kTokenMelRatio;
    }
    if (mel_end <= mel_start) {
      // Nothing to emit yet (e.g. prompt_pad consumed everything).
      restore_rng();
      return true;
    }
    const int mel_chunk_T = mel_end - mel_start;
    const float *mel_ptr =
        s.mel_output.data() + (size_t)mel_start * kMelDim;

    // 3) HiFT (with cross-chunk hamming crossfade + source/mel caches).
    std::vector<float> emit_pcm;
    auto t_hift0 = clk::now();
    if (!hift_->RunHiFTStreaming(s, sched, mel_ptr, mel_chunk_T,
                                 finalize, emit_pcm)) {
      RS_LOG_ERR("CosyVoice3-stream: RunHiFTStreaming failed");
      restore_rng();
      return false;
    }
    auto t_hift1 = clk::now();
    total_hift_ms += ms(t_hift0, t_hift1);
    chunks_emitted++;
    if (!emit_pcm.empty()) {
      emit(emit_pcm.data(), emit_pcm.size());
    }
    restore_rng();
    return true;
  };

  // Per-token callback fires after every newly sampled speech token has
  // been appended to s.speech_token_ids. Returns false to abort the LM
  // (mapped to a synthetic EOS inside RunLM).
  std::atomic<bool> stream_ok{true};
  set_lm_token_callback([&](int32_t /*tok*/) -> bool {
    const int n = (int)s.speech_token_ids.size();
    // The first chunk also has to swallow `prompt_pad` aligned tokens.
    const int extra_pad =
        (s.stream_token_offset == 0) ? prompt_pad : 0;
    const int target =
        s.stream_token_offset + cur_hop + extra_pad + kPreLookahead;
    if (n >= target) {
      const int slice_end = s.stream_token_offset + cur_hop + extra_pad
                            + kPreLookahead;
      if (!run_one_chunk(slice_end, /*finalize=*/false)) {
        stream_ok.store(false);
        return false;
      }
      s.stream_token_offset += cur_hop + extra_pad;
      cur_hop = std::min(kTokenHopMax, cur_hop * kStreamScale);
    }
    return true;
  });

  auto t_lm0 = clk::now();
  bool lm_ok = RunLM(s, sched);
  set_lm_token_callback(nullptr);
  auto t_lm1 = clk::now();
  if (!lm_ok || !stream_ok.load()) {
    RS_LOG_ERR("CosyVoice3-stream: RunLM aborted");
    return false;
  }

  // Final drain: any tokens past stream_token_offset get emitted as the
  // last (finalize=true) chunk so cache is cleared and the hamming tail
  // isn't deferred.
  const int total = (int)s.speech_token_ids.size();
  if (total > s.stream_token_offset) {
    if (!run_one_chunk(total, /*finalize=*/true)) return false;
  } else if (s.hift_stream.primed) {
    // No new tokens but a cached tail still owes the user audio. Emit
    // the cached speech tail directly.
    emit(s.hift_stream.speech.data(), s.hift_stream.speech.size());
    s.hift_stream = CosyVoice3State::HiFTStreamCache{};
  }
  auto t_done = clk::now();
  RS_LOG_INFO("CosyVoice3-stream timings: PrepareVoice=%.1fms LM=%.1fms "
              "Flow=%.1fms HiFT=%.1fms chunks=%d (%zu tok) total=%.1fms",
              ms(t_prep0, t_lm0), ms(t_lm0, t_lm1),
              total_flow_ms, total_hift_ms, chunks_emitted,
              s.speech_token_ids.size(), ms(t_prep0, t_done));
  return true;
}

// =====================================================================
// GetTranscription — CSV of speech-token ids (LLM-only carrier).
// =====================================================================

std::string CosyVoice3LMModel::GetTranscription(RSState &state) {
  auto &s = static_cast<CosyVoice3State &>(state);
  std::ostringstream oss;
  for (size_t i = 0; i < s.speech_token_ids.size(); ++i) {
    if (i) oss << ',';
    oss << s.speech_token_ids[i];
  }
  return oss.str();
}

// =====================================================================
// Static registration — both arch ids point at this same class.
// =====================================================================

namespace {
struct CosyVoice3Registrar {
  CosyVoice3Registrar() {
    auto creator = []() {
      return std::make_shared<CosyVoice3LMModel>();
    };
    rs_register_model_arch("cosyvoice3-llm", creator);
    rs_register_model_arch("cosyvoice3", creator);
  }
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
g_cv3_registrar;
} // namespace
