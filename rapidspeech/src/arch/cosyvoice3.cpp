#include "cosyvoice3.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llm_kv_cache.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

// =====================================================================
// Construction / metadata
// =====================================================================

CosyVoice3LMModel::CosyVoice3LMModel() {
  meta_.arch_name = "cosyvoice3-llm";
  // Information-only: the LLM head emits speech tokens at 25 Hz; the actual
  // 24 kHz waveform is produced downstream by flow + HiFT. We expose the
  // target sample rate so callers can pre-size buffers.
  meta_.audio_sample_rate = 24000;
  meta_.n_mels = 0;
  meta_.use_external_frontend = true;
}

// =====================================================================
// Load
// =====================================================================

bool CosyVoice3LMModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                             ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("CosyVoice3-LLM: invalid context");
    return false;
  }
  backend_ = backend;

  llm_model_ = std::make_shared<llm_model>();
  if (!llm_model_->load_metadata_from_gguf(ctx->ctx_gguf)) {
    RS_LOG_ERR("CosyVoice3-LLM: load_metadata_from_gguf failed");
    return false;
  }

  // Collect tensors from the merged gguf_data (already allocated on a backend
  // by rs_context_init_internal) into a name → tensor map.
  std::map<std::string, ggml_tensor *> tensors;
  const int n = gguf_get_n_tensors(ctx->ctx_gguf);
  for (int i = 0; i < n; ++i) {
    const char *name = gguf_get_tensor_name(ctx->ctx_gguf, i);
    ggml_tensor *t = ggml_get_tensor(ctx->gguf_data, name);
    if (t) tensors[name] = t;
  }

  if (!llm_model_->map_tensors_qwen2(tensors)) {
    RS_LOG_ERR("CosyVoice3-LLM: map_tensors_qwen2 failed");
    return false;
  }

  // Pull speech-vocab size out of the tensor itself (cosyvoice3.speech_embd
  // shape is [d_model, speech_vocab]). Fall back to upstream defaults if the
  // tensor is somehow missing — but map_tensors_qwen2 already errored above.
  if (llm_model_->speech_embd()) {
    speech_vocab_ = (int32_t)llm_model_->speech_embd()->ne[1];
  }
  if (llm_model_->speech_lm_head()) {
    const int32_t v_head = (int32_t)llm_model_->speech_lm_head()->ne[1];
    if (v_head != speech_vocab_) {
      RS_LOG_WARN("CosyVoice3-LLM: speech_embd vocab=%d, speech_lm_head=%d "
                  "(using head)", speech_vocab_, v_head);
      speech_vocab_ = v_head;
    }
  }
  // Codebook size from GGUF; fall back to speech_vocab - 1 if missing.
  {
    int key = gguf_find_key(ctx->ctx_gguf, "cosyvoice3.llm.speech_token_codebook");
    if (key >= 0) {
      speech_codebook_ = (int32_t)gguf_get_val_u32(ctx->ctx_gguf, key);
    } else {
      speech_codebook_ = speech_vocab_ - 1;
    }
  }
  stop_token_id_ = speech_codebook_;

  meta_.vocab_size = (int)llm_model_->hparams().n_vocab;

  RS_LOG_INFO("CosyVoice3-LLM loaded: layers=%u d_model=%u heads=%u/%u "
              "ff=%u text_vocab=%u speech_vocab=%d stop=%d",
              llm_model_->hparams().n_layer, llm_model_->hparams().n_embd,
              llm_model_->hparams().n_head, llm_model_->hparams().n_head_kv,
              llm_model_->hparams().n_ff, llm_model_->hparams().n_vocab,
              speech_vocab_, stop_token_id_);
  return true;
}

// =====================================================================
// State factory
// =====================================================================

std::shared_ptr<RSState> CosyVoice3LMModel::CreateState() {
  auto s = std::make_shared<CosyVoice3State>();
  s->sampler = default_sampler_;
  s->max_speech_tokens = max_tokens_;
  s->rng.seed(seed_);
  return s;
}

// =====================================================================
// PushText: BPE tokenize via llm_vocab (gpt2 byte-level merges from GGUF)
// =====================================================================

bool CosyVoice3LMModel::PushText(RSState &state, const char *text,
                                 const char *language, const char *instruct) {
  (void)language; (void)instruct;
  auto &s = static_cast<CosyVoice3State &>(state);
  if (!llm_model_) return false;
  if (!text || !*text) {
    s.text_token_ids.clear();
    return true;
  }
  // CosyVoice3 LLM does NOT use BOS — the upstream collator just feeds raw
  // BPE ids. add_bos=false matches reference behaviour.
  s.text_token_ids = llm_model_->vocab().tokenize(std::string(text), false);
  RS_LOG_INFO("CosyVoice3-LLM: tokenized %zu text ids from '%s'",
              s.text_token_ids.size(), text);
  return !s.text_token_ids.empty();
}

// =====================================================================
// GetTranscription: CSV of speech-token ids (Phase 2 carrier)
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
// Static registration
// =====================================================================

namespace {
struct CosyVoice3Registrar {
  CosyVoice3Registrar() {
    rs_register_model_arch("cosyvoice3-llm", []() {
      return std::make_shared<CosyVoice3LMModel>();
    });
  }
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
g_cv3_registrar;
} // namespace
