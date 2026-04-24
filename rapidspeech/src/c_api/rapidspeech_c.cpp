#include "core/rs_context.h"
#include "core/rs_model.h"
#include "core/rs_processor.h"
#include "rapidspeech.h"
#include "utils/rs_log.h"
#include <cstdarg>

#include <cstring>
#include <memory>
#include <string>

// ============================================
// Internal Thread-Local Error Storage
// ============================================

#if defined(_MSC_VER)
    #define THREAD_LOCAL __declspec(thread)
#else
    #define THREAD_LOCAL thread_local
#endif

static THREAD_LOCAL rs_error_info_t g_last_error = {RS_OK, ""};

static void set_error(rs_error_t code, const char *format, ...) {
  g_last_error.code = code;
  va_list args;
  va_start(args, format);
  vsnprintf(g_last_error.message, sizeof(g_last_error.message), format, args);
  va_end(args);
  RS_LOG_ERR("C-API Error: %s", g_last_error.message);
}

// ============================================
// Error Handling Implementation
// ============================================

RS_API rs_error_info_t rs_get_last_error(void) { return g_last_error; }

RS_API void rs_clear_error(void) {
  g_last_error.code = RS_OK;
  g_last_error.message[0] = '\0';
}

// ============================================
// Default Configuration Generators
// ============================================

RS_API rs_audio_config_t rs_audio_config_default(void) {
  rs_audio_config_t config;
  config.sample_rate = 16000;
  config.frame_size = 400; // 25ms @ 16k
  config.frame_step = 160; // 10ms @ 16k
  config.n_mels = 80;
  config.use_lfr = true;
  config.lfr_m = 7;
  config.lfr_n = 6;
  config.use_cmvn = true;
  return config;
}

RS_API rs_init_params_t rs_default_params(void) {
  rs_init_params_t params;
  params.model_path = nullptr;
  params.n_threads = 4;
  params.use_gpu = true;
  params.task_type = RS_TASK_ASR_OFFLINE;
  params.audio_config = rs_audio_config_default();
  return params;
}

// ============================================
// Lifecycle Management
// ============================================

RS_API rs_context_t *rs_init_from_file(rs_init_params_t params) {
  // Validate parameters
  if (!params.model_path) {
    set_error(RS_ERR_INVALID_ARGS, "model_path is NULL");
    return nullptr;
  }

  if (params.n_threads <= 0) {
    set_error(RS_ERR_INVALID_ARGS, "n_threads must be positive");
    return nullptr;
  }

  try {
    // Defined in rs_context.cpp
    extern rs_context_t *rs_context_init_internal(rs_init_params_t params);
    rs_context_t *ctx = rs_context_init_internal(params);

    if (!ctx) {
      set_error(RS_ERR_INIT_FAILED, "rs_context_init_internal returned NULL");
      return nullptr;
    }

    return ctx;

  } catch (const std::bad_alloc &e) {
    set_error(RS_ERR_OUT_OF_MEMORY, "Memory allocation failed: %s", e.what());
    return nullptr;
  } catch (const std::exception &e) {
    set_error(RS_ERR_INIT_FAILED, "Initialization failed: %s", e.what());
    return nullptr;
  } catch (...) {
    set_error(RS_ERR_INIT_FAILED, "Unknown exception during initialization");
    return nullptr;
  }
}

RS_API void rs_free(rs_context_t *ctx) {
  if (ctx) {
    delete ctx;
  }
}

// ============================================
// Model Information Query
// ============================================

RS_API rs_model_meta_t rs_get_model_meta(const rs_context_t *ctx) {
  rs_model_meta_t meta;
  std::memset(&meta, 0, sizeof(meta));

  if (!ctx || !ctx->model) {
    return meta; // Return zeroed struct
  }

  const auto &model_meta = ctx->model->GetMeta();

  // Safely copy architecture name
  std::strncpy(meta.arch_name, model_meta.arch_name.c_str(),
               sizeof(meta.arch_name) - 1);
  meta.arch_name[sizeof(meta.arch_name) - 1] = '\0';

  meta.audio_sample_rate = model_meta.audio_sample_rate;
  meta.n_mels = model_meta.n_mels;
  meta.vocab_size = model_meta.vocab_size;

  return meta;
}

RS_API bool rs_is_context_ready(const rs_context_t *ctx) {
  if (!ctx)
    return false;
  if (!ctx->processor)
    return false;
  if (!ctx->model)
    return false;
  if (!ctx->sched)
    return false;
  return true;
}

// ============================================
// Data Processing Interface
// ============================================

RS_API rs_error_t rs_push_audio(rs_context_t *ctx, const float *pcm,
                                int32_t n_samples) {
  if (!ctx) {
    set_error(RS_ERR_INVALID_ARGS, "Context is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  if (!ctx->processor) {
    set_error(RS_ERR_INVALID_ARGS, "Processor not initialized");
    return RS_ERR_INVALID_ARGS;
  }

  if (!pcm || n_samples <= 0) {
    set_error(RS_ERR_INVALID_ARGS, "Invalid audio data or n_samples <= 0");
    return RS_ERR_INVALID_ARGS;
  }

  try {
    ctx->processor->PushAudio(pcm, static_cast<size_t>(n_samples));
    return RS_OK;
  } catch (const std::exception &e) {
    set_error(RS_ERR_INFERENCE_FAILED, "PushAudio failed: %s", e.what());
    return RS_ERR_INFERENCE_FAILED;
  } catch (...) {
    set_error(RS_ERR_INFERENCE_FAILED, "PushAudio unknown error");
    return RS_ERR_INFERENCE_FAILED;
  }
}

RS_API rs_error_t rs_push_text(rs_context_t *ctx, const char *text) {
  if (!ctx) {
    set_error(RS_ERR_INVALID_ARGS, "Context is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  if (!text) {
    set_error(RS_ERR_INVALID_ARGS, "Text is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  // TODO: Implement TTS/LLM text push when TTS models are supported
  set_error(RS_ERR_NOT_IMPLEMENTED,
            "rs_push_text not implemented yet (TTS/LLM support pending)");
  return RS_ERR_NOT_IMPLEMENTED;
}

RS_API int32_t rs_process(rs_context_t *ctx) {
  if (!ctx || !ctx->processor) {
    set_error(RS_ERR_INVALID_ARGS, "Context or processor is NULL");
    return -1;
  }

  try {
    int result = ctx->processor->Process();

    if (result < 0) {
      set_error(RS_ERR_INFERENCE_FAILED, "Processing failed");
    }

    return result;
  } catch (const std::exception &e) {
    set_error(RS_ERR_INFERENCE_FAILED, "Process exception: %s", e.what());
    return -1;
  } catch (...) {
    set_error(RS_ERR_INFERENCE_FAILED, "Process unknown error");
    return -1;
  }
}

// ============================================
// Result Retrieval Interface
// ============================================

RS_API int32_t rs_get_audio_output(rs_context_t *ctx, float **out_pcm) {
  if (!ctx) {
    set_error(RS_ERR_INVALID_ARGS, "Context is NULL");
    return -1;
  }

  if (!out_pcm) {
    set_error(RS_ERR_INVALID_ARGS, "out_pcm pointer is NULL");
    return -1;
  }

  // TODO: Implement TTS audio output when TTS models are supported
  *out_pcm = nullptr;
  set_error(RS_ERR_NOT_IMPLEMENTED,
            "rs_get_audio_output not implemented yet (TTS support pending)");
  return -1;
}

RS_API const char *rs_get_text_output(rs_context_t *ctx) {
  // Static buffer to hold the result
  // Note: Using a simple static buffer for C API compatibility
  // For multi-threaded use, consider using thread-local storage with proper
  // cleanup
  static std::string temp_res;

  if (!ctx || !ctx->processor) {
    set_error(RS_ERR_INVALID_ARGS, "Context or processor is NULL");
    temp_res.clear();
    return temp_res.c_str();
  }

  try {
    temp_res = ctx->processor->GetTextResult();
    return temp_res.c_str();
  } catch (const std::exception &e) {
    set_error(RS_ERR_INFERENCE_FAILED, "GetTextResult failed: %s", e.what());
    temp_res.clear();
    return temp_res.c_str();
  } catch (...) {
    set_error(RS_ERR_INFERENCE_FAILED, "GetTextResult unknown error");
    temp_res.clear();
    return temp_res.c_str();
  }
}

RS_API bool rs_has_output(const rs_context_t *ctx) {
  if (!ctx || !ctx->processor) {
    return false;
  }
  // For ASR, check if there's accumulated text
  // This is a simplified check - could be enhanced based on task type
  const char *text = rs_get_text_output(const_cast<rs_context_t *>(ctx));
  return (text && std::strlen(text) > 0);
}

// ============================================
// Advanced Configuration
// ============================================

RS_API rs_error_t rs_set_n_threads(rs_context_t *ctx, int32_t n_threads) {
  if (!ctx) {
    set_error(RS_ERR_INVALID_ARGS, "Context is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  if (n_threads <= 0) {
    set_error(RS_ERR_INVALID_ARGS, "n_threads must be positive");
    return RS_ERR_INVALID_ARGS;
  }

  // Note: This would require exposing thread setting in rs_context_t
  // For now, return not implemented
  set_error(RS_ERR_NOT_IMPLEMENTED, "rs_set_n_threads not implemented yet");
  return RS_ERR_NOT_IMPLEMENTED;
}

RS_API rs_error_t rs_set_cmvn_params(rs_context_t *ctx, const float *means,
                                     const float *vars, int32_t n_mels) {
  if (!ctx || !ctx->processor) {
    set_error(RS_ERR_INVALID_ARGS, "Context or processor is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  if (!means || !vars) {
    set_error(RS_ERR_INVALID_ARGS, "means or vars is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  if (n_mels <= 0) {
    set_error(RS_ERR_INVALID_ARGS, "n_mels must be positive");
    return RS_ERR_INVALID_ARGS;
  }

  try {
    std::vector<float> means_vec(means, means + n_mels);
    std::vector<float> vars_vec(vars, vars + n_mels);
    ctx->processor->SetCMVN(means_vec, vars_vec);
    return RS_OK;
  } catch (const std::exception &e) {
    set_error(RS_ERR_INFERENCE_FAILED, "SetCMVN failed: %s", e.what());
    return RS_ERR_INFERENCE_FAILED;
  } catch (...) {
    set_error(RS_ERR_INFERENCE_FAILED, "SetCMVN unknown error");
    return RS_ERR_INFERENCE_FAILED;
  }
}

RS_API rs_error_t rs_set_chunk_size(rs_context_t *ctx, int32_t chunk_size_ms) {
  if (!ctx || !ctx->processor) {
    set_error(RS_ERR_INVALID_ARGS, "Context or processor is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  if (chunk_size_ms <= 0) {
    set_error(RS_ERR_INVALID_ARGS, "chunk_size_ms must be positive");
    return RS_ERR_INVALID_ARGS;
  }

  // TODO: This would require exposing chunk_size_samples_ in RSProcessor
  set_error(RS_ERR_NOT_IMPLEMENTED, "rs_set_chunk_size not implemented yet");
  return RS_ERR_NOT_IMPLEMENTED;
}

RS_API int32_t rs_get_audio_buffer_duration_ms(const rs_context_t *ctx) {
  if (!ctx || !ctx->processor) {
    return -1;
  }

  // Get buffer size in samples and convert to milliseconds
  size_t buffer_samples = ctx->processor->GetAudioBufferSize();
  int sample_rate = ctx->processor->GetAudioSampleRate();

  if (sample_rate <= 0) {
    return -1;
  }

  // duration_ms = (samples / sample_rate) * 1000
  int32_t duration_ms = static_cast<int32_t>(
      (static_cast<float>(buffer_samples) / sample_rate) * 1000.0f);
  return duration_ms;
}

RS_API rs_error_t rs_reset(rs_context_t *ctx) {
  if (!ctx || !ctx->processor) {
    set_error(RS_ERR_INVALID_ARGS, "Context or processor is NULL");
    return RS_ERR_INVALID_ARGS;
  }

  // TODO: Implement reset in RSProcessor
  set_error(RS_ERR_NOT_IMPLEMENTED, "rs_reset not implemented yet");
  return RS_ERR_NOT_IMPLEMENTED;
}

// ============================================
// Utility Functions
// ============================================

RS_API const char *rs_get_version(void) { return "0.1.0"; }

RS_API const char *rs_get_backend_name(const rs_context_t *ctx) {
  if (!ctx) {
    return "unknown";
  }

  // Return the primary backend name
  if (ctx->backends.empty()) {
    return "none";
  }

  // Get backend type from the first (primary) backend
  ggml_backend_t primary = ctx->backends[0];
  if (!primary) {
    return "none";
  }

  const char *backend_name = ggml_backend_name(primary);
  return backend_name ? backend_name : "unknown";
}
