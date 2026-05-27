/**
 * RapidSpeech WASM entry point — C API wrappers for Emscripten.
 *
 * Exports a minimal C-callable surface for both ASR and TTS.
 *
 * Model data is loaded through Emscripten's virtual filesystem:
 *   1. JS fetches .gguf from a URL
 *   2. JS writes it to /model.gguf via FS.writeFile()
 *   3. JS calls _rs_wasm_init("/model.gguf", task_type, n_threads)
 *
 * Task types (mirror rs_task_type_t):
 *   0 = ASR_OFFLINE   (default — matches the old single-arg init)
 *   1 = ASR_ONLINE
 *   2 = TTS_OFFLINE
 *   3 = TTS_ONLINE
 *   4 = E2E_SPEECH_LLM
 */

#include "rapidspeech.h"

#include <emscripten.h>
#include <cstdlib>
#include <cstring>

// Forward-declare explicit arch registrations so LTO cannot strip them.
void rs_register_sensevoice();

#ifdef __cplusplus
extern "C" {
#endif

// ── Globals (single-context for browser demos) ──────────────
static rs_context_t *g_ctx = nullptr;
static char g_last_text[4096] = {0};

// Buffer for TTS streaming PCM — owned by C, lifetime tied to g_ctx.
// Each successful TTS chunk is copied here so JS can read it via
// rs_wasm_get_audio_ptr / rs_wasm_get_audio_len without managing a
// separate heap allocation per chunk.
static float *g_audio_buf = nullptr;
static int    g_audio_cap = 0;
static int    g_audio_len = 0;

static void audio_buf_free(void) {
  std::free(g_audio_buf);
  g_audio_buf = nullptr;
  g_audio_cap = 0;
  g_audio_len = 0;
}

static int audio_buf_set(const float *src, int n) {
  if (n <= 0) {
    g_audio_len = 0;
    return 0;
  }
  if (n > g_audio_cap) {
    float *p = (float *)std::realloc(g_audio_buf, sizeof(float) * (size_t)n);
    if (!p) return -1;
    g_audio_buf = p;
    g_audio_cap = n;
  }
  std::memcpy(g_audio_buf, src, sizeof(float) * (size_t)n);
  g_audio_len = n;
  return 0;
}

// ── Init / Free ─────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_init_ex(const char *model_path, int task_type, int n_threads) {
  static bool archs_registered = false;
  if (!archs_registered) {
    rs_register_sensevoice();
    archs_registered = true;
  }
  if (g_ctx) {
    rs_free(g_ctx);
    g_ctx = nullptr;
  }
  audio_buf_free();

  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads  = n_threads;
  // Ask for GPU; rs_context falls back to CPU if no backend is compiled in.
  // For WASM the GPU backend is WebGPU (when RS_WASM_WEBGPU=ON at build time).
  params.use_gpu    = true;
  params.task_type  = (rs_task_type_t)task_type;

  g_ctx = rs_init_from_file(params);
  if (!g_ctx) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({
      console.error("rs_init_from_file failed: " + UTF8ToString($0), $1);
    }, err.message, err.code);
    return -1;
  }
  return 0;
}

// Backwards-compatible ASR init (defaults to RS_TASK_ASR_OFFLINE).
EMSCRIPTEN_KEEPALIVE
int rs_wasm_init(const char *model_path, int n_threads) {
  return rs_wasm_init_ex(model_path, (int)RS_TASK_ASR_OFFLINE, n_threads);
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_free(void) {
  if (g_ctx) {
    rs_free(g_ctx);
    g_ctx = nullptr;
  }
  audio_buf_free();
  g_last_text[0] = '\0';
  rs_clear_error();
}

// ── Audio / text input ──────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_audio(const float *pcm, int n_samples) {
  if (!g_ctx) return -1;
  return (int)rs_push_audio(g_ctx, pcm, n_samples);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_text(const char *text) {
  if (!g_ctx || !text) return -1;
  return (int)rs_push_text(g_ctx, text);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_reference_audio(const float *pcm, int n_samples, int sample_rate) {
  if (!g_ctx) return -1;
  return rs_push_reference_audio(g_ctx, pcm, n_samples, sample_rate);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_push_reference_text(const char *text) {
  if (!g_ctx || !text) return -1;
  return (int)rs_push_reference_text(g_ctx, text);
}

// ── Inference ───────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_process(void) {
  if (!g_ctx) return -1;
  int ret = rs_process(g_ctx);

  // ASR side-effect: capture latest text
  if (ret > 0) {
    const char *text = rs_get_text_output(g_ctx);
    if (text) {
      int i = 0;
      while (text[i] && i < 4095) { g_last_text[i] = text[i]; i++; }
      g_last_text[i] = '\0';
    }
  }

  // TTS side-effect: capture next PCM chunk if any
  float *chunk = nullptr;
  int n = rs_get_audio_output(g_ctx, &chunk);
  if (n > 0 && chunk) {
    audio_buf_set(chunk, n);
  } else {
    g_audio_len = 0;
  }

  return ret;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_text(void) {
  return g_last_text;
}

// Pointer to the most recent TTS PCM chunk (float32, host-endian).
// Valid until the next rs_wasm_process() / rs_wasm_reset() / rs_wasm_free().
EMSCRIPTEN_KEEPALIVE
const float *rs_wasm_get_audio_ptr(void) {
  return g_audio_buf;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_get_audio_len(void) {
  return g_audio_len;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_reset(void) {
  if (!g_ctx) return -1;
  g_last_text[0] = '\0';
  g_audio_len = 0;
  return (int)rs_reset(g_ctx);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_redecode(void) {
  if (!g_ctx) return -1;
  int ret = rs_redecode(g_ctx);
  if (ret > 0) {
    const char *text = rs_get_text_output(g_ctx);
    if (text) {
      int i = 0;
      while (text[i] && i < 4095) { g_last_text[i] = text[i]; i++; }
      g_last_text[i] = '\0';
    }
  }
  return ret;
}

// ── ASR knobs (FunASRNano) ──────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_user_input_prompt(const char *prompt) {
  if (!g_ctx || !prompt) return -1;
  return (int)rs_set_user_input_prompt(g_ctx, prompt);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_use_llm(int enable) {
  if (!g_ctx) return -1;
  return (int)rs_set_use_llm(g_ctx, enable != 0);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_ctc_precheck(int enable) {
  if (!g_ctx) return -1;
  return (int)rs_set_ctc_precheck(g_ctx, enable != 0);
}

// ── TTS knobs (OmniVoice) ───────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_tts_params(const char *instruct, const char *language, int seed) {
  if (!g_ctx) return -1;
  return (int)rs_set_tts_params(g_ctx, instruct, language, seed);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_set_tts_diffusion_steps(int n_steps) {
  if (!g_ctx) return -1;
  return (int)rs_set_tts_diffusion_steps(g_ctx, n_steps);
}

// ── Metadata ────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
int rs_wasm_get_sample_rate(void) {
  if (!g_ctx) return 16000;
  rs_model_meta_t meta = rs_get_model_meta(g_ctx);
  return meta.audio_sample_rate;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_arch_name(void) {
  if (!g_ctx) return "unknown";
  static char name[64];
  rs_model_meta_t meta = rs_get_model_meta(g_ctx);
  int i = 0;
  while (meta.arch_name[i] && i < 63) { name[i] = meta.arch_name[i]; i++; }
  name[i] = '\0';
  return name;
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_is_ready(void) {
  return g_ctx ? 1 : 0;
}

// ── VAD (independent of g_ctx — supports both silero-vad and firered-vad) ──
//
// Single-instance per WASM module; each tab creates its own MODULARIZE'd
// module so two simultaneous VADs can coexist across tabs.

static rs_vad_t *g_vad = nullptr;

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_init(const char *model_path, int n_threads) {
  if (g_vad) { rs_vad_free(g_vad); g_vad = nullptr; }
  // Pin VAD to CPU. Both silero-vad and firered-vad are tiny streaming
  // models invoked from a high-frequency AudioWorklet callback — WebGPU's
  // per-submit overhead dominates and the queue eventually fails with
  // "Queue work failed with status 3". CPU is both faster and async-safe
  // (no ASYNCIFY suspension inside vad.pushAudio).
  g_vad = rs_vad_init_from_file(model_path, n_threads, false);
  if (!g_vad) {
    rs_error_info_t err = rs_get_last_error();
    EM_ASM({
      console.error("rs_vad_init_from_file failed: " + UTF8ToString($0), $1);
    }, err.message, err.code);
    return -1;
  }
  return 0;
}

EMSCRIPTEN_KEEPALIVE
void rs_wasm_vad_free(void) {
  if (g_vad) { rs_vad_free(g_vad); g_vad = nullptr; }
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_reset(void) {
  if (!g_vad) return -1;
  return (int)rs_vad_reset(g_vad);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_set_threshold(float threshold) {
  if (!g_vad) return -1;
  return (int)rs_vad_set_threshold(g_vad, threshold);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_push_audio(const float *pcm, int n_samples) {
  if (!g_vad) return -1;
  return (int)rs_vad_push_audio(g_vad, pcm, n_samples);
}

EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_is_speech(void) {
  return g_vad ? rs_vad_is_speech(g_vad) : 0;
}

EMSCRIPTEN_KEEPALIVE
float rs_wasm_vad_get_probability(void) {
  return g_vad ? rs_vad_get_probability(g_vad) : 0.0f;
}

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_vad_get_arch(void) {
  return g_vad ? rs_vad_get_arch(g_vad) : "";
}

// out: caller-allocated buffer of `capacity` rs_vad_segment_t (8 bytes each).
// Returns number of segments written.
EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_drain_segments(rs_vad_segment_t *out, int capacity) {
  if (!g_vad) return 0;
  return (int)rs_vad_drain_segments(g_vad, out, capacity);
}

// out: caller-allocated buffer of `capacity` rs_vad_frame_t (24 bytes each).
EMSCRIPTEN_KEEPALIVE
int rs_wasm_vad_drain_frames(rs_vad_frame_t *out, int capacity) {
  if (!g_vad) return 0;
  return (int)rs_vad_drain_frames(g_vad, out, capacity);
}

// ── Utility ─────────────────────────────────────────────────

EMSCRIPTEN_KEEPALIVE
const char *rs_wasm_get_version(void) {
  return rs_get_version();
}

#ifdef __cplusplus
}
#endif
