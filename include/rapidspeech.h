#ifndef RAPIDSPEECH_H
#define RAPIDSPEECH_H

#include <stdint.h>
#include <stdbool.h>

// --- API Export Macro ---
#if defined(_WIN32)
#if defined(RAPIDSPEECH_BUILD)
#define RS_API __declspec(dllexport)
#else
#define RS_API __declspec(dllimport)
#endif
#else
#define RS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================
// Error Handling
// ============================================

// Error codes
typedef enum {
  RS_OK = 0,
  RS_ERR_INVALID_ARGS = -1,
  RS_ERR_INIT_FAILED = -2,
  RS_ERR_MODEL_LOAD_FAILED = -3,
  RS_ERR_INFERENCE_FAILED = -4,
  RS_ERR_OUT_OF_MEMORY = -5,
  RS_ERR_FILE_NOT_FOUND = -6,
  RS_ERR_UNSUPPORTED_FORMAT = -7,
  RS_ERR_BUFFER_FULL = -8,
  RS_ERR_NOT_IMPLEMENTED = -9
} rs_error_t;

// Error information (optional, for detailed error reporting)
typedef struct {
  rs_error_t code;
  char message[256];
} rs_error_info_t;

// Get last error info (thread-local)
RS_API rs_error_info_t rs_get_last_error(void);

// Clear last error
RS_API void rs_clear_error(void);

// ============================================
// Core Type Definitions
// ============================================

// Forward declaration for ggml types used by optional callbacks
struct ggml_cgraph;

// Context Handle (Opaque Pointer)
typedef struct rs_context_t rs_context_t;

// Task Types
typedef enum {
  RS_TASK_ASR_OFFLINE = 0,
  RS_TASK_ASR_ONLINE,
  RS_TASK_TTS_OFFLINE,
  RS_TASK_TTS_ONLINE,
  RS_TASK_E2E_SPEECH_LLM  // End-to-End Speech LLM
} rs_task_type_t;

// Model Metadata (read-only information about loaded model)
typedef struct {
  char arch_name[64];      // Model architecture name (e.g., "SenseVoiceSmall")
  int32_t audio_sample_rate;  // Expected sample rate (e.g., 16000)
  int32_t n_mels;          // Number of mel bins
  int32_t vocab_size;      // Vocabulary size
} rs_model_meta_t;

// Audio Processing Configuration
typedef struct {
  int32_t sample_rate;     // Audio sample rate (e.g., 16000)
  int32_t frame_size;      // Frame size in samples (e.g., 400 for 25ms)
  int32_t frame_step;      // Frame step in samples (e.g., 160 for 10ms)
  int32_t n_mels;          // Number of mel filterbanks
  bool use_lfr;            // Use Linear Frequency Reduction
  int32_t lfr_m;           // LFR stack frames
  int32_t lfr_n;           // LFR stride
  bool use_cmvn;           // Use CMVN normalization
} rs_audio_config_t;

// Default audio config generator
RS_API rs_audio_config_t rs_audio_config_default(void);

// Initialization Parameters
typedef struct {
  const char* model_path;   // GGUF model path
  int32_t n_threads;        // Number of CPU threads
  bool use_gpu;             // Whether to use GPU/NPU
  rs_task_type_t task_type; // Task type (ASR/TTS/LLM)
  rs_audio_config_t audio_config;  // Optional: override audio config from model
} rs_init_params_t;

// Default parameter generator
RS_API rs_init_params_t rs_default_params(void);

// --- Lifecycle Management ---

// Initialize context from file
// Returns: NULL on failure, use rs_get_last_error() for details
RS_API rs_context_t* rs_init_from_file(rs_init_params_t params);

// Free context and release all resources
RS_API void rs_free(rs_context_t* ctx);

// ============================================
// Model Information Query
// ============================================

// Get model metadata after successful initialization
// Returns: rs_model_meta_t struct (check arch_name[0] != '\0' for validity)
RS_API rs_model_meta_t rs_get_model_meta(const rs_context_t* ctx);

// Check if context is valid and ready for inference
RS_API bool rs_is_context_ready(const rs_context_t* ctx);

// ============================================
// Data Processing Interface (Unified Streaming Abstraction)
// ============================================

// Push audio data (ASR/E2E mode)
// pcm: 32-bit float audio data in range [-1.0, 1.0]
// n_samples: number of samples
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_push_audio(rs_context_t* ctx, const float* pcm, int32_t n_samples);

// Push text data (TTS/LLM mode)
// text: UTF-8 encoded null-terminated string
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_push_text(rs_context_t* ctx, const char* text);

// Push reference audio for voice cloning (TTS mode, optional)
// samples: 32-bit float reference audio data
// n_samples: number of samples
// sample_rate: reference audio sample rate
// Returns: 0 on success, -1 on error
RS_API int rs_push_reference_audio(rs_context_t* ctx, const float* samples,
                                   int32_t n_samples, int32_t sample_rate);

// Push reference text for voice cloning (TTS mode, optional)
// ref_text: UTF-8 encoded transcript of the reference audio
// Must be called before rs_process when using rs_push_reference_audio.
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_push_reference_text(rs_context_t* ctx, const char* ref_text);

// Execute single inference step
// Returns: 0=No output yet, 1=Has output, -1=Error (use rs_get_last_error)
RS_API int32_t rs_process(rs_context_t* ctx);

// ============================================
// Result Retrieval Interface
// ============================================

// Get generated audio (TTS mode)
// out_pcm: Pointer to internal buffer (do not free)
// Returns: number of samples available, 0 if none, -1 on error
RS_API int32_t rs_get_audio_output(rs_context_t* ctx, float** out_pcm);

// Get generated text (ASR mode)
// Returns: null-terminated UTF-8 string (do not free, valid until next rs_process)
RS_API const char* rs_get_text_output(rs_context_t* ctx);

// Check if there is pending output available
RS_API bool rs_has_output(const rs_context_t* ctx);

// ============================================
// Advanced Configuration (Optional)
// ============================================

// Set number of threads dynamically (for CPU backend)
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_set_n_threads(rs_context_t* ctx, int32_t n_threads);

// Override CMVN parameters (advanced users)
// means: array of n_mels mean values
// vars: array of n_mels variance values
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_set_cmvn_params(rs_context_t* ctx,
                                      const float* means,
                                      const float* vars,
                                      int32_t n_mels);

// Set audio chunk size for streaming processing
// chunk_size_ms: chunk duration in milliseconds (e.g., 1000 for 1 second)
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_set_chunk_size(rs_context_t* ctx, int32_t chunk_size_ms);

// Get current audio buffer size in milliseconds
// Returns: buffer duration in ms, -1 on error
RS_API int32_t rs_get_audio_buffer_duration_ms(const rs_context_t* ctx);

// Reset processing state (clear audio buffer, text accumulator, etc.)
// Returns: RS_OK on success, error code on failure
RS_API rs_error_t rs_reset(rs_context_t* ctx);

// ── TTS parameters ──

// Set TTS generation parameters (OmniVoice only).
// instruct: voice description, e.g. "male, young adult, moderate pitch"
// language: target language, e.g. "English" or "zh"
// seed: random seed for reproducible generation
// Returns: RS_OK on success
RS_API rs_error_t rs_set_tts_params(rs_context_t* ctx, const char* instruct,
                                     const char* language, int32_t seed);

// Set number of MaskGIT diffusion steps (OmniVoice TTS only, default 32).
// Fewer steps = faster but lower quality (min 1, max 128).
// Returns: RS_OK on success
RS_API rs_error_t rs_set_tts_diffusion_steps(rs_context_t* ctx, int32_t n_steps);

// ── 2-pass (CTC + LLM rescoring) support ──

// Set user input prompt for the LLM decoder (FunASRNano only).
// For example: "语音转写：" (default), "Transcribe speech to text:", etc.
// Returns: RS_OK on success
RS_API rs_error_t rs_set_user_input_prompt(rs_context_t* ctx, const char* prompt);

// Enable/disable LLM decoder at runtime (FunASRNano only, other models no-op).
// When disabled, Decode() uses CTC-greedy (fast, lower accuracy).
// When enabled,  Decode() uses the LLM (slower, higher accuracy).
// Typical 2-pass flow: push audio, process with LLM=off (first pass),
//   then rs_redecode with LLM=on (second pass).
// Returns: RS_OK on success
RS_API rs_error_t rs_set_use_llm(rs_context_t* ctx, bool use_llm);

// Enable/disable a CTC pre-check pass before LLM decoding (FunASRNano only).
// When enabled, a lightweight CTC decode runs first to detect silence —
// if no speech tokens are found the expensive LLM decode is skipped,
// preventing hallucinated output on noise/silence segments.
// Disabled by default.  Adds a small latency overhead (~ms) when enabled.
// Returns: RS_OK on success
RS_API rs_error_t rs_set_ctc_precheck(rs_context_t* ctx, bool enable);

// Re-run the decoder only (skip encoder) — used after rs_set_use_llm to
// rescore the same encoder output with different decoder settings.
// Caller must have completed at least one rs_process() first.
// Returns: 0=No output, 1=Has output, -1=Error
RS_API int32_t rs_redecode(rs_context_t* ctx);

// ============================================
// Voice Activity Detection (VAD)
// Independent handle (separate from rs_context_t). Auto-dispatches on the
// GGUF `general.architecture` field, currently {"silero-vad", "firered-vad"}.
// ============================================

typedef struct rs_vad_t rs_vad_t;

// Collapsed speech segment in seconds (relative to first pushed sample).
typedef struct {
    float start_s;
    float end_s;
} rs_vad_segment_t;

// Per-frame postprocessor output. Frame rate depends on the backend:
//   silero-vad : ~31.25 fps (512-sample window @ 16 kHz)
//   firered-vad: 100 fps (10 ms shift)
typedef struct {
    int32_t frame_idx;       // 1-based monotonic frame counter
    float   raw_prob;        // model output, [0..1]
    float   smoothed_prob;   // FireRed only; equals raw_prob on Silero
    int32_t is_speech;       // postprocessor decision (1=speech)
    int32_t is_speech_start; // 1 on the frame that begins a segment
    int32_t is_speech_end;   // 1 on the frame that ends a segment
} rs_vad_frame_t;

// Open a VAD model from a GGUF file. `use_gpu` is honored when a GPU backend
// is compiled in (e.g. WebGPU under WASM); CPU fallback is automatic.
RS_API rs_vad_t* rs_vad_init_from_file(const char* model_path,
                                       int32_t n_threads, bool use_gpu);

// Release a VAD model (state, weights, backend).
RS_API void rs_vad_free(rs_vad_t* vad);

// Reset all streaming state (caches, postprocessor, audio remainder,
// queued segments, queued frames). Sample counter is reset to 0.
RS_API rs_error_t rs_vad_reset(rs_vad_t* vad);

// Set the speech-probability threshold (0..1). Default per-model.
RS_API rs_error_t rs_vad_set_threshold(rs_vad_t* vad, float threshold);

// Push 16 kHz mono float PCM. Completed segments are appended to the
// internal segment queue; per-frame events to the frame queue.
RS_API rs_error_t rs_vad_push_audio(rs_vad_t* vad,
                                    const float* pcm, int32_t n_samples);

// Latest activity readouts (post-push).
RS_API int32_t     rs_vad_is_speech(const rs_vad_t* vad);     // 1 or 0
RS_API float       rs_vad_get_probability(const rs_vad_t* vad);
RS_API const char* rs_vad_get_arch(const rs_vad_t* vad);

// Drain queued speech segments. Returns number written to `out` (<= capacity).
// Drained entries are removed from the internal queue.
RS_API int32_t rs_vad_drain_segments(rs_vad_t* vad,
                                     rs_vad_segment_t* out, int32_t capacity);

// Drain queued per-frame events. Same semantics as drain_segments.
RS_API int32_t rs_vad_drain_frames(rs_vad_t* vad,
                                   rs_vad_frame_t* out, int32_t capacity);

// One-shot offline detection. Resets state, pushes the whole utterance,
// flushes any open segment, writes up to `capacity` segments into `out`.
// Returns the total number of segments detected (may exceed capacity).
RS_API int32_t rs_vad_detect_full(rs_vad_t* vad,
                                  const float* pcm, int32_t n_samples,
                                  rs_vad_segment_t* out, int32_t capacity);

// ============================================
// Utility Functions
// ============================================

// Get library version string
RS_API const char* rs_get_version(void);

// Get ggml backend name in use
RS_API const char* rs_get_backend_name(const rs_context_t* ctx);

//── Importance Matrix Collection (activation-aware quantization) ──

// Set a callback that is invoked after every ggml graph compute during TTS
// diffusion.  The callback receives userdata and the computed ggml_cgraph,
// which can be inspected to collect activation statistics for imatrix.
// Pass NULL for callback to unregister.
RS_API void rs_set_imatrix_callback(rs_context_t* ctx,
                                     void (*callback)(void* userdata,
                                                      struct ggml_cgraph* gf),
                                     void* userdata);

#ifdef __cplusplus
}
#endif

#endif // RAPIDSPEECH_H
