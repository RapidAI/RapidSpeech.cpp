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

// ============================================
// Utility Functions
// ============================================

// Get library version string
RS_API const char* rs_get_version(void);

// Get ggml backend name in use
RS_API const char* rs_get_backend_name(const rs_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // RAPIDSPEECH_H