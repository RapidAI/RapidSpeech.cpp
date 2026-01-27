#pragma once

#include "rapidspeech.h"
#include "core/rs_model.h"
#include "core/rs_processor.h"
#include "ggml-backend.h"
#include <memory>
#include <vector>

/**
 * Internal context structure definition.
 * This is the actual implementation of the opaque rs_context_t handle.
 * It manages the lifecycle of hardware backends, schedulers, model instances, and processors.
 */
struct rs_context_t {
  // Initialization parameters passed from the user
  rs_init_params_t params;

  // List of available GGML backends (e.g., [GPU, CPU])
  std::vector<ggml_backend_t> backends;

  // GGML backend scheduler for efficient tensor distribution and graph execution
  ggml_backend_sched_t sched = nullptr;

  // Model instance (holds weights and graph architecture)
  std::shared_ptr<ISpeechModel> model;

  // gguf ctx (holds heads from gguf file)
  gguf_context * ctx_gguf;
  // gguf data (holds weights from gguf file)
  ggml_context * gguf_data;


  // Main processor for audio/text orchestration and streaming logic
  std::unique_ptr<RSProcessor> processor;

  /**
     * Constructor and Destructor
   */
  rs_context_t();
  ~rs_context_t();

  /**
     * Initializes hardware backends and the scheduler based on the provided params.
     * This method will attempt to initialize high-performance backends (CUDA/Metal)
     * and always provide a CPU fallback via the scheduler.
     * @return true if successful, false otherwise.
   */
  bool init_backend();
};

/**
 * Internal factory to create models based on GGUF architecture metadata.
 */
class SpeechModelFactory {
public:
  /**
     * Factory method to create a concrete speech model.
     * @param model_path Path to the GGUF model file.
     * @param params Initialization parameters.
     * @return A shared pointer to the created model.
   */
  static std::shared_ptr<ISpeechModel> create_model(const char* model_path, const rs_init_params_t& params);
};