#pragma once

#include "core/rs_model.h"
#include "core/rs_processor.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "rapidspeech.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

/**
 * Internal context implementation.
 */
struct rs_context_t {
  rs_init_params_t params;
  std::vector<ggml_backend_buffer_t> weight_buffers;
  std::vector<ggml_backend_t> backends;
  ggml_backend_sched_t sched = nullptr;
  std::shared_ptr<ISpeechModel> model;
  std::unique_ptr<RSProcessor> processor;

  // GGUF related resources (required for Model::Load)
  gguf_context *ctx_gguf = nullptr;
  ggml_context *gguf_data = nullptr;

  // Second GPU backend + scheduler for multi-worker pipelining (CUDA only).
  // Shares the read-only weight buffers with `sched` but runs on an independent
  // CUDA stream so a second worker thread can compute Flow/HiFT concurrently
  // with LM decode. Metal is NOT supported (single command queue = no gain),
  // so create_extra_gpu_scheds(n) returns empty there and the caller falls
  // back to single-worker.
  std::vector<ggml_backend_t> extra_gpu_backends;
  std::vector<ggml_backend_t> extra_cpu_backends;
  std::vector<ggml_backend_sched_t> extra_scheds;

  rs_context_t();
  ~rs_context_t();

  bool init_backend(bool prefer_cpu = false);

  // Create a second GPU sched (independent CUDA backend + stream, shared
  // weights). Returns nullptr if the GPU backend is not CUDA (Metal/CPU) or
  // creation fails - caller should keep single-worker in that case.
  // Exported (rs-server links it directly; core lib is hidden-visibility).
#if defined(_WIN32)
  __declspec(dllexport)
#else
  __attribute__((visibility("default")))
#endif
  std::vector<ggml_backend_sched_t> create_extra_gpu_scheds(int n);
};

// --- Internal Framework Registry API ---
// These declarations ensure consistent signatures across the library

using ModelCreator = std::function<std::shared_ptr<ISpeechModel>()>;

/**
 * Registers a model architecture to the global registry.
 */
void rs_register_model_arch(const std::string &arch, ModelCreator creator);

/**
 * Internal entry point for context initialization.
 */
rs_context_t *rs_context_init_internal(rs_init_params_t params);

bool init_compute_ctx(struct ggml_context **ctx, struct ggml_cgraph **gf,
                      int n_nodes);