#include "core/rs_context.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-opencl.h"
#include "ggml-vulkan.h"
#include "ggml-cann.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "utils/rs_log.h"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

// --- Factory Infrastructure (Auto-Registration) ---

using ModelCreator = std::function<std::shared_ptr<ISpeechModel>()>;
static std::unordered_map<std::string, ModelCreator>& get_model_registry() {
  static std::unordered_map<std::string, ModelCreator> registry;
  return registry;
}

void rs_register_model_arch(const std::string& arch, ModelCreator creator) {
  get_model_registry()[arch] = creator;
}

// --- rs_context_t Implementation ---

rs_context_t::rs_context_t() : sched(nullptr) {
  // Member 'backends' is default initialized as an empty vector
}

rs_context_t::~rs_context_t() {
  RS_LOG_INFO("Releasing RapidSpeech context resources...");

  if (sched) {
    ggml_backend_sched_free(sched);
  }

  for (auto backend : backends) {
    ggml_backend_free(backend);
  }
}

bool rs_context_t::init_backend() {
  // 1. Try to initialize high-performance backends first

#ifdef RS_USE_CUDA
  if (params.use_gpu) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (cuda_backend) {
      RS_LOG_INFO("CUDA backend added to scheduler.");
      backends.push_back(cuda_backend);
    } else {
      RS_LOG_WARN("CUDA requested but initialization failed.");
    }
  }
#endif

#ifdef RS_USE_METAL
  if (params.use_gpu) {
    ggml_backend_t metal_backend = ggml_backend_metal_init();
    if (metal_backend) {
      RS_LOG_INFO("Metal backend added to scheduler.");
      backends.push_back(metal_backend);
    } else {
      RS_LOG_WARN("Metal requested but initialization failed.");
    }
  }
#endif

#ifdef RS_USE_VULKAN
  if (params.use_gpu) {
    ggml_backend_t vulkan_backend = ggml_backend_vk_init(0);
    if (vulkan_backend) {
      RS_LOG_INFO("Vulkan backend added to scheduler.");
      backends.push_back(vulkan_backend);
    } else {
      RS_LOG_WARN("Vulkan requested but initialization failed.");
    }
  }
#endif

#ifdef RS_USE_CANN
  if (params.use_gpu) {
    ggml_backend_t cann_backend = ggml_backend_cann_init(0);
    if (cann_backend) {
      RS_LOG_INFO("CANN backend added to scheduler.");
      backends.push_back(cann_backend);
    } else {
      RS_LOG_WARN("CANN requested but initialization failed.");
    }
  }
#endif

#ifdef RS_USE_OPENCL
  if (params.use_gpu) {
    ggml_backend_t opencl_backend = ggml_backend_opencl_init();
    if (opencl_backend) {
      RS_LOG_INFO("OpenCL backend added to scheduler.");
      backends.push_back(opencl_backend);
    } else {
      RS_LOG_WARN("OpenCL requested but initialization failed.");
    }
  }
#endif

  // 2. Always add CPU backend as a fallback
  ggml_backend_t cpu_backend = ggml_backend_cpu_init();
  if (cpu_backend) {
    backends.push_back(cpu_backend);
    RS_LOG_INFO("CPU backend added to scheduler.");
  } else {
    RS_LOG_ERR("Failed to initialize CPU backend.");
    return false;
  }

  // 3. Initialize the scheduler
  sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), SENSE_VOICE_MAX_GRAPH_SIZE, false, false);
  if (!sched) {
    RS_LOG_ERR("Failed to create ggml_backend_sched.");
    return false;
  }

  return true;
}

/**
 * Internal C++ implementation of context creation.
 * This function handles the heavy lifting and is called by the C-API wrapper.
 */
rs_context_t* rs_context_init_internal(rs_init_params_t params) {
  if (!params.model_path) {
    RS_LOG_ERR("Model path is NULL.");
    return nullptr;
  }

  auto ctx = std::make_unique<rs_context_t>();
  ctx->params = params;

  // 1. Initialize Backends and Scheduler
  if (!ctx->init_backend()) {
    return nullptr;
  }

  // 2. Open GGUF
  struct gguf_init_params gguf_params = {
      /* .no_alloc = */ true,
      /* .ctx      = */ &ctx->gguf_data,
  };
  ctx->ctx_gguf = gguf_init_from_file(params.model_path, gguf_params);
  if (!ctx->ctx_gguf) {
    RS_LOG_ERR("Failed to load GGUF file: %s", params.model_path);
    return nullptr;
  }


  RS_LOG_INFO("%s: gguf version: %d", __func__, gguf_get_version(ctx->ctx_gguf));
  RS_LOG_INFO("%s: gguf alignment: %zu", __func__, gguf_get_alignment(ctx->ctx_gguf));
  RS_LOG_INFO("%s: gguf data offset: %zu", __func__, gguf_get_data_offset(ctx->ctx_gguf));

  // 3. Resolve Architecture
  int key_id = gguf_find_key(ctx->ctx_gguf, "general.architecture");
  if (key_id == -1) {
    RS_LOG_ERR("GGUF file missing 'general.architecture' key.");
    gguf_free(ctx->ctx_gguf);
    return nullptr;
  }
  std::string arch = gguf_get_val_str(ctx->ctx_gguf, key_id);
  RS_LOG_INFO("Architecture detected: %s", arch.c_str());

  // 4. Create Model instance via Registry
  auto& registry = get_model_registry();
  auto it = registry.find(arch);
  if (it == registry.end()) {
    RS_LOG_ERR("Unsupported architecture: %s", arch.c_str());
    gguf_free(ctx->ctx_gguf);
    return nullptr;
  }

  ctx->model = it->second();
  if (!ctx->model) {
    RS_LOG_ERR("Model creator for '%s' returned NULL.", arch.c_str());
    gguf_free(ctx->ctx_gguf);
    return nullptr;
  }

  // 5. Load Weights
  if (ctx->model->Load(ctx, ctx->backends[0])) {

    gguf_free(ctx->ctx_gguf);

    // 6. Initialize Processor
    ctx->processor = std::make_unique<RSProcessor>(ctx->model, ctx->sched);

    RS_LOG_INFO("RapidSpeech context core initialized successfully.");
    return ctx.release();
  } else {
    RS_LOG_ERR("Failed to load model weights.");
    gguf_free(ctx->ctx_gguf);
    return nullptr;
  }
}