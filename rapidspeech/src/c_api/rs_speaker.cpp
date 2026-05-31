// Speaker-embedding C-API — dedicated handle around CAMPPlusModel.
//
// Mirrors `rs_vad.cpp` lifecycle: open the GGUF, pick the best available
// backend (GPU first, CPU fallback), allocate weights, hand off to
// `CAMPPlusModel::LoadDirect()`. The GGUF must declare
// `general.architecture = "campplus"` — otherwise we refuse to load to
// guard against accidentally feeding in an ASR model.

#include "arch/campplus.h"
#include "rapidspeech.h"
#include "utils/rs_log.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#ifdef RS_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef RS_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef RS_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef RS_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

#if defined(_MSC_VER)
#define SPK_THREAD_LOCAL __declspec(thread)
#else
#define SPK_THREAD_LOCAL thread_local
#endif

SPK_THREAD_LOCAL rs_error_info_t g_spk_error = {RS_OK, ""};

void spk_set_error(rs_error_t code, const char* msg) {
    g_spk_error.code = code;
    std::snprintf(g_spk_error.message, sizeof(g_spk_error.message), "%s", msg);
    RS_LOG_ERR("Speaker C-API: %s", msg);
}

} // namespace

// ============================================
// Internal context
// ============================================

struct rs_speaker_t {
    // GGUF resources
    gguf_context* ctx_gguf  = nullptr;
    ggml_context* gguf_data = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    // Compute backends ([GPU?, CPU])
    std::vector<ggml_backend_t> backends;
    ggml_backend_sched_t        sched = nullptr;

    // Model + state
    std::shared_ptr<CAMPPlusModel> model;
    std::shared_ptr<RSState>       state;

    ~rs_speaker_t() {
        // Drop state/model first so any backend buffers they hold are freed
        // while the scheduler/backends are still alive.
        state.reset();
        model.reset();

        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (sched)         ggml_backend_sched_free(sched);
        for (auto b : backends) ggml_backend_free(b);
        if (ctx_gguf)  gguf_free(ctx_gguf);
        if (gguf_data) ggml_free(gguf_data);
    }
};

// ============================================
// Backend init (mirrors vad_init_backends)
// ============================================

static bool spk_init_backends(rs_speaker_t* s, int n_threads, bool use_gpu) {
    bool gpu_ok = false;
    if (use_gpu) {
#ifdef RS_USE_CUDA
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_cuda_init(0);
            if (b) { s->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("Speaker: CUDA backend added."); }
        }
#endif
#ifdef RS_USE_METAL
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_metal_init();
            if (b) { s->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("Speaker: Metal backend added."); }
        }
#endif
#ifdef RS_USE_VULKAN
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_vk_init(0);
            if (b) { s->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("Speaker: Vulkan backend added."); }
        }
#endif
#ifdef RS_USE_WEBGPU
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_webgpu_init();
            if (b) { s->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("Speaker: WebGPU backend added."); }
        }
#endif
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) return false;
    ggml_backend_cpu_set_n_threads(cpu, n_threads > 0 ? n_threads : 4);
    s->backends.push_back(cpu);

    if (gpu_ok) {
        s->sched = ggml_backend_sched_new(s->backends.data(), nullptr,
                                          (int)s->backends.size(), 8192,
                                          false, true);
        RS_LOG_INFO("Speaker: scheduler with GPU+CPU (%d backends).",
                    (int)s->backends.size());
    } else {
        int cpu_idx = (int)s->backends.size() - 1;
        s->sched = ggml_backend_sched_new(&s->backends[cpu_idx], nullptr, 1,
                                          8192, false, false);
        RS_LOG_INFO("Speaker: scheduler CPU-only.");
    }
    return s->sched != nullptr;
}

// Load all tensor blobs from disk into the backend buffer.
static bool spk_load_tensor_data(rs_speaker_t* s, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    size_t data_offset = gguf_get_data_offset(s->ctx_gguf);
    int64_t n_tensors = gguf_get_n_tensors(s->ctx_gguf);
    std::vector<char> buf;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(s->ctx_gguf, i);
        ggml_tensor* t = ggml_get_tensor(s->gguf_data, name);
        if (!t) continue;
        size_t off  = gguf_get_tensor_offset(s->ctx_gguf, i);
        size_t size = ggml_nbytes(t);
        if (size == 0) continue;
        if (buf.size() < size) buf.resize(size);
        std::fseek(f, (long)(data_offset + off), SEEK_SET);
        if (std::fread(buf.data(), 1, size, f) != size) {
            std::fclose(f);
            return false;
        }
        ggml_backend_tensor_set(t, buf.data(), 0, size);
    }
    std::fclose(f);
    return true;
}

// ============================================
// Lifecycle
// ============================================

RS_API rs_speaker_t* rs_speaker_init_from_file(const char* model_path,
                                               int32_t n_threads, bool use_gpu) {
    if (!model_path) {
        spk_set_error(RS_ERR_INVALID_ARGS, "model_path is null");
        return nullptr;
    }

    auto s = std::make_unique<rs_speaker_t>();

    // 1. Open GGUF
    gguf_init_params gp = { /*no_alloc=*/true, &s->gguf_data };
    s->ctx_gguf = gguf_init_from_file(model_path, gp);
    if (!s->ctx_gguf) {
        spk_set_error(RS_ERR_MODEL_LOAD_FAILED, "gguf_init_from_file failed");
        return nullptr;
    }

    // 2. Validate arch — refuse anything that isn't CAMPPlus.
    int64_t arch_key = gguf_find_key(s->ctx_gguf, "general.architecture");
    if (arch_key < 0) {
        spk_set_error(RS_ERR_MODEL_LOAD_FAILED,
                      "GGUF missing general.architecture");
        return nullptr;
    }
    std::string arch = gguf_get_val_str(s->ctx_gguf, arch_key);
    if (arch != "campplus") {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "Speaker arch must be 'campplus', got '%s'",
                      arch.c_str());
        spk_set_error(RS_ERR_UNSUPPORTED_FORMAT, buf);
        return nullptr;
    }
    RS_LOG_INFO("Speaker architecture: %s", arch.c_str());

    // 3. Backends
    if (!spk_init_backends(s.get(), n_threads, use_gpu)) {
        spk_set_error(RS_ERR_INIT_FAILED, "backend init failed");
        return nullptr;
    }

    // 4. Allocate weights on the primary backend
    ggml_backend_t weight_backend = s->backends[0];
    s->weight_buffer = ggml_backend_alloc_ctx_tensors(s->gguf_data,
                                                     weight_backend);
    if (!s->weight_buffer) {
        spk_set_error(RS_ERR_OUT_OF_MEMORY, "weight buffer alloc failed");
        return nullptr;
    }

    // 5. Read tensor blobs from disk
    if (!spk_load_tensor_data(s.get(), model_path)) {
        spk_set_error(RS_ERR_MODEL_LOAD_FAILED, "tensor data read failed");
        return nullptr;
    }

    // 6. Construct + LoadDirect the CAMPPlus model
    s->model = std::make_shared<CAMPPlusModel>();
    if (!s->model->LoadDirect(s->gguf_data, s->ctx_gguf, weight_backend)) {
        spk_set_error(RS_ERR_MODEL_LOAD_FAILED,
                      "CAMPPlusModel::LoadDirect failed");
        return nullptr;
    }
    s->state = s->model->CreateState();

    RS_LOG_INFO("Speaker ready: campplus dim=%d sr=%d (n_threads=%d, gpu=%d)",
                s->model->GetEmbedDim(), s->model->GetSampleRate(),
                n_threads, (int)use_gpu);
    return s.release();
}

RS_API void rs_speaker_free(rs_speaker_t* sp) {
    delete sp;
}

// ============================================
// Read-outs
// ============================================

RS_API int32_t rs_speaker_dim(const rs_speaker_t* sp) {
    if (!sp || !sp->model) return 0;
    return (int32_t)sp->model->GetEmbedDim();
}

RS_API int32_t rs_speaker_sample_rate(const rs_speaker_t* sp) {
    if (!sp || !sp->model) return 0;
    return (int32_t)sp->model->GetSampleRate();
}

// ============================================
// Embed
// ============================================

RS_API rs_error_t rs_speaker_embed(rs_speaker_t* sp,
                                   const float* pcm, int32_t n_samples,
                                   float* out_emb, int32_t out_capacity) {
    if (!sp || !pcm || n_samples <= 0 || !out_emb) {
        spk_set_error(RS_ERR_INVALID_ARGS, "invalid embed arguments");
        return RS_ERR_INVALID_ARGS;
    }
    const int dim = sp->model ? sp->model->GetEmbedDim() : 0;
    if (dim <= 0) {
        spk_set_error(RS_ERR_INIT_FAILED, "model not initialised");
        return RS_ERR_INIT_FAILED;
    }
    if (out_capacity < dim) {
        spk_set_error(RS_ERR_BUFFER_FULL, "out_capacity smaller than dim");
        return RS_ERR_BUFFER_FULL;
    }

    ggml_backend_sched_reset(sp->sched);
    if (!sp->model->Embed(pcm, (int)n_samples, *sp->state, sp->sched)) {
        spk_set_error(RS_ERR_INFERENCE_FAILED, "CAMPPlusModel::Embed failed");
        return RS_ERR_INFERENCE_FAILED;
    }

    auto& cs = dynamic_cast<CAMPPlusState&>(*sp->state);
    if ((int)cs.embedding.size() != dim) {
        spk_set_error(RS_ERR_INFERENCE_FAILED,
                      "embedding size mismatch");
        return RS_ERR_INFERENCE_FAILED;
    }
    std::memcpy(out_emb, cs.embedding.data(), (size_t)dim * sizeof(float));
    return RS_OK;
}

// ============================================
// Cosine similarity utility
// ============================================

RS_API float rs_speaker_cosine(const float* a, const float* b, int32_t dim) {
    if (!a || !b || dim <= 0) return 0.0f;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int32_t i = 0; i < dim; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}
