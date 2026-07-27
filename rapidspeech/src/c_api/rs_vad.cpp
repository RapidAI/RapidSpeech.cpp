// VAD C-API — unified wrapper around SileroVadModel, FireRedVadModel and
// TenVadModel.
//
// Architecture is auto-detected from the GGUF `general.architecture` key:
//   "silero-vad"   -> SileroVadModel (32 ms windows, VADIterator hysteresis)
//   "firered-vad"  -> FireRedVadModel (10 ms shift, DFSMN + postprocessor)
//   "ten-vad"      -> TenVadModel (16 ms hop, mel+pitch frontend + LSTM)
//
// This file deliberately avoids the full rs_context_t pipeline. All VAD
// classes provide LoadDirect() entry points that work with bare gguf/ggml
// contexts and any backend handle, so we open the file once, pick a backend
// (with optional GPU), and dispatch.
//
// Streaming semantics:
//   rs_vad_push_audio() buffers samples, runs the model in its native chunk
//   size (Silero: 512 samples; TEN: 256; FireRed: arbitrary, model handles
//   remainder), and emits two queues:
//     - segments: closed [start_s, end_s] intervals
//     - frames:   per-window/per-frame postprocessor events
//   Both queues are drained by the matching rs_vad_drain_*() call.

#include "arch/silero_vad.h"
#include "arch/fireredvad.h"
#include "arch/tenvad.h"
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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

// Defined in rapidspeech_c.cpp.
extern "C" rs_error_info_t rs_get_last_error(void);
extern "C" void rs_clear_error(void);

// Thread-local error setter that mirrors the one in rapidspeech_c.cpp.
// We avoid linking against that symbol directly because it's static; instead
// we reuse `rs_get_last_error` and a local copy of the set helper.
namespace {

#if defined(_MSC_VER)
#define VAD_THREAD_LOCAL __declspec(thread)
#else
#define VAD_THREAD_LOCAL thread_local
#endif

VAD_THREAD_LOCAL rs_error_info_t g_vad_error = {RS_OK, ""};

void vad_set_error(rs_error_t code, const char* msg) {
    g_vad_error.code = code;
    std::snprintf(g_vad_error.message, sizeof(g_vad_error.message), "%s", msg);
    RS_LOG_ERR("VAD C-API: %s", msg);
}

constexpr int VAD_SAMPLE_RATE = 16000;
constexpr int SILERO_WINDOW   = 512;
constexpr int TENVAD_WINDOW   = 256;

} // namespace

// ============================================
// Internal VAD context
// ============================================

struct rs_vad_t {
    // GGUF resources
    gguf_context* ctx_gguf  = nullptr;
    ggml_context* gguf_data = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;

    // Compute backends
    std::vector<ggml_backend_t> backends;     // [GPU?, CPU]
    ggml_backend_sched_t        sched = nullptr;

    // Model + state
    std::string                       arch;
    std::shared_ptr<SileroVadModel>   silero;
    std::shared_ptr<FireRedVadModel>  firered;
    std::shared_ptr<TenVadModel>      tenvad;
    std::shared_ptr<RSState>          state;

    // Streaming buffering
    std::vector<float>  pending;     // windowed-model accumulator (512/256)
    int                 window_size  = SILERO_WINDOW;
    int64_t             samples_in   = 0;   // total samples ever pushed

    // Silero hysteresis tracking
    bool                in_speech    = false;
    int64_t             seg_start_sample = 0;
    int                 silero_frame_idx = 0;
    float               silero_exit_threshold = 0.35f;

    // Output queues
    std::deque<rs_vad_segment_t> seg_q;
    std::deque<rs_vad_frame_t>   frame_q;

    // Last-known activity
    float speech_prob = 0.0f;
    int   is_speech_flag = 0;

    ~rs_vad_t() {
        // Drop state/models first so any backend buffers they hold are freed
        // while the scheduler/backends are still alive.
        state.reset();
        silero.reset();
        firered.reset();
        tenvad.reset();

        if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
        if (sched)         ggml_backend_sched_free(sched);
        for (auto b : backends) ggml_backend_free(b);
        if (ctx_gguf)  gguf_free(ctx_gguf);
        if (gguf_data) ggml_free(gguf_data);
    }
};

// ============================================
// Backend init (mirrors rs_context_t::init_backend)
// ============================================

static bool vad_init_backends(rs_vad_t* v, int n_threads, bool use_gpu) {
    bool gpu_ok = false;
    if (use_gpu) {
#ifdef RS_USE_CUDA
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_cuda_init(0);
            if (b) { v->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("VAD: CUDA backend added."); }
        }
#endif
#ifdef RS_USE_METAL
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_metal_init();
            if (b) { v->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("VAD: Metal backend added."); }
        }
#endif
#ifdef RS_USE_VULKAN
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_vk_init(0);
            if (b) { v->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("VAD: Vulkan backend added."); }
        }
#endif
#ifdef RS_USE_WEBGPU
        if (!gpu_ok) {
            ggml_backend_t b = ggml_backend_webgpu_init();
            if (b) { v->backends.push_back(b); gpu_ok = true;
                     RS_LOG_INFO("VAD: WebGPU backend added."); }
        }
#endif
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) return false;
    ggml_backend_cpu_set_n_threads(cpu, n_threads > 0 ? n_threads : 4);
    v->backends.push_back(cpu);

    if (gpu_ok) {
        v->sched = ggml_backend_sched_new(v->backends.data(), nullptr,
                                          (int)v->backends.size(), 8192, false, true);
        RS_LOG_INFO("VAD: scheduler with GPU+CPU (%d backends).",
                    (int)v->backends.size());
    } else {
        int cpu_idx = (int)v->backends.size() - 1;
        v->sched = ggml_backend_sched_new(&v->backends[cpu_idx], nullptr, 1, 8192,
                                          false, false);
        RS_LOG_INFO("VAD: scheduler CPU-only.");
    }
    return v->sched != nullptr;
}

// Load all tensor blobs from disk into the backend buffer.
static bool vad_load_tensor_data(rs_vad_t* v, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    size_t data_offset = gguf_get_data_offset(v->ctx_gguf);
    int64_t n_tensors = gguf_get_n_tensors(v->ctx_gguf);
    std::vector<char> buf;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* name = gguf_get_tensor_name(v->ctx_gguf, i);
        ggml_tensor* t = ggml_get_tensor(v->gguf_data, name);
        if (!t) continue;
        size_t off  = gguf_get_tensor_offset(v->ctx_gguf, i);
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

RS_API rs_vad_t* rs_vad_init_from_file(const char* model_path,
                                       int32_t n_threads, bool use_gpu) {
    if (!model_path) {
        vad_set_error(RS_ERR_INVALID_ARGS, "model_path is null");
        return nullptr;
    }

    auto v = std::make_unique<rs_vad_t>();

    // 1. Open GGUF
    gguf_init_params gp = { /*no_alloc=*/true, &v->gguf_data };
    v->ctx_gguf = gguf_init_from_file(model_path, gp);
    if (!v->ctx_gguf) {
        vad_set_error(RS_ERR_MODEL_LOAD_FAILED, "gguf_init_from_file failed");
        return nullptr;
    }

    // 2. Detect arch
    int64_t arch_key = gguf_find_key(v->ctx_gguf, "general.architecture");
    if (arch_key < 0) {
        vad_set_error(RS_ERR_MODEL_LOAD_FAILED,
                      "GGUF missing general.architecture");
        return nullptr;
    }
    v->arch = gguf_get_val_str(v->ctx_gguf, arch_key);
    RS_LOG_INFO("VAD architecture: %s", v->arch.c_str());

    if (v->arch != "silero-vad" && v->arch != "firered-vad" &&
        v->arch != "ten-vad") {
        vad_set_error(RS_ERR_UNSUPPORTED_FORMAT,
                      "Unsupported VAD arch (expected silero-vad, firered-vad or ten-vad)");
        return nullptr;
    }

    // 3. Backends
    if (!vad_init_backends(v.get(), n_threads, use_gpu)) {
        vad_set_error(RS_ERR_INIT_FAILED, "backend init failed");
        return nullptr;
    }

    // 4. Allocate weights on the primary backend
    ggml_backend_t weight_backend = v->backends[0];
    v->weight_buffer = ggml_backend_alloc_ctx_tensors(v->gguf_data, weight_backend);
    if (!v->weight_buffer) {
        vad_set_error(RS_ERR_OUT_OF_MEMORY, "weight buffer alloc failed");
        return nullptr;
    }

    // 5. Read tensor blobs from disk
    if (!vad_load_tensor_data(v.get(), model_path)) {
        vad_set_error(RS_ERR_MODEL_LOAD_FAILED, "tensor data read failed");
        return nullptr;
    }

    // 6. Construct + LoadDirect the matching VAD model
    if (v->arch == "silero-vad") {
        v->silero = std::make_shared<SileroVadModel>();
        if (!v->silero->LoadDirect(v->gguf_data, v->ctx_gguf, weight_backend)) {
            vad_set_error(RS_ERR_MODEL_LOAD_FAILED, "SileroVadModel::LoadDirect failed");
            return nullptr;
        }
        v->state = v->silero->CreateState();
        v->window_size = SILERO_WINDOW;
        v->silero_exit_threshold = std::max(
            0.01f, v->silero->GetThreshold() - 0.15f);
    } else if (v->arch == "ten-vad") {
        v->tenvad = std::make_shared<TenVadModel>();
        if (!v->tenvad->LoadDirect(v->gguf_data, v->ctx_gguf, weight_backend)) {
            vad_set_error(RS_ERR_MODEL_LOAD_FAILED, "TenVadModel::LoadDirect failed");
            return nullptr;
        }
        v->state = v->tenvad->CreateState();
        if (!v->state) {
            vad_set_error(RS_ERR_INIT_FAILED, "TenVadModel::CreateState failed");
            return nullptr;
        }
        v->window_size = v->tenvad->GetHopSize() > 0 ? v->tenvad->GetHopSize()
                                                     : TENVAD_WINDOW;
        v->silero_exit_threshold = std::max(
            0.01f, v->tenvad->GetThreshold() - 0.15f);
    } else { // firered-vad
        v->firered = std::make_shared<FireRedVadModel>();
        if (!v->firered->LoadDirect(v->gguf_data, v->ctx_gguf, weight_backend)) {
            vad_set_error(RS_ERR_MODEL_LOAD_FAILED, "FireRedVadModel::LoadDirect failed");
            return nullptr;
        }
        v->state = v->firered->CreateState();
    }

    RS_LOG_INFO("VAD ready: %s (n_threads=%d, gpu=%d)",
                v->arch.c_str(), n_threads, (int)use_gpu);
    return v.release();
}

RS_API void rs_vad_free(rs_vad_t* vad) {
    delete vad;
}

// ============================================
// Configuration
// ============================================

RS_API rs_error_t rs_vad_reset(rs_vad_t* vad) {
    if (!vad) return RS_ERR_INVALID_ARGS;
    vad->pending.clear();
    vad->samples_in = 0;
    vad->in_speech = false;
    vad->seg_start_sample = 0;
    vad->silero_frame_idx = 0;
    vad->seg_q.clear();
    vad->frame_q.clear();
    vad->speech_prob = 0.0f;
    vad->is_speech_flag = 0;

    if (vad->silero) {
        // Recreate state — SileroVadState's ctor zeros h/c + context.
        vad->state = vad->silero->CreateState();
    } else if (vad->tenvad) {
        vad->tenvad->ResetState(*vad->state);
    } else if (vad->firered) {
        vad->firered->Reset(*vad->state);
    }
    return RS_OK;
}

RS_API rs_error_t rs_vad_set_threshold(rs_vad_t* vad, float threshold) {
    if (!vad) return RS_ERR_INVALID_ARGS;
    if (vad->silero) {
        vad->silero->SetThreshold(threshold);
        vad->silero_exit_threshold = std::max(0.01f, threshold - 0.15f);
    } else if (vad->tenvad) {
        vad->tenvad->SetThreshold(threshold);
        vad->silero_exit_threshold = std::max(0.01f, threshold - 0.15f);
    } else if (vad->firered) {
        vad->firered->SetThreshold(threshold);
    }
    return RS_OK;
}

RS_API const char* rs_vad_get_arch(const rs_vad_t* vad) {
    return vad ? vad->arch.c_str() : "";
}

RS_API int32_t rs_vad_is_speech(const rs_vad_t* vad) {
    return vad ? vad->is_speech_flag : 0;
}

RS_API float rs_vad_get_probability(const rs_vad_t* vad) {
    return vad ? vad->speech_prob : 0.0f;
}

// ============================================
// Push audio — windowed path (Silero 512 / TEN 256)
// ============================================

static void vad_process_windowed(rs_vad_t* v, const float* win, int n) {
    // n == v->window_size. Feed one window to the model's Encode().
    std::vector<float> input(win, win + n);
    ggml_backend_sched_reset(v->sched);
    float prob = 0.0f;
    float enter = 0.5f;
    if (v->silero) {
        if (!v->silero->Encode(input, *v->state, v->sched)) {
            RS_LOG_WARN("Silero VAD Encode failed");
            return;
        }
        prob  = v->silero->GetSpeechProbability(*v->state);
        enter = v->silero->GetThreshold();
    } else {
        if (!v->tenvad->Encode(input, *v->state, v->sched)) {
            RS_LOG_WARN("TEN VAD Encode failed");
            return;
        }
        prob  = v->tenvad->GetSpeechProbability(*v->state);
        enter = v->tenvad->GetThreshold();
    }
    v->speech_prob = prob;
    ++v->silero_frame_idx;

    float exit_  = v->silero_exit_threshold;
    bool was = v->in_speech;
    bool now = was ? (prob >= exit_) : (prob >= enter);

    // Window end sample (in absolute count):
    //   when this function runs we have already incremented samples_in to
    //   include `n` samples (caller does this before us).
    int64_t win_end = v->samples_in;
    int64_t win_start = win_end - n;

    rs_vad_frame_t fr = {};
    fr.frame_idx       = v->silero_frame_idx;
    fr.raw_prob        = prob;
    fr.smoothed_prob   = prob; // windowed models expose only one prob.
    fr.is_speech       = now ? 1 : 0;
    fr.is_speech_start = (!was && now) ? 1 : 0;
    fr.is_speech_end   = ( was && !now) ? 1 : 0;
    v->frame_q.push_back(fr);

    if (!was && now) {
        v->seg_start_sample = win_start;
    } else if (was && !now) {
        rs_vad_segment_t s;
        s.start_s = (float)v->seg_start_sample / VAD_SAMPLE_RATE;
        s.end_s   = (float)win_end           / VAD_SAMPLE_RATE;
        v->seg_q.push_back(s);
    }
    v->in_speech = now;
    v->is_speech_flag = now ? 1 : 0;
}

static rs_error_t vad_push_windowed(rs_vad_t* v, const float* pcm, int n) {
    const int window = v->window_size;
    v->pending.insert(v->pending.end(), pcm, pcm + n);
    while ((int)v->pending.size() >= window) {
        v->samples_in += window;
        vad_process_windowed(v, v->pending.data(), window);
        v->pending.erase(v->pending.begin(), v->pending.begin() + window);
    }
    return RS_OK;
}

// ============================================
// Push audio — FireRed path
// ============================================

static rs_error_t vad_push_firered(rs_vad_t* v, const float* pcm, int n) {
    std::vector<float> chunk(pcm, pcm + n);
    v->samples_in += n;
    auto results = v->firered->DetectStreamingChunk(chunk, *v->state);

    auto& fr_state = dynamic_cast<FireRedVadState&>(*v->state);
    v->speech_prob   = fr_state.speech_prob;
    v->is_speech_flag = fr_state.triggered ? 1 : 0;

    constexpr float FRAME_S = 0.01f; // 10 ms shift
    for (const auto& r : results) {
        rs_vad_frame_t fr = {};
        fr.frame_idx       = r.frame_idx;
        fr.raw_prob        = r.raw_prob;
        fr.smoothed_prob   = r.smoothed_prob;
        fr.is_speech       = r.is_speech ? 1 : 0;
        fr.is_speech_start = r.is_speech_start ? 1 : 0;
        fr.is_speech_end   = r.is_speech_end ? 1 : 0;
        v->frame_q.push_back(fr);

        if (r.is_speech_start) {
            v->seg_start_sample = (int64_t)(r.speech_start_frame - 1)
                                  * (VAD_SAMPLE_RATE / 100);
            v->in_speech = true;
        }
        if (r.is_speech_end && r.speech_start_frame > 0 && r.speech_end_frame > 0) {
            rs_vad_segment_t s;
            s.start_s = (float)(r.speech_start_frame - 1) * FRAME_S;
            s.end_s   = (float)r.speech_end_frame * FRAME_S;
            v->seg_q.push_back(s);
            v->in_speech = false;
        }
    }
    return RS_OK;
}

RS_API rs_error_t rs_vad_push_audio(rs_vad_t* vad, const float* pcm,
                                    int32_t n_samples) {
    if (!vad || !pcm || n_samples <= 0) return RS_ERR_INVALID_ARGS;
    if (vad->silero || vad->tenvad) return vad_push_windowed(vad, pcm, n_samples);
    if (vad->firered) return vad_push_firered(vad, pcm, n_samples);
    return RS_ERR_INIT_FAILED;
}

// ============================================
// Drains
// ============================================

RS_API int32_t rs_vad_drain_segments(rs_vad_t* vad,
                                     rs_vad_segment_t* out, int32_t capacity) {
    if (!vad || !out || capacity <= 0) return 0;
    int32_t n = (int32_t)std::min<size_t>(vad->seg_q.size(), (size_t)capacity);
    for (int32_t i = 0; i < n; ++i) {
        out[i] = vad->seg_q.front();
        vad->seg_q.pop_front();
    }
    return n;
}

RS_API int32_t rs_vad_drain_frames(rs_vad_t* vad,
                                   rs_vad_frame_t* out, int32_t capacity) {
    if (!vad || !out || capacity <= 0) return 0;
    int32_t n = (int32_t)std::min<size_t>(vad->frame_q.size(), (size_t)capacity);
    for (int32_t i = 0; i < n; ++i) {
        out[i] = vad->frame_q.front();
        vad->frame_q.pop_front();
    }
    return n;
}

// ============================================
// One-shot offline
// ============================================

RS_API int32_t rs_vad_detect_full(rs_vad_t* vad,
                                  const float* pcm, int32_t n_samples,
                                  rs_vad_segment_t* out, int32_t capacity) {
    if (!vad || !pcm || n_samples <= 0) return 0;

    rs_vad_reset(vad);
    rs_vad_push_audio(vad, pcm, n_samples);

    // Flush any in-progress segment using the latest sample timestamp.
    if (vad->in_speech) {
        rs_vad_segment_t s;
        s.start_s = (float)vad->seg_start_sample / VAD_SAMPLE_RATE;
        s.end_s   = (float)vad->samples_in / VAD_SAMPLE_RATE;
        vad->seg_q.push_back(s);
        vad->in_speech = false;
    }

    int32_t total = (int32_t)vad->seg_q.size();
    if (out && capacity > 0) {
        int32_t copy_n = std::min(total, capacity);
        for (int32_t i = 0; i < copy_n; ++i) {
            out[i] = vad->seg_q.front();
            vad->seg_q.pop_front();
        }
    }
    return total;
}
