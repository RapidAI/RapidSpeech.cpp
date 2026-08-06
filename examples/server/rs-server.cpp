// rs-server.cpp — native C++ speech server for RapidSpeech.cpp.
//
// Exposes the ASR and TTS pipelines over two protocols from one binary:
//
//   • OpenAI-compatible HTTP API
//       POST /v1/audio/transcriptions   (multipart file → text)   [ASR]
//       POST /v1/audio/speech           (JSON text → WAV audio)   [TTS]
//       GET  /v1/models                 (list loaded models)
//       GET  /health
//
//   • Model Context Protocol (MCP), JSON-RPC 2.0
//       tools: transcribe (path → text), synthesize (text → WAV file)
//       transports: stdio (--mcp-stdio, for Claude Desktop et al.)
//                   and HTTP  (POST /mcp, Streamable-HTTP style)
//
// One process can serve ASR, TTS, or both. Each model context is guarded by a
// mutex, so concurrent HTTP requests are serialized per model (ggml contexts
// are single-threaded).
//
// Usage:
//   rs-server --asr-model xasr.gguf --tts-model omnivoice.gguf --port 8080
//   rs-server --asr-model xasr.gguf --mcp-stdio        # MCP over stdio only

#include "httplib.h"
#include "rs_json.h"
#include "rs_ws.h"

#include "core/rs_context.h"
#include "frontend/audio_processor.h"
#include "rapidspeech.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"
#include "../common/rs_cli_utf8.h"

// TTS request schedulers. CosyVoice3 uses token-interleaved continuous
// batching; VoxCPM2 uses independent request states and backend schedulers.
#include "arch/cosyvoice3.h"
#include "arch/cosyvoice3_batch.h"
#include "arch/voxcpm2.h"
#include "arch/xasr.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <cstdarg>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define RS_DUP _dup
#define RS_DUP2 _dup2
#define RS_FILENO _fileno
#else
#include <unistd.h>
#define RS_DUP dup
#define RS_DUP2 dup2
#define RS_FILENO fileno
#endif

#ifdef _WIN32
#define RS_POPEN _popen
#define RS_PCLOSE _pclose
#else
#define RS_POPEN popen
#define RS_PCLOSE pclose
#endif

namespace fs = std::filesystem;

static std::string find_registry_path_from_argv0(const char *argv0) {
    std::vector<fs::path> candidates;
    candidates.push_back(fs::path("rapidspeech") / "python" / "_registry.json");
    if (argv0 && argv0[0]) {
        fs::path exe = fs::absolute(fs::path(argv0));
        fs::path dir = exe.parent_path();
        candidates.push_back(dir / ".." / "python" / "_registry.json");
        candidates.push_back(dir / ".." / "rapidspeech" / "python" /
                             "_registry.json");
        candidates.push_back(dir / ".." / ".." / "rapidspeech" / "python" /
                             "_registry.json");
    }
    for (const auto &candidate : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(candidate, ec)) {
            fs::path canon = fs::weakly_canonical(candidate, ec);
            return (ec ? candidate : canon).string();
        }
    }
    return "";
}

static const char *kServerName = "rapidspeech-server";

// ── audio helpers ───────────────────────────────────────────────────────

// Encode float [-1,1] mono PCM as a 16-bit WAV into a byte string.
static std::string wav_encode(const std::vector<float> &pcm, int sample_rate) {
    const int channels = 1, bits = 16;
    const uint32_t data_size = (uint32_t)(pcm.size() * (bits / 8));
    const uint32_t byte_rate = (uint32_t)sample_rate * channels * (bits / 8);
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t chunk_size = 36 + data_size;

    std::string out;
    out.reserve(44 + data_size);
    auto put = [&](const void *d, size_t n) { out.append((const char *)d, n); };
    auto u32 = [&](uint32_t v) { put(&v, 4); };
    auto u16 = [&](uint16_t v) { put(&v, 2); };

    put("RIFF", 4); u32(chunk_size); put("WAVE", 4);
    put("fmt ", 4); u32(16); u16(1); u16(channels);
    u32((uint32_t)sample_rate); u32(byte_rate); u16(block_align); u16(bits);
    put("data", 4); u32(data_size);
    for (float s : pcm) {
        float c = s < -1.f ? -1.f : (s > 1.f ? 1.f : s);
        int16_t v = (int16_t)lrintf(c * 32767.0f);
        put(&v, 2);
    }
    return out;
}

// Decode audio bytes (WAV) → mono float PCM at target_sr. Uses the shared
// rs_wav loader via a temp file (it handles WAV parsing + resampling).
static bool decode_audio(const std::string &bytes, int target_sr,
                         std::vector<float> &pcm) {
    std::error_code ec;
    fs::path tmp = fs::temp_directory_path(ec) /
                   ("rs_server_" + std::to_string((uintptr_t)&bytes) + ".wav");
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f) return false;
        f.write(bytes.data(), (std::streamsize)bytes.size());
    }
    bool ok = load_wav_file_resampled(tmp.string().c_str(), pcm, target_sr, nullptr);
    fs::remove(tmp, ec);
    return ok && !pcm.empty();
}

// ── model engines (one rs_context_t each, mutex-guarded) ────────────────

struct Engine {
    rs_context_t *ctx = nullptr;
    std::mutex mu;
    std::string arch;
    int sample_rate = 16000;

    void unload() {
        if (ctx) {
            rs_free(ctx);
            ctx = nullptr;
        }
        arch.clear();
        sample_rate = 16000;
    }

    bool load(const std::string &path, int threads, bool gpu,
              rs_task_type_t task = RS_TASK_ASR_OFFLINE) {
        unload();
        rs_init_params_t p = rs_default_params();
        p.model_path = path.c_str();
        p.n_threads = threads;
        p.use_gpu = gpu;
        p.task_type = task;
        ctx = rs_init_from_file(p);
        if (!ctx) return false;
        rs_model_meta_t m = rs_get_model_meta(ctx);
        arch = m.arch_name;
        sample_rate = m.audio_sample_rate;
        return true;
    }
    ~Engine() { unload(); }
};

static Engine g_asr, g_tts;
static bool g_have_asr = false, g_have_tts = false;
static std::string g_asr_model_path;
static std::string g_current_asr_model_spec;
static std::string g_current_tts_model_spec;
static std::string g_current_vad_model_spec;
static std::string g_current_speaker_model_spec;
static std::string g_registry_path;
static std::string g_python_executable;
static std::mutex g_model_switch_mu;
static int g_model_threads = 4;
static bool g_model_gpu = true;
static int g_asr_reload_every = 0; // 0 = disabled
static int g_asr_final_segments_since_reload = 0;
static bool g_mem_log = false;
static int g_asr_slots = 4;
static int g_asr_max_queued = 256;
static int g_asr_batch_wait_ms = 3;
static int g_asr_min_batch = 0;
static int g_asr_max_batch_wait_ms = -1;
static int g_asr_frontend_threads = 1;
static int g_asr_frontend_workers = 0;
static bool g_ws_log = false;
static std::atomic<uint64_t> g_asr_cmvn_version{0};
static std::vector<float> g_asr_cmvn_means;
static std::vector<float> g_asr_cmvn_vars;

static rs_speaker_t *g_speaker = nullptr;
static std::mutex g_speaker_mu;
static bool g_have_speaker = false;
static std::string g_speaker_model_path;
static std::string g_speaker_cluster = "spectral";
static float g_speaker_threshold = 0.5f;
static int g_speaker_num_speakers = 0;
static int g_speaker_max_speakers = 10;
static std::string g_speaker_log_path;

// CosyVoice3 continuous-batching scheduler. VoxCPM2 has a separate queued
// scheduler below because its diffusion/AR pipeline is request-oriented.
static std::unique_ptr<TtsBatchScheduler> g_tts_batch;
class VoxCPM2ServerScheduler {
public:
    using Result = std::pair<std::vector<float>, std::string>;
    using ChunkCallback = VoxCPM2Model::AudioChunkCallback;

    VoxCPM2ServerScheduler(VoxCPM2Model *model,
                           std::vector<ggml_backend_sched_t> scheds,
                           int slots)
        : model_(model), scheds_(std::move(scheds)), slots_(slots) {}

    ~VoxCPM2ServerScheduler() { Stop(); }

    void Start() {
        for (auto sched : scheds_) {
            workers_.emplace_back([this, sched] { Run(sched); });
        }
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto &worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        workers_.clear();
    }

    std::future<Result> Submit(const std::string &text,
                               const std::string &voice,
                               const std::string &language, int32_t seed,
                               int steps = 8) {
        return Enqueue(text, voice, language, seed, steps, {});
    }

    std::future<Result> SubmitStream(const std::string &text,
                                     const std::string &voice,
                                     const std::string &language,
                                     int32_t seed, int steps,
                                     ChunkCallback callback) {
        return Enqueue(text, voice, language, seed, steps,
                       std::move(callback));
    }

    int n_slots() const { return slots_; }
    int n_workers() const { return static_cast<int>(scheds_.size()); }
    int n_active() const { return active_.load(); }
    int n_queued() const {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<int>(queue_.size());
    }

private:
    struct Job {
        std::string text;
        std::string voice;
        std::string language;
        int32_t seed = 42;
        int steps = 8;
        ChunkCallback callback;
        std::promise<Result> promise;
    };

    std::future<Result> Enqueue(const std::string &text,
                                const std::string &voice,
                                const std::string &language, int32_t seed,
                                int steps, ChunkCallback callback) {
        auto job = std::make_shared<Job>();
        job->text = text;
        job->voice = voice;
        job->language = language;
        job->seed = seed;
        job->steps = steps > 0 ? steps : 8;
        job->callback = std::move(callback);
        auto future = job->promise.get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stopping_) {
                job->promise.set_value({{}, "TTS scheduler is stopping"});
                return future;
            }
            queue_.push_back(job);
        }
        cv_.notify_one();
        return future;
    }

    void Run(ggml_backend_sched_t sched) {
        while (true) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_ && queue_.empty()) return;
                job = std::move(queue_.front());
                queue_.pop_front();
                active_.fetch_add(1);
            }

            Result result;
            try {
                auto state = std::dynamic_pointer_cast<VoxCPM2State>(
                    model_->CreateState());
                if (!state || !model_->PushText(
                        *state, job->text.c_str(), job->language.c_str(),
                        job->voice.c_str())) {
                    result.second = "VoxCPM2 request initialization failed";
                } else {
                    model_->ConfigureState(
                        *state, static_cast<uint64_t>(
                            static_cast<uint32_t>(job->seed)), job->steps);
                    ggml_backend_sched_reset(sched);
                    const bool ok = job->callback
                        ? model_->DecodeStream(*state, sched, job->callback)
                        : model_->Decode(*state, sched);
                    if (!ok) {
                        result.second = "VoxCPM2 inference failed";
                    } else {
                        result.first = std::move(state->audio_output);
                    }
                }
            } catch (const std::exception &e) {
                result.second = e.what();
            } catch (...) {
                result.second = "VoxCPM2 inference failed";
            }
            job->promise.set_value(std::move(result));
            active_.fetch_sub(1);
        }
    }

    VoxCPM2Model *model_ = nullptr;
    std::vector<ggml_backend_sched_t> scheds_;
    int slots_ = 1;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> queue_;
    std::vector<std::thread> workers_;
    std::atomic<int> active_{0};
    bool stopping_ = false;
};

static std::unique_ptr<VoxCPM2ServerScheduler> g_voxcpm2_batch;
static int g_tts_slots = 2;
static int g_tts_voc_workers = 1;
static std::string g_tts_stream_mode = "true";
static int g_tts_segment_max_chars = 80;

struct AsrServerConfig {
    bool two_pass = false;
    bool ctc_precheck = false;
    bool ctc_only = false;
};
static AsrServerConfig g_asr_cfg;

static bool same_asr_config(const AsrServerConfig &a,
                            const AsrServerConfig &b) {
    return a.two_pass == b.two_pass &&
           a.ctc_precheck == b.ctc_precheck &&
           a.ctc_only == b.ctc_only;
}

// VAD is a separate handle (rs_vad_t), used by /asr/vad and HTTP VAD mode.
static rs_vad_t *g_vad = nullptr;
static std::mutex g_vad_mu;
static bool g_have_vad = false;
static int g_vad_workers = 1;
static std::vector<rs_vad_t *> g_vad_ws_pool;
static std::vector<bool> g_vad_ws_busy;
static std::mutex g_vad_ws_pool_mu;

struct VadServerConfig {
    float threshold = 0.5f;
    int end_silence_ms = 600;
    int partial_ms = 1500;
    int min_segment_ms = 500;
    int max_segment_ms = 0; // 0 = disabled
};
static VadServerConfig g_vad_cfg;

static void ws_logf(const char *fmt, ...) {
    if (!g_ws_log) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

static void release_vad_ws_worker(int index) {
    std::lock_guard<std::mutex> lk(g_vad_ws_pool_mu);
    if (index >= 0 && index < (int)g_vad_ws_busy.size()) {
        g_vad_ws_busy[index] = false;
    }
}

struct VadWorkerLease {
    rs_vad_t *vad = nullptr;
    int index = -1;

    VadWorkerLease() = default;
    VadWorkerLease(rs_vad_t *v, int i) : vad(v), index(i) {}
    VadWorkerLease(const VadWorkerLease &) = delete;
    VadWorkerLease &operator=(const VadWorkerLease &) = delete;
    VadWorkerLease(VadWorkerLease &&other) noexcept {
        vad = other.vad;
        index = other.index;
        other.vad = nullptr;
        other.index = -1;
    }
    ~VadWorkerLease() {
        if (vad) release_vad_ws_worker(index);
    }
};

static VadWorkerLease acquire_vad_ws_worker() {
    std::lock_guard<std::mutex> lk(g_vad_ws_pool_mu);
    for (int i = 0; i < (int)g_vad_ws_pool.size(); ++i) {
        if (!g_vad_ws_busy[i]) {
            g_vad_ws_busy[i] = true;
            return VadWorkerLease(g_vad_ws_pool[i], i);
        }
    }
    return {};
}

struct AsrSegment {
    int index = 0;
    float start_s = 0.0f;
    float end_s = 0.0f;
    std::string text;
    int speaker = -1;
    std::vector<float> pcm;
    std::vector<float> speaker_embedding;
};

struct SpeakerServerConfig {
    bool enabled = false;
    bool log = false;
    std::string cluster = "spectral";
    float threshold = 0.5f;
    int num_speakers = 0;
    int max_speakers = 10;
};

static std::string speaker_name(int speaker) {
    if (speaker < 0) return "";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "SPEAKER_%02d", speaker);
    return buf;
}

static std::string time_s(float seconds) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", seconds);
    return buf;
}

static bool speaker_embed_pcm(const std::vector<float> &pcm,
                              std::vector<float> &embedding,
                              std::string &err) {
    if (!g_have_speaker || !g_speaker) {
        err = "no speaker model loaded (use --spk-model)";
        return false;
    }
    std::lock_guard<std::mutex> lk(g_speaker_mu);
    const int dim = rs_speaker_dim(g_speaker);
    if (dim <= 0) {
        err = "speaker model has invalid embedding dimension";
        return false;
    }
    embedding.assign((size_t)dim, 0.0f);
    rs_error_t rc = rs_speaker_embed(g_speaker, pcm.data(), (int)pcm.size(),
                                     embedding.data(), dim);
    if (rc != RS_OK) {
        rs_error_info_t e = rs_get_last_error();
        err = e.message[0] ? e.message : "speaker embedding failed";
        return false;
    }
    return true;
}

static std::vector<int> cluster_speaker_embeddings_ahc(
    const std::vector<std::vector<float>> &embs, int dim, float threshold,
    int num_speakers) {
    const int n = (int)embs.size();
    std::vector<int> labels(n, 0);
    if (n <= 1) return labels;

    std::vector<float> sim((size_t)n * n, 1.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float c = rs_speaker_cosine(embs[i].data(), embs[j].data(), dim);
            sim[(size_t)i * n + j] = c;
            sim[(size_t)j * n + i] = c;
        }
    }

    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; ++i) clusters[i] = {i};

    auto avg_link = [&](const std::vector<int> &a,
                        const std::vector<int> &b) {
        double s = 0.0;
        for (int i : a) {
            for (int j : b) s += sim[(size_t)i * n + j];
        }
        return (float)(s / ((double)a.size() * (double)b.size()));
    };

    while ((int)clusters.size() > 1) {
        float best = -2.0f;
        int bi = -1;
        int bj = -1;
        for (size_t i = 0; i < clusters.size(); ++i) {
            for (size_t j = i + 1; j < clusters.size(); ++j) {
                float s = avg_link(clusters[i], clusters[j]);
                if (s > best) {
                    best = s;
                    bi = (int)i;
                    bj = (int)j;
                }
            }
        }
        bool do_merge = num_speakers > 0
            ? ((int)clusters.size() > num_speakers)
            : (best >= threshold);
        if (!do_merge) break;
        clusters[bi].insert(clusters[bi].end(), clusters[bj].begin(),
                            clusters[bj].end());
        clusters.erase(clusters.begin() + bj);
    }

    std::sort(clusters.begin(), clusters.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) {
                  return *std::min_element(a.begin(), a.end()) <
                         *std::min_element(b.begin(), b.end());
              });
    for (size_t k = 0; k < clusters.size(); ++k) {
        for (int idx : clusters[k]) labels[idx] = (int)k;
    }
    return labels;
}

static void jacobi_eigh(std::vector<double> &A, int n,
                        std::vector<double> &w,
                        std::vector<double> &V) {
    V.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) V[(size_t)i * n + i] = 1.0;

    double frob2 = 0.0;
    for (double v : A) frob2 += v * v;
    const double tol = 1e-12 * std::max(1.0, frob2);

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
            }
        }
        if (off < tol) break;

        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[(size_t)p * n + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = A[(size_t)p * n + p];
                const double aqq = A[(size_t)q * n + q];
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                    (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                for (int k = 0; k < n; ++k) {
                    const double akp = A[(size_t)k * n + p];
                    const double akq = A[(size_t)k * n + q];
                    A[(size_t)k * n + p] = c * akp - s * akq;
                    A[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = A[(size_t)p * n + k];
                    const double aqk = A[(size_t)q * n + k];
                    A[(size_t)p * n + k] = c * apk - s * aqk;
                    A[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vkp = V[(size_t)k * n + p];
                    const double vkq = V[(size_t)k * n + q];
                    V[(size_t)k * n + p] = c * vkp - s * vkq;
                    V[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    w.resize(n);
    for (int i = 0; i < n; ++i) w[i] = A[(size_t)i * n + i];

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return w[a] < w[b];
    });

    std::vector<double> w_sorted(n);
    std::vector<double> V_sorted((size_t)n * n);
    for (int k = 0; k < n; ++k) {
        w_sorted[k] = w[order[k]];
        for (int i = 0; i < n; ++i) {
            V_sorted[(size_t)i * n + k] = V[(size_t)i * n + order[k]];
        }
    }
    w = std::move(w_sorted);
    V = std::move(V_sorted);
}

static std::vector<int> kmeans_cluster(const std::vector<double> &X, int n,
                                       int d, int k) {
    std::vector<int> best_labels(n, 0);
    if (k <= 1 || n <= 1) return best_labels;
    k = std::min(k, n);

    std::mt19937 rng(42);
    const int n_init = 10;
    const int max_iter = 100;
    double best_inertia = std::numeric_limits<double>::max();

    auto dist2 = [&](const double *a, const double *b) {
        double s = 0.0;
        for (int j = 0; j < d; ++j) {
            const double diff = a[j] - b[j];
            s += diff * diff;
        }
        return s;
    };

    std::vector<double> centers((size_t)k * d);
    std::vector<double> min_d2(n);
    std::vector<int> labels(n);
    std::vector<int> counts(k);

    for (int init = 0; init < n_init; ++init) {
        std::uniform_int_distribution<int> uni(0, n - 1);
        int first = uni(rng);
        std::copy(X.begin() + (size_t)first * d,
                  X.begin() + (size_t)(first + 1) * d,
                  centers.begin());
        for (int i = 0; i < n; ++i) {
            min_d2[i] = dist2(&X[(size_t)i * d], &centers[0]);
        }

        for (int c = 1; c < k; ++c) {
            double total = 0.0;
            for (int i = 0; i < n; ++i) total += min_d2[i];
            int chosen = 0;
            if (total <= 0.0) {
                chosen = uni(rng);
            } else {
                std::uniform_real_distribution<double> ur(0.0, total);
                double r = ur(rng);
                double acc = 0.0;
                chosen = n - 1;
                for (int i = 0; i < n; ++i) {
                    acc += min_d2[i];
                    if (acc >= r) {
                        chosen = i;
                        break;
                    }
                }
            }
            std::copy(X.begin() + (size_t)chosen * d,
                      X.begin() + (size_t)(chosen + 1) * d,
                      centers.begin() + (size_t)c * d);
            for (int i = 0; i < n; ++i) {
                min_d2[i] = std::min(
                    min_d2[i],
                    dist2(&X[(size_t)i * d], &centers[(size_t)c * d]));
            }
        }

        std::fill(labels.begin(), labels.end(), -1);
        for (int iter = 0; iter < max_iter; ++iter) {
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                int best_c = 0;
                double best_d = std::numeric_limits<double>::max();
                for (int c = 0; c < k; ++c) {
                    const double dd =
                        dist2(&X[(size_t)i * d], &centers[(size_t)c * d]);
                    if (dd < best_d) {
                        best_d = dd;
                        best_c = c;
                    }
                }
                if (labels[i] != best_c) {
                    labels[i] = best_c;
                    changed = true;
                }
            }
            if (!changed) break;

            std::fill(centers.begin(), centers.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);
            for (int i = 0; i < n; ++i) {
                counts[labels[i]]++;
                for (int j = 0; j < d; ++j) {
                    centers[(size_t)labels[i] * d + j] += X[(size_t)i * d + j];
                }
            }
            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    for (int j = 0; j < d; ++j) {
                        centers[(size_t)c * d + j] /= (double)counts[c];
                    }
                } else {
                    int far_i = 0;
                    double far_d = -1.0;
                    for (int i = 0; i < n; ++i) {
                        const double dd = dist2(
                            &X[(size_t)i * d],
                            &centers[(size_t)labels[i] * d]);
                        if (dd > far_d) {
                            far_d = dd;
                            far_i = i;
                        }
                    }
                    std::copy(X.begin() + (size_t)far_i * d,
                              X.begin() + (size_t)(far_i + 1) * d,
                              centers.begin() + (size_t)c * d);
                }
            }
        }

        double inertia = 0.0;
        for (int i = 0; i < n; ++i) {
            inertia += dist2(&X[(size_t)i * d],
                             &centers[(size_t)labels[i] * d]);
        }
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_labels = labels;
        }
    }
    return best_labels;
}

static void renumber_labels(std::vector<int> &labels) {
    std::vector<int> remap;
    for (int &label : labels) {
        int found = -1;
        for (size_t i = 0; i < remap.size(); ++i) {
            if (remap[i] == label) {
                found = (int)i;
                break;
            }
        }
        if (found < 0) {
            remap.push_back(label);
            found = (int)remap.size() - 1;
        }
        label = found;
    }
}

static std::vector<int> spectral_cluster_embeddings(
    const std::vector<std::vector<float>> &embs, int dim, int max_speakers,
    int oracle_num) {
    const int n = (int)embs.size();
    std::vector<int> labels(n, 0);
    if (n <= 1) return labels;

    std::vector<double> M((size_t)n * n, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double c =
                (double)rs_speaker_cosine(embs[i].data(), embs[j].data(), dim);
            M[(size_t)i * n + j] = c;
            M[(size_t)j * n + i] = c;
        }
    }

    const float pval = 0.02f;
    const int min_pnum = 6;
    int n_elems = (int)((1.0f - pval) * (float)n);
    n_elems = std::min(n_elems, n - min_pnum);
    if (n_elems > 0) {
        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) idx[j] = j;
            double *row = &M[(size_t)i * n];
            std::sort(idx.begin(), idx.end(), [row](int a, int b) {
                return row[a] < row[b];
            });
            for (int e = 0; e < n_elems; ++e) row[idx[e]] = 0.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double v = 0.5 * (M[(size_t)i * n + j] +
                                    M[(size_t)j * n + i]);
            M[(size_t)i * n + j] = v;
            M[(size_t)j * n + i] = v;
        }
    }

    for (int i = 0; i < n; ++i) M[(size_t)i * n + i] = 0.0;
    std::vector<double> L((size_t)n * n);
    for (int i = 0; i < n; ++i) {
        double deg = 0.0;
        for (int j = 0; j < n; ++j) deg += std::fabs(M[(size_t)i * n + j]);
        for (int j = 0; j < n; ++j) L[(size_t)i * n + j] = -M[(size_t)i * n + j];
        L[(size_t)i * n + i] = deg;
    }

    std::vector<double> w;
    std::vector<double> V;
    jacobi_eigh(L, n, w, V);

    int num_spk = 1;
    if (oracle_num > 0) {
        num_spk = std::min(oracle_num, n);
    } else {
        const int hi = std::min(std::max(1, max_speakers) + 1, n);
        double best_gap = -std::numeric_limits<double>::max();
        for (int i = 0; i + 1 < hi; ++i) {
            const double gap = w[i + 1] - w[i];
            if (gap > best_gap) {
                best_gap = gap;
                num_spk = i + 1;
            }
        }
        num_spk = std::max(1, std::min(num_spk, n));
    }

    std::vector<double> spec_emb((size_t)n * num_spk);
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < num_spk; ++c) {
            spec_emb[(size_t)i * num_spk + c] = V[(size_t)i * n + c];
        }
    }

    labels = kmeans_cluster(spec_emb, n, num_spk, num_spk);
    renumber_labels(labels);
    return labels;
}

static bool apply_speaker_clustering(std::vector<AsrSegment> &segments,
                                     const SpeakerServerConfig &cfg,
                                     std::string &err) {
    if (!cfg.enabled) return true;
    if (!g_have_speaker || !g_speaker) {
        err = "speaker=true requires --spk-model";
        return false;
    }
    std::vector<std::vector<float>> embeddings;
    std::vector<int> segment_index;
    embeddings.reserve(segments.size());
    segment_index.reserve(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        if (segments[i].pcm.empty()) continue;
        std::vector<float> emb;
        if (!speaker_embed_pcm(segments[i].pcm, emb, err)) return false;
        segments[i].speaker_embedding = emb;
        embeddings.push_back(std::move(emb));
        segment_index.push_back((int)i);
    }
    if (embeddings.empty()) return true;
    const int dim = rs_speaker_dim(g_speaker);
    std::vector<int> labels = cfg.cluster == "ahc"
        ? cluster_speaker_embeddings_ahc(
              embeddings, dim, cfg.threshold, cfg.num_speakers)
        : spectral_cluster_embeddings(
              embeddings, dim, cfg.max_speakers, cfg.num_speakers);
    for (size_t i = 0; i < labels.size(); ++i) {
        segments[(size_t)segment_index[i]].speaker = labels[i];
    }
    return true;
}

static void write_speaker_log(const std::vector<AsrSegment> &segments,
                              const SpeakerServerConfig &cfg) {
    if (!cfg.log) return;
    std::ostream *out = &std::cerr;
    std::ofstream file;
    if (!g_speaker_log_path.empty() && g_speaker_log_path != "-") {
        file.open(g_speaker_log_path, std::ios::app);
        if (file) out = &file;
    }
    for (const auto &segment : segments) {
        if (segment.speaker < 0) continue;
        *out << "[speaker] segment=" << segment.index
             << " start=" << time_s(segment.start_s)
             << " end=" << time_s(segment.end_s)
             << " speaker=" << speaker_name(segment.speaker)
             << " text=" << segment.text << "\n";
    }
}

static bool parse_bool_value(const std::string &v, bool current);

static void log_asr_memory_locked(const char *label) {
    if (!g_mem_log || !g_asr.ctx || !g_asr.ctx->sched) return;
    fprintf(stderr, "[server] memory %s: ASR backend buffers", label ? label : "");
    for (auto b : g_asr.ctx->backends) {
        if (!b) continue;
        size_t bytes = ggml_backend_sched_get_buffer_size(g_asr.ctx->sched, b);
        const char *name = ggml_backend_name(b);
        fprintf(stderr, " %s=%.1fMiB", name ? name : "backend",
                (double)bytes / 1048576.0);
    }
    fprintf(stderr, "\n");
}

// Caller must hold g_asr.mu. This is intentionally opt-in because it unloads
// and reloads ASR weights, causing a latency spike but returning CUDA buffers
// to the driver between long stability-test segments.
static void note_asr_final_segment_locked(const char *label) {
    log_asr_memory_locked(label);
    if (g_asr_reload_every <= 0 || g_asr_model_path.empty()) return;
    g_asr_final_segments_since_reload++;
    if (g_asr_final_segments_since_reload < g_asr_reload_every) return;

    fprintf(stderr, "[server] reloading ASR after %d final segment(s) to release backend buffers\n",
            g_asr_final_segments_since_reload);
    g_asr_final_segments_since_reload = 0;
    g_asr.unload();
    if (!g_asr.load(g_asr_model_path, g_model_threads, g_model_gpu)) {
        g_have_asr = false;
        fprintf(stderr, "[server] ASR reload failed; ASR is now unavailable\n");
        return;
    }
    if (g_asr.ctx && g_asr.ctx->ctx_gguf) {
        load_cmvn_params(g_asr.ctx->ctx_gguf, g_asr_cmvn_means,
                         g_asr_cmvn_vars);
        g_asr_cmvn_version++;
    }
    g_have_asr = true;
    fprintf(stderr, "[server] ASR reloaded: %s @ %d Hz\n",
            g_asr.arch.c_str(), g_asr.sample_rate);
    log_asr_memory_locked("after-reload");
}

// ── core operations ─────────────────────────────────────────────────────

// Transcribe a mono float PCM buffer (at g_asr.sample_rate) → text.
// Caller must hold g_asr.mu. FunASR-Nano can run in:
//   - default mode: model default decoder
//   - ctc_only: fast CTC pass, useful for VAD partials
//   - two_pass: CTC first pass + LLM re-decode for the final text
static bool transcribe_pcm_locked(const float *pcm, int n, std::string &text,
                                  std::string &err, bool two_pass = false,
                                  bool ctc_precheck = false,
                                  bool ctc_only = false) {
    if (!g_asr.ctx) {
        err = "no ASR model loaded";
        return false;
    }
    rs_reset(g_asr.ctx);
    rs_set_ctc_precheck(g_asr.ctx, ctc_precheck);
    rs_set_use_llm(g_asr.ctx, !(two_pass || ctc_only));

    // Match rs-asr-vad-online: short growing VAD buffers need enough trailing
    // audio for ASR/CTC to run. Without this, two-pass partials at 600 ms can
    // return no output and appear to disappear.
    std::vector<float> padded;
    const int min_asr_samples = g_asr.sample_rate > 0 ? g_asr.sample_rate : 16000;
    if (n > 0 && n < min_asr_samples) {
        padded.assign(pcm, pcm + n);
        padded.resize(min_asr_samples, 0.0f);
        pcm = padded.data();
        n = (int)padded.size();
    }

    if (rs_push_audio(g_asr.ctx, pcm, n) != RS_OK) {
        err = "rs_push_audio failed";
        return false;
    }
    int32_t st;
    while ((st = rs_process(g_asr.ctx)) > 0) { /* drain */ }
    if (st < 0) { err = rs_get_last_error().message; return false; }
    const char *t = rs_get_text_output(g_asr.ctx);
    std::string first_pass = t ? t : "";

    if (two_pass && !ctc_only) {
        rs_set_use_llm(g_asr.ctx, true);
        int32_t st2 = rs_redecode(g_asr.ctx);
        if (st2 < 0) { err = rs_get_last_error().message; return false; }
        const char *t2 = st2 > 0 ? rs_get_text_output(g_asr.ctx) : nullptr;
        text = (t2 && t2[0]) ? t2 : first_pass;
    } else {
        text = first_pass;
    }
    return true;
}

static bool compute_asr_features(const std::vector<float> &pcm,
                                 std::vector<float> &features,
                                 std::string &err) {
    if (!g_have_asr || !g_asr.ctx || !g_asr.ctx->model) {
        err = "no ASR model loaded";
        return false;
    }

    RSModelMeta meta;
    {
        std::lock_guard<std::mutex> lk(g_asr.mu);
        if (!g_asr.ctx || !g_asr.ctx->model) {
            err = "no ASR model loaded";
            return false;
        }
        meta = g_asr.ctx->model->GetMeta();
    }

    if (meta.use_external_frontend) {
        features = pcm;
        return true;
    }

    STFTConfig config;
    config.sample_rate = meta.audio_sample_rate;
    config.n_mels = meta.n_mels;
    config.window_type = meta.window_type;
    config.use_lfr = true;
    config.lfr_m = 7;
    config.lfr_n = 6;
    config.fbank_num_threads = g_asr_frontend_threads;

    struct FrontendCache {
        STFTConfig config;
        uint64_t cmvn_version = 0;
        std::unique_ptr<AudioProcessor> proc;
    };
    thread_local FrontendCache cache;

    const uint64_t cmvn_version = g_asr_cmvn_version.load();
    auto same_config = [](const STFTConfig &a, const STFTConfig &b) {
        return a.sample_rate == b.sample_rate &&
               a.frame_size == b.frame_size &&
               a.frame_step == b.frame_step &&
               a.n_fft == b.n_fft &&
               a.n_mels == b.n_mels &&
               a.f_min == b.f_min &&
               a.f_max == b.f_max &&
               a.window_type == b.window_type &&
               a.snip_edges == b.snip_edges &&
               a.mel_low_hz == b.mel_low_hz &&
               a.mel_high_hz == b.mel_high_hz &&
               a.use_lfr == b.use_lfr &&
               a.lfr_m == b.lfr_m &&
               a.lfr_n == b.lfr_n &&
               a.use_cmvn == b.use_cmvn &&
               a.fbank_num_threads == b.fbank_num_threads;
    };
    if (!cache.proc || !same_config(cache.config, config) ||
        cache.cmvn_version != cmvn_version) {
        cache.config = config;
        cache.cmvn_version = cmvn_version;
        cache.proc = std::make_unique<AudioProcessor>(config);
        if (!g_asr_cmvn_means.empty() && !g_asr_cmvn_vars.empty()) {
            cache.proc->SetCMVN(g_asr_cmvn_means, g_asr_cmvn_vars);
        }
    }

    cache.proc->Compute(pcm, features);
    if (features.empty()) {
        err = "ASR frontend produced no features";
        return false;
    }
    return true;
}

static bool transcribe_encoded_state_locked(RSState &state,
                                            const AsrServerConfig &cfg,
                                            std::string &text,
                                            std::string &err) {
    if (!g_asr.ctx || !g_asr.ctx->model || !g_asr.ctx->sched) {
        err = "no ASR model loaded";
        return false;
    }

    auto &model = *g_asr.ctx->model;
    ggml_backend_sched_t sched = g_asr.ctx->sched;
    model.SetCTCPrecheck(cfg.ctc_precheck);
    model.SetUseLLM(!(cfg.two_pass || cfg.ctc_only));

    ggml_backend_sched_reset(sched);
    if (!model.Decode(state, sched)) {
        err = "model decode failed";
        return false;
    }
    std::string first_pass = model.GetTranscription(state);

    if (cfg.two_pass && !cfg.ctc_only) {
        model.SetUseLLM(true);
        ggml_backend_sched_reset(sched);
        if (!model.Decode(state, sched)) {
            err = "model re-decode failed";
            return false;
        }
        std::string second_pass = model.GetTranscription(state);
        text = second_pass.empty() ? first_pass : second_pass;
    } else {
        text = first_pass;
    }
    return true;
}

class AsrBatchScheduler {
public:
    struct Stats {
        int slots = 0;
        int max_queued = 0;
        int queued = 0;
        int raw_queued = 0;
        int preparing = 0;
        int frontend_workers = 0;
        int batch_wait_ms = 0;
        int min_batch = 0;
        int max_batch_wait_ms = 0;
        uint64_t submitted = 0;
        uint64_t completed = 0;
        uint64_t failed = 0;
        uint64_t rejected = 0;
        uint64_t batches = 0;
        uint64_t batched_requests = 0;
        uint64_t max_batch = 0;
        std::vector<uint64_t> batch_size_hist;
        uint64_t frontend_ms = 0;
        uint64_t queue_wait_ms = 0;
        uint64_t encode_ms = 0;
        uint64_t decode_ms = 0;
        uint64_t e2e_ms = 0;
    };

    AsrBatchScheduler(int n_slots, int max_queued, int batch_wait_ms,
                      int min_batch, int max_batch_wait_ms,
                      int frontend_workers)
        : n_slots_(std::max(1, n_slots)),
          max_queued_(std::max(1, max_queued)),
          batch_wait_ms_(std::max(0, batch_wait_ms)),
          min_batch_(std::max(1, min_batch)),
          max_batch_wait_ms_(std::max(batch_wait_ms_,
                                      max_batch_wait_ms < 0
                                          ? batch_wait_ms_
                                          : max_batch_wait_ms)),
          n_frontend_workers_(std::max(1, frontend_workers)) {
        min_batch_ = min_batch <= 0 ? n_slots_ : std::min(min_batch_, n_slots_);
        batch_size_hist_.resize((size_t)n_slots_ + 1, 0);
    }
    ~AsrBatchScheduler() { Stop(); }

    void Start() {
        stop_ = false;
        for (int i = 0; i < n_frontend_workers_; ++i) {
            frontend_workers_.emplace_back(&AsrBatchScheduler::FrontendLoop,
                                           this);
        }
        worker_ = std::thread(&AsrBatchScheduler::RunLoop, this);
    }

    void Stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        frontend_cv_.notify_all();
        for (auto &t : frontend_workers_) {
            if (t.joinable()) t.join();
        }
        frontend_workers_.clear();
        if (worker_.joinable()) worker_.join();
        std::lock_guard<std::mutex> lk(mu_);
        while (!raw_pending_.empty()) {
            raw_pending_.front().promise.set_value({"", "server shutting down"});
            raw_pending_.pop_front();
        }
        while (!pending_.empty()) {
            pending_.front().promise.set_value({"", "server shutting down"});
            pending_.pop_front();
        }
    }

    std::future<std::pair<std::string, std::string>>
    Submit(std::vector<float> pcm, AsrServerConfig cfg) {
        PendingReq req;
        req.pcm = std::move(pcm);
        req.cfg = cfg;
        req.enqueue_time = std::chrono::steady_clock::now();
        auto fut = req.promise.get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            submitted_++;
            if ((int)raw_pending_.size() + (int)pending_.size() +
                    preparing_ >=
                max_queued_) {
                rejected_++;
                failed_++;
                completed_++;
                req.promise.set_value({"", "ASR queue full"});
                return fut;
            }
            raw_pending_.push_back(std::move(req));
        }
        frontend_cv_.notify_one();
        return fut;
    }

    int n_queued() {
        std::lock_guard<std::mutex> lk(mu_);
        return (int)raw_pending_.size() + (int)pending_.size() + preparing_;
    }

    Stats snapshot() {
        std::lock_guard<std::mutex> lk(mu_);
        Stats s;
        s.slots = n_slots_;
        s.max_queued = max_queued_;
        s.queued = (int)pending_.size();
        s.raw_queued = (int)raw_pending_.size();
        s.preparing = preparing_;
        s.frontend_workers = n_frontend_workers_;
        s.batch_wait_ms = batch_wait_ms_;
        s.min_batch = min_batch_;
        s.max_batch_wait_ms = max_batch_wait_ms_;
        s.submitted = submitted_;
        s.completed = completed_;
        s.failed = failed_;
        s.rejected = rejected_;
        s.batches = batches_;
        s.batched_requests = batched_requests_;
        s.max_batch = max_batch_;
        s.batch_size_hist = batch_size_hist_;
        s.frontend_ms = frontend_ms_;
        s.queue_wait_ms = queue_wait_ms_;
        s.encode_ms = encode_ms_;
        s.decode_ms = decode_ms_;
        s.e2e_ms = e2e_ms_;
        return s;
    }

private:
    struct PendingReq {
        std::vector<float> pcm;
        std::vector<float> features;
        AsrServerConfig cfg;
        std::chrono::steady_clock::time_point enqueue_time;
        std::chrono::steady_clock::time_point ready_time;
        std::promise<std::pair<std::string, std::string>> promise;
    };

    static uint64_t elapsed_ms(std::chrono::steady_clock::time_point start,
                               std::chrono::steady_clock::time_point end) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start).count();
    }

    bool Prepare(PendingReq &req) {
        if (!req.features.empty()) return true;
        std::string err;
        if (!compute_asr_features(req.pcm, req.features, err)) {
            req.promise.set_value({"", err});
            std::lock_guard<std::mutex> lk(mu_);
            failed_++;
            completed_++;
            e2e_ms_ += elapsed_ms(req.enqueue_time,
                                  std::chrono::steady_clock::now());
            return false;
        }
        req.ready_time = std::chrono::steady_clock::now();
        return true;
    }

    bool CompatibleWith(const PendingReq &req,
                        const PendingReq &first) const {
        return req.features.size() == first.features.size() &&
               same_asr_config(req.cfg, first.cfg);
    }

    void restore_skipped(std::deque<PendingReq> &skipped) {
        if (skipped.empty()) return;
        std::lock_guard<std::mutex> lk(mu_);
        while (!skipped.empty()) {
            pending_.push_front(std::move(skipped.back()));
            skipped.pop_back();
        }
    }

    void FrontendLoop() {
        for (;;) {
            PendingReq req;
            {
                std::unique_lock<std::mutex> lk(mu_);
                frontend_cv_.wait(lk, [&] {
                    return stop_.load() || !raw_pending_.empty();
                });
                if (stop_ && raw_pending_.empty()) break;
                req = std::move(raw_pending_.front());
                raw_pending_.pop_front();
                preparing_++;
            }

            std::string err;
            auto frontend_start = std::chrono::steady_clock::now();
            bool prepared = compute_asr_features(req.pcm, req.features, err);
            auto frontend_end = std::chrono::steady_clock::now();
            req.ready_time = frontend_end;

            {
                std::lock_guard<std::mutex> lk(mu_);
                preparing_--;
                frontend_ms_ += elapsed_ms(frontend_start, frontend_end);
                if (stop_) {
                    failed_++;
                    completed_++;
                    e2e_ms_ += elapsed_ms(req.enqueue_time, frontend_end);
                    req.promise.set_value({"", "server shutting down"});
                } else if (!prepared) {
                    failed_++;
                    completed_++;
                    e2e_ms_ += elapsed_ms(req.enqueue_time, frontend_end);
                    req.promise.set_value({"", err});
                } else {
                    pending_.push_back(std::move(req));
                }
            }
            cv_.notify_one();
            frontend_cv_.notify_one();
        }
    }

    void RunBatch(std::vector<PendingReq> &batch) {
        auto batch_start = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(mu_);
            batches_++;
            batched_requests_ += batch.size();
            max_batch_ = std::max<uint64_t>(max_batch_, batch.size());
            if (batch.size() < batch_size_hist_.size()) {
                batch_size_hist_[batch.size()]++;
            }
            for (auto &req : batch) {
                queue_wait_ms_ += elapsed_ms(req.ready_time, batch_start);
            }
        }

        std::vector<std::shared_ptr<RSState>> owned_states;
        std::vector<RSState *> states;
        std::vector<const std::vector<float> *> inputs;
        owned_states.reserve(batch.size());
        states.reserve(batch.size());
        inputs.reserve(batch.size());

        int successful = 0;
        std::vector<std::pair<std::string, std::string>> results(batch.size());
        {
            std::lock_guard<std::mutex> lk(g_asr.mu);
            if (!g_asr.ctx || !g_asr.ctx->model || !g_asr.ctx->sched) {
                for (auto &r : results) r.second = "no ASR model loaded";
            } else {
                for (auto &req : batch) {
                    auto st = g_asr.ctx->model->CreateState();
                    states.push_back(st.get());
                    inputs.push_back(&req.features);
                    owned_states.push_back(std::move(st));
                }
                ggml_backend_sched_reset(g_asr.ctx->sched);
                auto encode_start = std::chrono::steady_clock::now();
                bool ok = g_asr.ctx->model->EncodeBatch(inputs, states,
                                                        g_asr.ctx->sched);
                auto encode_end = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> mlk(mu_);
                    encode_ms_ += elapsed_ms(encode_start, encode_end);
                }
                if (!ok) {
                    for (auto &r : results) r.second = "model encode batch failed";
                } else {
                    auto &model = *g_asr.ctx->model;
                    const AsrServerConfig cfg = batch.front().cfg;
                    model.SetCTCPrecheck(cfg.ctc_precheck);
                    model.SetUseLLM(!(cfg.two_pass || cfg.ctc_only));

                    ggml_backend_sched_reset(g_asr.ctx->sched);
                    auto decode_start = std::chrono::steady_clock::now();
                    ok = model.DecodeBatch(states, g_asr.ctx->sched);
                    if (!ok) {
                        for (auto &r : results) r.second = "model decode batch failed";
                    } else {
                        for (size_t i = 0; i < batch.size(); ++i) {
                            std::string first_pass = model.GetTranscription(*states[i]);
                            if (cfg.two_pass && !cfg.ctc_only) {
                                model.SetUseLLM(true);
                                ggml_backend_sched_reset(g_asr.ctx->sched);
                                if (!model.Decode(*states[i], g_asr.ctx->sched)) {
                                    results[i].second = "model re-decode failed";
                                    continue;
                                }
                                std::string second_pass =
                                    model.GetTranscription(*states[i]);
                                results[i].first = second_pass.empty()
                                                       ? first_pass
                                                       : second_pass;
                            } else {
                                results[i].first = first_pass;
                            }
                            successful++;
                        }
                    }
                    auto decode_end = std::chrono::steady_clock::now();
                    {
                        std::lock_guard<std::mutex> mlk(mu_);
                        decode_ms_ += elapsed_ms(decode_start, decode_end);
                    }
                }
                for (int i = 0; i < successful; ++i) {
                    note_asr_final_segment_locked("asr-batch-final");
                }
            }
        }

        auto finish_time = std::chrono::steady_clock::now();
        uint64_t failed = 0;
        for (const auto &r : results) {
            if (!r.second.empty()) failed++;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            completed_ += batch.size();
            failed_ += failed;
            for (auto &req : batch) {
                e2e_ms_ += elapsed_ms(req.enqueue_time, finish_time);
            }
        }
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].promise.set_value(std::move(results[i]));
        }
    }

    void RunLoop() {
        while (!stop_) {
            PendingReq first;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_.load() || !pending_.empty(); });
                if (stop_) break;
                first = std::move(pending_.front());
                pending_.pop_front();
            }
            if (!Prepare(first)) continue;

            std::vector<PendingReq> batch;
            batch.push_back(std::move(first));

            auto batch_open = std::chrono::steady_clock::now();
            auto soft_deadline = batch_open +
                                 std::chrono::milliseconds(batch_wait_ms_);
            auto hard_deadline = batch_open +
                                 std::chrono::milliseconds(max_batch_wait_ms_);
            std::deque<PendingReq> skipped;
            const int scan_budget = std::max(n_slots_, n_slots_ * 4);
            int scanned = 0;
            while ((int)batch.size() < n_slots_ && scanned < scan_budget) {
                PendingReq cand;
                bool have = false;
                {
                    std::unique_lock<std::mutex> lk(mu_);
                    if (pending_.empty()) {
                        auto now = std::chrono::steady_clock::now();
                        const int known_demand =
                            (int)batch.size() + (int)pending_.size() +
                            preparing_ + (int)raw_pending_.size();
                        const bool can_fill_batch = known_demand >= n_slots_;
                        const bool below_target = (int)batch.size() < min_batch_;
                        auto deadline =
                            (below_target || (preparing_ > 0 && can_fill_batch))
                                ? hard_deadline
                                : soft_deadline;
                        if (now >= deadline) break;
                        cv_.wait_until(lk, deadline, [&] {
                            return stop_.load() || !pending_.empty();
                        });
                    }
                    if (stop_ || pending_.empty()) break;
                    cand = std::move(pending_.front());
                    pending_.pop_front();
                    have = true;
                }
                if (!have) break;
                scanned++;
                if (!Prepare(cand)) continue;
                if (CompatibleWith(cand, batch.front())) {
                    batch.push_back(std::move(cand));
                } else {
                    skipped.push_back(std::move(cand));
                }
                auto now = std::chrono::steady_clock::now();
                if (now >= hard_deadline) break;
                if (now >= soft_deadline && (int)batch.size() >= min_batch_) {
                    std::lock_guard<std::mutex> lk(mu_);
                    const int known_demand =
                        (int)batch.size() + (int)pending_.size() +
                        preparing_ + (int)raw_pending_.size();
                    if (pending_.empty() && known_demand < n_slots_) break;
                }
            }
            restore_skipped(skipped);

            RunBatch(batch);
        }
    }

    int n_slots_;
    int max_queued_;
    int batch_wait_ms_;
    int min_batch_;
    int max_batch_wait_ms_;
    int n_frontend_workers_;
    std::deque<PendingReq> raw_pending_;
    std::deque<PendingReq> pending_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable frontend_cv_;
    std::thread worker_;
    std::vector<std::thread> frontend_workers_;
    std::atomic<bool> stop_{false};
    int preparing_ = 0;
    uint64_t submitted_ = 0;
    uint64_t completed_ = 0;
    uint64_t failed_ = 0;
    uint64_t rejected_ = 0;
    uint64_t batches_ = 0;
    uint64_t batched_requests_ = 0;
    uint64_t max_batch_ = 0;
    std::vector<uint64_t> batch_size_hist_;
    uint64_t frontend_ms_ = 0;
    uint64_t queue_wait_ms_ = 0;
    uint64_t encode_ms_ = 0;
    uint64_t decode_ms_ = 0;
    uint64_t e2e_ms_ = 0;
};

static std::unique_ptr<AsrBatchScheduler> g_asr_batch;

static void set_process_env(const std::string &key, const std::string &value) {
#ifdef _WIN32
    _putenv_s(key.c_str(), value.c_str());
#else
    setenv(key.c_str(), value.c_str(), 1);
#endif
}

struct RegistryDependency {
    std::string name;
    std::string env;
    std::string filename;
    std::string path;
    int64_t size_bytes = 0;
};

static void maybe_start_asr_batch_scheduler() {
    if (g_asr_slots > 1 &&
        (g_asr.arch == "SenseVoiceSmall" || g_asr.arch == "FunASRNano")) {
        const int asr_max_batch_wait_ms =
            g_asr_max_batch_wait_ms < 0
                ? std::max(g_asr_batch_wait_ms, g_asr_batch_wait_ms * 4)
                : g_asr_max_batch_wait_ms;
        const int asr_frontend_workers =
            g_asr_frontend_workers > 0 ? g_asr_frontend_workers : g_asr_slots;
        const int asr_min_batch =
            g_asr_min_batch <= 0 ? g_asr_slots
                                 : std::min(g_asr_min_batch, g_asr_slots);
        g_asr_batch = std::make_unique<AsrBatchScheduler>(
            g_asr_slots, g_asr_max_queued, g_asr_batch_wait_ms,
            asr_min_batch, asr_max_batch_wait_ms, asr_frontend_workers);
        g_asr_batch->Start();
        fprintf(stderr,
                "[server] ASR scheduler: slots=%d max_queued=%d "
                "batch_wait=%dms min_batch=%d max_wait=%dms "
                "frontend_workers=%d frontend_threads=%d\n",
                g_asr_slots, g_asr_max_queued, g_asr_batch_wait_ms,
                asr_min_batch, asr_max_batch_wait_ms, asr_frontend_workers,
                g_asr_frontend_threads);
    } else if (g_asr_slots > 1 && !g_asr.arch.empty()) {
        fprintf(stderr,
                "[server] ASR batching unavailable for arch=%s; using serialized path\n",
                g_asr.arch.c_str());
    }
}

static bool load_asr_runtime_model(const std::string &path,
                                   const std::string &spec,
                                   std::string &err) {
    std::lock_guard<std::mutex> switch_lk(g_model_switch_mu);
    g_asr_batch.reset();
    {
        std::lock_guard<std::mutex> lk(g_asr.mu);
        g_have_asr = false;
        g_asr_model_path = path;
        g_asr_final_segments_since_reload = 0;
        g_asr.unload();
        if (!g_asr.load(path, g_model_threads, g_model_gpu)) {
            err = "failed to load ASR model: " + path;
            return false;
        }
        if (g_asr.ctx && g_asr.ctx->ctx_gguf) {
            load_cmvn_params(g_asr.ctx->ctx_gguf, g_asr_cmvn_means,
                             g_asr_cmvn_vars);
            g_asr_cmvn_version++;
        }
        g_have_asr = true;
        g_current_asr_model_spec = spec;
        fprintf(stderr, "[server] ASR ready: %s @ %d Hz (%s)\n",
                g_asr.arch.c_str(), g_asr.sample_rate, spec.c_str());
        fprintf(stderr, "[server] ASR decode: two-pass=%s ctc-precheck=%s\n",
                g_asr_cfg.two_pass ? "on" : "off",
                g_asr_cfg.ctc_precheck ? "on" : "off");
        if (g_mem_log) log_asr_memory_locked("after-load");
    }
    if (g_asr_reload_every > 0) {
        fprintf(stderr,
                "[server] ASR auto-reload enabled: every %d final segment(s)\n",
                g_asr_reload_every);
    }
    maybe_start_asr_batch_scheduler();
    return true;
}

static void stop_tts_runtime_schedulers() {
    g_tts_batch.reset();
    g_voxcpm2_batch.reset();
}

static void maybe_start_tts_schedulers() {
    if (g_tts_slots > 1 && g_tts.arch == "cosyvoice3") {
        auto *cv3 = dynamic_cast<CosyVoice3LMModel *>(g_tts.ctx->model.get());
        if (cv3) {
            auto scheds_voc =
                g_tts.ctx->create_extra_gpu_scheds(g_tts_voc_workers);
            g_tts_batch = std::make_unique<TtsBatchScheduler>(
                cv3, g_tts.ctx->sched, std::move(scheds_voc), g_tts_slots);
            g_tts_batch->Start();
            fprintf(stderr, "[server] TTS continuous batching: %d slots\n",
                    g_tts_slots);
        } else {
            fprintf(stderr,
                    "[server] TTS batching unavailable (model cast failed); "
                    "using serialized path\n");
        }
    }
    if (g_tts_slots > 1 && g_tts.arch == "voxcpm2") {
        auto *voxcpm2 = dynamic_cast<VoxCPM2Model *>(g_tts.ctx->model.get());
        if (voxcpm2) {
            auto extra = g_tts.ctx->create_extra_gpu_scheds(g_tts_slots - 1);
            std::vector<ggml_backend_sched_t> scheds;
            scheds.push_back(g_tts.ctx->sched);
            scheds.insert(scheds.end(), extra.begin(), extra.end());
            g_voxcpm2_batch =
                std::make_unique<VoxCPM2ServerScheduler>(
                    voxcpm2, std::move(scheds), g_tts_slots);
            g_voxcpm2_batch->Start();
            fprintf(stderr,
                    "[server] VoxCPM2 TTS scheduler: %d slots, %d worker(s)\n",
                    g_tts_slots, g_voxcpm2_batch->n_workers());
        } else {
            fprintf(stderr,
                    "[server] VoxCPM2 batching unavailable (model cast failed)\n");
        }
    }
}

static bool load_tts_runtime_model(const std::string &path,
                                   const std::string &spec,
                                   const std::string &voice_path,
                                   const std::vector<RegistryDependency> &dependencies,
                                   std::string &err) {
    std::lock_guard<std::mutex> switch_lk(g_model_switch_mu);
    if (!voice_path.empty()) set_process_env("RS_KOKORO_VOICE_PATH", voice_path);
    for (const auto &dep : dependencies) {
        if (!dep.env.empty()) set_process_env(dep.env, dep.path);
    }
    stop_tts_runtime_schedulers();
    {
        std::lock_guard<std::mutex> lk(g_tts.mu);
        g_have_tts = false;
        g_tts.unload();
        if (!g_tts.load(path, g_model_threads, g_model_gpu, RS_TASK_TTS_ONLINE)) {
            err = "failed to load TTS model: " + path;
            return false;
        }
        g_have_tts = true;
        g_current_tts_model_spec = spec;
        fprintf(stderr, "[server] TTS ready: %s @ %d Hz (%s)\n",
                g_tts.arch.c_str(), g_tts.sample_rate, spec.c_str());
    }
    maybe_start_tts_schedulers();
    return true;
}

static bool load_vad_runtime_model(const std::string &path,
                                   const std::string &spec,
                                   std::string &err) {
    std::lock_guard<std::mutex> switch_lk(g_model_switch_mu);
    std::lock_guard<std::mutex> lk(g_vad_mu);
    if (g_vad) {
        rs_vad_free(g_vad);
        g_vad = nullptr;
    }
    for (auto *worker : g_vad_ws_pool) {
        if (worker) rs_vad_free(worker);
    }
    g_vad_ws_pool.clear();
    g_vad_ws_busy.clear();
    g_have_vad = false;

    g_vad = rs_vad_init_from_file(path.c_str(), g_model_threads, g_model_gpu);
    if (!g_vad) {
        err = "failed to load VAD model: " + path;
        return false;
    }
    rs_vad_set_threshold(g_vad, g_vad_cfg.threshold);
    for (int i = 0; i < g_vad_workers; ++i) {
        rs_vad_t *worker =
            rs_vad_init_from_file(path.c_str(), g_model_threads, g_model_gpu);
        if (!worker) {
            err = "failed to load VAD worker";
            return false;
        }
        rs_vad_set_threshold(worker, g_vad_cfg.threshold);
        g_vad_ws_pool.push_back(worker);
        g_vad_ws_busy.push_back(false);
    }
    g_have_vad = true;
    g_current_vad_model_spec = spec;
    fprintf(stderr,
            "[server] VAD ready: %s threshold=%.2f ws-workers=%d (%s)\n",
            rs_vad_get_arch(g_vad), g_vad_cfg.threshold, g_vad_workers,
            spec.c_str());
    return true;
}

static bool load_speaker_runtime_model(const std::string &path,
                                       const std::string &spec,
                                       std::string &err) {
    std::lock_guard<std::mutex> switch_lk(g_model_switch_mu);
    std::lock_guard<std::mutex> lk(g_speaker_mu);
    if (g_speaker) {
        rs_speaker_free(g_speaker);
        g_speaker = nullptr;
    }
    g_have_speaker = false;
    g_speaker = rs_speaker_init_from_file(path.c_str(), g_model_threads,
                                          g_model_gpu);
    if (!g_speaker) {
        rs_error_info_t e = rs_get_last_error();
        err = e.message[0] ? e.message : "failed to load speaker model";
        return false;
    }
    g_have_speaker = true;
    g_speaker_model_path = path;
    g_current_speaker_model_spec = spec;
    fprintf(stderr, "[server] Speaker ready: CAMPPlus dim=%d (%s)\n",
            rs_speaker_dim(g_speaker), spec.c_str());
    return true;
}

static bool transcribe_pcm_server(const float *pcm, int n,
                                  const AsrServerConfig &cfg,
                                  std::string &text, std::string &err) {
    if (g_asr_batch) {
        std::vector<float> owned(pcm, pcm + n);
        auto fut = g_asr_batch->Submit(std::move(owned), cfg);
        auto result = fut.get();
        text = std::move(result.first);
        err = std::move(result.second);
        return err.empty();
    }

    std::lock_guard<std::mutex> lk(g_asr.mu);
    bool ok = transcribe_pcm_locked(pcm, n, text, err,
                                    cfg.two_pass, cfg.ctc_precheck,
                                    cfg.ctc_only);
    if (ok) note_asr_final_segment_locked("http-final");
    return ok;
}

// Caller holds g_vad_mu for the stateful VAD handle. ASR is locked only while
// a finalized segment is actually transcribed, so VAD scanning does not block
// unrelated ASR requests.
static bool transcribe_pcm_vad_locked(const std::vector<float> &pcm,
                                      const AsrServerConfig &asr_cfg,
                                      const VadServerConfig &cfg,
                                      std::vector<AsrSegment> &segments,
                                      std::string &text, std::string &err) {
    if (rs_vad_reset(g_vad) != RS_OK) {
        err = "rs_vad_reset failed";
        return false;
    }
    if (rs_vad_set_threshold(g_vad, cfg.threshold) != RS_OK) {
        err = "rs_vad_set_threshold failed";
        return false;
    }

    const int sr = g_asr.sample_rate > 0 ? g_asr.sample_rate : 16000;
    const int chunk_samples = std::max(1, sr / 10); // 100 ms
    std::vector<float> seg;
    bool in_speech = false;
    bool pending_end = false;
    int pending_silence_samples = 0;
    int seg_index = 0;
    int64_t seg_start_sample = 0;
    int64_t total_samples = 0;

    auto append_text = [&](const std::string &seg_text) {
        if (seg_text.empty()) return;
        if (!text.empty()) text += "\n";
        text += seg_text;
    };

    auto transcribe_seg = [&](int64_t end_sample) -> bool {
        if (seg.empty()) return true;
        int min_samples = (sr * cfg.min_segment_ms) / 1000;
        if ((int)seg.size() < min_samples) return true;

        std::string seg_text;
        if (!transcribe_pcm_server(seg.data(), (int)seg.size(), asr_cfg,
                                   seg_text, err)) {
            return false;
        }
        if (!seg_text.empty()) {
            AsrSegment out;
            out.index = seg_index;
            out.start_s = (float)((double)seg_start_sample / sr);
            out.end_s = (float)((double)end_sample / sr);
            out.text = seg_text;
            segments.push_back(out);
            append_text(seg_text);
        }
        return true;
    };

    auto finalize_seg = [&](int64_t end_sample) -> bool {
        if (!transcribe_seg(end_sample)) return false;
        in_speech = false;
        pending_end = false;
        pending_silence_samples = 0;
        seg_index++;
        seg.clear();
        return true;
    };

    auto force_split_seg = [&](int64_t end_sample) -> bool {
        if (!finalize_seg(end_sample)) return false;
        in_speech = true;
        seg_start_sample = end_sample;
        return true;
    };

    auto handle_frames = [&]() {
        rs_vad_frame_t frames[64];
        int nf;
        while ((nf = rs_vad_drain_frames(g_vad, frames, 64)) > 0) {
            for (int i = 0; i < nf; i++) {
                if (frames[i].is_speech_start) {
                    if (pending_end) {
                        pending_end = false;
                        pending_silence_samples = 0;
                    } else if (!in_speech) {
                        in_speech = true;
                        seg.clear();
                        seg_start_sample = total_samples;
                    }
                }
                if (frames[i].is_speech_end && in_speech) {
                    pending_end = true;
                    pending_silence_samples = 0;
                }
            }
            if (nf < 64) break;
        }
    };

    for (size_t offset = 0; offset < pcm.size();) {
        int n = (int)std::min<size_t>((size_t)chunk_samples, pcm.size() - offset);
        const float *chunk = pcm.data() + offset;
        if (rs_vad_push_audio(g_vad, chunk, n) != RS_OK) {
            err = "rs_vad_push_audio failed";
            return false;
        }
        handle_frames();
        if (in_speech) {
            seg.insert(seg.end(), chunk, chunk + n);
            if (pending_end) pending_silence_samples += n;
        }

        total_samples += n;
        if (pending_end &&
            pending_silence_samples >= (sr * cfg.end_silence_ms) / 1000) {
            if (!finalize_seg(total_samples)) return false;
        }
        if (in_speech && cfg.max_segment_ms > 0 &&
            seg.size() >= (size_t)((sr * cfg.max_segment_ms) / 1000)) {
            if (pending_end) {
                if (!finalize_seg(total_samples)) return false;
            } else if (!force_split_seg(total_samples)) {
                return false;
            }
        }
        offset += n;
    }
    if (in_speech) {
        if (!finalize_seg(total_samples)) return false;
    }
    return true;
}

static bool detect_pcm_vad_segments_locked(const std::vector<float> &pcm,
                                           const VadServerConfig &cfg,
                                           std::vector<AsrSegment> &segments,
                                           std::string &err) {
    if (rs_vad_reset(g_vad) != RS_OK) {
        err = "rs_vad_reset failed";
        return false;
    }
    if (rs_vad_set_threshold(g_vad, cfg.threshold) != RS_OK) {
        err = "rs_vad_set_threshold failed";
        return false;
    }

    const int sr = g_asr.sample_rate > 0 ? g_asr.sample_rate : 16000;
    const int chunk_samples = std::max(1, sr / 10);
    std::vector<float> seg;
    bool in_speech = false;
    bool pending_end = false;
    int pending_silence_samples = 0;
    int seg_index = 0;
    int64_t seg_start_sample = 0;
    int64_t total_samples = 0;

    auto push_segment = [&](int64_t end_sample) {
        if (seg.empty()) return true;
        int min_samples = (sr * cfg.min_segment_ms) / 1000;
        if ((int)seg.size() < min_samples) return true;
        AsrSegment out;
        out.index = seg_index;
        out.start_s = (float)((double)seg_start_sample / sr);
        out.end_s = (float)((double)end_sample / sr);
        out.pcm = seg;
        segments.push_back(std::move(out));
        return true;
    };

    auto finalize_seg = [&](int64_t end_sample) {
        if (!push_segment(end_sample)) return false;
        in_speech = false;
        pending_end = false;
        pending_silence_samples = 0;
        seg_index++;
        seg.clear();
        return true;
    };

    auto force_split_seg = [&](int64_t end_sample) {
        if (!finalize_seg(end_sample)) return false;
        in_speech = true;
        seg_start_sample = end_sample;
        return true;
    };

    auto handle_frames = [&]() {
        rs_vad_frame_t frames[64];
        int nf;
        while ((nf = rs_vad_drain_frames(g_vad, frames, 64)) > 0) {
            for (int i = 0; i < nf; i++) {
                if (frames[i].is_speech_start) {
                    if (pending_end) {
                        pending_end = false;
                        pending_silence_samples = 0;
                    } else if (!in_speech) {
                        in_speech = true;
                        seg.clear();
                        seg_start_sample = total_samples;
                    }
                }
                if (frames[i].is_speech_end && in_speech) {
                    pending_end = true;
                    pending_silence_samples = 0;
                }
            }
            if (nf < 64) break;
        }
    };

    for (size_t offset = 0; offset < pcm.size();) {
        int n = (int)std::min<size_t>((size_t)chunk_samples, pcm.size() - offset);
        const float *chunk = pcm.data() + offset;
        if (rs_vad_push_audio(g_vad, chunk, n) != RS_OK) {
            err = "rs_vad_push_audio failed";
            return false;
        }
        handle_frames();
        if (in_speech) {
            seg.insert(seg.end(), chunk, chunk + n);
            if (pending_end) pending_silence_samples += n;
        }

        total_samples += n;
        if (pending_end &&
            pending_silence_samples >= (sr * cfg.end_silence_ms) / 1000) {
            if (!finalize_seg(total_samples)) return false;
        }
        if (in_speech && cfg.max_segment_ms > 0 &&
            seg.size() >= (size_t)((sr * cfg.max_segment_ms) / 1000)) {
            if (pending_end) {
                if (!finalize_seg(total_samples)) return false;
            } else if (!force_split_seg(total_samples)) {
                return false;
            }
        }
        offset += n;
    }
    if (in_speech) {
        if (!finalize_seg(total_samples)) return false;
    }
    return true;
}

static bool transcribe_detected_segments(std::vector<AsrSegment> &segments,
                                         const AsrServerConfig &cfg,
                                         std::string &text,
                                         std::string &err) {
    text.clear();
    if (segments.empty()) return true;

    if (g_asr_batch) {
        std::vector<std::future<std::pair<std::string, std::string>>> futures;
        futures.reserve(segments.size());
        for (const auto &segment : segments) {
            futures.push_back(g_asr_batch->Submit(segment.pcm, cfg));
        }
        for (size_t i = 0; i < futures.size(); ++i) {
            auto result = futures[i].get();
            if (!result.second.empty()) {
                err = result.second;
                return false;
            }
            segments[i].text = std::move(result.first);
        }
    } else {
        for (auto &segment : segments) {
            if (!transcribe_pcm_server(segment.pcm.data(), (int)segment.pcm.size(),
                                       cfg, segment.text, err)) {
                return false;
            }
        }
    }

    std::vector<AsrSegment> non_empty;
    non_empty.reserve(segments.size());
    for (auto &segment : segments) {
        if (segment.text.empty()) continue;
        if (!text.empty()) text += "\n";
        text += segment.text;
        segment.index = (int)non_empty.size();
        non_empty.push_back(std::move(segment));
    }
    segments = std::move(non_empty);
    return true;
}

static bool do_transcribe_with_config(const std::string &audio_bytes,
                                      const AsrServerConfig &cfg,
                                      std::string &text, std::string &err) {
    if (!g_have_asr) { err = "no ASR model loaded"; return false; }
    std::vector<float> pcm;
    if (!decode_audio(audio_bytes, g_asr.sample_rate, pcm)) {
        err = "could not decode audio (expected WAV)";
        return false;
    }
    return transcribe_pcm_server(pcm.data(), (int)pcm.size(), cfg, text, err);
}

// Transcribe WAV bytes → text (HTTP/MCP path). Decodes then locks + transcribes.
static bool do_transcribe(const std::string &audio_bytes, std::string &text,
                          std::string &err) {
    return do_transcribe_with_config(audio_bytes, g_asr_cfg, text, err);
}

static bool do_transcribe_vad(const std::string &audio_bytes,
                              const AsrServerConfig &asr_cfg,
                              const VadServerConfig &vad_cfg,
                              const SpeakerServerConfig &spk_cfg,
                              std::vector<AsrSegment> &segments,
                              std::string &text, std::string &err) {
    if (!g_have_asr) { err = "no ASR model loaded"; return false; }
    if (!g_have_vad) { err = "no VAD model loaded (use --vad-model)"; return false; }
    if (spk_cfg.enabled && !g_have_speaker) {
        err = "speaker=true requires --spk-model";
        return false;
    }
    std::vector<float> pcm;
    if (!decode_audio(audio_bytes, g_asr.sample_rate, pcm)) {
        err = "could not decode audio (expected WAV)";
        return false;
    }
    {
        std::unique_lock<std::mutex> vlk(g_vad_mu);
        if (!detect_pcm_vad_segments_locked(pcm, vad_cfg, segments, err)) {
            return false;
        }
    }
    if (!transcribe_detected_segments(segments, asr_cfg, text, err)) {
        return false;
    }
    if (!apply_speaker_clustering(segments, spk_cfg, err)) {
        return false;
    }
    write_speaker_log(segments, spk_cfg);
    return true;
}

// Synthesize text → mono float PCM at g_tts.sample_rate.
static bool do_synthesize(const std::string &text, const std::string &instruct,
                          const std::string &language, int seed, int steps,
                          std::vector<float> &pcm, std::string &err) {
    if (!g_have_tts) { err = "no TTS model loaded"; return false; }

    if (g_voxcpm2_batch) {
        auto fut = g_voxcpm2_batch->Submit(
            text, instruct, language, seed, steps > 0 ? steps : 8);
        auto result = fut.get();
        if (!result.second.empty()) { err = result.second; return false; }
        pcm = std::move(result.first);
        return true;
    }

    // CosyVoice3 with --tts-slots > 1: route through the continuous-batching
    // scheduler so N requests decode concurrently (token-interleaved).
    if (g_tts_batch) {
        auto fut = g_tts_batch->Submit(text, instruct, language, seed);
        auto result = fut.get();
        if (!result.second.empty()) { err = result.second; return false; }
        pcm = std::move(result.first);
        return true;
    }

    std::lock_guard<std::mutex> lk(g_tts.mu);
    rs_reset(g_tts.ctx);
    rs_set_tts_params(g_tts.ctx, instruct.c_str(), language.c_str(), seed);
    if (steps > 0) rs_set_tts_diffusion_steps(g_tts.ctx, steps);
    if (rs_push_text(g_tts.ctx, text.c_str()) != RS_OK) {
        err = "rs_push_text failed";
        return false;
    }
    int32_t ret;
    while ((ret = rs_process(g_tts.ctx)) >= 0) {
        float *chunk = nullptr;
        int n = rs_get_audio_output(g_tts.ctx, &chunk);
        if (n > 0 && chunk) pcm.insert(pcm.end(), chunk, chunk + n);
        if (ret == 0) break;
    }
    if (ret < 0) { err = rs_get_last_error().message; return false; }
    return true;
}

// ── OpenAI HTTP handlers ────────────────────────────────────────────────

static void send_json(httplib::Response &res, int status, const rsjson::Value &v) {
    res.status = status;
    res.set_content(v.dump(), "application/json");
}
static void send_error(httplib::Response &res, int status, const std::string &msg,
                       const std::string &type = "invalid_request_error") {
    rsjson::Value e = rsjson::Value::object();
    e["error"]["message"] = msg;
    e["error"]["type"] = type;
    send_json(res, status, e);
}

static std::string read_text_file(const fs::path &path, std::string &err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = "could not open " + path.string();
        return "";
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static const rsjson::Value *object_get(const rsjson::Value &obj,
                                       const std::string &key) {
    if (!obj.is_object()) return nullptr;
    for (const auto &kv : obj.members()) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool looks_like_path(const std::string &s) {
    return s.find('/') != std::string::npos ||
           s.find('\\') != std::string::npos ||
           ends_with(s, ".gguf");
}

static fs::path expand_user_path(const std::string &s) {
    if (s == "~" || s.rfind("~/", 0) == 0) {
        const char *home = std::getenv(
#ifdef _WIN32
            "USERPROFILE"
#else
            "HOME"
#endif
        );
        if (home) {
            return s == "~" ? fs::path(home) : fs::path(home) / s.substr(2);
        }
    }
    return fs::path(s);
}

static fs::path rapidspeech_cache_home() {
    if (const char *p = std::getenv("RAPIDSPEECH_HOME")) {
        return expand_user_path(p);
    }
    const char *home = std::getenv(
#ifdef _WIN32
        "USERPROFILE"
#else
        "HOME"
#endif
    );
    if (!home || !home[0]) return fs::path(".rapidspeech-cache");
    return fs::path(home) / ".cache" / "rapidspeech";
}

static fs::path rapidspeech_models_dir() {
    return rapidspeech_cache_home() / "models";
}

static std::string basename_from_url_path(const std::string &path) {
    size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

static bool cached_file_ok(const std::string &filename, int64_t size_bytes) {
    std::error_code ec;
    fs::path path = rapidspeech_models_dir() / filename;
    if (!fs::is_regular_file(path, ec)) return false;
    if (size_bytes <= 0) return true;
    return (int64_t)fs::file_size(path, ec) == size_bytes && !ec;
}

static bool load_registry_root(rsjson::Value &root, std::string &err) {
    if (g_registry_path.empty()) {
        err = "model registry path is not configured";
        return false;
    }
    std::string text = read_text_file(g_registry_path, err);
    if (!err.empty()) return false;
    try {
        root = rsjson::parse(text);
    } catch (const std::exception &e) {
        err = std::string("could not parse model registry: ") + e.what();
        return false;
    }
    if (!root.is_object() || !root.contains("models")) {
        err = "model registry is missing the models object";
        return false;
    }
    return true;
}

struct ResolvedRegistryModel {
    std::string task;
    std::string name;
    std::string variant;
    std::string canonical_spec;
    std::string filename;
    std::string path;
    int64_t size_bytes = 0;
    std::string known_issues;
    std::string default_voice;
    std::string voice_filename;
    std::string voice_path;
    int64_t voice_size_bytes = 0;
    std::vector<RegistryDependency> dependencies;
};

static bool file_entry_info(const rsjson::Value &entry, std::string &filename,
                            int64_t &size_bytes, std::string &err) {
    const rsjson::Value *files = object_get(entry, "files");
    if (!files || !files->is_object() || files->members().empty()) {
        err = "registry entry has no files";
        return false;
    }
    const std::string rel = files->members().front().second.as_string();
    if (rel.empty()) {
        err = "registry entry has an empty file path";
        return false;
    }
    filename = basename_from_url_path(rel);
    size_bytes = (int64_t)entry["size_bytes"].as_number(0);
    return true;
}

static bool split_model_spec(const std::string &spec, std::string &name,
                             std::string &variant) {
    size_t pos = spec.find(':');
    if (pos == std::string::npos) {
        name = spec;
        variant.clear();
    } else {
        name = spec.substr(0, pos);
        variant = spec.substr(pos + 1);
    }
    while (!name.empty() && std::isspace((unsigned char)name.front()))
        name.erase(name.begin());
    while (!name.empty() && std::isspace((unsigned char)name.back()))
        name.pop_back();
    while (!variant.empty() && std::isspace((unsigned char)variant.front()))
        variant.erase(variant.begin());
    while (!variant.empty() && std::isspace((unsigned char)variant.back()))
        variant.pop_back();
    return !name.empty();
}

static bool resolve_registry_model(const std::string &spec,
                                   const std::string &requested_task,
                                   const std::string &voice,
                                   ResolvedRegistryModel &out,
                                   std::string &err) {
    out = ResolvedRegistryModel{};
    rsjson::Value root;
    if (!load_registry_root(root, err)) return false;

    std::string name, variant;
    if (spec.empty()) {
        const rsjson::Value &defaults = root["defaults"];
        name = defaults[requested_task].as_string();
        if (name.empty()) {
            err = "registry has no default model for task '" + requested_task + "'";
            return false;
        }
    } else if (!split_model_spec(spec, name, variant)) {
        err = "empty model spec";
        return false;
    }

    const rsjson::Value *model = object_get(root["models"], name);
    if (!model) {
        err = "unknown model '" + name + "'";
        return false;
    }
    out.task = (*model)["task"].as_string();
    if (!requested_task.empty() && out.task != requested_task) {
        err = "model '" + name + "' is task '" + out.task +
              "', not '" + requested_task + "'";
        return false;
    }
    if (variant.empty()) variant = (*model)["default_variant"].as_string();
    const rsjson::Value *variants = object_get(*model, "variants");
    const rsjson::Value *var = variants ? object_get(*variants, variant) : nullptr;
    if (!var) {
        err = "model '" + name + "' has no variant '" + variant + "'";
        return false;
    }

    out.name = name;
    out.variant = variant;
    out.canonical_spec = name + ":" + variant;
    out.known_issues = (*var)["known_issues"].as_string();
    if (!file_entry_info(*var, out.filename, out.size_bytes, err)) return false;
    out.path = (rapidspeech_models_dir() / out.filename).string();

    if (out.task == "tts" && model->contains("voices")) {
        out.default_voice = voice.empty() ? (*model)["default_voice"].as_string()
                                          : voice;
        const rsjson::Value *voices = object_get(*model, "voices");
        const rsjson::Value *voice_entry =
            (voices && !out.default_voice.empty())
                ? object_get(*voices, out.default_voice)
                : nullptr;
        if (voice_entry) {
            if (!file_entry_info(*voice_entry, out.voice_filename,
                                 out.voice_size_bytes, err)) {
                return false;
            }
            out.voice_path =
                (rapidspeech_models_dir() / out.voice_filename).string();
        }
    }

    const rsjson::Value *deps = object_get(*model, "dependencies");
    if (deps && deps->is_object()) {
        for (const auto &kv : deps->members()) {
            RegistryDependency dep;
            dep.name = kv.first;
            dep.env = kv.second["env"].as_string();
            if (!file_entry_info(kv.second, dep.filename, dep.size_bytes, err)) {
                err = "dependency '" + dep.name + "': " + err;
                return false;
            }
            dep.path = (rapidspeech_models_dir() / dep.filename).string();
            out.dependencies.push_back(dep);
        }
    }
    return true;
}

static std::string shell_quote(const std::string &s) {
#ifdef _WIN32
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
#else
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
#endif
}

static bool run_command_capture(const std::string &cmd, std::string &output,
                                int &status) {
    output.clear();
    std::string full = cmd + " 2>&1";
    FILE *pipe = RS_POPEN(full.c_str(), "r");
    if (!pipe) {
        status = -1;
        output = "failed to start command";
        return false;
    }
    std::array<char, 4096> buf{};
    while (fgets(buf.data(), (int)buf.size(), pipe)) {
        output += buf.data();
    }
    status = RS_PCLOSE(pipe);
    return status == 0;
}

static bool ensure_registry_model_downloaded(ResolvedRegistryModel &model,
                                             bool &downloaded,
                                             std::string &err) {
    downloaded = false;
    bool model_ok = cached_file_ok(model.filename, model.size_bytes);
    bool voice_ok = model.voice_filename.empty() ||
                    cached_file_ok(model.voice_filename,
                                   model.voice_size_bytes);
    bool deps_ok = true;
    for (const auto &dep : model.dependencies) {
        deps_ok = deps_ok && cached_file_ok(dep.filename, dep.size_bytes);
    }
    if (model_ok && voice_ok && deps_ok) return true;
    if (g_python_executable.empty()) {
        err = "model is not cached and RAPIDSPEECH_PYTHON_EXECUTABLE is not set";
        return false;
    }
    std::string cmd = shell_quote(g_python_executable) +
        " -m rapidspeech.python.cli pull " + shell_quote(model.canonical_spec);
    std::string output;
    int status = 0;
    fprintf(stderr, "[server] downloading model via: %s\n", cmd.c_str());
    if (!run_command_capture(cmd, output, status)) {
        err = "download failed for " + model.canonical_spec +
              " (exit status " + std::to_string(status) + "):\n" + output;
        return false;
    }
    model_ok = cached_file_ok(model.filename, model.size_bytes);
    voice_ok = model.voice_filename.empty() ||
               cached_file_ok(model.voice_filename, model.voice_size_bytes);
    deps_ok = true;
    for (const auto &dep : model.dependencies) {
        deps_ok = deps_ok && cached_file_ok(dep.filename, dep.size_bytes);
    }
    if (!model_ok || !voice_ok || !deps_ok) {
        err = "download finished but expected files are still missing";
        return false;
    }
    downloaded = true;
    return true;
}

static void add_registry_file_status(rsjson::Value &dst,
                                     const rsjson::Value &entry,
                                     const std::string &spec) {
    std::string filename;
    int64_t size_bytes = 0;
    std::string err;
    if (file_entry_info(entry, filename, size_bytes, err)) {
        dst["filename"] = filename;
        dst["size_bytes"] = size_bytes;
        dst["cached"] = cached_file_ok(filename, size_bytes);
    } else {
        dst["cached"] = false;
    }
    if (!spec.empty()) dst["spec"] = spec;
    std::string issue = entry["known_issues"].as_string();
    if (!issue.empty()) dst["known_issues"] = issue;
}

static void handle_model_registry(const httplib::Request &,
                                  httplib::Response &res) {
    rsjson::Value root;
    std::string err;
    if (!load_registry_root(root, err)) {
        send_error(res, 503, err, "server_error");
        return;
    }
    rsjson::Value out = rsjson::Value::object();
    out["registry_path"] = g_registry_path;
    out["cache_dir"] = rapidspeech_models_dir().string();
    out["download_enabled"] = !g_python_executable.empty();
    out["defaults"] = root["defaults"];
    out["loaded"]["asr"] = g_have_asr ? g_current_asr_model_spec : "";
    out["loaded"]["tts"] = g_have_tts ? g_current_tts_model_spec : "";
    out["loaded"]["vad"] = g_have_vad ? g_current_vad_model_spec : "";
    out["loaded"]["speaker"] =
        g_have_speaker ? g_current_speaker_model_spec : "";

    rsjson::Value models = rsjson::Value::array();
    for (const auto &kv : root["models"].members()) {
        const std::string &name = kv.first;
        const rsjson::Value &entry = kv.second;
        rsjson::Value item = rsjson::Value::object();
        item["id"] = name;
        item["task"] = entry["task"].as_string();
        item["description"] = entry["description"].as_string();
        item["default_variant"] = entry["default_variant"].as_string();
        item["default_voice"] = entry["default_voice"].as_string();
        item["loaded"] =
            ((item["task"].as_string() == "asr" &&
              g_current_asr_model_spec.rfind(name + ":", 0) == 0) ||
             (item["task"].as_string() == "tts" &&
              g_current_tts_model_spec.rfind(name + ":", 0) == 0) ||
             (item["task"].as_string() == "vad" &&
              g_current_vad_model_spec.rfind(name + ":", 0) == 0) ||
             (item["task"].as_string() == "speaker" &&
              g_current_speaker_model_spec.rfind(name + ":", 0) == 0));

        rsjson::Value vars = rsjson::Value::array();
        const rsjson::Value *variants = object_get(entry, "variants");
        if (variants) {
            for (const auto &vv : variants->members()) {
                rsjson::Value v = rsjson::Value::object();
                v["id"] = vv.first;
                v["default"] = vv.first == item["default_variant"].as_string();
                add_registry_file_status(v, vv.second, name + ":" + vv.first);
                vars.push_back(v);
            }
        }
        item["variants"] = vars;

        rsjson::Value deps = rsjson::Value::array();
        const rsjson::Value *dep_entries = object_get(entry, "dependencies");
        if (dep_entries) {
            for (const auto &vv : dep_entries->members()) {
                rsjson::Value d = rsjson::Value::object();
                d["id"] = vv.first;
                d["env"] = vv.second["env"].as_string();
                add_registry_file_status(d, vv.second, "");
                deps.push_back(d);
            }
        }
        item["dependencies"] = deps;

        rsjson::Value voices = rsjson::Value::array();
        const rsjson::Value *voice_entries = object_get(entry, "voices");
        if (voice_entries) {
            for (const auto &vv : voice_entries->members()) {
                rsjson::Value v = rsjson::Value::object();
                v["id"] = vv.first;
                v["default"] = vv.first == item["default_voice"].as_string();
                add_registry_file_status(v, vv.second, "");
                voices.push_back(v);
            }
        }
        item["voices"] = voices;
        models.push_back(item);
    }
    out["models"] = models;
    send_json(res, 200, out);
}

static void handle_model_load(const httplib::Request &req,
                              httplib::Response &res) {
    rsjson::Value body;
    try {
        body = req.body.empty() ? rsjson::Value::object()
                                : rsjson::parse(req.body);
    } catch (const std::exception &e) {
        send_error(res, 400, e.what());
        return;
    }
    std::string task = body["task"].as_string();
    std::string model_spec = body["model"].as_string();
    std::string voice = body["voice"].as_string();
    std::string err;
    std::string model_path;
    std::string voice_path;
    std::string canonical_spec = model_spec;
    std::vector<RegistryDependency> dependencies;
    bool downloaded = false;

    if (looks_like_path(model_spec)) {
        if (task.empty()) {
            send_error(res, 400, "task is required when loading a direct .gguf path");
            return;
        }
        fs::path p = expand_user_path(model_spec);
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) {
            send_error(res, 404, "model file not found: " + p.string());
            return;
        }
        model_path = p.string();
    } else {
        if (task.empty() && model_spec.empty()) {
            send_error(res, 400, "task or model is required");
            return;
        }
        ResolvedRegistryModel resolved;
        if (!resolve_registry_model(model_spec, task, voice, resolved, err)) {
            send_error(res, 400, err);
            return;
        }
        if (!ensure_registry_model_downloaded(resolved, downloaded, err)) {
            send_error(res, 500, err, "server_error");
            return;
        }
        task = resolved.task;
        model_path = resolved.path;
        voice_path = resolved.voice_path;
        canonical_spec = resolved.canonical_spec;
        dependencies = resolved.dependencies;
    }

    bool ok = false;
    if (task == "asr") {
        ok = load_asr_runtime_model(model_path, canonical_spec, err);
    } else if (task == "tts") {
        ok = load_tts_runtime_model(model_path, canonical_spec, voice_path,
                                    dependencies, err);
    } else if (task == "vad") {
        ok = load_vad_runtime_model(model_path, canonical_spec, err);
    } else if (task == "speaker") {
        ok = load_speaker_runtime_model(model_path, canonical_spec, err);
    } else {
        send_error(res, 400, "unsupported model task: " + task);
        return;
    }
    if (!ok) {
        send_error(res, 500, err, "server_error");
        return;
    }

    rsjson::Value out = rsjson::Value::object();
    out["ok"] = true;
    out["task"] = task;
    out["model"] = canonical_spec;
    out["path"] = model_path;
    out["downloaded"] = downloaded;
    if (!voice_path.empty()) out["voice_path"] = voice_path;
    if (task == "asr") out["arch"] = g_asr.arch;
    if (task == "tts") out["arch"] = g_tts.arch;
    send_json(res, 200, out);
}

static bool request_has_value(const httplib::Request &req, const char *key) {
    return req.has_file(key) || req.has_param(key);
}

static std::string request_value(const httplib::Request &req, const char *key,
                                 const std::string &current = "") {
    if (req.has_file(key)) return req.get_file_value(key).content;
    if (req.has_param(key)) return req.get_param_value(key);
    return current;
}

static bool request_bool(const httplib::Request &req, const char *key,
                         bool current) {
    if (!request_has_value(req, key)) return current;
    std::string v = request_value(req, key);
    return parse_bool_value(v.empty() ? "1" : v, current);
}

static float request_float(const httplib::Request &req, const char *key,
                           float current, float lo, float hi) {
    if (!request_has_value(req, key)) return current;
    std::string v = request_value(req, key);
    char *end = nullptr;
    float parsed = std::strtof(v.c_str(), &end);
    if (end == v.c_str()) return current;
    return std::max(lo, std::min(hi, parsed));
}

static int request_int(const httplib::Request &req, const char *key,
                       int current, int lo, int hi) {
    if (!request_has_value(req, key)) return current;
    std::string v = request_value(req, key);
    char *end = nullptr;
    long parsed = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str()) return current;
    return std::max(lo, std::min(hi, (int)parsed));
}

static AsrServerConfig request_asr_config(const httplib::Request &req) {
    AsrServerConfig cfg = g_asr_cfg;
    cfg.two_pass = request_bool(req, "two_pass", cfg.two_pass);
    cfg.ctc_precheck = request_bool(req, "ctc_precheck", cfg.ctc_precheck);
    cfg.ctc_only = request_bool(req, "ctc_only", cfg.ctc_only);
    return cfg;
}

static VadServerConfig request_vad_config(const httplib::Request &req) {
    VadServerConfig cfg = g_vad_cfg;
    cfg.threshold = request_float(req, "vad_threshold", cfg.threshold, 0.0f, 1.0f);
    cfg.threshold = request_float(req, "threshold", cfg.threshold, 0.0f, 1.0f);
    cfg.end_silence_ms = request_int(req, "vad_end_silence_ms", cfg.end_silence_ms, 0, 60000);
    cfg.end_silence_ms = request_int(req, "end_silence_ms", cfg.end_silence_ms, 0, 60000);
    cfg.end_silence_ms = request_int(req, "silence_ms", cfg.end_silence_ms, 0, 60000);
    cfg.partial_ms = request_int(req, "vad_partial_ms", cfg.partial_ms, 0, 60000);
    cfg.partial_ms = request_int(req, "partial_ms", cfg.partial_ms, 0, 60000);
    cfg.min_segment_ms = request_int(req, "vad_min_segment_ms", cfg.min_segment_ms, 0, 60000);
    cfg.min_segment_ms = request_int(req, "min_segment_ms", cfg.min_segment_ms, 0, 60000);
    cfg.max_segment_ms = request_int(req, "vad_max_segment_ms", cfg.max_segment_ms, 0, 3600000);
    cfg.max_segment_ms = request_int(req, "max_segment_ms", cfg.max_segment_ms, 0, 3600000);
    return cfg;
}

static SpeakerServerConfig request_speaker_config(const httplib::Request &req) {
    SpeakerServerConfig cfg;
    cfg.enabled = request_bool(req, "speaker", false);
    cfg.enabled = request_bool(req, "diarization", cfg.enabled);
    cfg.enabled = request_bool(req, "speaker_diarization", cfg.enabled);
    cfg.enabled = request_bool(req, "spk", cfg.enabled);
    bool explicit_log = request_has_value(req, "speaker_log");
    cfg.log = request_bool(req, "speaker_log", !g_speaker_log_path.empty());
    if (explicit_log && cfg.log) {
        cfg.enabled = true;
    }
    cfg.cluster = request_value(req, "spk_cluster", g_speaker_cluster);
    cfg.cluster = request_value(req, "speaker_cluster", cfg.cluster);
    if (cfg.cluster != "ahc" && cfg.cluster != "spectral") {
        cfg.cluster = "spectral";
    }
    cfg.threshold = request_float(req, "spk_threshold", g_speaker_threshold,
                                  -1.0f, 1.0f);
    cfg.threshold = request_float(req, "speaker_threshold", cfg.threshold,
                                  -1.0f, 1.0f);
    cfg.num_speakers = request_int(req, "num_speakers",
                                   g_speaker_num_speakers, 0, 256);
    cfg.max_speakers = request_int(req, "max_speakers",
                                   g_speaker_max_speakers, 1, 256);
    return cfg;
}

static bool request_wants_vad(const httplib::Request &req) {
    if (request_has_value(req, "vad")) return request_bool(req, "vad", false);
    if (request_has_value(req, "use_vad")) return request_bool(req, "use_vad", false);
    if (request_has_value(req, "vad_segments"))
        return request_bool(req, "vad_segments", false);
    return request_has_value(req, "vad_threshold") ||
           request_has_value(req, "vad_end_silence_ms") ||
           request_has_value(req, "vad_min_segment_ms") ||
           request_has_value(req, "vad_max_segment_ms");
}

static void handle_transcriptions(const httplib::Request &req, httplib::Response &res) {
    if (!g_have_asr) { send_error(res, 503, "no ASR model loaded"); return; }
    if (!req.has_file("file")) {
        send_error(res, 400, "missing 'file' (multipart/form-data)");
        return;
    }
    const auto &file = req.get_file_value("file");
    std::string text, err;
    std::vector<AsrSegment> segments;
    AsrServerConfig asr_cfg = request_asr_config(req);
    VadServerConfig vad_cfg = request_vad_config(req);
    SpeakerServerConfig spk_cfg = request_speaker_config(req);
    bool use_vad = request_wants_vad(req) || spk_cfg.enabled;
    bool ok = use_vad
                  ? do_transcribe_vad(file.content, asr_cfg, vad_cfg,
                                      spk_cfg, segments, text, err)
                  : do_transcribe_with_config(file.content, asr_cfg, text, err);
    if (!ok) {
        send_error(res, 400, err, "server_error");
        return;
    }
    std::string fmt = request_value(req, "response_format", "json");
    if (fmt == "text") {
        res.status = 200;
        res.set_content(text, "text/plain; charset=utf-8");
    } else {
        rsjson::Value o = rsjson::Value::object();
        o["text"] = text;
        if (use_vad) {
            o["vad"] = true;
            rsjson::Value arr = rsjson::Value::array();
            for (const auto &s : segments) {
                rsjson::Value item = rsjson::Value::object();
                item["id"] = s.index;
                item["start"] = s.start_s;
                item["end"] = s.end_s;
                item["text"] = s.text;
                if (s.speaker >= 0) {
                    item["speaker"] = speaker_name(s.speaker);
                    item["speaker_id"] = s.speaker;
                }
                arr.push_back(item);
            }
            o["segments"] = arr;
            if (spk_cfg.enabled) {
                o["speaker"] = true;
                o["speaker_cluster"] = spk_cfg.cluster;
                int n_speakers = 0;
                for (const auto &s : segments) {
                    if (s.speaker >= 0) n_speakers = std::max(n_speakers, s.speaker + 1);
                }
                o["num_speakers"] = n_speakers;
            }
        }
        send_json(res, 200, o);
    }
}

static void handle_speech(const httplib::Request &req, httplib::Response &res) {
    if (!g_have_tts) { send_error(res, 503, "no TTS model loaded"); return; }
    rsjson::Value body;
    try { body = rsjson::parse(req.body); }
    catch (const std::exception &e) { send_error(res, 400, e.what()); return; }

    std::string text = body["input"].as_string();
    if (text.empty()) { send_error(res, 400, "missing 'input'"); return; }
    // OpenAI's 'voice' maps to our instruct/voice description; extra knobs are
    // accepted as passthrough fields.
    std::string instruct = body.contains("voice") ? body["voice"].as_string("male") : "male";
    if (body.contains("instruct")) instruct = body["instruct"].as_string(instruct);
    std::string language = body["language"].as_string("English");
    int seed = body.contains("seed") ? body["seed"].as_int(42) : 42;
    int steps = body.contains("n_steps") ? body["n_steps"].as_int(0) : 0;

    std::vector<float> pcm;
    std::string err;
    if (!do_synthesize(text, instruct, language, seed, steps, pcm, err)) {
        send_error(res, 400, err, "server_error");
        return;
    }
    std::string wav = wav_encode(pcm, g_tts.sample_rate);
    res.status = 200;
    res.set_content(wav, "audio/wav");
}

static void handle_models(const httplib::Request &, httplib::Response &res) {
    rsjson::Value o = rsjson::Value::object();
    o["object"] = "list";
    rsjson::Value data = rsjson::Value::array();
    auto add = [&](const std::string &id, const std::string &arch) {
        rsjson::Value m = rsjson::Value::object();
        m["id"] = id; m["object"] = "model"; m["owned_by"] = "rapidspeech";
        m["arch"] = arch;
        data.push_back(m);
    };
    if (g_have_asr) add("rapidspeech-asr", g_asr.arch);
    if (g_have_tts) add("rapidspeech-tts", g_tts.arch);
    if (g_have_speaker) add("rapidspeech-speaker", "campplus");
    o["data"] = data;
    send_json(res, 200, o);
}

// ── MCP (JSON-RPC 2.0) ──────────────────────────────────────────────────

static rsjson::Value mcp_tools_list() {
    rsjson::Value tools = rsjson::Value::array();
    if (g_have_asr) {
        rsjson::Value t = rsjson::Value::object();
        t["name"] = "transcribe";
        t["description"] = "Transcribe a local audio file (WAV) to text using the ASR model.";
        rsjson::Value schema = rsjson::Value::object();
        schema["type"] = "object";
        rsjson::Value props = rsjson::Value::object();
        rsjson::Value path = rsjson::Value::object();
        path["type"] = "string";
        path["description"] = "Absolute path to a WAV audio file.";
        props["path"] = path;
        schema["properties"] = props;
        rsjson::Value req = rsjson::Value::array();
        req.push_back("path");
        schema["required"] = req;
        t["inputSchema"] = schema;
        tools.push_back(t);
    }
    if (g_have_tts) {
        rsjson::Value t = rsjson::Value::object();
        t["name"] = "synthesize";
        t["description"] = "Synthesize speech from text and write a WAV file; returns the path.";
        rsjson::Value schema = rsjson::Value::object();
        schema["type"] = "object";
        rsjson::Value props = rsjson::Value::object();
        auto strprop = [&](const char *desc) {
            rsjson::Value p = rsjson::Value::object();
            p["type"] = "string"; p["description"] = desc; return p;
        };
        props["text"] = strprop("Text to synthesize.");
        props["output_path"] = strprop("Where to write the WAV (optional; a temp file is used if omitted).");
        props["voice"] = strprop("Voice / speaker description (optional).");
        props["language"] = strprop("Language name, e.g. English / Chinese (optional).");
        schema["properties"] = props;
        rsjson::Value req = rsjson::Value::array();
        req.push_back("text");
        schema["required"] = req;
        t["inputSchema"] = schema;
        tools.push_back(t);
    }
    rsjson::Value r = rsjson::Value::object();
    r["tools"] = tools;
    return r;
}

// Build an MCP tools/call result with a single text block.
static rsjson::Value mcp_text_result(const std::string &text, bool is_error = false) {
    rsjson::Value r = rsjson::Value::object();
    rsjson::Value content = rsjson::Value::array();
    rsjson::Value block = rsjson::Value::object();
    block["type"] = "text";
    block["text"] = text;
    content.push_back(block);
    r["content"] = content;
    r["isError"] = is_error;
    return r;
}

static rsjson::Value mcp_call_tool(const std::string &name, const rsjson::Value &args) {
    if (name == "transcribe") {
        std::string path = args["path"].as_string();
        if (path.empty()) return mcp_text_result("error: 'path' is required", true);
        std::ifstream f(path, std::ios::binary);
        if (!f) return mcp_text_result("error: cannot open file: " + path, true);
        std::string bytes((std::istreambuf_iterator<char>(f)), {});
        std::string text, err;
        if (!do_transcribe(bytes, text, err))
            return mcp_text_result("error: " + err, true);
        return mcp_text_result(text);
    }
    if (name == "synthesize") {
        std::string text = args["text"].as_string();
        if (text.empty()) return mcp_text_result("error: 'text' is required", true);
        std::string instruct = args["voice"].as_string("male");
        std::string language = args["language"].as_string("English");
        std::vector<float> pcm;
        std::string err;
        if (!do_synthesize(text, instruct, language, 42, 0, pcm, err))
            return mcp_text_result("error: " + err, true);
        std::string out_path = args["output_path"].as_string();
        if (out_path.empty()) {
            std::error_code ec;
            out_path = (fs::temp_directory_path(ec) /
                        ("rs_tts_" + std::to_string((uintptr_t)&text) + ".wav")).string();
        }
        std::ofstream of(out_path, std::ios::binary);
        if (!of) return mcp_text_result("error: cannot write: " + out_path, true);
        std::string wav = wav_encode(pcm, g_tts.sample_rate);
        of.write(wav.data(), (std::streamsize)wav.size());
        double dur = (double)pcm.size() / g_tts.sample_rate;
        char msg[512];
        snprintf(msg, sizeof(msg), "Wrote %.2fs of audio to %s", dur, out_path.c_str());
        return mcp_text_result(msg);
    }
    return mcp_text_result("error: unknown tool: " + name, true);
}

// Dispatch one JSON-RPC message. Returns the response Value; sets has_response
// to false for notifications (which get no reply).
static rsjson::Value mcp_dispatch(const rsjson::Value &msg, bool &has_response) {
    has_response = true;
    const std::string method = msg["method"].as_string();
    const rsjson::Value &id = msg["id"];
    const bool is_notification = !msg.contains("id");

    rsjson::Value resp = rsjson::Value::object();
    resp["jsonrpc"] = "2.0";
    if (!is_notification) resp["id"] = id;

    auto ok = [&](rsjson::Value result) { resp["result"] = std::move(result); return resp; };
    auto rpc_err = [&](int code, const std::string &m) {
        resp["error"]["code"] = code;
        resp["error"]["message"] = m;
        return resp;
    };

    if (is_notification) { has_response = false; return resp; }

    if (method == "initialize") {
        rsjson::Value r = rsjson::Value::object();
        r["protocolVersion"] = "2024-11-05";
        rsjson::Value caps = rsjson::Value::object();
        caps["tools"] = rsjson::Value::object();
        r["capabilities"] = caps;
        rsjson::Value info = rsjson::Value::object();
        info["name"] = kServerName;
        info["version"] = rs_get_version();
        r["serverInfo"] = info;
        return ok(r);
    }
    if (method == "ping") return ok(rsjson::Value::object());
    if (method == "tools/list") return ok(mcp_tools_list());
    if (method == "tools/call") {
        const rsjson::Value &params = msg["params"];
        std::string name = params["name"].as_string();
        return ok(mcp_call_tool(name, params["arguments"]));
    }
    return rpc_err(-32601, "method not found: " + method);
}

// MCP over stdio: newline-delimited JSON-RPC on stdin/stdout. `out` is the
// real stdout (fd 1 is redirected to stderr in main so library printf logs
// don't corrupt the JSON-RPC stream); responses go only through `out`.
static int run_mcp_stdio(FILE *out) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        rsjson::Value msg;
        try { msg = rsjson::parse(line); }
        catch (const std::exception &) { continue; }
        bool has_response = true;
        rsjson::Value resp = mcp_dispatch(msg, has_response);
        if (has_response) {
            std::string s = resp.dump();
            s += '\n';
            fwrite(s.data(), 1, s.size(), out);
            fflush(out);
        }
    }
    return 0;
}

// ── WebSocket streaming endpoints ───────────────────────────────────────
//
// Wire format: binary frames carry raw 16-bit signed little-endian mono PCM;
// text frames carry JSON control/result messages. ASR input is 16 kHz; TTS
// output is the model's rate, announced in a leading {"type":"audio_info"}.
//
// ggml scheduler/model objects are still protected by per-model mutexes, but
// streaming ASR keeps per-connection state and only locks around each chunk.

static void pcm16_to_float(const std::string &b, std::vector<float> &out) {
    size_t n = b.size() / 2;
    out.resize(n);
    const int16_t *s = (const int16_t *)b.data();
    for (size_t i = 0; i < n; i++) out[i] = s[i] / 32768.0f;
}

static std::string float_to_pcm16(const float *pcm, size_t n) {
    std::string b((n * 2), '\0');
    int16_t *d = (int16_t *)&b[0];
    for (size_t i = 0; i < n; i++) {
        float c = pcm[i] < -1.f ? -1.f : (pcm[i] > 1.f ? 1.f : pcm[i]);
        d[i] = (int16_t)lrintf(c * 32767.0f);
    }
    return b;
}

static bool ws_send_json(rsws::Conn &c, const rsjson::Value &v) {
    return c.send_text(v.dump());
}
static bool ws_msg(rsws::Conn &c, const char *type, const std::string &key = "",
                   const std::string &val = "") {
    rsjson::Value o = rsjson::Value::object();
    o["type"] = type;
    if (!key.empty()) o[key] = val;
    return ws_send_json(c, o);
}

static bool query_get(const std::string &query, const std::string &key,
                      std::string &value) {
    size_t pos = 0;
    while (pos <= query.size()) {
        size_t amp = query.find('&', pos);
        std::string item = query.substr(pos, amp == std::string::npos
                                             ? std::string::npos
                                             : amp - pos);
        size_t eq = item.find('=');
        std::string k = eq == std::string::npos ? item : item.substr(0, eq);
        if (k == key) {
            value = eq == std::string::npos ? "" : item.substr(eq + 1);
            return true;
        }
        if (amp == std::string::npos) break;
        pos = amp + 1;
    }
    return false;
}

static float query_float(const std::string &query, const char *key,
                         float current, float lo, float hi) {
    std::string v;
    if (!query_get(query, key, v) || v.empty()) return current;
    char *end = nullptr;
    float parsed = std::strtof(v.c_str(), &end);
    if (end == v.c_str()) return current;
    return std::max(lo, std::min(hi, parsed));
}

static int query_int(const std::string &query, const char *key,
                     int current, int lo, int hi) {
    std::string v;
    if (!query_get(query, key, v) || v.empty()) return current;
    char *end = nullptr;
    long parsed = std::strtol(v.c_str(), &end, 10);
    if (end == v.c_str()) return current;
    return std::max(lo, std::min(hi, (int)parsed));
}

static bool parse_bool_value(const std::string &v, bool current) {
    if (v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on")
        return true;
    if (v == "0" || v == "false" || v == "FALSE" || v == "no" || v == "off")
        return false;
    return current;
}

static bool query_bool(const std::string &query, const char *key, bool current) {
    std::string v;
    if (!query_get(query, key, v)) return current;
    return parse_bool_value(v.empty() ? "1" : v, current);
}

static std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::string normalize_tts_stream_mode(const std::string &mode) {
    std::string m = lower_ascii(mode);
    if (m == "hybrid" || m == "mixed") return "hybrid";
    if (m == "auto") return "auto";
    return "true";
}

static size_t utf8_char_bytes(const std::string &s, size_t pos) {
    if (pos >= s.size()) return 0;
    const unsigned char c = static_cast<unsigned char>(s[pos]);
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0 && pos + 1 < s.size()) return 2;
    if ((c & 0xF0) == 0xE0 && pos + 2 < s.size()) return 3;
    if ((c & 0xF8) == 0xF0 && pos + 3 < s.size()) return 4;
    return 1;
}

static std::string trim_copy(const std::string &s) {
    const size_t begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

static bool starts_with_at(const std::string &s, size_t pos,
                           const char *needle) {
    const size_t n = std::strlen(needle);
    return pos + n <= s.size() && s.compare(pos, n, needle) == 0;
}

static bool is_soft_segment_break(const std::string &s, size_t pos,
                                  size_t *break_end) {
    const char c = s[pos];
    if (c == ',' || c == ':' || c == ' ' || c == '\t') {
        *break_end = pos + 1;
        return true;
    }
    static const char *breaks[] = {"，", "、", "：", "——"};
    for (const char *b : breaks) {
        if (starts_with_at(s, pos, b)) {
            *break_end = pos + std::strlen(b);
            return true;
        }
    }
    return false;
}

static void append_bounded_segment(std::vector<std::string> &out,
                                   const std::string &segment,
                                   int max_chars) {
    const std::string text = trim_copy(segment);
    if (text.empty()) return;
    if (max_chars <= 0) {
        out.push_back(text);
        return;
    }

    size_t start = 0;
    while (start < text.size()) {
        while (start < text.size() &&
               std::isspace(static_cast<unsigned char>(text[start]))) {
            ++start;
        }
        if (start >= text.size()) break;

        size_t pos = start;
        size_t last_break = std::string::npos;
        int chars = 0;
        while (pos < text.size() && chars < max_chars) {
            size_t soft_end = 0;
            if (is_soft_segment_break(text, pos, &soft_end)) {
                last_break = soft_end;
            }
            const size_t step = utf8_char_bytes(text, pos);
            if (step == 0) break;
            pos += step;
            ++chars;
        }

        size_t cut = pos;
        if (pos < text.size() && last_break != std::string::npos &&
            last_break > start) {
            cut = last_break;
        }
        std::string part = trim_copy(text.substr(start, cut - start));
        if (!part.empty()) out.push_back(std::move(part));
        start = cut;
    }
}

// Split text into synthesis segments at sentence-ending punctuation (CJK + ASCII).
static std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> segs;
    std::string cur;
    static const char *enders[] = {"。", "！", "？", "；", "\n"}; // multi-byte CJK
    auto flush = [&]() {
        size_t s = cur.find_first_not_of(" \t\r\n");
        if (s != std::string::npos) segs.push_back(cur.substr(s));
        cur.clear();
    };
    for (size_t i = 0; i < text.size();) {
        // ASCII sentence enders
        char c = text[i];
        if (c == '.' || c == '!' || c == '?' || c == ';' || c == '\n') {
            cur += c; ++i; flush(); continue;
        }
        bool matched = false;
        for (const char *e : enders) {
            size_t el = strlen(e);
            if (text.compare(i, el, e) == 0) {
                cur.append(text, i, el); i += el; flush(); matched = true; break;
            }
        }
        if (!matched) { cur += c; ++i; }
    }
    flush();
    if (segs.empty() && !text.empty()) segs.push_back(text);
    return segs;
}

static std::vector<std::string> split_sentences_bounded(
    const std::string &text, int max_chars) {
    std::vector<std::string> out;
    for (const std::string &segment : split_sentences(text)) {
        append_bounded_segment(out, segment, max_chars);
    }
    if (out.empty() && !text.empty()) out.push_back(text);
    return out;
}

static bool ws_emit_pcm(rsws::Conn &c, const float *pcm, size_t n) {
    if (!pcm || n == 0) return true;
    std::string bytes = float_to_pcm16(pcm, n);
    return c.send_binary(bytes.data(), bytes.size());
}

static bool ws_emit_pcm(rsws::Conn &c, const std::vector<float> &pcm) {
    return ws_emit_pcm(c, pcm.data(), pcm.size());
}

// Independent hybrid segments can contain decoder padding at their edges.
// Keep a short natural attack/release, but avoid concatenating long silent
// tails/heads between sentences.
static void trim_tts_boundary_silence(std::vector<float> &pcm) {
    const int sample_rate = std::max(1, g_tts.sample_rate);
    if (pcm.size() < static_cast<size_t>(sample_rate / 4)) return;

    float peak = 0.0f;
    for (float sample : pcm) peak = std::max(peak, std::fabs(sample));
    if (peak < 0.02f) return;
    const float threshold = std::max(0.008f, peak * 0.02f);

    size_t first = 0;
    while (first < pcm.size() && std::fabs(pcm[first]) <= threshold) ++first;
    size_t last = pcm.size();
    while (last > first && std::fabs(pcm[last - 1]) <= threshold) --last;
    if (first == 0 && last == pcm.size()) return;

    const size_t keep_head = static_cast<size_t>(sample_rate * 15 / 1000);
    const size_t keep_tail = static_cast<size_t>(sample_rate * 45 / 1000);
    const size_t begin = first > keep_head ? first - keep_head : 0;
    const size_t end = std::min(pcm.size(), last + keep_tail);
    if (begin < end && (begin > 0 || end < pcm.size()))
    {
        pcm.erase(pcm.begin(), pcm.begin() + static_cast<ptrdiff_t>(begin));
        pcm.erase(pcm.begin() + static_cast<ptrdiff_t>(end - begin), pcm.end());
    }
}

static bool stream_tts_context(rsws::Conn &c, const std::string &text,
                               const std::string &voice,
                               const std::string &lang, int seed, int steps,
                               std::string &err) {
    std::lock_guard<std::mutex> lk(g_tts.mu);
    rs_reset(g_tts.ctx);
    rs_set_tts_params(g_tts.ctx, voice.c_str(), lang.c_str(), seed);
    if (steps > 0) rs_set_tts_diffusion_steps(g_tts.ctx, steps);
    if (rs_push_text(g_tts.ctx, text.c_str()) != RS_OK) {
        err = "rs_push_text failed";
        return false;
    }
    int32_t ret;
    while ((ret = rs_process(g_tts.ctx)) >= 0) {
        float *chunk = nullptr;
        int n = rs_get_audio_output(g_tts.ctx, &chunk);
        if (n > 0 && chunk && !ws_emit_pcm(c, chunk, static_cast<size_t>(n))) {
            err = "WebSocket send failed";
            return false;
        }
        if (ret == 0) break;
    }
    if (ret < 0) {
        err = rs_get_last_error().message;
        return false;
    }
    return true;
}

static bool ws_wait_emit(std::future<std::pair<std::vector<float>, std::string>> &future,
                         rsws::Conn &c, std::string &err) {
    auto result = future.get();
    if (!result.second.empty()) {
        err = result.second;
        return false;
    }
    trim_tts_boundary_silence(result.first);
    if (!ws_emit_pcm(c, result.first)) {
        err = "WebSocket send failed";
        return false;
    }
    return true;
}

static bool ws_tts_stream_true_impl(rsws::Conn &c, const std::string &text,
                                    const std::string &voice,
                                    const std::string &lang, int seed,
                                    int steps, std::string &err) {
    auto callback = [&c](const float *pcm, size_t n) {
        (void)ws_emit_pcm(c, pcm, n);
    };

    if (g_voxcpm2_batch) {
        auto future = g_voxcpm2_batch->SubmitStream(
            text, voice, lang, seed, steps, callback);
        auto result = future.get();
        if (!result.second.empty()) {
            err = result.second;
            return false;
        }
        return true;
    }

    if (g_tts_batch && g_tts_batch->CanStreamFirstSentence()) {
        if (!g_tts_batch->StreamFirstSentence(text, voice, lang, seed,
                                              callback)) {
            err = "CosyVoice3 streaming failed";
            return false;
        }
        return true;
    }
    if (g_tts_batch) {
        auto future = g_tts_batch->Submit(text, voice, lang, seed);
        return ws_wait_emit(future, c, err);
    }

    return stream_tts_context(c, text, voice, lang, seed, steps, err);
}

static bool ws_tts_stream_hybrid_impl(rsws::Conn &c, const std::string &text,
                                      const std::string &voice,
                                      const std::string &lang, int seed,
                                      int steps, int segment_max_chars,
                                      uint64_t session_id, std::string &err) {
    std::vector<std::string> segs =
        split_sentences_bounded(text, segment_max_chars);
    if (segs.empty()) return true;
    if (segs.size() == 1) {
        return ws_tts_stream_true_impl(c, segs.front(), voice, lang, seed,
                                       steps, err);
    }

    auto callback = [&c](const float *pcm, size_t n) {
        (void)ws_emit_pcm(c, pcm, n);
    };

    if (g_voxcpm2_batch) {
        auto first = g_voxcpm2_batch->SubmitStream(
            segs[0], voice, lang, seed, steps, callback);
        std::vector<std::future<VoxCPM2ServerScheduler::Result>> tail;
        tail.reserve(segs.size() - 1);
        for (size_t i = 1; i < segs.size(); ++i) {
            tail.push_back(g_voxcpm2_batch->Submit(
                segs[i], voice, lang, seed, steps));
        }
        auto first_result = first.get();
        if (!first_result.second.empty()) {
            err = first_result.second;
            return false;
        }
        for (auto &future : tail) {
            if (!ws_wait_emit(future, c, err)) return false;
        }
        return true;
    }

    if (g_tts_batch && g_tts_batch->CanStreamFirstSentence()) {
        std::vector<std::future<std::pair<std::vector<float>, std::string>>>
            tail;
        tail.reserve(segs.size() - 1);
        for (size_t i = 1; i < segs.size(); ++i) {
            tail.push_back(g_tts_batch->Submit(
                segs[i], voice, lang, seed, session_id));
        }
        if (!g_tts_batch->StreamFirstSentence(segs[0], voice, lang, seed,
                                              callback)) {
            err = "CosyVoice3 first-segment streaming failed";
            return false;
        }
        for (auto &future : tail) {
            if (!ws_wait_emit(future, c, err)) return false;
        }
        return true;
    }
    if (g_tts_batch) {
        std::vector<std::future<std::pair<std::vector<float>, std::string>>>
            futures;
        futures.reserve(segs.size());
        for (size_t i = 0; i < segs.size(); ++i) {
            futures.push_back(g_tts_batch->Submit(
                segs[i], voice, lang, seed, session_id));
        }
        for (auto &future : futures) {
            if (!ws_wait_emit(future, c, err)) return false;
        }
        return true;
    }

    // No independent streaming scheduler is available. Keep correctness and
    // bounded per-segment synthesis; only the first segment is true-streamed.
    if (!stream_tts_context(c, segs[0], voice, lang, seed, steps, err)) {
        return false;
    }
    for (size_t i = 1; i < segs.size(); ++i) {
        std::vector<float> pcm;
        if (!do_synthesize(segs[i], voice, lang, seed, steps, pcm, err)) {
            return false;
        }
        trim_tts_boundary_silence(pcm);
        if (!ws_emit_pcm(c, pcm)) {
            err = "WebSocket send failed";
            return false;
        }
    }
    return true;
}

// (1) Pure streaming ASR (X-ASR). Client streams PCM; server emits partials as
// tokens update and a final on {"type":"eos"} / close.
static void ws_asr_stream(rsws::Conn &c) {
    if (!g_have_asr) { ws_msg(c, "error", "error", "no ASR model loaded"); return; }
    {
        std::lock_guard<std::mutex> lk(g_asr.mu);
        auto *xasr = g_asr.ctx ? dynamic_cast<XASRModel *>(g_asr.ctx->model.get()) : nullptr;
        if (!xasr) {
            ws_msg(c, "error", "error", "model does not support streaming (need X-ASR)");
            return;
        }
    }

    int chunk_len = query_int(c.query(), "chunk_len", 32, 16, 192);
    if (chunk_len != 16 && chunk_len != 32 && chunk_len != 48 &&
        chunk_len != 96 && chunk_len != 192) {
        chunk_len = 32;
    }
    auto stream_state = std::make_shared<XASRState>();
    ws_msg(c, "ready");

    std::vector<float> pcm;
    int op; std::string msg;
    while (c.recv(op, msg)) {
        if (op == rsws::OP_BINARY) {
            pcm16_to_float(msg, pcm);
            bool ok = false;
            bool changed = false;
            std::string partial;
            {
                std::lock_guard<std::mutex> lk(g_asr.mu);
                auto *xasr = g_asr.ctx ? dynamic_cast<XASRModel *>(g_asr.ctx->model.get()) : nullptr;
                if (xasr) {
                    const size_t before = stream_state->tokens.size();
                    xasr->SetChunkLen(chunk_len);
                    ok = xasr->EncodeStreamingChunk(pcm.data(), pcm.size(),
                                                    *stream_state,
                                                    g_asr.ctx->sched);
                    changed = ok && stream_state->tokens.size() != before;
                    if (changed) partial = xasr->GetTranscription(*stream_state);
                }
            }
            if (!ok) {
                ws_msg(c, "error", "error", "ASR streaming chunk failed");
                return;
            }
            if (changed) ws_msg(c, "partial", "text", partial);
        } else if (op == rsws::OP_TEXT) {
            rsjson::Value j;
            try { j = rsjson::parse(msg); } catch (...) { continue; }
            if (j["type"].as_string() == "eos") break;
        }
    }
    bool ok = false;
    std::string final_text;
    {
        std::lock_guard<std::mutex> lk(g_asr.mu);
        auto *xasr = g_asr.ctx ? dynamic_cast<XASRModel *>(g_asr.ctx->model.get()) : nullptr;
        if (xasr) {
            xasr->SetChunkLen(chunk_len);
            ok = xasr->FinishStream(*stream_state, g_asr.ctx->sched);
            if (ok) final_text = xasr->GetTranscription(*stream_state);
        }
    }
    if (!ok) {
        ws_msg(c, "error", "error", "ASR streaming finish failed");
        return;
    }
    ws_msg(c, "final", "text", final_text);
}

// (2) VAD + offline ASR. Client streams PCM; VAD segments it, each segment gets
// interim partials (transcribing the growing buffer) and a final on segment end.
static void ws_asr_vad(rsws::Conn &c) {
    const uint64_t conn_id =
        (uint64_t)(uintptr_t)&c ^ (uint64_t)std::chrono::steady_clock::now()
                                        .time_since_epoch()
                                        .count();
    ws_logf("[ws/asr/vad:%llx] enter query='%s'\n",
            (unsigned long long)conn_id, c.query().c_str());
    if (!g_have_asr) {
        ws_logf("[ws/asr/vad:%llx] reject: no ASR\n",
                (unsigned long long)conn_id);
        ws_msg(c, "error", "error", "no ASR model loaded");
        return;
    }
    if (!g_have_vad) {
        ws_logf("[ws/asr/vad:%llx] reject: no VAD\n",
                (unsigned long long)conn_id);
        ws_msg(c, "error", "error", "no VAD model (use --vad-model)");
        return;
    }
    AsrServerConfig asr_cfg = g_asr_cfg;
    VadServerConfig cfg = g_vad_cfg;
    const std::string &q = c.query();
    asr_cfg.two_pass = query_bool(q, "two_pass", asr_cfg.two_pass);
    asr_cfg.ctc_precheck = query_bool(q, "ctc_precheck", asr_cfg.ctc_precheck);
    asr_cfg.ctc_only = query_bool(q, "ctc_only", asr_cfg.ctc_only);
    cfg.threshold = query_float(q, "vad_threshold", cfg.threshold, 0.0f, 1.0f);
    cfg.threshold = query_float(q, "threshold", cfg.threshold, 0.0f, 1.0f);
    cfg.end_silence_ms = query_int(q, "vad_end_silence_ms", cfg.end_silence_ms, 0, 60000);
    cfg.end_silence_ms = query_int(q, "end_silence_ms", cfg.end_silence_ms, 0, 60000);
    cfg.end_silence_ms = query_int(q, "silence_ms", cfg.end_silence_ms, 0, 60000);
    cfg.partial_ms = query_int(q, "vad_partial_ms", cfg.partial_ms, 0, 60000);
    cfg.partial_ms = query_int(q, "partial_ms", cfg.partial_ms, 0, 60000);
    cfg.min_segment_ms = query_int(q, "vad_min_segment_ms", cfg.min_segment_ms, 0, 60000);
    cfg.min_segment_ms = query_int(q, "min_segment_ms", cfg.min_segment_ms, 0, 60000);
    cfg.max_segment_ms = query_int(q, "vad_max_segment_ms", cfg.max_segment_ms, 0, 3600000);
    cfg.max_segment_ms = query_int(q, "max_segment_ms", cfg.max_segment_ms, 0, 3600000);

    auto vad_worker = acquire_vad_ws_worker();
    if (!vad_worker.vad) {
        ws_logf("[ws/asr/vad:%llx] reject: VAD busy\n",
                (unsigned long long)conn_id);
        ws_msg(c, "error", "error", "VAD busy");
        return;
    }
    ws_logf("[ws/asr/vad:%llx] acquired worker=%d\n",
            (unsigned long long)conn_id, vad_worker.index);
    if (rs_vad_reset(vad_worker.vad) != RS_OK) {
        ws_logf("[ws/asr/vad:%llx] rs_vad_reset failed\n",
                (unsigned long long)conn_id);
        ws_msg(c, "error", "error", "rs_vad_reset failed");
        return;
    }
    if (rs_vad_set_threshold(vad_worker.vad, cfg.threshold) != RS_OK) {
        ws_logf("[ws/asr/vad:%llx] rs_vad_set_threshold failed\n",
                (unsigned long long)conn_id);
        ws_msg(c, "error", "error", "rs_vad_set_threshold failed");
        return;
    }

    rsjson::Value ready = rsjson::Value::object();
    ready["type"] = "ready";
    ready["vad_threshold"] = cfg.threshold;
    ready["vad_end_silence_ms"] = cfg.end_silence_ms;
    ready["vad_partial_ms"] = cfg.partial_ms;
    ready["vad_min_segment_ms"] = cfg.min_segment_ms;
    ready["vad_max_segment_ms"] = cfg.max_segment_ms;
    ready["two_pass"] = asr_cfg.two_pass;
    ready["ctc_precheck"] = asr_cfg.ctc_precheck;
    ready["ctc_only"] = asr_cfg.ctc_only;
    if (!ws_send_json(c, ready)) {
        ws_logf("[ws/asr/vad:%llx] send ready failed\n",
                (unsigned long long)conn_id);
        return;
    }
    ws_logf("[ws/asr/vad:%llx] ready sent worker=%d threshold=%.3f "
            "silence=%dms partial=%dms\n",
            (unsigned long long)conn_id, vad_worker.index, cfg.threshold,
            cfg.end_silence_ms, cfg.partial_ms);

    const int sr = g_asr.sample_rate;
    std::vector<float> seg;       // current speech segment
    bool in_speech = false;
    bool pending_end = false;     // VAD ended, waiting for hangover silence
    int pending_silence_samples = 0;
    int seg_index = 0;
    size_t last_partial_len = 0;  // samples at last partial

    auto transcribe_seg = [&](const char *kind) -> bool {
        if (seg.empty()) return true;
        int min_samples = (sr * cfg.min_segment_ms) / 1000;
        if ((int)seg.size() < min_samples) return true;
        std::string text, err;
        bool is_partial = strcmp(kind, "partial") == 0;
        AsrServerConfig seg_cfg = asr_cfg;
        if (is_partial) {
            seg_cfg.two_pass = false;
            seg_cfg.ctc_precheck = false;
            seg_cfg.ctc_only = true;
        }
        bool ok = transcribe_pcm_server(seg.data(), (int)seg.size(), seg_cfg,
                                        text, err);
        if (!ok) {
            ws_logf("[ws/asr/vad:%llx] %s ASR failed: %s\n",
                    (unsigned long long)conn_id, kind,
                    err.empty() ? "ASR transcribe failed" : err.c_str());
            ws_msg(c, "error", "error", err.empty() ? "ASR transcribe failed" : err);
            return false;
        }
        if (!text.empty()) {
            rsjson::Value o = rsjson::Value::object();
            o["type"] = kind; o["segment"] = seg_index; o["text"] = text;
            o["two_pass"] = seg_cfg.two_pass && !seg_cfg.ctc_only;
            o["ctc_only"] = seg_cfg.ctc_only;
            ws_send_json(c, o);
            ws_logf("[ws/asr/vad:%llx] sent %s segment=%d samples=%zu "
                    "text_len=%zu\n",
                    (unsigned long long)conn_id, kind, seg_index, seg.size(),
                    text.size());
        }
        return true;
    };

    auto finalize_seg = [&]() -> bool {
        if (!transcribe_seg("final")) return false;
        in_speech = false;
        pending_end = false;
        pending_silence_samples = 0;
        seg_index++;
        seg.clear();
        last_partial_len = 0;
        return true;
    };

    auto force_split_seg = [&]() -> bool {
        if (!finalize_seg()) return false;
        in_speech = true;
        return true;
    };

    std::vector<float> pcm;
    int op; std::string msg;
    bool saw_binary = false;
    auto handle_frames = [&]() {
        rs_vad_frame_t frames[64];
        int nf;
        while ((nf = rs_vad_drain_frames(vad_worker.vad, frames, 64)) > 0) {
            for (int i = 0; i < nf; i++) {
                if (frames[i].is_speech_start) {
                    if (pending_end) {
                        // Short pause inside one utterance: keep the current
                        // buffer and cancel the tentative endpoint.
                        pending_end = false;
                        pending_silence_samples = 0;
                    } else if (!in_speech) {
                        in_speech = true;
                        seg.clear();
                        last_partial_len = 0;
                    }
                }
                if (frames[i].is_speech_end && in_speech) {
                    pending_end = true;
                    pending_silence_samples = 0;
                }
            }
            if (nf < 64) break;
        }
    };

    while (c.recv(op, msg)) {
        if (op == rsws::OP_BINARY) {
            pcm16_to_float(msg, pcm);
            if (!saw_binary) {
                saw_binary = true;
                ws_logf("[ws/asr/vad:%llx] first binary frame samples=%zu\n",
                        (unsigned long long)conn_id, pcm.size());
            }
            if (rs_vad_push_audio(vad_worker.vad, pcm.data(),
                                  (int)pcm.size()) != RS_OK) {
                ws_logf("[ws/asr/vad:%llx] rs_vad_push_audio failed\n",
                        (unsigned long long)conn_id);
                ws_msg(c, "error", "error", "rs_vad_push_audio failed");
                return;
            }
            handle_frames();
            if (in_speech) {
                seg.insert(seg.end(), pcm.begin(), pcm.end());
                if (pending_end) pending_silence_samples += (int)pcm.size();
            }
            // Emit partials on a configurable cadence while speech is active.
            if (in_speech && !pending_end && cfg.partial_ms > 0 &&
                seg.size() - last_partial_len >= (size_t)((sr * cfg.partial_ms) / 1000)) {
                last_partial_len = seg.size();
                if (!transcribe_seg("partial")) return;
            }
            if (pending_end &&
                pending_silence_samples >= (sr * cfg.end_silence_ms) / 1000) {
                if (!finalize_seg()) return;
            }
            if (in_speech && cfg.max_segment_ms > 0 &&
                seg.size() >= (size_t)((sr * cfg.max_segment_ms) / 1000)) {
                if (pending_end) {
                    if (!finalize_seg()) return;
                } else if (!force_split_seg()) {
                    return;
                }
            }
        } else if (op == rsws::OP_TEXT) {
            rsjson::Value j;
            try { j = rsjson::parse(msg); } catch (...) { continue; }
            if (j["type"].as_string() == "eos") {
                ws_logf("[ws/asr/vad:%llx] eos received\n",
                        (unsigned long long)conn_id);
                break;
            }
        }
    }
    if (in_speech && !transcribe_seg("final")) return;
    ws_logf("[ws/asr/vad:%llx] done segments=%d\n",
            (unsigned long long)conn_id, seg_index + (in_speech ? 1 : 0));
    ws_msg(c, "done");
}

// (3) Segmented streaming TTS: split text into sentences, synthesize each fully,
// stream back per-segment audio (works for any TTS model).
static void ws_tts_segmented(rsws::Conn &c) {
    if (!g_have_tts) { ws_msg(c, "error", "error", "no TTS model loaded"); return; }

    // Batch mode: no lock needed — each segment is Submit()ed to the
    // scheduler and decodes concurrently with other connections.
    const uint64_t sid = reinterpret_cast<uint64_t>(&c) ^ 0xDEADBEEF;
    int op; std::string msg;
    if (!c.recv(op, msg) || op != rsws::OP_TEXT) return;
    rsjson::Value req;
    try { req = rsjson::parse(msg); } catch (...) { ws_msg(c, "error", "error", "bad json"); return; }
    std::string text = req["text"].as_string();
    if (text.empty()) { ws_msg(c, "error", "error", "missing 'text'"); return; }
    std::string voice = req["voice"].as_string("male");
    std::string lang = req["language"].as_string("English");

    rsjson::Value info = rsjson::Value::object();
    info["type"] = "audio_info"; info["sample_rate"] = g_tts.sample_rate;
    info["format"] = "pcm_s16le";
    ws_send_json(c, info);

    std::unique_lock<std::mutex> lk(g_tts.mu, std::defer_lock);
    if (!g_tts_batch && !g_voxcpm2_batch) lk.lock();

    auto segs = split_sentences(text);
    for (size_t i = 0; i < segs.size(); i++) {
        std::vector<float> pcm;
        if (g_voxcpm2_batch) {
            auto fut = g_voxcpm2_batch->Submit(
                segs[i], voice, lang, 42, 8);
            auto result = fut.get();
            if (!result.second.empty()) {
                ws_msg(c, "error", "error", result.second);
                continue;
            }
            pcm = std::move(result.first);
        } else if (g_tts_batch) {
            auto fut = g_tts_batch->Submit(segs[i], voice, lang, 42, sid);
            auto result = fut.get();
            if (!result.second.empty()) {
                ws_msg(c, "error", "error", result.second);
                continue;
            }
            pcm = std::move(result.first);
        } else {
            // do_synthesize locks g_tts.mu; we already hold it — inline the drive.
            rs_reset(g_tts.ctx);
            rs_set_tts_params(g_tts.ctx, voice.c_str(), lang.c_str(), 42);
            if (rs_push_text(g_tts.ctx, segs[i].c_str()) != RS_OK) continue;
            int32_t ret;
            while ((ret = rs_process(g_tts.ctx)) >= 0) {
                float *ch = nullptr; int n = rs_get_audio_output(g_tts.ctx, &ch);
                if (n > 0 && ch) pcm.insert(pcm.end(), ch, ch + n);
                if (ret == 0) break;
            }
        }
        rsjson::Value m = rsjson::Value::object();
        m["type"] = "segment"; m["index"] = (int)i; m["text"] = segs[i];
        ws_send_json(c, m);
        std::string b = float_to_pcm16(pcm.data(), pcm.size());
        c.send_binary(b.data(), b.size());
    }
    ws_msg(c, "final");
}

// (4) Streaming TTS. stream_mode=true preserves model-native streaming over the
// full text. stream_mode=hybrid streams segment 0 and queues later segments for
// bounded, concurrent synthesis while preserving output order.
static void ws_tts_stream(rsws::Conn &c) {
    if (!g_have_tts) { ws_msg(c, "error", "error", "no TTS model loaded"); return; }

    const uint64_t sid = reinterpret_cast<uint64_t>(&c) ^ 0xDEADBEEF;
    int op; std::string msg;
    if (!c.recv(op, msg) || op != rsws::OP_TEXT) return;
    rsjson::Value req;
    try { req = rsjson::parse(msg); } catch (...) { ws_msg(c, "error", "error", "bad json"); return; }
    std::string text = req["text"].as_string();
    if (text.empty()) { ws_msg(c, "error", "error", "missing 'text'"); return; }
    std::string voice = req["voice"].as_string("male");
    std::string lang = req["language"].as_string("English");
    int seed = req.contains("seed") ? req["seed"].as_int(42) : 42;
    int steps = req.contains("n_steps") ? req["n_steps"].as_int(8) : 8;
    std::string stream_mode = g_tts_stream_mode;
    if (req.contains("stream_mode")) {
        stream_mode = req["stream_mode"].as_string(stream_mode);
    } else if (req.contains("tts_stream_mode")) {
        stream_mode = req["tts_stream_mode"].as_string(stream_mode);
    }
    stream_mode = normalize_tts_stream_mode(stream_mode);
    int segment_max_chars = g_tts_segment_max_chars;
    if (req.contains("segment_max_chars")) {
        segment_max_chars = req["segment_max_chars"].as_int(segment_max_chars);
    } else if (req.contains("tts_segment_max_chars")) {
        segment_max_chars = req["tts_segment_max_chars"].as_int(segment_max_chars);
    }
    segment_max_chars = std::max(16, std::min(512, segment_max_chars));

    if (stream_mode == "auto") {
        stream_mode = split_sentences_bounded(text, segment_max_chars).size() > 1
            ? "hybrid" : "true";
    }

    rsjson::Value info = rsjson::Value::object();
    info["type"] = "audio_info"; info["sample_rate"] = g_tts.sample_rate;
    info["format"] = "pcm_s16le";
    info["stream_mode"] = stream_mode;
    if (stream_mode == "hybrid") {
        info["segment_max_chars"] = segment_max_chars;
    }
    ws_send_json(c, info);

    std::string err;
    bool ok = false;
    if (stream_mode == "hybrid") {
        ok = ws_tts_stream_hybrid_impl(c, text, voice, lang, seed, steps,
                                       segment_max_chars, sid, err);
    } else {
        ok = ws_tts_stream_true_impl(c, text, voice, lang, seed, steps, err);
    }
    if (!ok) {
        ws_msg(c, "error", "error", err.empty() ? "TTS streaming failed" : err);
        return;
    }
    ws_msg(c, "final");
}

// ── main ────────────────────────────────────────────────────────────────

static void usage(const char *argv0) {
    fprintf(stderr,
        "RapidSpeech server — OpenAI-compatible HTTP API + MCP\n"
        "Usage: %s [--asr-model asr.gguf] [--tts-model tts.gguf] [options]\n"
        "  --asr-model PATH   ASR GGUF (enables /v1/audio/transcriptions + transcribe tool)\n"
        "  --tts-model PATH   TTS GGUF (enables /v1/audio/speech + synthesize tool)\n"
        "  --vad-model PATH   VAD GGUF (enables VAD segmented ASR and /asr/vad)\n"
        "  --vad-workers N    Independent /asr/vad WebSocket VAD workers (default 1)\n"
        "  --vad-threshold F  VAD speech threshold (default 0.5)\n"
        "  --vad-end-silence-ms N  Silence before finalizing segment (default 600)\n"
        "  --vad-partial-ms N Partial ASR interval for /asr/vad (default 1500, 0=off)\n"
        "  --vad-min-segment-ms N  Minimum segment to transcribe (default 500)\n"
        "  --vad-max-segment-ms N  Force-finalize long speech (default 0=off)\n"
        "  --spk-model PATH   CAMPPlus speaker GGUF (enables speaker=true requests)\n"
        "  --speaker PATH     Alias for --spk-model\n"
        "  --spk-cluster MODE Speaker clustering: spectral or ahc (default spectral)\n"
        "  --spk-threshold F  AHC cosine merge threshold (default 0.5)\n"
        "  --num-speakers N   Fixed speaker count, 0 = auto (default 0)\n"
        "  --max-speakers N   Spectral auto-estimation cap (default 10)\n"
        "  --speaker-log PATH Append speaker transcript log TSV, '-' = stderr\n"
        "  --two-pass         FunASR-Nano: CTC first pass + LLM final re-decode\n"
        "  --ctc-precheck     FunASR-Nano: skip LLM on CTC-empty silence/noise\n"
        "  --asr-slots N      ASR shape-bucketed batch slots (default 4, 1=off)\n"
        "  --asr-max-queued N Maximum queued ASR requests (default 256)\n"
        "  --asr-batch-wait-ms N  ASR microbatch wait window (default 3)\n"
        "  --asr-min-batch N  Prefer waiting for at least N ASR requests (default=asr-slots)\n"
        "  --asr-max-batch-wait-ms N  Hard ASR microbatch wait cap (default=4x batch-wait)\n"
        "  --asr-frontend-threads N  CPU fbank threads per ASR request (default 1)\n"
        "  --asr-frontend-workers N  Concurrent ASR frontend jobs (default=asr-slots)\n"
        "  --tts-slots N      TTS concurrent requests via scheduler/batching\n"
        "  --tts-voc-workers N  Flow/HiFT worker threads (CUDA streams, default 1)\n"
        "                     (CosyVoice3/VoxCPM2; default 2, 1=serialized)\n"
        "  --tts-stream-mode MODE  /tts/stream mode: true, hybrid, auto\n"
        "  --tts-segment-max-chars N  Max chars per hybrid TTS segment (default 80)\n"
        "  --mem-log          Log ASR scheduler compute-buffer high-water marks\n"
        "  --asr-reload-every N  Reload ASR after N final segments to release GPU buffers (0=off)\n"
        "  --ws-log           Log WebSocket connection lifecycle events\n"
        "  --host HOST        HTTP bind host (default 127.0.0.1)\n"
        "  --port PORT        HTTP port (default 8080)\n"
        "  --ws-port PORT     WebSocket port (default HTTP port + 1)\n"
        "  --web-dir DIR      Serve a static web UI dir at / (e.g. examples/server)\n"
        "                     WebUI can call /v1/models/registry and /v1/models/load\n"
        "                     to download and hot-load registry models.\n"
        "  --mcp-stdio        Run as an MCP server over stdio (no HTTP)\n"
        "  -t, --threads N    CPU threads per model (default 4)\n"
        "  --cpu              Disable GPU\n"
        "  -h, --help\n"
        "HTTP transcription params:\n"
        "  vad=true           VAD-segmented offline ASR; segments are batch-decoded when ASR scheduler is on\n"
        "  speaker=true       CAMPPlus embedding + clustering per VAD segment (requires --spk-model)\n"
        "  speaker_log=true   Write timestamped speaker log to --speaker-log path or stderr\n"
        "  spk_cluster=MODE   spectral or ahc; also accepts num_speakers/max_speakers/spk_threshold\n"
        "WebSocket endpoints (binary = int16 mono PCM, text = JSON):\n"
        "  /asr/stream      pure streaming ASR (X-ASR)     partial+final\n"
        "  /asr/vad         VAD-segmented offline ASR      partial+final per segment\n"
        "                   query overrides: vad_threshold, vad_end_silence_ms,\n"
        "                   vad_partial_ms,\n"
        "                   vad_min_segment_ms, vad_max_segment_ms,\n"
        "                   two_pass, ctc_precheck, ctc_only\n"
        "  /tts/segmented   sentence-by-sentence TTS       audio per segment\n"
        "  /tts/stream      pure streaming TTS             audio chunks as generated\n"
        "If no model is provided, start the WebUI and load one from the registry.\n",
        argv0);
}

int main(int argc_raw, char **argv_raw) {
    rs::cli::Utf8Args utf8(argc_raw, argv_raw);
    int argc = utf8.argc();
    char **argv = utf8.argv();

    // Multi-worker uses two CUDA schedulers concurrently; CUDA graph capture
    // coordinates via a global lock that would serialize them. Disable it so
    // the two streams can overlap. (Harmless on non-CUDA - env is ignored.)
#ifdef _WIN32
    if (!std::getenv("GGML_CUDA_DISABLE_GRAPHS")) {
        _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
    }
#else
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);
#endif

    if (const char *p = std::getenv("RAPIDSPEECH_REGISTRY_PATH")) {
        g_registry_path = p;
    }
    if (g_registry_path.empty()) {
        g_registry_path = find_registry_path_from_argv0(argv[0]);
    }
    if (const char *p = std::getenv("RAPIDSPEECH_PYTHON_EXECUTABLE")) {
        g_python_executable = p;
    }
    if (g_python_executable.empty()) {
#ifdef _WIN32
        g_python_executable = "python";
#else
        g_python_executable = "python3";
#endif
    }

    std::string asr_model, tts_model, vad_model, host = "127.0.0.1", web_dir;
    int port = 8080, ws_port = -1, threads = 4;
    bool gpu = true, mcp_stdio = false;
    if (const char *p = std::getenv("RS_TTS_STREAM_MODE")) {
        g_tts_stream_mode = normalize_tts_stream_mode(p);
    }
    if (const char *p = std::getenv("RS_TTS_SEGMENT_MAX_CHARS")) {
        g_tts_segment_max_chars = std::max(16, std::min(512, std::atoi(p)));
    }

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs an argument\n", name); exit(1); }
            return argv[++i];
        };
        if (a == "--asr-model") asr_model = next("--asr-model");
        else if (a == "--tts-model") tts_model = next("--tts-model");
        else if (a == "--vad-model") vad_model = next("--vad-model");
        else if (a == "--vad-workers") {
            g_vad_workers = std::max(1, atoi(next("--vad-workers")));
        }
        else if (a == "--vad-threshold") {
            g_vad_cfg.threshold = std::max(0.0f, std::min(1.0f, (float)atof(next("--vad-threshold"))));
        }
        else if (a == "--vad-end-silence-ms" || a == "--vad-silence-ms" || a == "--silence-ms") {
            g_vad_cfg.end_silence_ms = std::max(0, atoi(next(a.c_str())));
        }
        else if (a == "--vad-partial-ms") {
            g_vad_cfg.partial_ms = std::max(0, atoi(next("--vad-partial-ms")));
        }
        else if (a == "--vad-min-segment-ms") {
            g_vad_cfg.min_segment_ms = std::max(0, atoi(next("--vad-min-segment-ms")));
        }
        else if (a == "--vad-max-segment-ms") {
            g_vad_cfg.max_segment_ms = std::max(0, atoi(next("--vad-max-segment-ms")));
        }
        else if (a == "--spk-model" || a == "--speaker-model" ||
                 a == "--speaker") {
            g_speaker_model_path = next(a.c_str());
        }
        else if (a == "--spk-cluster" || a == "--speaker-cluster") {
            g_speaker_cluster = next(a.c_str());
            if (g_speaker_cluster != "ahc" &&
                g_speaker_cluster != "spectral") {
                fprintf(stderr,
                        "[server] warning: unsupported speaker cluster '%s'; "
                        "using spectral\n",
                        g_speaker_cluster.c_str());
                g_speaker_cluster = "spectral";
            }
        }
        else if (a == "--spk-threshold" || a == "--speaker-threshold") {
            g_speaker_threshold =
                std::max(-1.0f, std::min(1.0f, (float)atof(next(a.c_str()))));
        }
        else if (a == "--num-speakers") {
            g_speaker_num_speakers = std::max(0, atoi(next("--num-speakers")));
        }
        else if (a == "--max-speakers") {
            g_speaker_max_speakers = std::max(1, atoi(next("--max-speakers")));
        }
        else if (a == "--speaker-log" || a == "--spk-log") {
            g_speaker_log_path = next(a.c_str());
        }
        else if (a == "--two-pass") g_asr_cfg.two_pass = true;
        else if (a == "--ctc-precheck") g_asr_cfg.ctc_precheck = true;
        else if (a == "--asr-slots") {
            g_asr_slots = std::max(1, atoi(next("--asr-slots")));
        }
        else if (a == "--asr-max-queued") {
            g_asr_max_queued = std::max(1, atoi(next("--asr-max-queued")));
        }
        else if (a == "--asr-batch-wait-ms") {
            g_asr_batch_wait_ms = std::max(0, atoi(next("--asr-batch-wait-ms")));
        }
        else if (a == "--asr-min-batch") {
            g_asr_min_batch = std::max(1, atoi(next("--asr-min-batch")));
        }
        else if (a == "--asr-max-batch-wait-ms") {
            g_asr_max_batch_wait_ms =
                std::max(0, atoi(next("--asr-max-batch-wait-ms")));
        }
        else if (a == "--asr-frontend-threads") {
            g_asr_frontend_threads =
                std::max(1, atoi(next("--asr-frontend-threads")));
        }
        else if (a == "--asr-frontend-workers") {
            g_asr_frontend_workers =
                std::max(1, atoi(next("--asr-frontend-workers")));
        }
        else if (a == "--tts-slots") {
            g_tts_slots = std::max(1, atoi(next("--tts-slots")));
        }
        else if (a == "--tts-voc-workers" || a == "--voc-workers") {
            g_tts_voc_workers = std::max(1, atoi(next("--tts-voc-workers")));
        }
        else if (a == "--tts-stream-mode") {
            g_tts_stream_mode = normalize_tts_stream_mode(
                next("--tts-stream-mode"));
        }
        else if (a == "--tts-segment-max-chars") {
            g_tts_segment_max_chars = std::max(
                16, std::min(512, atoi(next("--tts-segment-max-chars"))));
        }
        else if (a == "--mem-log") g_mem_log = true;
        else if (a == "--ws-log") g_ws_log = true;
        else if (a == "--asr-reload-every") {
            g_asr_reload_every = std::max(0, atoi(next("--asr-reload-every")));
        }
        else if (a == "--host") host = next("--host");
        else if (a == "--port") port = atoi(next("--port"));
        else if (a == "--ws-port") ws_port = atoi(next("--ws-port"));
        else if (a == "--web-dir") web_dir = next("--web-dir");
        else if (a == "--mcp-stdio") mcp_stdio = true;
        else if (a == "-t" || a == "--threads") threads = atoi(next("--threads"));
        else if (a == "--cpu") gpu = false;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (ws_port < 0) ws_port = port + 1;
    g_asr_model_path = asr_model;
    g_model_threads = threads;
    g_model_gpu = gpu;
    if (asr_model.empty() && tts_model.empty()) {
        fprintf(stderr, "[server] warning: no model given; inference will return "
                        "errors. Pass --asr-model / --tts-model.\n");
    }

    // In stdio-MCP mode the JSON-RPC stream owns stdout, but the framework's
    // RS_LOG (and any stray printf) also write to stdout — which would corrupt
    // it. Redirect fd 1 → stderr and keep the real stdout for JSON-RPC only.
    FILE *rpc_out = stdout;
    if (mcp_stdio) {
        rs_log_set_level(RSLogLevel::RS_LOG_LEVEL_WARN);
        int saved = RS_DUP(RS_FILENO(stdout));
        RS_DUP2(RS_FILENO(stderr), RS_FILENO(stdout));
        rpc_out = fdopen(saved, "w");
        if (!rpc_out) rpc_out = stderr;
    }

    if (!asr_model.empty()) {
        fprintf(stderr, "[server] loading ASR: %s\n", asr_model.c_str());
        if (!g_asr.load(asr_model, threads, gpu)) {
            fprintf(stderr, "[server] failed to load ASR model\n");
            return 1;
        }
        g_have_asr = true;
        g_current_asr_model_spec = asr_model;
        if (g_asr.ctx && g_asr.ctx->ctx_gguf) {
            load_cmvn_params(g_asr.ctx->ctx_gguf, g_asr_cmvn_means,
                             g_asr_cmvn_vars);
            g_asr_cmvn_version++;
        }
        fprintf(stderr, "[server] ASR ready: %s @ %d Hz\n", g_asr.arch.c_str(), g_asr.sample_rate);
        fprintf(stderr, "[server] ASR decode: two-pass=%s ctc-precheck=%s\n",
                g_asr_cfg.two_pass ? "on" : "off",
                g_asr_cfg.ctc_precheck ? "on" : "off");
        if (g_mem_log) {
            std::lock_guard<std::mutex> lk(g_asr.mu);
            log_asr_memory_locked("after-load");
        }
        if (g_asr_reload_every > 0) {
            fprintf(stderr, "[server] ASR auto-reload enabled: every %d final segment(s)\n",
                    g_asr_reload_every);
        }
        if (g_asr_slots > 1 &&
            (g_asr.arch == "SenseVoiceSmall" || g_asr.arch == "FunASRNano")) {
            const int asr_max_batch_wait_ms =
                g_asr_max_batch_wait_ms < 0
                    ? std::max(g_asr_batch_wait_ms,
                               g_asr_batch_wait_ms * 4)
                    : g_asr_max_batch_wait_ms;
            const int asr_frontend_workers =
                g_asr_frontend_workers > 0
                    ? g_asr_frontend_workers
                    : g_asr_slots;
            const int asr_min_batch =
                g_asr_min_batch <= 0
                    ? g_asr_slots
                    : std::min(g_asr_min_batch, g_asr_slots);
            g_asr_batch = std::make_unique<AsrBatchScheduler>(
                g_asr_slots, g_asr_max_queued, g_asr_batch_wait_ms,
                g_asr_min_batch, asr_max_batch_wait_ms,
                asr_frontend_workers);
            g_asr_batch->Start();
            fprintf(stderr,
                    "[server] ASR scheduler: slots=%d max_queued=%d "
                    "batch_wait=%dms min_batch=%d max_wait=%dms "
                    "frontend_workers=%d frontend_threads=%d\n",
                    g_asr_slots, g_asr_max_queued, g_asr_batch_wait_ms,
                    asr_min_batch, asr_max_batch_wait_ms,
                    asr_frontend_workers, g_asr_frontend_threads);
        } else if (g_asr_slots > 1) {
            fprintf(stderr,
                    "[server] ASR batching unavailable for arch=%s; using serialized path\n",
                    g_asr.arch.c_str());
        }
    }
    if (!g_speaker_model_path.empty()) {
        fprintf(stderr, "[server] loading speaker model: %s\n",
                g_speaker_model_path.c_str());
        g_speaker = rs_speaker_init_from_file(g_speaker_model_path.c_str(),
                                              threads, gpu);
        if (!g_speaker) {
            rs_error_info_t e = rs_get_last_error();
            fprintf(stderr, "[server] failed to load speaker model: %s\n",
                    e.message[0] ? e.message : "unknown error");
            return 1;
        }
        g_have_speaker = true;
        g_current_speaker_model_spec = g_speaker_model_path;
        fprintf(stderr,
                "[server] Speaker ready: CAMPPlus dim=%d sr=%d cluster=%s "
                "threshold=%.2f num_speakers=%d max_speakers=%d log=%s\n",
                rs_speaker_dim(g_speaker), rs_speaker_sample_rate(g_speaker),
                g_speaker_cluster.c_str(), g_speaker_threshold,
                g_speaker_num_speakers, g_speaker_max_speakers,
                g_speaker_log_path.empty() ? "off" : g_speaker_log_path.c_str());
    }
    if (!tts_model.empty()) {
        fprintf(stderr, "[server] loading TTS: %s\n", tts_model.c_str());
        // Load as TTS_ONLINE so rs_process streams per-chunk where the model
        // supports it; other TTS models fall back to offline but
        // still work through the same collect loop.
        if (!g_tts.load(tts_model, threads, gpu, RS_TASK_TTS_ONLINE)) {
            fprintf(stderr, "[server] failed to load TTS model\n");
            return 1;
        }
        g_have_tts = true;
        g_current_tts_model_spec = tts_model;
        fprintf(stderr, "[server] TTS ready: %s @ %d Hz\n", g_tts.arch.c_str(), g_tts.sample_rate);

        // CosyVoice3 uses continuous LM batching. VoxCPM2 is initialized below
        // with independent per-request states and backend schedulers.
        if (g_tts_slots > 1 && g_tts.arch == "cosyvoice3") {
            auto *cv3 = dynamic_cast<CosyVoice3LMModel *>(g_tts.ctx->model.get());
            if (cv3) {
                // Try to create a 2nd GPU sched for LM || Flow/HiFT concurrency
                // (CUDA only - shares weights, independent stream). Metal/CPU
                // returns nullptr -> single-worker path.
                auto scheds_voc =g_tts.ctx->create_extra_gpu_scheds(g_tts_voc_workers);
                g_tts_batch = std::make_unique<TtsBatchScheduler>(
                    cv3, g_tts.ctx->sched, std::move(scheds_voc), g_tts_slots);
                g_tts_batch->Start();
                fprintf(stderr,
                        "[server] TTS continuous batching: %d slots (%s)\n",
                        g_tts_slots, scheds_voc.empty() ? "single-worker" : (std::to_string(scheds_voc.size()) + "-worker CUDA").c_str());
            } else {
                fprintf(stderr,
                        "[server] TTS batching unavailable (model cast failed); "
                        "using serialized path\n");
            }
        }
        if (g_tts_slots > 1 && g_tts.arch == "voxcpm2") {
            auto *voxcpm2 =
                dynamic_cast<VoxCPM2Model *>(g_tts.ctx->model.get());
            if (voxcpm2) {
                auto extra = g_tts.ctx->create_extra_gpu_scheds(
                    g_tts_slots - 1);
                std::vector<ggml_backend_sched_t> scheds;
                scheds.push_back(g_tts.ctx->sched);
                scheds.insert(scheds.end(), extra.begin(), extra.end());
                g_voxcpm2_batch =
                    std::make_unique<VoxCPM2ServerScheduler>(
                        voxcpm2, std::move(scheds), g_tts_slots);
                g_voxcpm2_batch->Start();
                fprintf(stderr,
                        "[server] VoxCPM2 TTS scheduler: %d slots, %d "
                        "compute worker(s)%s\n",
                        g_tts_slots, g_voxcpm2_batch->n_workers(),
                        g_voxcpm2_batch->n_workers() > 1
                            ? " (parallel CUDA schedulers)"
                            : " (serialized backend queue)");
            } else {
                fprintf(stderr,
                        "[server] VoxCPM2 batching unavailable "
                        "(model cast failed)\n");
            }
        }
    }
    if (!vad_model.empty()) {
        fprintf(stderr, "[server] loading VAD: %s\n", vad_model.c_str());
        g_vad = rs_vad_init_from_file(vad_model.c_str(), threads, gpu);
        if (!g_vad) {
            fprintf(stderr, "[server] failed to load VAD model\n");
            return 1;
        }
        rs_vad_set_threshold(g_vad, g_vad_cfg.threshold);
        for (int i = 0; i < g_vad_workers; ++i) {
            rs_vad_t *worker =
                rs_vad_init_from_file(vad_model.c_str(), threads, gpu);
            if (!worker) {
                fprintf(stderr,
                        "[server] failed to load VAD worker %d/%d\n",
                        i + 1, g_vad_workers);
                return 1;
            }
            rs_vad_set_threshold(worker, g_vad_cfg.threshold);
            g_vad_ws_pool.push_back(worker);
            g_vad_ws_busy.push_back(false);
        }
        g_have_vad = true;
        g_current_vad_model_spec = vad_model;
        fprintf(stderr,
                "[server] VAD ready: %s  threshold=%.2f silence=%dms partial=%dms "
                "min-seg=%dms max-seg=%dms ws-workers=%d\n",
                rs_vad_get_arch(g_vad), g_vad_cfg.threshold,
                g_vad_cfg.end_silence_ms, g_vad_cfg.partial_ms, g_vad_cfg.min_segment_ms,
                g_vad_cfg.max_segment_ms, g_vad_workers);
    }

    if (mcp_stdio) {
        fprintf(stderr, "[server] MCP stdio mode\n");
        return run_mcp_stdio(rpc_out);
    }

    httplib::Server srv;
    // CORS: allow the web UI (or any browser client) to call the API, including
    // when opened from file://. Also answer preflight OPTIONS for every path.
    srv.set_post_routing_handler([](const httplib::Request &, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    });
    srv.Options(R"(.*)", [](const httplib::Request &, httplib::Response &res) {
        res.status = 204;
    });
    // Optional static web UI (examples/server/webui.html): serve it same-origin
    // so getUserMedia (mic) works. Enable with --web-dir <dir>.
    if (!web_dir.empty()) {
        if (!srv.set_mount_point("/", web_dir))
            fprintf(stderr, "[server] warning: could not mount web dir %s\n", web_dir.c_str());
    }
    srv.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        rsjson::Value o = rsjson::Value::object();
        o["status"] = "ok";
        if (g_asr_batch) {
            auto s = g_asr_batch->snapshot();
            rsjson::Value b = rsjson::Value::object();
            b["slots"] = s.slots;
            b["queued"] = s.queued;
            b["raw_queued"] = s.raw_queued;
            b["preparing"] = s.preparing;
            b["max_queued"] = s.max_queued;
            b["batch_wait_ms"] = s.batch_wait_ms;
            b["min_batch"] = s.min_batch;
            b["max_batch_wait_ms"] = s.max_batch_wait_ms;
            b["frontend_threads"] = g_asr_frontend_threads;
            b["frontend_workers"] = s.frontend_workers;
            b["submitted"] = (int64_t)s.submitted;
            b["completed"] = (int64_t)s.completed;
            b["failed"] = (int64_t)s.failed;
            b["rejected"] = (int64_t)s.rejected;
            b["batches"] = (int64_t)s.batches;
            b["batched_requests"] = (int64_t)s.batched_requests;
            b["max_batch"] = (int64_t)s.max_batch;
            rsjson::Value hist = rsjson::Value::array();
            for (uint64_t v : s.batch_size_hist) {
                hist.push_back((int64_t)v);
            }
            b["batch_size_hist"] = hist;
            b["avg_batch"] = s.batches
                                 ? (double)s.batched_requests / (double)s.batches
                                 : 0.0;
            b["avg_queue_wait_ms"] = s.completed
                                         ? (double)s.queue_wait_ms /
                                               (double)s.completed
                                         : 0.0;
            b["avg_frontend_ms"] = s.completed
                                       ? (double)s.frontend_ms /
                                             (double)s.completed
                                       : 0.0;
            b["avg_e2e_ms"] = s.completed
                                  ? (double)s.e2e_ms / (double)s.completed
                                  : 0.0;
            b["avg_encode_ms_per_batch"] = s.batches
                                               ? (double)s.encode_ms /
                                                     (double)s.batches
                                               : 0.0;
            b["avg_decode_ms_per_batch"] = s.batches
                                               ? (double)s.decode_ms /
                                                     (double)s.batches
                                               : 0.0;
            o["asr_batch"] = b;
        }
        if (g_tts_batch) {
            o["tts_batch"]["slots"] = g_tts_batch->n_slots();
            o["tts_batch"]["active"] = g_tts_batch->n_active();
            o["tts_batch"]["queued"] = g_tts_batch->n_queued();
        } else if (g_voxcpm2_batch) {
            o["tts_batch"]["slots"] = g_voxcpm2_batch->n_slots();
            o["tts_batch"]["workers"] = g_voxcpm2_batch->n_workers();
            o["tts_batch"]["active"] = g_voxcpm2_batch->n_active();
            o["tts_batch"]["queued"] = g_voxcpm2_batch->n_queued();
        }
        if (g_have_vad) {
            std::lock_guard<std::mutex> lk(g_vad_ws_pool_mu);
            int busy = 0;
            for (bool v : g_vad_ws_busy) {
                if (v) busy++;
            }
            o["vad_workers"]["slots"] = (int)g_vad_ws_pool.size();
            o["vad_workers"]["busy"] = busy;
        }
        if (g_have_speaker && g_speaker) {
            std::lock_guard<std::mutex> lk(g_speaker_mu);
            rsjson::Value spk = rsjson::Value::object();
            spk["model"] = "campplus";
            spk["loaded"] = true;
            spk["dim"] = rs_speaker_dim(g_speaker);
            spk["sample_rate"] = rs_speaker_sample_rate(g_speaker);
            spk["cluster"] = g_speaker_cluster;
            spk["threshold"] = g_speaker_threshold;
            spk["num_speakers"] = g_speaker_num_speakers;
            spk["max_speakers"] = g_speaker_max_speakers;
            spk["log"] = !g_speaker_log_path.empty();
            o["speaker"] = spk;
        }
        send_json(res, 200, o);
    });
    srv.Get("/v1/models", handle_models);
    srv.Get("/v1/models/registry", handle_model_registry);
    srv.Post("/v1/models/load", handle_model_load);
    srv.Post("/v1/audio/transcriptions", handle_transcriptions);
    srv.Post("/v1/audio/speech", handle_speech);

    // MCP over Streamable HTTP: one JSON-RPC message per POST. Requests get a
    // JSON response; notifications get 202 with no body.
    srv.Post("/mcp", [](const httplib::Request &req, httplib::Response &res) {
        rsjson::Value msg;
        try { msg = rsjson::parse(req.body); }
        catch (const std::exception &e) {
            rsjson::Value err = rsjson::Value::object();
            err["jsonrpc"] = "2.0";
            err["error"]["code"] = -32700;
            err["error"]["message"] = std::string("parse error: ") + e.what();
            send_json(res, 400, err);
            return;
        }
        bool has_response = true;
        rsjson::Value resp = mcp_dispatch(msg, has_response);
        if (has_response) send_json(res, 200, resp);
        else res.status = 202;
    });

    // WebSocket server on its own port/thread (cpp-httplib is HTTP-only).
    rsws::Server ws;
    ws.route("/asr/stream", ws_asr_stream);
    ws.route("/asr/vad", ws_asr_vad);
    ws.route("/tts/segmented", ws_tts_segmented);
    ws.route("/tts/stream", ws_tts_stream);
    std::thread ws_thread([&]() {
        if (!ws.listen(host, ws_port))
            fprintf(stderr, "[server] failed to bind WebSocket %s:%d\n", host.c_str(), ws_port);
    });
    ws_thread.detach();

    fprintf(stderr, "[server] listening on http://%s:%d  (ASR=%s TTS=%s VAD=%s)\n",
            host.c_str(), port, g_have_asr ? "on" : "off",
            g_have_tts ? "on" : "off", g_have_vad ? "on" : "off");
    fprintf(stderr, "[server]   OpenAI:  POST /v1/audio/transcriptions, /v1/audio/speech\n");
    fprintf(stderr, "[server]   MCP:     POST /mcp\n");
    fprintf(stderr, "[server]   WS:      ws://%s:%d/{asr/stream,asr/vad,tts/segmented,tts/stream}\n",
            host.c_str(), ws_port);
    bool listened = srv.listen(host, port);
    if (g_speaker) {
        rs_speaker_free(g_speaker);
        g_speaker = nullptr;
        g_have_speaker = false;
    }
    if (!listened) {
        fprintf(stderr, "[server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
