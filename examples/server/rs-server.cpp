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
#include "rapidspeech.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"
#include "../common/rs_cli_utf8.h"

// TTS continuous batching (CosyVoice3 only; other archs fall back to the
// mutex-serialized path).
#include "arch/cosyvoice3.h"
#include "arch/cosyvoice3_batch.h"
#include "arch/xasr.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
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

namespace fs = std::filesystem;

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
static int g_model_threads = 4;
static bool g_model_gpu = true;
static int g_asr_reload_every = 0; // 0 = disabled
static int g_asr_final_segments_since_reload = 0;
static bool g_mem_log = false;

// TTS continuous-batching scheduler (CosyVoice3 only). When non-null, HTTP
// and WS TTS handlers Submit() to it instead of locking g_tts.mu, so up to
// --tts-slots requests decode concurrently (token-interleaved).
static std::unique_ptr<TtsBatchScheduler> g_tts_batch;
static int g_tts_slots = 2;
static int g_tts_voc_workers = 1;

struct AsrServerConfig {
    bool two_pass = false;
    bool ctc_precheck = false;
};
static AsrServerConfig g_asr_cfg;

// VAD is a separate handle (rs_vad_t), used by /asr/vad and HTTP VAD mode.
static rs_vad_t *g_vad = nullptr;
static std::mutex g_vad_mu;
static bool g_have_vad = false;

struct VadServerConfig {
    float threshold = 0.5f;
    int end_silence_ms = 600;
    int partial_ms = 600;
    int min_segment_ms = 500;
    int max_segment_ms = 0; // 0 = disabled
};
static VadServerConfig g_vad_cfg;

struct AsrSegment {
    int index = 0;
    float start_s = 0.0f;
    float end_s = 0.0f;
    std::string text;
};

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
        {
            std::lock_guard<std::mutex> alk(g_asr.mu);
            if (!transcribe_pcm_locked(seg.data(), (int)seg.size(), seg_text, err,
                                      asr_cfg.two_pass, asr_cfg.ctc_precheck)) {
                return false;
            }
            note_asr_final_segment_locked("http-vad-final");
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

static bool do_transcribe_with_config(const std::string &audio_bytes,
                                      const AsrServerConfig &cfg,
                                      std::string &text, std::string &err) {
    if (!g_have_asr) { err = "no ASR model loaded"; return false; }
    std::vector<float> pcm;
    if (!decode_audio(audio_bytes, g_asr.sample_rate, pcm)) {
        err = "could not decode audio (expected WAV)";
        return false;
    }
    std::lock_guard<std::mutex> lk(g_asr.mu);
    bool ok = transcribe_pcm_locked(pcm.data(), (int)pcm.size(), text, err,
                                    cfg.two_pass,
                                    cfg.ctc_precheck);
    if (ok) note_asr_final_segment_locked("http-final");
    return ok;
}

// Transcribe WAV bytes → text (HTTP/MCP path). Decodes then locks + transcribes.
static bool do_transcribe(const std::string &audio_bytes, std::string &text,
                          std::string &err) {
    return do_transcribe_with_config(audio_bytes, g_asr_cfg, text, err);
}

static bool do_transcribe_vad(const std::string &audio_bytes,
                              const AsrServerConfig &asr_cfg,
                              const VadServerConfig &vad_cfg,
                              std::vector<AsrSegment> &segments,
                              std::string &text, std::string &err) {
    if (!g_have_asr) { err = "no ASR model loaded"; return false; }
    if (!g_have_vad) { err = "no VAD model loaded (use --vad-model)"; return false; }
    std::vector<float> pcm;
    if (!decode_audio(audio_bytes, g_asr.sample_rate, pcm)) {
        err = "could not decode audio (expected WAV)";
        return false;
    }
    std::unique_lock<std::mutex> vlk(g_vad_mu);
    return transcribe_pcm_vad_locked(pcm, asr_cfg, vad_cfg, segments, text, err);
}

// Synthesize text → mono float PCM at g_tts.sample_rate.
static bool do_synthesize(const std::string &text, const std::string &instruct,
                          const std::string &language, int seed, int steps,
                          std::vector<float> &pcm, std::string &err) {
    if (!g_have_tts) { err = "no TTS model loaded"; return false; }

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
    bool use_vad = request_wants_vad(req);
    bool ok = use_vad
                  ? do_transcribe_vad(file.content, asr_cfg, vad_cfg,
                                      segments, text, err)
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
                arr.push_back(item);
            }
            o["segments"] = arr;
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

static void ws_send_json(rsws::Conn &c, const rsjson::Value &v) {
    c.send_text(v.dump());
}
static void ws_msg(rsws::Conn &c, const char *type, const std::string &key = "",
                   const std::string &val = "") {
    rsjson::Value o = rsjson::Value::object();
    o["type"] = type;
    if (!key.empty()) o[key] = val;
    ws_send_json(c, o);
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
    if (!g_have_asr) { ws_msg(c, "error", "error", "no ASR model loaded"); return; }
    if (!g_have_vad) { ws_msg(c, "error", "error", "no VAD model (use --vad-model)"); return; }
    std::unique_lock<std::mutex> vlk(g_vad_mu);
    AsrServerConfig asr_cfg = g_asr_cfg;
    VadServerConfig cfg = g_vad_cfg;
    const std::string &q = c.query();
    asr_cfg.two_pass = query_bool(q, "two_pass", asr_cfg.two_pass);
    asr_cfg.ctc_precheck = query_bool(q, "ctc_precheck", asr_cfg.ctc_precheck);
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

    rs_vad_reset(g_vad);
    rs_vad_set_threshold(g_vad, cfg.threshold);

    rsjson::Value ready = rsjson::Value::object();
    ready["type"] = "ready";
    ready["vad_threshold"] = cfg.threshold;
    ready["vad_end_silence_ms"] = cfg.end_silence_ms;
    ready["vad_partial_ms"] = cfg.partial_ms;
    ready["vad_min_segment_ms"] = cfg.min_segment_ms;
    ready["vad_max_segment_ms"] = cfg.max_segment_ms;
    ready["two_pass"] = asr_cfg.two_pass;
    ready["ctc_precheck"] = asr_cfg.ctc_precheck;
    ws_send_json(c, ready);

    const int sr = g_asr.sample_rate;
    std::vector<float> seg;       // current speech segment
    bool in_speech = false;
    bool pending_end = false;     // VAD ended, waiting for hangover silence
    int pending_silence_samples = 0;
    int seg_index = 0;
    size_t last_partial_len = 0;  // samples at last partial

    auto transcribe_seg = [&](const char *kind) {
        if (seg.empty()) return;
        int min_samples = (sr * cfg.min_segment_ms) / 1000;
        if ((int)seg.size() < min_samples) return;
        std::string text, err;
        bool is_partial = strcmp(kind, "partial") == 0;
        bool ok = false;
        {
            std::lock_guard<std::mutex> alk(g_asr.mu);
            ok = transcribe_pcm_locked(seg.data(), (int)seg.size(), text, err,
                                       asr_cfg.two_pass && !is_partial,
                                       asr_cfg.ctc_precheck,
                                       asr_cfg.two_pass && is_partial);
            if (ok && !is_partial) note_asr_final_segment_locked("ws-vad-final");
        }
        if (!ok) {
            ws_msg(c, "error", "error", err.empty() ? "ASR transcribe failed" : err);
            return;
        }
        if (is_partial || !text.empty()) {
            rsjson::Value o = rsjson::Value::object();
            o["type"] = kind; o["segment"] = seg_index; o["text"] = text;
            o["two_pass"] = asr_cfg.two_pass && !is_partial;
            o["ctc_only"] = asr_cfg.two_pass && is_partial;
            ws_send_json(c, o);
        }
    };

    auto finalize_seg = [&]() {
        transcribe_seg("final");
        in_speech = false;
        pending_end = false;
        pending_silence_samples = 0;
        seg_index++;
        seg.clear();
        last_partial_len = 0;
    };

    auto force_split_seg = [&]() {
        finalize_seg();
        in_speech = true;
    };

    std::vector<float> pcm;
    int op; std::string msg;
    auto handle_frames = [&]() {
        rs_vad_frame_t frames[64];
        int nf;
        while ((nf = rs_vad_drain_frames(g_vad, frames, 64)) > 0) {
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
            rs_vad_push_audio(g_vad, pcm.data(), (int)pcm.size());
            handle_frames();
            if (in_speech) {
                seg.insert(seg.end(), pcm.begin(), pcm.end());
                if (pending_end) pending_silence_samples += (int)pcm.size();
            }
            // Emit partials on a configurable cadence while speech is active.
            if (in_speech && !pending_end && cfg.partial_ms > 0 &&
                seg.size() - last_partial_len >= (size_t)((sr * cfg.partial_ms) / 1000)) {
                last_partial_len = seg.size();
                transcribe_seg("partial");
            }
            if (pending_end &&
                pending_silence_samples >= (sr * cfg.end_silence_ms) / 1000) {
                finalize_seg();
            }
            if (in_speech && cfg.max_segment_ms > 0 &&
                seg.size() >= (size_t)((sr * cfg.max_segment_ms) / 1000)) {
                if (pending_end) finalize_seg();
                else force_split_seg();
            }
        } else if (op == rsws::OP_TEXT) {
            rsjson::Value j;
            try { j = rsjson::parse(msg); } catch (...) { continue; }
            if (j["type"].as_string() == "eos") break;
        }
    }
    if (in_speech) transcribe_seg("final");
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
    if (!g_tts_batch) lk.lock();

    auto segs = split_sentences(text);
    for (size_t i = 0; i < segs.size(); i++) {
        std::vector<float> pcm;
        if (g_tts_batch) {
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

// (4) Pure streaming TTS: drive rs_process and emit each audio chunk immediately
// (low first-chunk latency for CosyVoice3; one big chunk for offline-only models).
// Batch mode: sentence-level pipelining — sentence N+1 decodes in the scheduler
// while sentence N streams to the client, and multiple connections interleave
// at token granularity instead of getting "TTS model busy".
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

    rsjson::Value info = rsjson::Value::object();
    info["type"] = "audio_info"; info["sample_rate"] = g_tts.sample_rate;
    info["format"] = "pcm_s16le";
    ws_send_json(c, info);

    std::unique_lock<std::mutex> lk(g_tts.mu, std::defer_lock);
    if (!g_tts_batch) lk.lock();

    if (g_tts_batch) {
        auto segs = split_sentences(text);
        if (segs.empty()) { ws_msg(c, "final"); return; }

        // First sentence: true streaming via DecodeStream (chunk-level audio,
        // first chunk ~350 ms — under 500 ms target). Runs on the vocoder sched
        // so the main LM thread can continue batched decode of other sessions.
        if (!segs.empty()) {
            struct ChunkState { rsws::Conn *c; std::string pcm16; };
            ChunkState cs{&c, {}};
            bool ok = g_tts_batch->StreamFirstSentence(
                segs[0], voice, lang, 42,
                [&cs](const float *pcm, size_t n) {
                    cs.c->send_binary(float_to_pcm16(pcm, n).data(), n * 2);
                });
            if (!ok) {
                ws_msg(c, "error", "error", "stream first sentence failed");
                return;
            }
        }

        // Remaining sentences: normal batch Submit (batched decode + flow/hift,
        // concurrent with other sessions' work on the main LM sched).
        if (segs.size() > 1) {
            std::vector<std::future<std::pair<std::vector<float>, std::string>>> futs;
            futs.reserve(segs.size() - 1);
            for (size_t i = 1; i < segs.size(); ++i)
                futs.push_back(g_tts_batch->Submit(segs[i], voice, lang, 42, sid));
            for (auto &fut : futs) {
                auto result = fut.get();
                if (!result.second.empty()) {
                    ws_msg(c, "error", "error", result.second);
                    continue;
                }
                std::string b = float_to_pcm16(result.first.data(), result.first.size());
                c.send_binary(b.data(), b.size());
            }
        }
        ws_msg(c, "final");
        return;
    }

    rs_reset(g_tts.ctx);
    rs_set_tts_params(g_tts.ctx, voice.c_str(), lang.c_str(), 42);
    if (rs_push_text(g_tts.ctx, text.c_str()) != RS_OK) {
        ws_msg(c, "error", "error", "rs_push_text failed");
        return;
    }
    int32_t ret;
    while ((ret = rs_process(g_tts.ctx)) >= 0) {
        float *ch = nullptr; int n = rs_get_audio_output(g_tts.ctx, &ch);
        if (n > 0 && ch) {
            std::string b = float_to_pcm16(ch, n);
            c.send_binary(b.data(), b.size());
        }
        if (ret == 0) break;
    }
    ws_msg(c, "final");
}

// ── main ────────────────────────────────────────────────────────────────

static void usage(const char *argv0) {
    fprintf(stderr,
        "RapidSpeech server — OpenAI-compatible HTTP API + MCP\n"
        "Usage: %s [--asr-model asr.gguf] [--tts-model tts.gguf] [options]\n"
        "  --asr-model PATH   ASR GGUF (enables /v1/audio/transcriptions + transcribe tool)"
        "  --tts-model PATH   TTS GGUF (enables /v1/audio/speech + synthesize tool)"
        "  --vad-model PATH   VAD GGUF (enables the /asr/vad WebSocket)"
        "  --vad-threshold F  VAD speech threshold (default 0.5)"
        "  --vad-end-silence-ms N  Silence before finalizing segment (default 600)"
        "  --vad-partial-ms N Partial ASR interval for /asr/vad (default 600, 0=off)"
        "  --vad-min-segment-ms N  Minimum segment to transcribe (default 500)"
        "  --vad-max-segment-ms N  Force-finalize long speech (default 0=off)"
        "  --two-pass        FunASR-Nano: CTC first pass + LLM final re-decode"
        "  --ctc-precheck    FunASR-Nano: skip LLM on CTC-empty silence/noise"
        "  --tts-slots N     TTS concurrent requests via continuous batching"
        "  --tts-voc-workers N  Flow/HiFT worker threads (CUDA streams, default 1)"
        "                    (CosyVoice3 only; default 2, 1=serialized)"
        "  --mem-log         Log ASR scheduler compute-buffer high-water marks"
        "  --asr-reload-every N  Reload ASR after N final segments to release GPU buffers (0=off)"
        "  --host HOST        HTTP bind host (default 127.0.0.1)"
        "  --port PORT        HTTP port (default 8080)"
        "  --ws-port PORT     WebSocket port (default HTTP port + 1)"
        "  --web-dir DIR      Serve a static web UI dir at / (e.g. examples/server)"
        "  --mcp-stdio        Run as an MCP server over stdio (no HTTP)"
        "  -t, --threads N    CPU threads per model (default 4)"
        "  --cpu              Disable GPU"
        "  -h, --help\n"
        "WebSocket endpoints (binary = int16 mono PCM, text = JSON):"
        "  /asr/stream      pure streaming ASR (X-ASR)     partial+final"
        "  /asr/vad         VAD-segmented offline ASR      partial+final per segment"
        "                   query overrides: vad_threshold, vad_end_silence_ms,"
        "                   vad_partial_ms,"
        "                   vad_min_segment_ms, vad_max_segment_ms,"
        "                   two_pass, ctc_precheck"
        "  /tts/segmented   sentence-by-sentence TTS       audio per segment"
        "  /tts/stream      pure streaming TTS             audio chunks as generated\n"
        "At least one of --asr-model / --tts-model is required.\n",
        argv0);
}

int main(int argc_raw, char **argv_raw) {
    rs::cli::Utf8Args utf8(argc_raw, argv_raw);
    int argc = utf8.argc();
    char **argv = utf8.argv();

    // Multi-worker uses two CUDA schedulers concurrently; CUDA graph capture
    // coordinates via a global lock that would serialize them. Disable it so
    // the two streams can overlap. (Harmless on non-CUDA - env is ignored.)
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 0);

    std::string asr_model, tts_model, vad_model, host = "127.0.0.1", web_dir;
    int port = 8080, ws_port = -1, threads = 4;
    bool gpu = true, mcp_stdio = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char *name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs an argument\n", name); exit(1); }
            return argv[++i];
        };
        if (a == "--asr-model") asr_model = next("--asr-model");
        else if (a == "--tts-model") tts_model = next("--tts-model");
        else if (a == "--vad-model") vad_model = next("--vad-model");
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
        else if (a == "--two-pass") g_asr_cfg.two_pass = true;
        else if (a == "--ctc-precheck") g_asr_cfg.ctc_precheck = true;
        else if (a == "--tts-slots") {
            g_tts_slots = std::max(1, atoi(next("--tts-slots")));
        }
        else if (a == "--tts-voc-workers" || a == "--voc-workers") {
            g_tts_voc_workers = std::max(1, atoi(next("--tts-voc-workers")));
        }
        else if (a == "--mem-log") g_mem_log = true;
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
    }
    if (!tts_model.empty()) {
        fprintf(stderr, "[server] loading TTS: %s\n", tts_model.c_str());
        // Load as TTS_ONLINE so rs_process streams per-chunk where the model
        // supports it (CosyVoice3); other TTS models fall back to offline but
        // still work through the same collect loop.
        if (!g_tts.load(tts_model, threads, gpu, RS_TASK_TTS_ONLINE)) {
            fprintf(stderr, "[server] failed to load TTS model\n");
            return 1;
        }
        g_have_tts = true;
        fprintf(stderr, "[server] TTS ready: %s @ %d Hz\n", g_tts.arch.c_str(), g_tts.sample_rate);

        // Continuous batching: CosyVoice3 only (its LM decode is split into
        // per-token steps the scheduler can interleave). Other archs keep the
        // mutex-serialized path.
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
    }
    if (!vad_model.empty()) {
        fprintf(stderr, "[server] loading VAD: %s\n", vad_model.c_str());
        g_vad = rs_vad_init_from_file(vad_model.c_str(), threads, gpu);
        if (!g_vad) {
            fprintf(stderr, "[server] failed to load VAD model\n");
            return 1;
        }
        rs_vad_set_threshold(g_vad, g_vad_cfg.threshold);
        g_have_vad = true;
        fprintf(stderr,
                "[server] VAD ready: %s  threshold=%.2f silence=%dms partial=%dms "
                "min-seg=%dms max-seg=%dms\n",
                rs_vad_get_arch(g_vad), g_vad_cfg.threshold,
                g_vad_cfg.end_silence_ms, g_vad_cfg.partial_ms, g_vad_cfg.min_segment_ms,
                g_vad_cfg.max_segment_ms);
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
        send_json(res, 200, o);
    });
    srv.Get("/v1/models", handle_models);
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
    if (!srv.listen(host, port)) {
        fprintf(stderr, "[server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
