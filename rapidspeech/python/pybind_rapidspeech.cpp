#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rapidspeech.h"

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static std::string rs_format_error(const char *prefix) {
    rs_error_info_t err = rs_get_last_error();
    std::string msg = prefix ? prefix : "rapidspeech error";
    if (err.code != RS_OK) {
        msg += " (code=";
        msg += std::to_string((int)err.code);
        msg += "): ";
        msg += err.message;
    }
    return msg;
}

static void check_rs(rs_error_t e, const char *what) {
    if (e != RS_OK) {
        throw std::runtime_error(rs_format_error(what));
    }
}

// -----------------------------------------------------------------------------
// ASR Offline
// -----------------------------------------------------------------------------

class RSAsrOffline {
public:
    RSAsrOffline(const std::string &model_path,
                 int n_threads = 4,
                 bool use_gpu = true) {
        rs_init_params_t p = rs_default_params();
        p.model_path = model_path.c_str();
        p.n_threads  = n_threads;
        p.use_gpu    = use_gpu;
        p.task_type  = RS_TASK_ASR_OFFLINE;

        ctx_ = rs_init_from_file(p);
        if (!ctx_) {
            throw std::runtime_error(rs_format_error("Failed to initialize ASR context"));
        }
    }

    ~RSAsrOffline() {
        if (ctx_) {
            rs_free(ctx_);
            ctx_ = nullptr;
        }
    }

    void push_audio(py::array_t<float, py::array::c_style | py::array::forcecast> pcm) {
        auto buf = pcm.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("PCM must be a 1-D float32 array");
        }
        float *data = static_cast<float *>(buf.ptr);
        int n = static_cast<int>(buf.shape[0]);
        check_rs(rs_push_audio(ctx_, data, n), "rs_push_audio failed");
    }

    int process() { return rs_process(ctx_); }

    int redecode() { return rs_redecode(ctx_); }

    void reset() { check_rs(rs_reset(ctx_), "rs_reset failed"); }

    std::string get_text() {
        const char *res = rs_get_text_output(ctx_);
        return res ? std::string(res) : std::string();
    }

    void set_user_input_prompt(const std::string &prompt) {
        check_rs(rs_set_user_input_prompt(ctx_, prompt.c_str()),
                 "rs_set_user_input_prompt failed");
    }

    void set_use_llm(bool enable) {
        check_rs(rs_set_use_llm(ctx_, enable), "rs_set_use_llm failed");
    }

    void set_ctc_precheck(bool enable) {
        check_rs(rs_set_ctc_precheck(ctx_, enable), "rs_set_ctc_precheck failed");
    }

    py::dict get_model_meta() const {
        rs_model_meta_t m = rs_get_model_meta(ctx_);
        py::dict d;
        d["arch_name"]         = std::string(m.arch_name);
        d["audio_sample_rate"] = m.audio_sample_rate;
        d["n_mels"]            = m.n_mels;
        d["vocab_size"]        = m.vocab_size;
        return d;
    }

    std::string get_backend_name() const {
        const char *n = rs_get_backend_name(ctx_);
        return n ? std::string(n) : std::string();
    }

private:
    rs_context_t *ctx_ = nullptr;
};

// -----------------------------------------------------------------------------
// TTS Synthesizer
// -----------------------------------------------------------------------------

class RSTTSSynthesizer {
public:
    RSTTSSynthesizer(const std::string &model_path,
                     int n_threads = 4,
                     bool use_gpu = true) {
        rs_init_params_t p = rs_default_params();
        p.model_path = model_path.c_str();
        p.n_threads  = n_threads;
        p.use_gpu    = use_gpu;
        p.task_type  = RS_TASK_TTS_ONLINE;

        ctx_ = rs_init_from_file(p);
        if (!ctx_) {
            throw std::runtime_error(rs_format_error("Failed to initialize TTS context"));
        }
    }

    ~RSTTSSynthesizer() {
        if (ctx_) {
            rs_free(ctx_);
            ctx_ = nullptr;
        }
    }

    // ---- Configuration ----

    void set_params(const std::string &instruct,
                    const std::string &language,
                    int seed) {
        check_rs(rs_set_tts_params(ctx_, instruct.c_str(), language.c_str(), seed),
                 "rs_set_tts_params failed");
    }

    void set_diffusion_steps(int n_steps) {
        check_rs(rs_set_tts_diffusion_steps(ctx_, n_steps),
                 "rs_set_tts_diffusion_steps failed");
    }

    // ---- Voice cloning ----

    void set_reference_audio(py::array_t<float, py::array::c_style | py::array::forcecast> pcm,
                             int sample_rate = 16000) {
        auto buf = pcm.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("PCM must be a 1-D float32 array");
        }
        if (rs_push_reference_audio(ctx_,
                                    static_cast<float *>(buf.ptr),
                                    static_cast<int>(buf.shape[0]),
                                    sample_rate) != 0) {
            throw std::runtime_error(rs_format_error("rs_push_reference_audio failed"));
        }
    }

    void set_reference_text(const std::string &text) {
        check_rs(rs_push_reference_text(ctx_, text.c_str()),
                 "rs_push_reference_text failed");
    }

    // ---- Synthesis ----

    py::array_t<float> synthesize(const std::string &text) {
        rs_reset(ctx_);
        check_rs(rs_push_text(ctx_, text.c_str()), "rs_push_text failed");

        std::vector<float> all_pcm;
        int ret;
        while ((ret = rs_process(ctx_)) >= 0) {
            float *chunk = nullptr;
            int n = rs_get_audio_output(ctx_, &chunk);
            if (n > 0 && chunk) {
                all_pcm.insert(all_pcm.end(), chunk, chunk + n);
            }
            if (ret == 0) break;
        }
        if (ret < 0) {
            throw std::runtime_error(rs_format_error("TTS inference error"));
        }
        auto result = py::array_t<float>(static_cast<py::ssize_t>(all_pcm.size()));
        std::memcpy(result.mutable_data(), all_pcm.data(), all_pcm.size() * sizeof(float));
        return result;
    }

    py::list synthesize_streaming(const std::string &text) {
        rs_reset(ctx_);
        check_rs(rs_push_text(ctx_, text.c_str()), "rs_push_text failed");

        py::list chunks;
        int ret;
        while ((ret = rs_process(ctx_)) >= 0) {
            float *chunk = nullptr;
            int n = rs_get_audio_output(ctx_, &chunk);
            if (n > 0 && chunk) {
                auto arr = py::array_t<float>(n);
                std::memcpy(arr.mutable_data(), chunk, n * sizeof(float));
                chunks.append(arr);
            }
            if (ret == 0) break;
        }
        if (ret < 0) {
            throw std::runtime_error(rs_format_error("TTS inference error"));
        }
        return chunks;
    }

    // ---- Metadata ----

    py::dict get_model_meta() const {
        rs_model_meta_t m = rs_get_model_meta(ctx_);
        py::dict d;
        d["arch_name"]         = std::string(m.arch_name);
        d["audio_sample_rate"] = m.audio_sample_rate;
        d["n_mels"]            = m.n_mels;
        d["vocab_size"]        = m.vocab_size;
        return d;
    }

    std::string get_backend_name() const {
        const char *n = rs_get_backend_name(ctx_);
        return n ? std::string(n) : std::string();
    }

    int get_sample_rate() const {
        rs_model_meta_t m = rs_get_model_meta(ctx_);
        return m.audio_sample_rate;
    }

private:
    rs_context_t *ctx_ = nullptr;
};

// -----------------------------------------------------------------------------
// VAD (silero-vad / firered-vad — auto-detected from GGUF `general.architecture`)
// -----------------------------------------------------------------------------

class RSVad {
public:
    RSVad(const std::string &model_path, int n_threads = 2, bool use_gpu = false) {
        vad_ = rs_vad_init_from_file(model_path.c_str(), n_threads, use_gpu);
        if (!vad_) {
            throw std::runtime_error(rs_format_error("Failed to initialize VAD"));
        }
    }

    ~RSVad() {
        if (vad_) {
            rs_vad_free(vad_);
            vad_ = nullptr;
        }
    }

    void reset() { check_rs(rs_vad_reset(vad_), "rs_vad_reset failed"); }

    void set_threshold(float threshold) {
        check_rs(rs_vad_set_threshold(vad_, threshold), "rs_vad_set_threshold failed");
    }

    void push_audio(py::array_t<float, py::array::c_style | py::array::forcecast> pcm) {
        auto buf = pcm.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("PCM must be a 1-D float32 array");
        }
        check_rs(rs_vad_push_audio(vad_,
                                   static_cast<float *>(buf.ptr),
                                   static_cast<int>(buf.shape[0])),
                 "rs_vad_push_audio failed");
    }

    bool is_speech() const { return rs_vad_is_speech(vad_) != 0; }
    float get_probability() const { return rs_vad_get_probability(vad_); }
    std::string get_arch() const {
        const char *a = rs_vad_get_arch(vad_);
        return a ? std::string(a) : std::string();
    }

    py::list drain_segments(int max_count = 64) {
        std::vector<rs_vad_segment_t> buf(max_count);
        int n = rs_vad_drain_segments(vad_, buf.data(), max_count);
        py::list out;
        for (int i = 0; i < n; ++i) {
            py::dict d;
            d["start_s"] = buf[i].start_s;
            d["end_s"]   = buf[i].end_s;
            out.append(d);
        }
        return out;
    }

    py::list drain_frames(int max_count = 256) {
        std::vector<rs_vad_frame_t> buf(max_count);
        int n = rs_vad_drain_frames(vad_, buf.data(), max_count);
        py::list out;
        for (int i = 0; i < n; ++i) {
            py::dict d;
            d["frame_idx"]       = buf[i].frame_idx;
            d["raw_prob"]        = buf[i].raw_prob;
            d["smoothed_prob"]   = buf[i].smoothed_prob;
            d["is_speech"]       = buf[i].is_speech != 0;
            d["is_speech_start"] = buf[i].is_speech_start != 0;
            d["is_speech_end"]   = buf[i].is_speech_end != 0;
            out.append(d);
        }
        return out;
    }

    // One-shot: reset, push the whole clip, flush any open segment.
    py::list detect_full(py::array_t<float, py::array::c_style | py::array::forcecast> pcm,
                         int max_segments = 1024) {
        auto buf = pcm.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("PCM must be a 1-D float32 array");
        }
        std::vector<rs_vad_segment_t> seg(max_segments);
        int n = rs_vad_detect_full(vad_,
                                   static_cast<float *>(buf.ptr),
                                   static_cast<int>(buf.shape[0]),
                                   seg.data(), max_segments);
        py::list out;
        for (int i = 0; i < std::min(n, max_segments); ++i) {
            py::dict d;
            d["start_s"] = seg[i].start_s;
            d["end_s"]   = seg[i].end_s;
            out.append(d);
        }
        return out;
    }

private:
    rs_vad_t *vad_ = nullptr;
};

// -----------------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(rapidspeech, m) {
    m.doc() = "RapidSpeech Python bindings — offline/online ASR & TTS on ggml";

    m.def("version", &rs_get_version, "Get the rapidspeech library version string");

    // ── ASR ────────────────────────────────────────────────────────
    py::class_<RSAsrOffline>(m, "asr_offline")
        .def(py::init<const std::string &, int, bool>(),
             py::arg("model_path"),
             py::arg("n_threads") = 4,
             py::arg("use_gpu")   = true,
             "Load an offline ASR model (e.g. SenseVoice / FunASRNano).")

        .def("push_audio", &RSAsrOffline::push_audio,
             py::arg("pcm"),
             py::call_guard<py::gil_scoped_release>(),
             "Push mono float32 PCM in [-1, 1] at the model's native sample rate.")

        .def("process", &RSAsrOffline::process,
             py::call_guard<py::gil_scoped_release>(),
             "Run one inference step. Returns 0 (no output), 1 (has output), -1 (error).")

        .def("redecode", &RSAsrOffline::redecode,
             py::call_guard<py::gil_scoped_release>(),
             "Re-run the decoder on the cached encoder output (e.g. after set_use_llm).")

        .def("reset", &RSAsrOffline::reset,
             "Clear audio buffer and text accumulator.")

        .def("get_text", &RSAsrOffline::get_text,
             "Return the transcribed text from the last process() / redecode().")

        .def("set_user_input_prompt", &RSAsrOffline::set_user_input_prompt,
             py::arg("prompt"),
             "Set the LLM decoder prompt (default: \"语音转写：\"; FunASRNano only).")

        .def("set_use_llm", &RSAsrOffline::set_use_llm,
             py::arg("enable"),
             "Toggle the 2nd-pass LLM decoder (FunASRNano only). False = CTC greedy.")

        .def("set_ctc_precheck", &RSAsrOffline::set_ctc_precheck,
             py::arg("enable"),
             "Skip LLM decode on silence by running a quick CTC precheck first.")

        .def("get_model_meta", &RSAsrOffline::get_model_meta,
             "Return a dict with arch_name, audio_sample_rate, n_mels, vocab_size.")

        .def("get_backend_name", &RSAsrOffline::get_backend_name,
             "Return the ggml backend in use (e.g. \"CUDA0\", \"Metal\", \"CPU\").");

    // ── TTS ────────────────────────────────────────────────────────
    py::class_<RSTTSSynthesizer>(m, "tts_synthesizer")
        .def(py::init<const std::string &, int, bool>(),
             py::arg("model_path"),
             py::arg("n_threads") = 4,
             py::arg("use_gpu")   = true,
             "Load a TTS model (e.g. OmniVoice / OpenVoice2).")

        .def("set_params", &RSTTSSynthesizer::set_params,
             py::arg("instruct") = "male",
             py::arg("language") = "English",
             py::arg("seed")     = 42,
             "Set voice description, target language, and RNG seed (OmniVoice only).")

        .def("set_diffusion_steps", &RSTTSSynthesizer::set_diffusion_steps,
             py::arg("n_steps"),
             "Set MaskGIT diffusion steps (1-128, default 32; OmniVoice only).")

        .def("set_reference_audio", &RSTTSSynthesizer::set_reference_audio,
             py::arg("pcm"), py::arg("sample_rate") = 16000,
             "Set reference audio for voice cloning (mono float32 PCM).")

        .def("set_reference_text", &RSTTSSynthesizer::set_reference_text,
             py::arg("text"),
             "Set the transcript of the reference audio (required for voice cloning).")

        .def("synthesize", &RSTTSSynthesizer::synthesize,
             py::arg("text"),
             py::call_guard<py::gil_scoped_release>(),
             "Synthesize text and return the full PCM as a 1-D float32 numpy array.")

        .def("synthesize_streaming", &RSTTSSynthesizer::synthesize_streaming,
             py::arg("text"),
             py::call_guard<py::gil_scoped_release>(),
             "Synthesize text and return a list of PCM chunks (1-D float32 arrays).")

        .def("get_sample_rate", &RSTTSSynthesizer::get_sample_rate,
             "Return the audio sample rate produced by the TTS model.")

        .def("get_model_meta", &RSTTSSynthesizer::get_model_meta,
             "Return a dict with arch_name, audio_sample_rate, n_mels, vocab_size.")

        .def("get_backend_name", &RSTTSSynthesizer::get_backend_name,
             "Return the ggml backend in use (e.g. \"CUDA0\", \"Metal\", \"CPU\").");

    // Backwards-compatible alias (old name from earlier examples)
    m.attr("AsrOffline")      = m.attr("asr_offline");
    m.attr("TtsSynthesizer")  = m.attr("tts_synthesizer");

    // ── VAD ────────────────────────────────────────────────────────
    py::class_<RSVad>(m, "vad")
        .def(py::init<const std::string &, int, bool>(),
             py::arg("model_path"),
             py::arg("n_threads") = 2,
             py::arg("use_gpu")   = false,
             "Load a VAD model (silero-vad or firered-vad, auto-detected from GGUF).")

        .def("reset", &RSVad::reset,
             "Reset all streaming state (caches, postprocessor, sample counter).")

        .def("set_threshold", &RSVad::set_threshold,
             py::arg("threshold"),
             "Set speech-probability threshold (0..1).")

        .def("push_audio", &RSVad::push_audio,
             py::arg("pcm"),
             py::call_guard<py::gil_scoped_release>(),
             "Push 16 kHz mono float32 PCM. Drives the segment/frame queues.")

        .def("is_speech", &RSVad::is_speech,
             "Latest post-push speech-state.")

        .def("get_probability", &RSVad::get_probability,
             "Latest post-push speech probability (0..1).")

        .def("get_arch", &RSVad::get_arch,
             "Architecture string (\"silero-vad\" or \"firered-vad\").")

        .def("drain_segments", &RSVad::drain_segments,
             py::arg("max_count") = 64,
             "Drain queued [{start_s, end_s}, ...] segments. Empties internal queue.")

        .def("drain_frames", &RSVad::drain_frames,
             py::arg("max_count") = 256,
             "Drain queued per-frame events. Empties internal queue.")

        .def("detect_full", &RSVad::detect_full,
             py::arg("pcm"),
             py::arg("max_segments") = 1024,
             py::call_guard<py::gil_scoped_release>(),
             "One-shot offline detection: reset, push the whole clip, "
             "flush any open segment, return segments list.");

    m.attr("Vad") = m.attr("vad");
}
