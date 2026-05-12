#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rapidspeech.h"

namespace py = pybind11;

class RSAsrOffline {
public:
    RSAsrOffline(
        const std::string& model_path,
        int n_threads = 4,
        bool use_gpu = true
    ) {
        rs_init_params_t p = rs_default_params();
        p.model_path = model_path.c_str();
        p.n_threads  = n_threads;
        p.use_gpu    = use_gpu;

        ctx_ = rs_init_from_file(p);
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize RapidSpeech context");
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
            throw std::runtime_error("PCM must be 1-D float32 array");
        }

        float* data = static_cast<float*>(buf.ptr);
        int n = static_cast<int>(buf.shape[0]);

        if (rs_push_audio(ctx_, data, n) != 0) {
            throw std::runtime_error("rs_push_audio failed");
        }
    }

    int process() {
        return rs_process(ctx_);
    }

    std::string get_text() {
        const char* res = rs_get_text_output(ctx_);
        return res ? std::string(res) : std::string();
    }

    void set_user_input_prompt(const std::string& prompt) {
        if (rs_set_user_input_prompt(ctx_, prompt.c_str()) != RS_OK) {
            throw std::runtime_error("Failed to set user input prompt");
        }
    }

private:
    rs_context_t* ctx_ = nullptr;
};

// TTS Synthesizer: text-to-speech with optional voice cloning
class RSTTSSynthesizer {
public:
    RSTTSSynthesizer(
        const std::string& model_path,
        int n_threads = 4,
        bool use_gpu = true
    ) {
        rs_init_params_t p = rs_default_params();
        p.model_path = model_path.c_str();
        p.n_threads  = n_threads;
        p.use_gpu    = use_gpu;
        p.task_type  = RS_TASK_TTS_ONLINE;

        ctx_ = rs_init_from_file(p);
        if (!ctx_) {
            throw std::runtime_error("Failed to initialize TTS context");
        }
    }

    ~RSTTSSynthesizer() {
        if (ctx_) { rs_free(ctx_); ctx_ = nullptr; }
    }

    void set_reference(py::array_t<float, py::array::c_style | py::array::forcecast> pcm, int sample_rate = 16000) {
        auto buf = pcm.request();
        if (buf.ndim != 1) throw std::runtime_error("PCM must be 1-D float32 array");

        if (rs_push_reference_audio(ctx_,
                static_cast<float*>(buf.ptr),
                static_cast<int>(buf.shape[0]),
                sample_rate) != 0) {
            throw std::runtime_error("rs_push_reference_audio failed");
        }
    }

    py::array_t<float> synthesize(const std::string& text) {
        rs_reset(ctx_);
        if (rs_push_text(ctx_, text.c_str()) != 0) {
            throw std::runtime_error("rs_push_text failed");
        }

        std::vector<float> all_pcm;
        int ret;
        while ((ret = rs_process(ctx_)) >= 0) {
            float* chunk = nullptr;
            int n = rs_get_audio_output(ctx_, &chunk);
            if (n > 0 && chunk) {
                all_pcm.insert(all_pcm.end(), chunk, chunk + n);
            }
            if (ret == 0) break;
        }
        if (ret < 0) throw std::runtime_error("TTS inference error");

        auto result = py::array_t<float>(static_cast<py::ssize_t>(all_pcm.size()));
        auto r = result.mutable_unchecked<1>();
        for (size_t i = 0; i < all_pcm.size(); i++) r(i) = all_pcm[i];
        return result;
    }

    py::list synthesize_streaming(const std::string& text) {
        rs_reset(ctx_);
        if (rs_push_text(ctx_, text.c_str()) != 0) {
            throw std::runtime_error("rs_push_text failed");
        }

        py::list chunks;
        int ret;
        while ((ret = rs_process(ctx_)) >= 0) {
            float* chunk = nullptr;
            int n = rs_get_audio_output(ctx_, &chunk);
            if (n > 0 && chunk) {
                auto arr = py::array_t<float>(n);
                auto r = arr.mutable_unchecked<1>();
                for (int i = 0; i < n; i++) r(i) = chunk[i];
                chunks.append(arr);
            }
            if (ret == 0) break;
        }
        if (ret < 0) throw std::runtime_error("TTS inference error");
        return chunks;
    }

private:
    rs_context_t* ctx_ = nullptr;
};

/* -------- Python Module -------- */

PYBIND11_MODULE(rapidspeech, m) {
    m.doc() = "RapidSpeech Python bindings";

    py::class_<RSAsrOffline>(m, "asr_offline")
        .def(py::init<
            const std::string&,
            int,
            bool
        >(),
        py::arg("model_path"),
        py::arg("n_threads") = 4,
        py::arg("use_gpu") = true)

        .def("push_audio", &RSAsrOffline::push_audio,
             py::arg("pcm"),
             py::call_guard<py::gil_scoped_release>())

        .def("process", &RSAsrOffline::process, py::call_guard<py::gil_scoped_release>())

        .def("get_text", &RSAsrOffline::get_text)

        .def("set_user_input_prompt", &RSAsrOffline::set_user_input_prompt,
             py::arg("prompt"),
             "Set the user input prompt for the LLM decoder (default: 语音转写：)");

    // TTS Synthesizer
    py::class_<RSTTSSynthesizer>(m, "tts_synthesizer")
        .def(py::init<const std::string&, int, bool>(),
             py::arg("model_path"),
             py::arg("n_threads") = 4,
             py::arg("use_gpu") = true)
        .def("set_reference", &RSTTSSynthesizer::set_reference,
             py::arg("pcm"), py::arg("sample_rate") = 16000,
             "Set reference audio for voice cloning (optional)")
        .def("synthesize", &RSTTSSynthesizer::synthesize,
             py::arg("text"),
             py::call_guard<py::gil_scoped_release>(),
             "Synthesize text to audio, returns full PCM numpy array")
        .def("synthesize_streaming", &RSTTSSynthesizer::synthesize_streaming,
             py::arg("text"),
             py::call_guard<py::gil_scoped_release>(),
             "Synthesize text to audio chunks (list of numpy arrays)");
}