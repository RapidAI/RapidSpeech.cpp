/**
 * rs-imatrix — Importance Matrix collection tool for activation-aware quantization.
 *
 * Supports two modes:
 *   TTS calibration (OmniVoice): runs built-in calibration sentences through
 *     the model's diffusion + LLM forward and collects activation stats.
 *   ASR calibration (FunASRNano): pushes a list of wav files through the
 *     encoder/CTC/LLM and collects activation stats.
 *
 * Mode is auto-detected from the model architecture; --audio-list selects
 * ASR mode explicitly.
 *
 * Usage:
 *   rs-imatrix -m <model.gguf> -o <output.dat> [--audio-list <paths.txt>] [options]
 *
 * Options (common):
 *   -m, --model       <path>   Input GGUF model (required)
 *   -o, --output      <path>   Output importance matrix .dat file (required)
 *       --threads     <n>      CPU threads (default: 4)
 *       --gpu                  Enable GPU acceleration (default: off)
 *   -h, --help                 Show help
 *
 * Options (TTS):
 *       --n-steps     <n>      Diffusion steps for calibration (default: 8)
 *
 * Options (ASR):
 *       --audio-list  <file>   Text file with one wav path per line
 *       --use-llm              Run the LLM 2nd-pass decoder too (default: off)
 *       --ctc-precheck         Run a CTC precheck before LLM (default: off)
 */

#include "imatrix_collector.h"
#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define LOG_INFO(fmt, ...)  std::printf("[imatrix] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) std::fprintf(stderr, "[imatrix] ERROR: " fmt "\n", ##__VA_ARGS__)

struct ImatrixArgs {
    const char *model_path = nullptr;
    const char *output_path = nullptr;
    const char *audio_list = nullptr;
    int n_steps = 8;
    int n_threads = 4;
    bool use_gpu = false;
    bool use_llm = false;
    bool ctc_precheck = false;
};

// TTS calibration sentences (used when no --audio-list is given and the
// model is a TTS model).
static const char *kCalibrationTexts[] = {
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis technology has advanced rapidly.",
    "Hello world, this is a test of the text to speech system.",
    "Machine learning models can generate natural sounding speech.",
    "The weather today is sunny with a chance of rain.",
    "Please say something so we can verify the audio quality.",
    "Artificial intelligence is transforming how we interact with computers.",
    "Once upon a time there was a beautiful princess.",
    "One two three four five six seven eight nine ten.",
    "The conference on natural language processing starts tomorrow.",
    "Good morning everyone, thank you for attending.",
    "Music and art are essential parts of human culture.",
    "Scientific research requires careful observation and measurement.",
    "The package should arrive by the end of the week.",
    "Reading books helps improve vocabulary and comprehension.",
};

static void print_usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -m <model.gguf> -o <output.dat> [options]\n\n"
        << "Common:\n"
        << "  -m, --model       <path>   Input GGUF model (required)\n"
        << "  -o, --output      <path>   Output importance matrix .dat file (required)\n"
        << "      --threads     <n>      CPU threads (default: 4)\n"
        << "      --gpu                  Enable GPU acceleration (default: off)\n"
        << "  -h, --help                 Show this help\n\n"
        << "TTS (OmniVoice):\n"
        << "      --n-steps     <n>      Diffusion steps (default: 8)\n\n"
        << "ASR (FunASRNano):\n"
        << "      --audio-list  <file>   Text file: one wav path per line\n"
        << "      --use-llm              Also run the 2nd-pass LLM decoder\n"
        << "      --ctc-precheck         Run CTC precheck before LLM\n"
        << std::endl;
}

static bool parse_args(int argc, char **argv, ImatrixArgs &args) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            args.model_path = argv[++i];
        } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (a == "--n-steps" && i + 1 < argc) {
            args.n_steps = std::stoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            args.n_threads = std::stoi(argv[++i]);
        } else if (a == "--gpu") {
            args.use_gpu = true;
        } else if (a == "--audio-list" && i + 1 < argc) {
            args.audio_list = argv[++i];
        } else if (a == "--use-llm") {
            args.use_llm = true;
        } else if (a == "--ctc-precheck") {
            args.ctc_precheck = true;
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return false;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return false;
        }
    }
    if (!args.model_path) { std::cerr << "Error: --model is required\n"; return false; }
    if (!args.output_path) { std::cerr << "Error: --output is required\n"; return false; }
    return true;
}

// Static collector and callback for C API
static IMatrixCollector *g_collector = nullptr;

static void imatrix_callback(void *userdata, struct ggml_cgraph *gf) {
    (void)userdata;
    if (g_collector) {
        g_collector->collect_from_graph(gf);
    }
}

// Read a list of wav file paths (one per line; '#' comments and blank
// lines are ignored).
static std::vector<std::string> read_audio_list(const char *path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    if (!in) {
        LOG_ERROR("Failed to open audio list: %s", path);
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue;
        if (line[start] == '#') continue;
        size_t end = line.find_last_not_of(" \t\r");
        out.emplace_back(line.substr(start, end - start + 1));
    }
    return out;
}

static int run_tts(rs_context_t *ctx, const ImatrixArgs &args) {
    rs_set_tts_params(ctx, "male", "English", 42);
    rs_set_tts_diffusion_steps(ctx, args.n_steps);
    LOG_INFO("TTS calibration hooked (%d diffusion steps)", args.n_steps);

    int n_texts = sizeof(kCalibrationTexts) / sizeof(kCalibrationTexts[0]);
    LOG_INFO("Running %d calibration texts...", n_texts);

    int n_success = 0;
    for (int i = 0; i < n_texts; i++) {
        LOG_INFO("[%2d/%2d] \"%s\"", i + 1, n_texts, kCalibrationTexts[i]);
        rs_reset(ctx);
        if (rs_push_text(ctx, kCalibrationTexts[i]) != RS_OK) {
            LOG_ERROR("PushText failed for text %d", i);
            continue;
        }
        bool ok = true;
        while (true) {
            int ret = rs_process(ctx);
            if (ret < 0) { ok = false; break; }
            if (ret == 0) break;
            float *chunk = nullptr;
            while (rs_get_audio_output(ctx, &chunk) > 0) {}
        }
        if (ok) n_success++;
    }
    LOG_INFO("TTS calibration done: %d/%d texts succeeded", n_success, n_texts);
    return n_success;
}

static int run_asr(rs_context_t *ctx, const ImatrixArgs &args) {
    auto wavs = read_audio_list(args.audio_list);
    if (wavs.empty()) {
        LOG_ERROR("Audio list is empty: %s", args.audio_list);
        return 0;
    }
    LOG_INFO("ASR calibration: %zu wav files (use_llm=%s, ctc_precheck=%s)",
             wavs.size(), args.use_llm ? "on" : "off",
             args.ctc_precheck ? "on" : "off");

    rs_set_use_llm(ctx, args.use_llm);
    rs_set_ctc_precheck(ctx, args.ctc_precheck);

    const int kExpectedSampleRate = 16000;
    int n_success = 0;
    for (size_t i = 0; i < wavs.size(); ++i) {
        const std::string &path = wavs[i];
        LOG_INFO("[%3zu/%zu] %s", i + 1, wavs.size(), path.c_str());

        std::vector<float> pcm;
        int sr = 0;
        if (!load_wav_file(path.c_str(), pcm, &sr)) {
            LOG_ERROR("Failed to load wav: %s", path.c_str());
            continue;
        }
        if (sr != kExpectedSampleRate) {
            LOG_ERROR("Skipping %s: sample rate %d != %d",
                      path.c_str(), sr, kExpectedSampleRate);
            continue;
        }
        if (pcm.size() < (size_t)kExpectedSampleRate) {
            // Pad to 1 second so RSProcessor::Process() has enough audio
            pcm.resize(kExpectedSampleRate, 0.f);
        }

        rs_reset(ctx);
        if (rs_push_audio(ctx, pcm.data(), (int32_t)pcm.size()) != RS_OK) {
            LOG_ERROR("rs_push_audio failed for %s", path.c_str());
            continue;
        }
        int32_t ret = rs_process(ctx);
        if (ret < 0) {
            LOG_ERROR("rs_process failed for %s", path.c_str());
            continue;
        }
        const char *text = rs_get_text_output(ctx);
        if (text && text[0]) {
            LOG_INFO("    → %s", text);
        }
        n_success++;
    }
    LOG_INFO("ASR calibration done: %d/%zu wavs succeeded", n_success, wavs.size());
    return n_success;
}

int main(int argc, char **argv) {
    ImatrixArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    const bool asr_mode = args.audio_list != nullptr;

    LOG_INFO("Importance Matrix Collection Tool");
    LOG_INFO("Mode: %s", asr_mode ? "ASR" : "TTS");
    LOG_INFO("Model: %s", args.model_path);
    LOG_INFO("Output: %s", args.output_path);
    LOG_INFO("Threads: %d  GPU: %s",
             args.n_threads, args.use_gpu ? "ON" : "OFF");

    // ---- Init via C API ----
    rs_init_params_t params = rs_default_params();
    params.model_path = args.model_path;
    params.n_threads = args.n_threads;
    params.use_gpu = args.use_gpu;
    params.task_type = asr_mode ? RS_TASK_ASR_OFFLINE : RS_TASK_TTS_ONLINE;

    rs_context_t *ctx = rs_init_from_file(params);
    if (!ctx) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("Init failed: %s", err.message);
        return 1;
    }

    rs_model_meta_t meta = rs_get_model_meta(ctx);
    LOG_INFO("Arch: %s  SampleRate: %d  Backend: %s",
             meta.arch_name, meta.audio_sample_rate, rs_get_backend_name(ctx));

    // ---- Setup collector and hook ----
    IMatrixCollector collector;
    g_collector = &collector;
    rs_set_imatrix_callback(ctx, imatrix_callback, nullptr);

    // ---- Run calibration ----
    auto t_start = std::chrono::steady_clock::now();
    int n_success = asr_mode ? run_asr(ctx, args) : run_tts(ctx, args);
    auto t_end = std::chrono::steady_clock::now();
    float elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0f;
    LOG_INFO("Elapsed: %.1f s", elapsed);

    if (n_success == 0) {
        LOG_ERROR("No calibration samples succeeded - nothing to save");
        rs_set_imatrix_callback(ctx, nullptr, nullptr);
        rs_free(ctx);
        return 1;
    }

    collector.save(args.output_path);
    LOG_INFO("Importance matrix saved to %s", args.output_path);

    rs_set_imatrix_callback(ctx, nullptr, nullptr);
    rs_free(ctx);
    return 0;
}
