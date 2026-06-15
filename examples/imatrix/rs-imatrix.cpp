/**
 * rs-imatrix — Importance Matrix collection tool for activation-aware quantization.
 *
 * Supports four modes:
 *   TTS calibration (OmniVoice): runs built-in calibration sentences through
 *     the model's diffusion + LLM forward and collects activation stats.
 *   ASR calibration (FunASRNano): pushes a list of wav files through the
 *     encoder/CTC/LLM and collects activation stats.
 *   CV3 calibration (CosyVoice3): runs ~40 zh+en sentences across multiple
 *     seeds (default 8) through LM + Flow DiT, collecting stats on both.
 *     Uses the GGUF's baked cv3.default_voice.* tuple by default;
 *     --voice <baked-voice.gguf> overrides it with an external bake.
 *   Kokoro calibration (Kokoro v1.1-zh): runs a ZH+EN calibration corpus
 *     through the text encoder, PLBert, prosody predictor, decoder, and
 *     iSTFTNet generator. Requires --voice to point at a voice pack GGUF.
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
 *
 * Options (CosyVoice3):
 *       --voice       <path>   Optional baked voice GGUF (overrides the
 *                              unified GGUF's cv3.default_voice.* tuple)
 *       --n-seeds     <n>      Seeds per sentence (default: 8, ~320 passes total)
 *       --text-list   <file>   Override built-in 40-sentence corpus
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
    const char *voice_path = nullptr;
    const char *text_list = nullptr;
    int n_steps = 8;
    int n_seeds = 8;
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

// CosyVoice3 calibration sentences — 20 Chinese + 20 English. Combined with
// --n-seeds (default 8) the LM + Flow DiT see ~320 forward passes covering
// short/long, news/dialog/literary, and common phonetic regimes. The pair
// (language, text) is run as a single tuple.
struct Cv3Sentence { const char *lang; const char *text; };
static const Cv3Sentence kCv3Sentences[] = {
    // ----- Chinese -----
    {"Chinese", "你好，欢迎使用语音合成系统。"},
    {"Chinese", "今天天气真不错，适合出去走走。"},
    {"Chinese", "我们一起去公园散步好吗？"},
    {"Chinese", "请问最近的地铁站在哪里？"},
    {"Chinese", "这家餐厅的菜品非常美味，服务也很周到。"},
    {"Chinese", "人工智能正在改变我们的生活方式。"},
    {"Chinese", "希望你以后能够做得比我还好。"},
    {"Chinese", "春天来了，柳树发芽，花儿盛开。"},
    {"Chinese", "他用一辈子的努力，证明了知识可以改变命运。"},
    {"Chinese", "孩子们在操场上欢快地奔跑、追逐、嬉戏。"},
    {"Chinese", "经过仔细分析，我们决定采用第二个方案。"},
    {"Chinese", "明天上午九点的会议改到下午三点举行。"},
    {"Chinese", "国家发展改革委发布了新一轮的产业政策。"},
    {"Chinese", "这本书讲述了一位年轻科学家的成长故事。"},
    {"Chinese", "无论遇到什么困难，都不要轻易放弃自己的梦想。"},
    {"Chinese", "深夜的城市，霓虹闪烁，街道空旷得让人有些恍惚。"},
    {"Chinese", "他端起茶杯，轻轻抿了一口，眼神望向窗外。"},
    {"Chinese", "音乐响起的那一刻，所有人都静静地坐了下来。"},
    {"Chinese", "一二三四五六七八九十，百千万亿。"},
    {"Chinese", "二零二六年六月十二日，星期五。"},

    // ----- English -----
    {"English", "Hello, welcome to the speech synthesis system."},
    {"English", "The weather today is wonderful for a walk in the park."},
    {"English", "Could you please tell me where the nearest subway station is?"},
    {"English", "This restaurant has delicious food and excellent service."},
    {"English", "Artificial intelligence is transforming the way we live and work."},
    {"English", "I hope that one day you will surpass even my best work."},
    {"English", "Spring has arrived; the willows are budding and flowers are blooming."},
    {"English", "He spent his entire life proving that knowledge can change destiny."},
    {"English", "Children were running, chasing, and laughing on the playground."},
    {"English", "After careful analysis, we decided to adopt the second proposal."},
    {"English", "Tomorrow's nine o'clock meeting has been moved to three in the afternoon."},
    {"English", "The agency announced a new round of industrial development policies."},
    {"English", "This book tells the story of a young scientist's coming of age."},
    {"English", "No matter what difficulties you face, never give up on your dreams."},
    {"English", "Late at night the city's neon lights flicker over empty streets."},
    {"English", "He raised the teacup, took a small sip, and gazed out the window."},
    {"English", "The moment the music began, everyone quietly took their seats."},
    {"English", "One, two, three, four, five, six, seven, eight, nine, ten."},
    {"English", "On June twelfth, twenty twenty-six, a Friday afternoon."},
    {"English", "Quantization aware calibration reduces perplexity at the same bit width."},
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
        << "      --ctc-precheck         Run CTC precheck before LLM\n\n"
        << "CosyVoice3:\n"
        << "      --voice       <path>   Optional baked voice GGUF (overrides GGUF default_voice)\n"
        << "      --n-seeds     <n>      Seeds per sentence (default: 8, ~320 passes total)\n"
        << "      --text-list   <file>   Override built-in 40-sentence corpus\n"
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
        } else if (a == "--voice" && i + 1 < argc) {
            args.voice_path = argv[++i];
        } else if (a == "--n-seeds" && i + 1 < argc) {
            args.n_seeds = std::stoi(argv[++i]);
        } else if (a == "--text-list" && i + 1 < argc) {
            args.text_list = argv[++i];
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

static void imatrix_callback(void *userdata, struct ggml_tensor *node) {
    (void)userdata;
    if (g_collector) {
        g_collector->collect_node(node);
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
            std::vector<float> resampled;
            if (!resample_pcm(pcm, sr, resampled, kExpectedSampleRate)) {
                LOG_ERROR("Resampling failed for %s (%d -> %d)",
                          path.c_str(), sr, kExpectedSampleRate);
                continue;
            }
            pcm = std::move(resampled);
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

static int run_cv3(rs_context_t *ctx, const ImatrixArgs &args,
                   const std::vector<std::pair<std::string, std::string>> &sentences) {
    LOG_INFO("CosyVoice3 calibration: %zu sentences × %d seeds = %d forward passes",
             sentences.size(), args.n_seeds,
             (int)sentences.size() * args.n_seeds);

    int n_success = 0;
    int n_total = 0;
    for (int seed_idx = 0; seed_idx < args.n_seeds; ++seed_idx) {
        const int seed = 42 + seed_idx * 17;
        for (size_t i = 0; i < sentences.size(); ++i) {
            const auto &[lang, text] = sentences[i];
            ++n_total;
            LOG_INFO("[%3d/%3d] seed=%d lang=%s text=\"%s\"",
                     n_total, (int)sentences.size() * args.n_seeds, seed,
                     lang.c_str(), text.c_str());

            rs_set_tts_params(ctx, "male", lang.c_str(), seed);
            rs_reset(ctx);
            if (rs_push_text(ctx, text.c_str()) != RS_OK) {
                LOG_ERROR("PushText failed at %d", n_total);
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
    }
    LOG_INFO("CV3 calibration done: %d/%d passes succeeded", n_success, n_total);
    return n_success;
}

// Kokoro calibration: same multi-language corpus as CV3 — except Kokoro's
// PushText only routes through the ZH G2P (the EN G2P is invoked internally
// for Latin-script segments embedded inside Chinese strings). English-only
// sentences are rewritten to lang="Chinese" so the ZH G2P fans them out via
// its EN fallback. Single deterministic forward per sentence (no --n-seeds).
static int run_kokoro(rs_context_t *ctx, const ImatrixArgs &args,
                      const std::vector<std::pair<std::string, std::string>> &sentences) {
    (void)args;
    LOG_INFO("Kokoro calibration: %zu sentences (deterministic, 1 pass each)",
             sentences.size());

    int n_success = 0;
    for (size_t i = 0; i < sentences.size(); ++i) {
        const auto &orig = sentences[i];
        // Kokoro only ships a ZH G2P; route everything through it. The EN
        // fallback inside ZHG2P handles Latin-script segments.
        const std::string lang = "Chinese";
        const std::string &text = orig.second;
        LOG_INFO("[%3zu/%zu] orig_lang=%s text=\"%s\"",
                 i + 1, sentences.size(), orig.first.c_str(), text.c_str());
        rs_set_tts_params(ctx, "male", lang.c_str(), 42);
        rs_reset(ctx);
        if (rs_push_text(ctx, text.c_str()) != RS_OK) {
            LOG_ERROR("PushText failed at %zu", i + 1);
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
    LOG_INFO("Kokoro calibration done: %d/%zu sentences succeeded",
             n_success, sentences.size());
    return n_success;
}

static std::vector<std::pair<std::string, std::string>>
load_cv3_sentences(const ImatrixArgs &args) {
    std::vector<std::pair<std::string, std::string>> out;
    if (args.text_list) {
        // Format: "<lang>\t<text>" per line. Lang defaults to English when
        // no tab present; comment lines start with '#'.
        std::ifstream in(args.text_list);
        if (!in) {
            LOG_ERROR("Failed to open --text-list: %s", args.text_list);
            return out;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t tab = line.find('\t');
            if (tab == std::string::npos) {
                out.emplace_back("English", line);
            } else {
                out.emplace_back(line.substr(0, tab), line.substr(tab + 1));
            }
        }
        LOG_INFO("Loaded %zu sentences from %s", out.size(), args.text_list);
    } else {
        int n = (int)(sizeof(kCv3Sentences) / sizeof(kCv3Sentences[0]));
        out.reserve((size_t)n);
        for (int i = 0; i < n; ++i) {
            out.emplace_back(kCv3Sentences[i].lang, kCv3Sentences[i].text);
        }
    }
    return out;
}

int main(int argc, char **argv) {
    ImatrixArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    const bool asr_mode = args.audio_list != nullptr;

    // CV3 expects its baked voice via env var, which must be set BEFORE
    // rs_init_from_file (Load reads it). Kokoro uses RS_KOKORO_VOICE_PATH for
    // the same purpose. Harmless for archs that don't read these.
    if (args.voice_path) {
        setenv("RS_CV3_VOICE_PATH", args.voice_path, 1);
        setenv("RS_KOKORO_VOICE_PATH", args.voice_path, 1);
    }

    LOG_INFO("Importance Matrix Collection Tool");
    LOG_INFO("Mode: %s", asr_mode ? "ASR" : "TTS/CV3/Kokoro");
    LOG_INFO("Model: %s", args.model_path);
    LOG_INFO("Output: %s", args.output_path);
    LOG_INFO("Threads: %d  GPU: %s",
             args.n_threads, args.use_gpu ? "ON" : "OFF");
    if (args.voice_path) {
        LOG_INFO("Voice: %s", args.voice_path);
    }

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

    const bool cv3_mode = !asr_mode &&
        std::string(meta.arch_name).rfind("cosyvoice3", 0) == 0;
    const bool kokoro_mode = !asr_mode &&
        std::string(meta.arch_name) == "kokoro";
    if (cv3_mode && !args.voice_path) {
        LOG_INFO("CV3: using GGUF baked default_voice (no --voice override)");
    }
    if (kokoro_mode && !args.voice_path) {
        LOG_ERROR("Kokoro: --voice <voice_pack.gguf> is required for calibration");
        rs_free(ctx);
        return 1;
    }

    // ---- Setup collector and hook ----
    IMatrixCollector collector;
    g_collector = &collector;
    rs_set_imatrix_callback(ctx, imatrix_callback, nullptr);

    // ---- Run calibration ----
    auto t_start = std::chrono::steady_clock::now();
    int n_success;
    if (asr_mode) {
        n_success = run_asr(ctx, args);
    } else if (cv3_mode) {
        n_success = run_cv3(ctx, args, load_cv3_sentences(args));
    } else if (kokoro_mode) {
        n_success = run_kokoro(ctx, args, load_cv3_sentences(args));
    } else {
        n_success = run_tts(ctx, args);
    }
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

    collector.print_skip_stats();
    collector.save(args.output_path);
    LOG_INFO("Importance matrix saved to %s", args.output_path);

    rs_set_imatrix_callback(ctx, nullptr, nullptr);
    rs_free(ctx);
    return 0;
}
