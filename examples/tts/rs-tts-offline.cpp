/**
 * rs-tts-offline.cpp — Offline Text-to-Speech (OmniVoice / OpenVoice2 /
 *                       CosyVoice3-LLM Phase 2)
 *
 * Usage:
 *   rs-tts-offline -m <tts.gguf> -t "text" [options]
 *
 * Options:
 *   -m, --model <path>       TTS model path (required)
 *   -t, --text <text>        Text to synthesize (required)
 *   -o, --output <path>      Output WAV file path (default: output.wav). For
 *                            models that emit speech tokens instead of audio
 *                            (e.g. cosyvoice3-llm Phase 2), the int32 LE token
 *                            sequence is written here.
 *       --instruct <text>    Voice description (default: male)
 *       --lang <lang>        Target language (default: English)
 *       --seed <n>           Random seed (default: 42)
 *       --n-steps <n>        Diffusion steps (1-128, default: 32)
 *       --threads <n>        CPU threads (default: 4)
 *       --gpu <true|false>   Enable GPU acceleration (default: true)
 *       --dump-step0-logits <path>
 *                            Debug: write the first speech-token logits
 *                            (float32) to this path. CosyVoice3-LLM only.
 *   -h, --help               Show help
 */

#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define LOG_INFO(fmt, ...) std::printf("[tts] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) std::fprintf(stderr, "[tts] ERROR: " fmt "\n", ##__VA_ARGS__)

struct TtsArgs {
  const char *model_path = nullptr;
  const char *text = nullptr;
  const char *output_path = "output.wav";
  const char *instruct = "male";
  const char *language = "English";
  const char *ref_path = nullptr;       // reference audio for voice cloning
  const char *ref_text = nullptr;       // transcript of reference audio
  const char *bert_path = nullptr;      // ZH BERT (1024-dim) GGUF, OpenVoice2 ZH
  const char *mbert_path = nullptr;     // multilingual BERT (768-dim) GGUF
  const char *dump_step0 = nullptr;     // CosyVoice3-LLM step-0 logits dump
  const char *speech_tokenizer_path = nullptr; // CosyVoice3 speech tokenizer GGUF
  const char *campplus_path = nullptr;         // CosyVoice3 CAM++ speaker GGUF
  const char *prompt_tokens_bin = nullptr;     // CosyVoice3 prompt-tokens int32 LE blob
  int seed = 42;
  int n_steps = 32;
  int n_threads = 4;
  bool use_gpu = true;
};

static bool parse_bool(const std::string &v) {
  return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

static void print_usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " -m <model.gguf> -t \"text\" [options]\n\n"
      << "Options:\n"
      << "  -m, --model <path>       TTS model path (required)\n"
      << "  -t, --text <text>        Text to synthesize (required)\n"
      << "  -o, --output <path>      Output WAV file path (default: output.wav)\n"
      << "      --ref <path>         Reference audio for voice cloning (WAV)\n"
	      << "      --ref-text <text>    Transcript of the reference audio\n"
      << "      --bert <path>        ZH BERT GGUF (1024-dim, OpenVoice2 ZH only)\n"
      << "      --mbert <path>       Multilingual BERT GGUF (768-dim)\n"
      << "      --speech-tokenizer <path>\n"
      << "                          CosyVoice3 speech tokenizer GGUF (required for --ref)\n"
      << "      --campplus <path>    CosyVoice3 CAM++ speaker embedding GGUF (required for --ref)\n"
      << "      --prompt-tokens-bin <path>\n"
      << "                          CosyVoice3 prompt-tokens int32 LE blob (debug override)\n"
      << "      --instruct <text>    Voice description (default: male)\n"
      << "      --lang <lang>        Target language (default: English)\n"
      << "      --seed <n>           Random seed (default: 42)\n"
      << "      --n-steps <n>        Diffusion steps 1-128 (default: 32)\n"
      << "      --threads <n>        CPU threads (default: 4)\n"
      << "      --gpu <true|false>   Enable GPU acceleration (default: true)\n"
      << "      --dump-step0-logits <path>\n"
      << "                          Debug: dump first speech-token logits "
         "(CosyVoice3-LLM only)\n"
      << "  -h, --help               Show this help\n"
      << std::endl;
}

static bool parse_args(int argc, char **argv, TtsArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model") && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if ((a == "-t" || a == "--text") && i + 1 < argc) {
      args.text = argv[++i];
    } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (a == "--ref" && i + 1 < argc) {
      args.ref_path = argv[++i];
    } else if (a == "--ref-text" && i + 1 < argc) {
      args.ref_text = argv[++i];
    } else if (a == "--bert" && i + 1 < argc) {
      args.bert_path = argv[++i];
    } else if (a == "--mbert" && i + 1 < argc) {
      args.mbert_path = argv[++i];
    } else if (a == "--speech-tokenizer" && i + 1 < argc) {
      args.speech_tokenizer_path = argv[++i];
    } else if (a == "--campplus" && i + 1 < argc) {
      args.campplus_path = argv[++i];
    } else if (a == "--prompt-tokens-bin" && i + 1 < argc) {
      args.prompt_tokens_bin = argv[++i];
    } else if (a == "--instruct" && i + 1 < argc) {
      args.instruct = argv[++i];
    } else if (a == "--lang" && i + 1 < argc) {
      args.language = argv[++i];
    } else if (a == "--seed" && i + 1 < argc) {
      args.seed = std::stoi(argv[++i]);
    } else if (a == "--n-steps" && i + 1 < argc) {
      args.n_steps = std::stoi(argv[++i]);
    } else if (a == "--threads" && i + 1 < argc) {
      args.n_threads = std::stoi(argv[++i]);
    } else if (a == "--gpu" && i + 1 < argc) {
      args.use_gpu = parse_bool(argv[++i]);
    } else if (a == "--dump-step0-logits" && i + 1 < argc) {
      args.dump_step0 = argv[++i];
    } else if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      return false;
    }
  }
  if (!args.model_path) {
    std::cerr << "Error: --model is required\n";
    return false;
  }
  if (!args.text) {
    std::cerr << "Error: --text is required\n";
    return false;
  }
  return true;
}

static bool write_wav_file(const char *filename, const std::vector<float> &pcm,
                           int sample_rate) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return false;

  int num_samples = (int)pcm.size();
  int num_channels = 1;
  int bits_per_sample = 16;
  int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
  int block_align = num_channels * bits_per_sample / 8;
  int data_size = num_samples * block_align;
  int chunk_size = 36 + data_size;

  // RIFF header
  file.write("RIFF", 4);
  file.write(reinterpret_cast<const char *>(&chunk_size), 4);
  file.write("WAVE", 4);

  // fmt chunk
  file.write("fmt ", 4);
  int subchunk1_size = 16;
  short audio_format = 1; // PCM
  file.write(reinterpret_cast<const char *>(&subchunk1_size), 4);
  file.write(reinterpret_cast<const char *>(&audio_format), 2);
  file.write(reinterpret_cast<const char *>(&num_channels), 2);
  file.write(reinterpret_cast<const char *>(&sample_rate), 4);
  file.write(reinterpret_cast<const char *>(&byte_rate), 4);
  file.write(reinterpret_cast<const char *>(&block_align), 2);
  file.write(reinterpret_cast<const char *>(&bits_per_sample), 2);

  // data chunk
  file.write("data", 4);
  file.write(reinterpret_cast<const char *>(&data_size), 4);

  // Convert float [-1,1] to int16
  for (float s : pcm) {
    float clipped = std::max(-1.0f, std::min(1.0f, s));
    int16_t sample = static_cast<int16_t>(clipped * 32767.0f);
    file.write(reinterpret_cast<const char *>(&sample), sizeof(int16_t));
  }

  return true;
}

int main(int argc, char **argv) {
  TtsArgs args;
  if (!parse_args(argc, argv, args)) return 1;

  // Init TTS model
  LOG_INFO("RapidSpeech.cpp v%s", rs_get_version());
  LOG_INFO("TTS model: %s", args.model_path);
  LOG_INFO("Threads: %d  GPU: %s", args.n_threads, args.use_gpu ? "ON" : "OFF");

  // BERT paths are consumed by OpenVoice2Model::Load via env vars; setting them
  // here keeps the C API stable while still exposing a CLI flag to the user.
  auto set_env_var = [](const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, /*overwrite=*/1);
#endif
  };
  if (args.bert_path && args.bert_path[0]) {
    set_env_var("RS_ZH_BERT_PATH", args.bert_path);
    LOG_INFO("ZH BERT: %s", args.bert_path);
  }
  if (args.mbert_path && args.mbert_path[0]) {
    set_env_var("RS_MBERT_PATH", args.mbert_path);
    LOG_INFO("mBERT:   %s", args.mbert_path);
  }
  if (args.speech_tokenizer_path && args.speech_tokenizer_path[0]) {
    set_env_var("RS_CV3_SPEECH_TOKENIZER_PATH", args.speech_tokenizer_path);
    LOG_INFO("CV3 speech tokenizer: %s", args.speech_tokenizer_path);
  }
  if (args.campplus_path && args.campplus_path[0]) {
    set_env_var("RS_CV3_CAMPPLUS_PATH", args.campplus_path);
    LOG_INFO("CV3 CAM++: %s", args.campplus_path);
  }
  if (args.prompt_tokens_bin && args.prompt_tokens_bin[0]) {
    set_env_var("RS_CV3_PROMPT_TOKENS_BIN", args.prompt_tokens_bin);
    LOG_INFO("CV3 prompt-tokens bin: %s", args.prompt_tokens_bin);
  }
  if (args.dump_step0 && args.dump_step0[0]) {
    // CosyVoice3-LLM reads this env var inside Decode (Phase-2 debug hook).
    set_env_var("RS_CV3_DUMP_STEP0_LOGITS", args.dump_step0);
    LOG_INFO("Step-0 logits dump: %s", args.dump_step0);
  }

  rs_init_params_t tts_params = rs_default_params();
  tts_params.model_path = args.model_path;
  tts_params.n_threads = args.n_threads;
  tts_params.use_gpu = args.use_gpu;
  tts_params.task_type = RS_TASK_TTS_ONLINE;

  rs_context_t *tts_ctx = rs_init_from_file(tts_params);
  if (!tts_ctx) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("TTS init failed: %s", err.message);
    return 1;
  }

  rs_model_meta_t meta = rs_get_model_meta(tts_ctx);
  LOG_INFO("Architecture: %s  SampleRate: %d", meta.arch_name,
           meta.audio_sample_rate);
  LOG_INFO("Backend: %s", rs_get_backend_name(tts_ctx));

  // Set TTS params
  rs_set_tts_params(tts_ctx, args.instruct, args.language, args.seed);
  rs_set_tts_diffusion_steps(tts_ctx, args.n_steps);
  LOG_INFO("Instruct: %s  Language: %s  Seed: %d  Steps: %d", args.instruct, args.language, args.seed, args.n_steps);

  // Push reference audio for voice cloning (optional)
  if (args.ref_path && args.ref_path[0]) {
    std::vector<float> ref_pcm;
    int ref_sr = 0;
    if (!load_wav_file(args.ref_path, ref_pcm, &ref_sr)) {
      LOG_ERROR("Failed to load reference WAV: %s", args.ref_path);
      rs_free(tts_ctx);
      return 1;
    }
    LOG_INFO("Reference audio: %zu samples @ %d Hz", ref_pcm.size(), ref_sr);
    if (rs_push_reference_audio(tts_ctx, ref_pcm.data(), (int32_t)ref_pcm.size(),
                                ref_sr) != 0) {
      rs_error_info_t err = rs_get_last_error();
      LOG_ERROR("rs_push_reference_audio failed: %s", err.message);
      rs_free(tts_ctx);
      return 1;
    }
    if (args.ref_text && args.ref_text[0]) {
      if (rs_push_reference_text(tts_ctx, args.ref_text) != RS_OK) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("rs_push_reference_text failed: %s", err.message);
        rs_free(tts_ctx);
        return 1;
      }
      LOG_INFO("Reference text: \"%s\"", args.ref_text);
    }
  }

  // Push text
  LOG_INFO("Synthesizing: \"%s\"", args.text);

  auto t0 = std::chrono::steady_clock::now();

  if (rs_push_text(tts_ctx, args.text) != RS_OK) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("rs_push_text failed: %s", err.message);
    rs_free(tts_ctx);
    return 1;
  }

  // Streaming synthesis loop
  std::vector<float> all_pcm;
  int ret;
  int chunks = 0;
  while ((ret = rs_process(tts_ctx)) >= 0) {
    float *chunk = nullptr;
    int n = rs_get_audio_output(tts_ctx, &chunk);
    if (n > 0 && chunk) {
      all_pcm.insert(all_pcm.end(), chunk, chunk + n);
      chunks++;
    }
    if (ret == 0) break;
  }

  if (ret < 0) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("TTS inference failed: %s", err.message);
    rs_free(tts_ctx);
    return 1;
  }

  auto t1 = std::chrono::steady_clock::now();
  float elapsed_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1e3f;

  // ----- Token-emitting models (e.g. CosyVoice3-LLM Phase 2) ----------------
  // These don't return PCM; they expose a CSV of int32 speech-token ids via
  // rs_get_text_output. Persist them as a packed little-endian int32 blob so
  // downstream tooling (flow + HiFT integration, validation harnesses) can
  // consume them.
  if (all_pcm.empty()) {
    const std::string arch = meta.arch_name;
    if (arch == "cosyvoice3-llm") {
      const char *csv = rs_get_text_output(tts_ctx);
      std::vector<int32_t> ids;
      if (csv && *csv) {
        std::string buf;
        for (const char *p = csv; ; ++p) {
          if (*p == ',' || *p == '\0') {
            if (!buf.empty()) ids.push_back((int32_t)std::atoi(buf.c_str()));
            buf.clear();
            if (*p == '\0') break;
          } else {
            buf.push_back(*p);
          }
        }
      }
      std::ofstream out(args.output_path, std::ios::binary);
      if (!out) {
        LOG_ERROR("Cannot write tokens to %s", args.output_path);
        rs_free(tts_ctx);
        return 1;
      }
      if (!ids.empty()) {
        out.write(reinterpret_cast<const char *>(ids.data()),
                  (std::streamsize)(ids.size() * sizeof(int32_t)));
      }
      LOG_INFO("Generated %zu speech tokens in %.0f ms → %s", ids.size(),
               elapsed_ms, args.output_path);
      rs_free(tts_ctx);
      rs_clear_error();
      return 0;
    }
    LOG_ERROR("No audio generated");
    rs_free(tts_ctx);
    return 1;
  }

  float audio_dur = (float)all_pcm.size() / meta.audio_sample_rate;
  float rtf = audio_dur > 0 ? (elapsed_ms / 1e3f) / audio_dur : 0.f;

  LOG_INFO("Generated %zu samples (%.2f s) in %d chunks, %.0f ms, RTF: %.3f",
           all_pcm.size(), audio_dur, chunks, elapsed_ms, rtf);

  // Write WAV
  if (write_wav_file(args.output_path, all_pcm, meta.audio_sample_rate)) {
    LOG_INFO("Wrote: %s", args.output_path);
  } else {
    LOG_ERROR("Failed to write WAV: %s", args.output_path);
    rs_free(tts_ctx);
    return 1;
  }

  rs_free(tts_ctx);
  rs_clear_error();
  return 0;
}
