#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Simple logging macros (avoiding dependency on internal rs_log.h)
#define RS_CLI_LOG_INFO(fmt, ...)                                              \
  std::printf("[RapidSpeech] " fmt "\n", ##__VA_ARGS__)
#define RS_CLI_LOG_ERROR(fmt, ...)                                             \
  std::fprintf(stderr, "[RapidSpeech] Error: " fmt "\n", ##__VA_ARGS__)

static void print_usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " --model <model.gguf> [options]\n\n"
      << "Options:\n"
      << "  -m, --model <path>     Model file path (required)\n"
      << "  -w, --wav <path>       WAV file path (optional)\n"
      << "  -t, --threads <num>    Number of threads (default: 4)\n"
      << "      --gpu <true|false> Enable GPU acceleration (default: true)\n"
      << "  -h, --help             Show this help message\n"
      << std::endl;
}

static bool parse_bool(const std::string &v) {
  return (v == "1" || v == "true" || v == "TRUE" || v == "yes");
}

static void print_model_info(const rs_model_meta_t &meta) {
  std::cout << "\n=== Model Information ===" << std::endl;
  std::cout << "  Architecture : " << meta.arch_name << std::endl;
  std::cout << "  Sample Rate  : " << meta.audio_sample_rate << " Hz"
            << std::endl;
  std::cout << "  Mel Bins     : " << meta.n_mels << std::endl;
  std::cout << "  Vocab Size   : " << meta.vocab_size << std::endl;
  std::cout << "=========================\n" << std::endl;
}

// ---------------- main ----------------

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  // -------- defaults --------
  const char *model_path = nullptr;
  const char *wav_path = nullptr;
  int n_threads = 4;
  bool use_gpu = true;

  // -------- parse argv --------
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
      model_path = argv[++i];
    } else if ((arg == "-w" || arg == "--wav") && i + 1 < argc) {
      wav_path = argv[++i];
    } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
      n_threads = std::stoi(argv[++i]);
    } else if (arg == "--gpu" && i + 1 < argc) {
      use_gpu = parse_bool(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  if (!model_path) {
    std::cerr << "Error: --model is required\n";
    print_usage(argv[0]);
    return 1;
  }

  // -------- init params with new API --------
  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads = n_threads;
  params.use_gpu = use_gpu;

  std::cout << "[RapidSpeech] Library Version: " << rs_get_version()
            << std::endl;
  std::cout << "[RapidSpeech] Model   : " << model_path << std::endl;
  std::cout << "[RapidSpeech] Threads : " << n_threads << std::endl;
  std::cout << "[RapidSpeech] GPU     : " << (use_gpu ? "ON" : "OFF")
            << std::endl;

  // -------- create context --------
  rs_context_t *ctx = rs_init_from_file(params);
  if (!ctx) {
    rs_error_info_t err = rs_get_last_error();
    std::cerr << "[RapidSpeech] Failed to initialize context." << std::endl;
    std::cerr << "  Error Code: " << err.code << std::endl;
    std::cerr << "  Message   : " << err.message << std::endl;
    return 1;
  }

  // -------- print model info --------
  rs_model_meta_t meta = rs_get_model_meta(ctx);
  if (meta.arch_name[0] != '\0') {
    print_model_info(meta);
  }

  // -------- check context readiness --------
  if (!rs_is_context_ready(ctx)) {
    std::cerr << "[RapidSpeech] Context is not ready for inference."
              << std::endl;
    rs_free(ctx);
    return 1;
  }

  // -------- load audio --------
  std::vector<float> pcm;
  int sample_rate = 16000;

  if (wav_path) {
    RS_CLI_LOG_INFO("Loading audio: %s", wav_path);
    if (!load_wav_file(wav_path, pcm, &sample_rate)) {
      RS_CLI_LOG_ERROR("Failed to load WAV file");
      rs_free(ctx);
      return 1;
    }
    RS_CLI_LOG_INFO("Loaded %zu samples @ %d Hz", pcm.size(), sample_rate);

    // Check if sample rate matches model expectation
    if (sample_rate != meta.audio_sample_rate) {
      std::cerr << "[RapidSpeech] Warning: Audio sample rate (" << sample_rate
                << ") differs from model expected (" << meta.audio_sample_rate
                << ")" << std::endl;
    }
  } else {
    RS_CLI_LOG_INFO("No WAV provided, generating 1s sine wave");
    pcm.resize(16000);
    for (int i = 0; i < 16000; ++i) {
      pcm[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f);
    }
  }

  if (pcm.empty()) {
    std::cerr << "[RapidSpeech] No audio data." << std::endl;
    rs_free(ctx);
    return 1;
  }

  // -------- push audio with error checking --------
  rs_error_t err =
      rs_push_audio(ctx, pcm.data(), static_cast<int32_t>(pcm.size()));
  if (err != RS_OK) {
    rs_error_info_t err_info = rs_get_last_error();
    std::cerr << "[RapidSpeech] Failed to push audio: " << err_info.message
              << std::endl;
    rs_free(ctx);
    return 1;
  }

  // -------- inference --------
  std::cout << "[RapidSpeech] Starting inference..." << std::endl;
  std::cout << "[RapidSpeech] Backend: " << rs_get_backend_name(ctx)
            << std::endl;

  int process_count = 0;
  while (true) {
    int32_t status = rs_process(ctx);

    if (status < 0) {
      rs_error_info_t err_info = rs_get_last_error();
      std::cerr << "[RapidSpeech] Inference error: " << err_info.message
                << std::endl;
      break;
    } else if (status == 0) {
      // No more output
      break;
    } else {
      process_count++;
      const char *text = rs_get_text_output(ctx);
      if (text && std::strlen(text) > 0) {
        std::cout << "\r[RapidSpeech] Result: " << text << std::flush;
      }
    }
  }

  std::cout << std::endl;

  if (process_count > 0) {
    std::cout << "[RapidSpeech] Finished. Processed " << process_count
              << " iteration(s)." << std::endl;
  } else {
    std::cout << "[RapidSpeech] Finished. No transcription result."
              << std::endl;
  }

  // -------- cleanup --------
  rs_free(ctx);
  rs_clear_error();

  return 0;
}
