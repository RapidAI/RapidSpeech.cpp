#include "rapidspeech.h"
#include "utils/rs_wav.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path_gguf> [wav_file_path]" << std::endl;
    return 1;
  }

  const char* model_path = argv[1];
  const char* wav_path = (argc > 2) ? argv[2] : nullptr;

  // 1. Initialize Parameters
  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads = 4;
  params.use_gpu = true;

  std::cout << "[rs-cli] Initializing RapidSpeech model: " << model_path << std::endl;

  // 2. Create Context
  rs_context_t* ctx = rs_init_from_file(params);
  if (!ctx) {
    std::cerr << "[rs-cli] Failed to initialize context. Check model path and logs." << std::endl;
    return 1;
  }

  // 3. Prepare Input Audio
  std::vector<float> pcm;
  int sample_rate = 16000;

  if (wav_path) {
    std::cout << "[rs-cli] Loading audio from: " << wav_path << std::endl;
    if (!load_wav_file(wav_path, pcm, &sample_rate)) {
      std::cerr << "[rs-cli] Failed to load WAV file: " << wav_path << std::endl;
      rs_free(ctx);
      return 1;
    }
    std::cout << "[rs-cli] Loaded " << pcm.size() << " samples @ " << sample_rate << "Hz" << std::endl;
  } else {
    std::cout << "[rs-cli] No WAV file provided. Using dummy audio (1s sine wave)..." << std::endl;
    int n_samples = 16000;
    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
      pcm[i] = sinf(2.0f * 3.14159f * 440.0f * i / 16000.0f);
    }
  }

  // Sanity check: Ensure we actually have samples before pushing
  if (pcm.empty()) {
    std::cerr << "[rs-cli] No audio data to process." << std::endl;
    rs_free(ctx);
    return 1;
  }

  // 4. Push Audio to Framework
  // The crash here was caused by a struct mismatch in the library.
  // By including the correct internal header in rapidspeech_c.cpp, this is resolved.
  if (rs_push_audio(ctx, pcm.data(), static_cast<int>(pcm.size())) != 0) {
    std::cerr << "[rs-cli] Failed to push audio." << std::endl;
    rs_free(ctx);
    return 1;
  }

  // 5. Inference Loop
  std::cout << "[rs-cli] Starting inference..." << std::endl;

  bool running = true;
  while (running) {
    int status = rs_process(ctx);

    if (status < 0) {
      std::cerr << "[rs-cli] Error occurred during inference loop." << std::endl;
      break;
    } else if (status == 0) {
      // No more data to process (offline mode completion)
      running = false;
    } else {
      // New content available
      const char* text = rs_get_text_output(ctx);
      if (text && strlen(text) > 0) {
        // Using \r for real-time update effect in terminal
        std::cout << "\r[rs-cli] Result: " << text << std::flush;
      }
    }
  }
  std::cout << std::endl;

  // 6. Cleanup
  std::cout << "[rs-cli] Processing finished, cleaning up resources." << std::endl;
  rs_free(ctx);

  return 0;
}