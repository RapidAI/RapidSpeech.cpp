#pragma once

#include <cstdint>
#include <random>
#include <vector>

/**
 * Repetition-Aware Sampling (RAS) for CosyVoice3 speech-token AR.
 *
 * Mirrors `cosyvoice3_tts_sample_ras` in CrispASR / the FunAudioLLM upstream:
 *   1. softmax(logits / max(temp, 1e-6))
 *   2. truncate to top_k
 *   3. truncate to nucleus (top_p) on the surviving distribution
 *   4. multinomial sample → candidate id
 *   5. if candidate appears >= win_size * tau_r times in the last `win_size`
 *      history entries, fall back to a multinomial sample over the FULL
 *      (post-temperature) distribution — i.e. same logits but no top-k/top-p
 *      filtering — to break the repeat.
 *
 * temperature == 0 → greedy argmax (debug / determinism path).
 *
 * `win_size`/`tau_r` defaults match the Python reference: 10 / 0.1.
 */
struct ras_params {
  int32_t top_k       = 25;
  float   top_p       = 0.8f;
  float   temperature = 1.0f;
  int32_t win_size    = 10;
  float   tau_r       = 0.1f;
};

int32_t cosyvoice3_sample_ras(const float *logits, int32_t n_vocab,
                              const std::vector<int32_t> &history,
                              const ras_params &p, std::mt19937 &rng);
