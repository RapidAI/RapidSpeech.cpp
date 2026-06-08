#pragma once

#include <vector>

/**
 * 80-mel @ 24 kHz extractor for the CosyVoice3 Flow `prompt_feat` input.
 *
 * Config (matches FunAudioLLM CosyVoice3 upstream `kaldi-style` 80-mel):
 *   - sample_rate=24000, n_fft=1920, hop=480, win=1920
 *   - 80 Slaney mel bins, [0, sr/2]
 *   - log(max(mel, 1e-5))  (natural log, no global normalization)
 *
 * Output layout: out[t * 80 + m] for frame t, mel bin m  (mel fastest).
 */
int compute_log_mel_80_24k(const float *pcm_24k, int n_samples,
                           std::vector<float> &out);

/**
 * 128-mel front-end for the CosyVoice3 speech_tokenizer_v3.
 *
 * Configuration (matches Funaudio S3 V3 inputs `feats [1, 128, T]`):
 *   - n_fft = 400, hop = 160, win = 400 (25 ms / 10 ms @ 16 kHz)
 *   - Slaney mel filterbank (htk=False), 128 bins, [0, sr/2]
 *   - Hann window (symmetric)
 *   - Reflection padding (center=True) of n_fft/2 samples each side
 *   - Per-frame log: log(max(mel, eps))    — natural log, no Whisper-style
 *                                            global normalization
 *
 * The log convention here intentionally differs from `WhisperMelExtractor`:
 * Whisper applies a global `(max(v, mmax) + 4) / 4` post-normalization that
 * the V3 tokenizer was NOT trained against. Until we have a verified PyTorch
 * reference dump, we ship the raw natural-log path and document it loudly.
 *
 * Output layout: `out[t * 128 + m]` for frame t, mel bin m  (mel fastest).
 */
class CosyVoice3MelExtractor {
public:
  CosyVoice3MelExtractor();
  ~CosyVoice3MelExtractor();

  // Returns number of frames written.
  int Compute(const float *pcm_16k, int n_samples,
              std::vector<float> &out_log_mel) const;

  int n_mels()     const { return 128; }
  int hop_length() const { return 160; }

private:
  void InitTables();
  void InitMelFilters();

  std::vector<double> hann_window_;       // length 400
  std::vector<float>  mel_filters_;       // [128, 201] row-major
  std::vector<double> cos_tab_;           // length 400, cos(2π i / N)
  std::vector<double> sin_tab_;           // length 400, sin(2π i / N)
};
