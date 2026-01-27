#include "frontend/audio_processor.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cassert>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PREEMPH_COEFF 0.97f

// External reference to rdft from fftsg.cc or similar library
extern "C" {
void rdft(int n, int isgn, double *a, int *ip, double *w);
}

AudioProcessor::AudioProcessor(const STFTConfig& config) : config_(config) {
  // Ensure n_fft is power of two
  config_.n_fft = RoundToPowerOfTwo(config_.frame_size);

  InitTables();
  InitMelFilters();
}

AudioProcessor::~AudioProcessor() {}

int AudioProcessor::RoundToPowerOfTwo(int n) {
  n--;
  n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
  return n + 1;
}

void AudioProcessor::InitTables() {
  // 1. Hamming Window
  hamming_window_.resize(config_.frame_size);
  for (int i = 0; i < config_.frame_size; i++) {
    hamming_window_[i] = 0.54 - 0.46 * cos((2.0 * M_PI * i) / (config_.frame_size));
  }

  // 2. FFT Tables
  fft_ip_.assign(2 + static_cast<int>(sqrt(config_.n_fft / 2)) + 1, 0);
  fft_w_.assign(config_.n_fft / 2, 0.0);
  // Initialize ip[0] to 0 to trigger internal initialization on first rdft call
  fft_ip_[0] = 0;
}

void AudioProcessor::InitMelFilters() {
  int n_fft = config_.n_fft;
  int n_mels = config_.n_mels;
  int sample_rate = config_.sample_rate;
  int num_bins = n_fft / 2 + 1;

  mel_filters_.assign(n_mels * num_bins, 0.0f);

  auto hz_to_mel = [](float hz) { return 1127.0f * logf(1.0f + hz / 700.0f); };
  auto mel_to_hz = [](float mel) { return 700.0f * (expf(mel / 1127.0f) - 1.0f); };

  float min_mel = hz_to_mel(config_.f_min);
  float max_mel = hz_to_mel(config_.f_max);
  float mel_step = (max_mel - min_mel) / (n_mels + 1);

  for (int i = 0; i < n_mels; i++) {
    float f0 = mel_to_hz(min_mel + i * mel_step);
    float f1 = mel_to_hz(min_mel + (i + 1) * mel_step);
    float f2 = mel_to_hz(min_mel + (i + 2) * mel_step);

    for (int j = 0; j < num_bins; j++) {
      float freq = (float)j * sample_rate / n_fft;
      if (freq >= f0 && freq <= f1) {
        mel_filters_[i * num_bins + j] = (freq - f0) / (f1 - f0);
      } else if (freq > f1 && freq <= f2) {
        mel_filters_[i * num_bins + j] = (f2 - freq) / (f2 - f1);
      }
    }
  }
}

void AudioProcessor::RealFFT(std::vector<double>& window) {
  rdft(config_.n_fft, 1, window.data(), fft_ip_.data(), fft_w_.data());
}

void AudioProcessor::ComputeFbank(const std::vector<float>& samples, std::vector<float>& output_mel) {
  int n_samples = samples.size();
  int n_frames = (n_samples - config_.frame_size) / config_.frame_step + 1;
  output_mel.resize(n_frames * config_.n_mels);

  std::vector<double> window(config_.n_fft);

  for (int i = 0; i < n_frames; i++) {
    int offset = i * config_.frame_step;

    // 1. Copy and Pad
    for (int j = 0; j < config_.n_fft; j++) {
      if (j < config_.frame_size && (offset + j) < n_samples) {
        window[j] = static_cast<double>(samples[offset + j]);
      } else {
        window[j] = 0.0;
      }
    }

    // 2. Remove DC
    double sum = 0.0;
    for (int j = 0; j < config_.frame_size; j++) sum += window[j];
    double mean = sum / config_.frame_size;
    for (int j = 0; j < config_.frame_size; j++) window[j] -= mean;

    // 3. Pre-emphasis
    for (int j = config_.frame_size - 1; j > 0; j--) {
      window[j] -= PREEMPH_COEFF * window[j - 1];
    }
    window[0] -= PREEMPH_COEFF * window[0];

    // 4. Hamming window
    for (int j = 0; j < config_.frame_size; j++) {
      window[j] *= hamming_window_[j];
    }

    // 5. FFT
    RealFFT(window);

    // 6. Power Spectrum
    // rdft output: [Re0, Re(n/2), Re1, Im1, Re2, Im2 ... Re(n/2-1), Im(n/2-1)]
    std::vector<double> power_spec(config_.n_fft / 2 + 1);
    power_spec[0] = window[0] * window[0]; // DC
    power_spec[config_.n_fft / 2] = window[1] * window[1]; // Nyquist
    for (int j = 1; j < config_.n_fft / 2; j++) {
      power_spec[j] = window[2 * j] * window[2 * j] + window[2 * j + 1] * window[2 * j + 1];
    }

    // 7. Mel Filtering
    int num_bins = config_.n_fft / 2 + 1;
    for (int j = 0; j < config_.n_mels; j++) {
      double mel_energy = 0.0;
      for (int k = 0; k < num_bins; k++) {
        mel_energy += power_spec[k] * mel_filters_[j * num_bins + k];
      }
      // Log & Clamp
      output_mel[i * config_.n_mels + j] = static_cast<float>(log(std::max(mel_energy, 1.19e-7)));
    }
  }
}

void AudioProcessor::ApplyLFR(const std::vector<float>& input_mel, int n_frames, std::vector<float>& output_lfr) {
  int m = config_.lfr_m;
  int n = config_.lfr_n;
  int n_mels = config_.n_mels;

  int T_lfr = static_cast<int>(ceil(1.0 * n_frames / n));
  output_lfr.resize(T_lfr * m * n_mels);

  int left_pad = (m - 1) / 2;

  for (int i = 0; i < T_lfr; i++) {
    for (int j = 0; j < m; j++) {
      // Find source frame index with repeat padding
      int source_frame_idx = i * n - left_pad + j;
      if (source_frame_idx < 0) source_frame_idx = 0;
      if (source_frame_idx >= n_frames) source_frame_idx = n_frames - 1;

      // Copy 80-dim mel feature
      std::memcpy(output_lfr.data() + (i * m * n_mels) + (j * n_mels),
                  input_mel.data() + (source_frame_idx * n_mels),
                  n_mels * sizeof(float));
    }
  }
}

void AudioProcessor::SetCMVN(const std::vector<float>& means, const std::vector<float>& vars) {
  cmvn_.means = means;
  cmvn_.vars = vars;
}

void AudioProcessor::ApplyCMVN(std::vector<float>& feats) {
  if (cmvn_.means.empty() || cmvn_.vars.empty()) return;

  int feat_dim = config_.lfr_m * config_.n_mels;
  int n_frames = feats.size() / feat_dim;

  for (int i = 0; i < n_frames; i++) {
    for (int j = 0; j < feat_dim; j++) {
      int idx = i * feat_dim + j;
      feats[idx] = (feats[idx] + cmvn_.means[j]) * cmvn_.vars[j];
    }
  }
}

void AudioProcessor::Compute(const std::vector<float>& input_pcm, std::vector<float>& output_feats) {
  if (input_pcm.empty()) return;

  // 1. PCM -> Fbank
  std::vector<float> mel_feats;
  ComputeFbank(input_pcm, mel_feats);
  int n_frames = mel_feats.size() / config_.n_mels;

  // 2. Fbank -> LFR
  if (config_.use_lfr) {
    std::vector<float> lfr_feats;
    ApplyLFR(mel_feats, n_frames, lfr_feats);
    output_feats = std::move(lfr_feats);
  } else {
    output_feats = std::move(mel_feats);
  }

  // 3. LFR -> CMVN
  if (config_.use_cmvn) {
    ApplyCMVN(output_feats);
  }
}