#include "utils/rs_wav.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

// Simple logging macros to avoid dependency on rs_log.cpp
#define RS_WAV_LOG_INFO(fmt, ...)                                              \
  std::printf("[rs_wav] " fmt "\n", ##__VA_ARGS__)
#define RS_WAV_LOG_WARN(fmt, ...)                                              \
  std::fprintf(stderr, "[rs_wav] Warning: " fmt "\n", ##__VA_ARGS__)
#define RS_WAV_LOG_ERROR(fmt, ...)                                             \
  std::fprintf(stderr, "[rs_wav] Error: " fmt "\n", ##__VA_ARGS__)

bool load_wav_file(const char *filename, std::vector<float> &data,
                   int *sample_rate) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  // Read initial 12 bytes to verify RIFF/WAVE
  char riff_header[12];
  file.read(riff_header, 12);
  if (std::strncmp(riff_header, "RIFF", 4) != 0 ||
      std::strncmp(riff_header + 8, "WAVE", 4) != 0) {
    std::cerr << "[rs_wav] Error: Invalid RIFF/WAVE file." << std::endl;
    return false;
  }

  uint16_t num_channels = 0;
  uint32_t samples_per_sec = 0;
  uint16_t bits_per_sample = 0;
  uint32_t data_size = 0;

  // Robustly search for 'fmt ' and 'data' chunks
  char chunk_id[4];
  uint32_t chunk_size;
  while (file.read(chunk_id, 4) &&
         file.read(reinterpret_cast<char *>(&chunk_size), 4)) {
    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      uint16_t audio_format;
      file.read(reinterpret_cast<char *>(&audio_format), 2);
      file.read(reinterpret_cast<char *>(&num_channels), 2);
      file.read(reinterpret_cast<char *>(&samples_per_sec), 4);
      file.seekg(6, std::ios::cur); // Skip byteRate and blockAlign
      file.read(reinterpret_cast<char *>(&bits_per_sample), 2);

      if (audio_format != 1 || bits_per_sample != 16) {
        std::cerr << "[rs_wav] Error: Only 16-bit PCM is supported."
                  << std::endl;
        return false;
      }
      // Skip remaining fmt chunk if any (e.g. for non-PCM)
      if (chunk_size > 16)
        file.seekg(chunk_size - 16, std::ios::cur);
    } else if (std::strncmp(chunk_id, "data", 4) == 0) {
      data_size = chunk_size;
      break; // Found the data chunk
    } else {
      // Skip unknown chunks
      file.seekg(chunk_size, std::ios::cur);
    }
  }

  if (num_channels == 0 || data_size == 0) {
    std::cerr << "[rs_wav] Error: Could not find audio data." << std::endl;
    return false;
  }

  *sample_rate = static_cast<int>(samples_per_sec);

  // num_samples here is the number of audio frames (time steps)
  int bytes_per_sample = bits_per_sample / 8;
  int num_samples = data_size / (num_channels * bytes_per_sample);
  data.resize(num_samples);

  // Read samples
  for (int i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    file.read(reinterpret_cast<char *>(&sample), sizeof(int16_t));
    data[i] = static_cast<float>(sample) / 32768.0f;

    // Correctly skip the remaining channels for multi-channel files
    if (num_channels > 1) {
      file.seekg(bytes_per_sample * (num_channels - 1), std::ios::cur);
    }
  }

  return true;
}

// ─────────────────────────────────────────────────────
// Polyphase Kaiser-windowed sinc resampler (mono float PCM).
//
// Designed around a rational ratio L/M = dst_sr / src_sr (gcd-reduced).
// For each input sample we contribute it to a span of output positions
// through an FIR low-pass whose cutoff is min(L, M) (in the L*src_sr
// shared upsampled grid). Output sample y[n] is the inner product of the
// FIR phase taps for (n*M) mod L with M input samples around n*M/L.
// ─────────────────────────────────────────────────────
namespace {

double bessel_i0(double x) {
  // Series approximation; sufficient accuracy for Kaiser window design.
  double t = x / 3.75;
  if (std::fabs(x) < 3.75) {
    double t2 = t * t;
    return 1.0 + t2 * (3.5156229 +
                       t2 * (3.0899424 +
                             t2 * (1.2067492 +
                                   t2 * (0.2659732 +
                                         t2 * (0.0360768 + t2 * 0.0045813)))));
  } else {
    double ax = std::fabs(x);
    double y = 3.75 / ax;
    return (std::exp(ax) / std::sqrt(ax)) *
           (0.39894228 +
            y * (0.01328592 +
                 y * (0.00225319 +
                      y * (-0.00157565 +
                           y * (0.00916281 +
                                y * (-0.02057706 +
                                     y * (0.02635537 +
                                          y * (-0.01647633 +
                                               y * 0.00392377))))))));
  }
}

struct ResamplerKernel {
  int L = 1;            // up factor
  int M = 1;            // down factor
  int taps_per_phase;   // M taps per output
  int half_zc;          // sinc half-width in zero-crossings (relative to low-rate)
  std::vector<float> taps; // size L * taps_per_phase, indexed [phase * taps_per_phase + t]
};

ResamplerKernel build_kernel(int src_sr, int dst_sr) {
  ResamplerKernel k;
  int g = std::gcd(src_sr, dst_sr);
  k.L = dst_sr / g;
  k.M = src_sr / g;

  // ~16 zero-crossings on the slower side gives ~80 dB stop-band with β=8.6
  k.half_zc = 16;
  k.taps_per_phase = 2 * k.half_zc;

  const double cutoff = (double)std::min(k.L, k.M) / (double)std::max(k.L, k.M);
  const double beta = 8.6; // Kaiser β; ~80 dB stop-band
  const double i0_beta = bessel_i0(beta);

  const int N = k.L * k.taps_per_phase;
  k.taps.resize(N);

  // Build a continuous prototype low-pass at L*fs_in and split into L phases.
  // Tap m (m=0..N-1) corresponds to time offset
  //   x_m = (m - (N-1)/2) / L   (in units of input samples)
  // multiplied by the sinc cutoff.
  const double center = 0.5 * (double)(N - 1);
  double sum = 0.0;
  for (int m = 0; m < N; ++m) {
    double x = ((double)m - center) / (double)k.L;
    double sinc_arg = M_PI * cutoff * x;
    double sinc_val =
        (std::fabs(sinc_arg) < 1e-12) ? 1.0 : std::sin(sinc_arg) / sinc_arg;
    double w_arg = ((double)m - center) / center;
    double w = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - w_arg * w_arg))) /
               i0_beta;
    double tap = cutoff * sinc_val * w;
    k.taps[m] = (float)tap;
    sum += tap;
  }

  // Per-phase normalization so DC gain == 1 on every output phase.
  for (int phase = 0; phase < k.L; ++phase) {
    double psum = 0.0;
    for (int t = 0; t < k.taps_per_phase; ++t) {
      int m = t * k.L + phase;
      psum += k.taps[m];
    }
    if (psum > 0.0) {
      double scale = 1.0 / psum;
      for (int t = 0; t < k.taps_per_phase; ++t) {
        int m = t * k.L + phase;
        k.taps[m] = (float)(k.taps[m] * scale);
      }
    }
  }
  (void)sum;
  return k;
}

} // namespace

bool resample_pcm(const std::vector<float> &in, int src_sr,
                  std::vector<float> &out, int dst_sr) {
  if (src_sr <= 0 || dst_sr <= 0) {
    RS_WAV_LOG_ERROR("resample_pcm: invalid sample rates src=%d dst=%d",
                     src_sr, dst_sr);
    return false;
  }
  if (in.empty()) {
    out.clear();
    return true;
  }
  if (src_sr == dst_sr) {
    out = in;
    return true;
  }

  ResamplerKernel k = build_kernel(src_sr, dst_sr);

  const int N = (int)in.size();
  // Output length: ceil(N * dst_sr / src_sr) == ceil(N * L / M)
  const int64_t out_len =
      ((int64_t)N * (int64_t)k.L + (int64_t)k.M - 1) / (int64_t)k.M;
  out.assign((size_t)out_len, 0.f);

  // For output index n, the corresponding low-rate index is n*M/L; phase
  // is (n*M) mod L. Convolution centers on the prototype filter midpoint
  // (half_zc input samples), so input start index is
  //   base = floor(n*M / L) - half_zc + 1
  // and we read taps_per_phase samples from there.
  const int half = k.half_zc;
  const int taps = k.taps_per_phase;

  for (int64_t n = 0; n < out_len; ++n) {
    int64_t nm = n * (int64_t)k.M;
    int64_t base_low = nm / (int64_t)k.L;
    int phase = (int)(nm - base_low * (int64_t)k.L);
    int64_t in_start = base_low - (half - 1);

    const float *phase_taps = &k.taps[(size_t)phase * (size_t)taps];
    double acc = 0.0;
    for (int t = 0; t < taps; ++t) {
      int64_t idx = in_start + t;
      float s = 0.f;
      if (idx >= 0 && idx < N) {
        s = in[(size_t)idx];
      }
      acc += (double)phase_taps[t] * (double)s;
    }
    out[(size_t)n] = (float)acc;
  }
  return true;
}

bool load_wav_file_resampled(const char *filename, std::vector<float> &data,
                             int target_sample_rate, int *src_sample_rate) {
  std::vector<float> raw;
  int sr = 0;
  if (!load_wav_file(filename, raw, &sr)) {
    return false;
  }
  if (src_sample_rate) {
    *src_sample_rate = sr;
  }
  if (target_sample_rate <= 0 || sr == target_sample_rate) {
    data = std::move(raw);
    return true;
  }
  return resample_pcm(raw, sr, data, target_sample_rate);
}

/**
 * Helper to load CMVN from GGUF or fallback to defaults.
 * @param ctx_gguf The loaded GGUF context.
 * @param means Target vector for mean values.
 * @param vars Target vector for variance values.
 */
void load_cmvn_params(struct gguf_context *ctx_gguf, std::vector<float> &means,
                      std::vector<float> &vars) {
  bool loaded = false;

  // 1. Try to load from GGUF KV
  int key_means = gguf_find_key(ctx_gguf, "model.cmvn_means");
  int key_vars = gguf_find_key(ctx_gguf, "model.cmvn_vars");

  if (key_means != -1 && key_vars != -1) {
    const float *data_means =
        (const float *)gguf_get_arr_data(ctx_gguf, key_means);
    const float *data_vars =
        (const float *)gguf_get_arr_data(ctx_gguf, key_vars);
    int n_means = gguf_get_arr_n(ctx_gguf, key_means);
    int n_vars = gguf_get_arr_n(ctx_gguf, key_vars);

    if (n_means == 560 && n_vars == 560) {
      means.assign(data_means, data_means + 560);
      vars.assign(data_vars, data_vars + 560);
      RS_WAV_LOG_INFO("CMVN parameters loaded from GGUF KV metadata.");
      loaded = true;
    } else {
      RS_WAV_LOG_WARN(
          "CMVN metadata size mismatch (expected 560, got %d). Falling back.",
          n_means);
    }
  }
}