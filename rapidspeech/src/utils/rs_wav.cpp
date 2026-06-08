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
// Sinc-interp Hann resampler (mono float PCM).
//
// This is a direct port of torchaudio.functional.resample with the library's
// defaults (lowpass_filter_width=6, rolloff=0.99, sinc_interp_hann window).
// Bit-exact with `torchaudio.transforms.Resample(orig, new)` to within float
// precision, which matches CosyVoice / FunAudioLLM's wav loader.
// ─────────────────────────────────────────────────────
namespace {

struct ResamplerKernel {
  int orig_g;           // src_sr / gcd
  int new_g;            // dst_sr / gcd
  int width;            // = ceil(lowpass_filter_width * orig_g / base_freq)
  int taps_per_phase;   // = 2*width + orig_g
  // Phase-major taps: kernel[j, p] = taps[j * taps_per_phase + p],
  // j ∈ [0, new_g), p ∈ [0, taps_per_phase).
  std::vector<float> taps;
};

ResamplerKernel build_kernel(int src_sr, int dst_sr) {
  ResamplerKernel k;
  const int g = std::gcd(src_sr, dst_sr);
  k.orig_g = src_sr / g;
  k.new_g  = dst_sr / g;

  // torchaudio defaults
  const int    lowpass_filter_width = 6;
  const double rolloff              = 0.99;
  const double base_freq =
      (double)std::min(k.orig_g, k.new_g) * rolloff;

  k.width = (int)std::ceil((double)lowpass_filter_width *
                           (double)k.orig_g / base_freq);
  k.taps_per_phase = 2 * k.width + k.orig_g;

  const double scale = base_freq / (double)k.orig_g;

  k.taps.assign((size_t)k.new_g * (size_t)k.taps_per_phase, 0.f);
  for (int j = 0; j < k.new_g; ++j) {
    for (int p = 0; p < k.taps_per_phase; ++p) {
      // idx_p (in input-sample units, relative to current frame start):
      //   idx_p = (-width + p) / orig_g
      // Output-phase time offset (in output-sample units): -j / new_g.
      // t = (-j/new_g + idx_p) * base_freq, clamped to ±lowpass_filter_width.
      double t = (-(double)j / (double)k.new_g +
                  (double)(-k.width + p) / (double)k.orig_g) *
                 base_freq;
      const double cw = (double)lowpass_filter_width;
      if (t < -cw) t = -cw;
      if (t >  cw) t =  cw;
      const double w_cos = std::cos(t * M_PI / cw / 2.0);
      const double w     = w_cos * w_cos;
      const double t_pi  = t * M_PI;
      const double sval  = (std::fabs(t_pi) < 1e-30) ? 1.0
                                                     : std::sin(t_pi) / t_pi;
      k.taps[(size_t)j * (size_t)k.taps_per_phase + (size_t)p] =
          (float)(sval * w * scale);
    }
  }
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

  const int64_t N        = (int64_t)in.size();
  const int     orig_g   = k.orig_g;
  const int     new_g    = k.new_g;
  const int     width    = k.width;
  const int     taps     = k.taps_per_phase;

  // torchaudio pads the input with (width) zeros on the left and (width +
  // orig_g) zeros on the right, runs conv1d with stride=orig_g and kernel
  // shape [new_g, 1, taps], transposes [B, new_g, n_conv] → [B, n_conv,
  // new_g] and flattens, then trims to ceil(new_g * N / orig_g).
  const int64_t n_conv     = N / (int64_t)orig_g + 1;
  const int64_t total      = (int64_t)new_g * n_conv;
  const int64_t target_len =
      ((int64_t)new_g * N + (int64_t)orig_g - 1) / (int64_t)orig_g;
  const int64_t out_len    = std::min(target_len, total);
  out.assign((size_t)out_len, 0.f);

  for (int64_t m = 0; m < out_len; ++m) {
    const int64_t n     = m / (int64_t)new_g;          // conv row
    const int     j     = (int)(m % (int64_t)new_g);   // output phase
    const float  *ktaps = &k.taps[(size_t)j * (size_t)taps];
    // First input index touched by this conv row (after un-padding):
    //   in_padded[n*orig_g .. n*orig_g + taps − 1]
    //   in_padded[i] = (i < width) ? 0
    //                : (i − width < N ? in[i − width] : 0)
    const int64_t base = n * (int64_t)orig_g - (int64_t)width;
    double acc = 0.0;
    for (int p = 0; p < taps; ++p) {
      const int64_t idx = base + (int64_t)p;
      float s = 0.f;
      if ((unsigned long long)idx < (unsigned long long)N) {
        s = in[(size_t)idx];
      }
      acc += (double)ktaps[p] * (double)s;
    }
    out[(size_t)m] = (float)acc;
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