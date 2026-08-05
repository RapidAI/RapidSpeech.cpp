#include "frontend/whisper_mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Real-input DFT for arbitrary N. Ooura's rdft silently truncates non-pow2 N
// to the largest pow2 below (n_fft=400 → 256), so it cannot be used here —
// see the cosyvoice3_mel.cpp note on the same trap.
//   power[k] = |X[k]|^2,  k = 0 .. N/2
inline void real_power_spectrum(const double *x, int N,
                                const double *cos_tab, const double *sin_tab,
                                double *power) {
  for (int k = 0; k <= N / 2; ++k) {
    double re = 0.0, im = 0.0;
    int phase = 0;
    for (int n = 0; n < N; ++n) {
      re += x[n] * cos_tab[phase];
      im -= x[n] * sin_tab[phase];
      phase += k;
      if (phase >= N) phase -= N;
    }
    power[k] = re * re + im * im;
  }
}

// Convert Hz to Slaney mel (matches librosa.filters.mel(htk=False)).
//   Below 1000 Hz: linear ramp at 200/3 Hz per mel.
//   Above 1000 Hz: log ramp through 6.4 octaves spanning the remainder.
inline double hz_to_mel_slaney(double hz) {
  const double f_min        = 0.0;
  const double f_sp         = 200.0 / 3.0;
  const double min_log_hz   = 1000.0;
  const double min_log_mel  = (min_log_hz - f_min) / f_sp; // 15
  const double logstep      = std::log(6.4) / 27.0;
  if (hz >= min_log_hz) {
    return min_log_mel + std::log(hz / min_log_hz) / logstep;
  }
  return (hz - f_min) / f_sp;
}

inline double mel_to_hz_slaney(double mel) {
  const double f_min        = 0.0;
  const double f_sp         = 200.0 / 3.0;
  const double min_log_hz   = 1000.0;
  const double min_log_mel  = (min_log_hz - f_min) / f_sp;
  const double logstep      = std::log(6.4) / 27.0;
  if (mel >= min_log_mel) {
    return min_log_hz * std::exp(logstep * (mel - min_log_mel));
  }
  return f_min + f_sp * mel;
}

} // namespace

WhisperMelExtractor::WhisperMelExtractor(const WhisperMelConfig &config)
    : cfg_(config) {
  InitTables();
  InitMelFilters();
}

WhisperMelExtractor::~WhisperMelExtractor() = default;

void WhisperMelExtractor::InitTables() {
  // Hann window over n_fft samples. Symmetric matches torch.hann_window with
  // periodic=False (original Qwen3-ASR bring-up); periodic matches
  // torch.hann_window default / HF WhisperFeatureExtractor's
  // window_function(400, "hann").
  hann_window_.resize(cfg_.n_fft);
  const double denom =
      cfg_.periodic_window ? (double)cfg_.n_fft : (double)(cfg_.n_fft - 1);
  for (int i = 0; i < cfg_.n_fft; ++i) {
    hann_window_[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / denom);
  }

  // DFT twiddle tables (direct DFT — n_fft=400 is not a power of 2, which
  // Ooura's rdft silently truncates).
  cos_tab_.resize(cfg_.n_fft);
  sin_tab_.resize(cfg_.n_fft);
  for (int i = 0; i < cfg_.n_fft; ++i) {
    cos_tab_[i] = std::cos(2.0 * M_PI * i / (double)cfg_.n_fft);
    sin_tab_[i] = std::sin(2.0 * M_PI * i / (double)cfg_.n_fft);
  }
}

void WhisperMelExtractor::InitMelFilters() {
  const int n_mels  = cfg_.n_mels;
  const int n_bins  = cfg_.n_fft / 2 + 1;
  mel_filters_.assign((size_t)n_mels * n_bins, 0.0f);

  const double mel_lo = hz_to_mel_slaney(cfg_.f_min);
  const double mel_hi = hz_to_mel_slaney(cfg_.f_max);
  std::vector<double> mel_points(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    mel_points[i] = mel_lo + (mel_hi - mel_lo) * i / (double)(n_mels + 1);
  }
  std::vector<double> hz_points(n_mels + 2);
  for (int i = 0; i < n_mels + 2; ++i) {
    hz_points[i] = mel_to_hz_slaney(mel_points[i]);
  }

  const double fft_bin_hz = (double)cfg_.sample_rate / (double)cfg_.n_fft;
  for (int m = 0; m < n_mels; ++m) {
    const double f_left   = hz_points[m];
    const double f_center = hz_points[m + 1];
    const double f_right  = hz_points[m + 2];
    // Slaney area-normalization factor.
    const double enorm = 2.0 / (f_right - f_left);
    for (int k = 0; k < n_bins; ++k) {
      const double hz       = fft_bin_hz * k;
      const double up       = (hz - f_left) / (f_center - f_left);
      const double down     = (f_right - hz) / (f_right - f_center);
      const double weight   = std::max(0.0, std::min(up, down)) * enorm;
      mel_filters_[(size_t)m * n_bins + k] = (float)weight;
    }
  }
}

int WhisperMelExtractor::Compute(const std::vector<float> &pcm,
                                 std::vector<float> &out) const {
  const int n_fft   = cfg_.n_fft;
  const int hop     = cfg_.hop_length;
  const int n_mels  = cfg_.n_mels;
  const int n_bins  = n_fft / 2 + 1;
  const int chunk   = cfg_.chunk_size;
  const int pad     = n_fft / 2;

  if (pcm.empty()) {
    out.clear();
    return 0;
  }

  // 1. Reflection padding (matches WhisperFeatureExtractor(center=True)).
  std::vector<float> padded;
  padded.resize((size_t)pcm.size() + 2 * pad);
  // Reflect-without-endpoint: mirror across index 0 / index N-1.
  for (int i = 0; i < pad; ++i) {
    int src = pad - i;             // 1..pad
    if (src >= (int)pcm.size()) src = (int)pcm.size() - 1;
    padded[i] = pcm[src];
  }
  std::memcpy(padded.data() + pad, pcm.data(), pcm.size() * sizeof(float));
  for (int i = 0; i < pad; ++i) {
    int src = (int)pcm.size() - 2 - i; // N-2, N-3, ...
    if (src < 0) src = 0;
    padded[pad + pcm.size() + i] = pcm[src];
  }

  // 2. Frame count after centering: floor((N + 2*pad - n_fft) / hop) + 1
  //    With center=True this is equivalent to ceil(N / hop).
  const int n_padded = (int)padded.size();
  if (n_padded < n_fft) {
    out.clear();
    return 0;
  }
  int n_frames = (n_padded - n_fft) / hop + 1;
  // HF WhisperFeatureExtractor drops the trailing STFT frame
  // (log_spec[:, :-1]): 480000 samples -> 3001 frames -> 3000.
  if (cfg_.max_frames > 0 && n_frames > cfg_.max_frames) {
    n_frames = cfg_.max_frames;
  }

  // 3. STFT + |X|^2 + mel projection + log10 for every frame.
  //    Output layout: mel-bin slow, frame fast. log_mel[m * n_frames + t].
  std::vector<float> log_mel((size_t)n_mels * n_frames, 0.0f);

  auto frame_worker = [&](int t_start, int t_end) {
    std::vector<double> buf(n_fft);
    std::vector<double> power(n_bins);
    for (int t = t_start; t < t_end; ++t) {
      const int offset = t * hop;
      for (int j = 0; j < n_fft; ++j) {
        buf[j] = (double)padded[offset + j] * hann_window_[j];
      }
      real_power_spectrum(buf.data(), n_fft, cos_tab_.data(),
                          sin_tab_.data(), power.data());
      for (int m = 0; m < n_mels; ++m) {
        const float *row = &mel_filters_[(size_t)m * n_bins];
        double acc = 0.0;
        for (int k = 0; k < n_bins; ++k) {
          acc += (double)row[k] * power[k];
        }
        log_mel[(size_t)m * n_frames + t] =
            (float)std::log10(std::max(acc, 1e-10));
      }
    }
  };

#if defined(_OPENMP)
#pragma omp parallel
  {
    int nthr = omp_get_num_threads();
    int tid  = omp_get_thread_num();
    int per  = (n_frames + nthr - 1) / nthr;
    int s    = std::min(tid * per, n_frames);
    int e    = std::min(s + per, n_frames);
    frame_worker(s, e);
  }
#else
  frame_worker(0, n_frames);
#endif

  // 4. Whisper-style normalization: clamp(log_mel, max-8) shifted to [0, 1+].
  //    Computed across the whole utterance once (so a longer wav stays
  //    self-consistent across chunks).
  float mmax = -1e30f;
  for (float v : log_mel) {
    if (v > mmax) mmax = v;
  }
  const float floor_v = mmax - 8.0f;
  for (float &v : log_mel) {
    if (v < floor_v) v = floor_v;
    v = (v + 4.0f) / 4.0f;
  }

  // 5. Pad frame count up to a multiple of chunk_size; copy data and zero-fill
  //    the trailing frames. Output layout: frame fast, mel-bin slow, so the
  //    encoder graph can view it as [chunk_size, n_mel, 1, n_chunks] with
  //    nb[1] = n_out_frames * sizeof(float).
  const int n_eff   = std::min(n_frames, (int)pcm.size() / hop + 1);
  const int n_chunks = (n_eff + chunk - 1) / chunk;
  const int n_out_frames = std::max(1, n_chunks) * chunk;

  out.assign((size_t)n_mels * n_out_frames, 0.0f);
  const int n_copy = std::min(n_eff, n_frames);
  for (int m = 0; m < n_mels; ++m) {
    std::memcpy(out.data() + (size_t)m * n_out_frames,
                log_mel.data() + (size_t)m * n_frames,
                (size_t)n_copy * sizeof(float));
    // Trailing frames stay zero-filled (the floor value of normalized log-mel
    // is 0, since (mmax-8 + 4)/4 - we put 0 which is below the floor — but
    // the encoder ignores these positions via chunk masking, so the exact
    // value does not matter).
  }
  return n_out_frames;
}
