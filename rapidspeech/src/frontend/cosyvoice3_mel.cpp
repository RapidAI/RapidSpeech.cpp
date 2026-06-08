#include "frontend/cosyvoice3_mel.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

// Real-input DFT for arbitrary N (not just powers of 2). Ooura's rdft silently
// truncates non-power-of-2 N to the largest pow2 below — e.g. N=400 → 256 — so
// it cannot be used here (n_fft=400 and 1920 are both non-pow2).
//   power[k] = |X[k]|^2,  k = 0 .. N/2
// cos_tab[i] = cos(2π i / N),  sin_tab[i] = sin(2π i / N),  i = 0..N-1
static inline void real_power_spectrum(const double *x, int N,
                                       const double *cos_tab,
                                       const double *sin_tab,
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

constexpr int kSr     = 16000;
constexpr int kNFft   = 400;
constexpr int kHop    = 160;
constexpr int kNMels  = 128;
constexpr int kNBins  = kNFft / 2 + 1;

inline double hz_to_mel_slaney(double hz) {
  const double f_sp        = 200.0 / 3.0;
  const double min_log_hz  = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep     = std::log(6.4) / 27.0;
  return hz >= min_log_hz
             ? min_log_mel + std::log(hz / min_log_hz) / logstep
             : hz / f_sp;
}

inline double mel_to_hz_slaney(double mel) {
  const double f_sp        = 200.0 / 3.0;
  const double min_log_hz  = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep     = std::log(6.4) / 27.0;
  return mel >= min_log_mel
             ? min_log_hz * std::exp(logstep * (mel - min_log_mel))
             : mel * f_sp;
}

} // namespace

CosyVoice3MelExtractor::CosyVoice3MelExtractor() {
  InitTables();
  InitMelFilters();
}
CosyVoice3MelExtractor::~CosyVoice3MelExtractor() = default;

void CosyVoice3MelExtractor::InitTables() {
  // Match torch.hann_window(N_FFT) which defaults to periodic=True:
  //   w[i] = 0.5 − 0.5 * cos(2π * i / N)   for i = 0..N−1
  // (Note: periodic, NOT symmetric — the symmetric version divides by N−1.)
  hann_window_.resize(kNFft);
  for (int i = 0; i < kNFft; ++i)
    hann_window_[i] =
        0.5 - 0.5 * std::cos(2.0 * M_PI * i / (double)kNFft);
  cos_tab_.resize(kNFft);
  sin_tab_.resize(kNFft);
  for (int i = 0; i < kNFft; ++i) {
    cos_tab_[i] = std::cos(2.0 * M_PI * i / (double)kNFft);
    sin_tab_[i] = std::sin(2.0 * M_PI * i / (double)kNFft);
  }
}

void CosyVoice3MelExtractor::InitMelFilters() {
  mel_filters_.assign((size_t)kNMels * kNBins, 0.0f);
  const double mel_lo = hz_to_mel_slaney(0.0);
  const double mel_hi = hz_to_mel_slaney((double)kSr / 2.0);
  std::vector<double> mel_pts(kNMels + 2);
  for (int i = 0; i < kNMels + 2; ++i)
    mel_pts[i] = mel_lo + (mel_hi - mel_lo) * i / (double)(kNMels + 1);
  std::vector<double> hz_pts(kNMels + 2);
  for (int i = 0; i < kNMels + 2; ++i)
    hz_pts[i] = mel_to_hz_slaney(mel_pts[i]);
  const double fft_bin_hz = (double)kSr / (double)kNFft;
  for (int m = 0; m < kNMels; ++m) {
    const double fl = hz_pts[m], fc = hz_pts[m + 1], fr = hz_pts[m + 2];
    const double enorm = 2.0 / (fr - fl);
    for (int k = 0; k < kNBins; ++k) {
      const double hz   = fft_bin_hz * k;
      const double up   = (hz - fl) / (fc - fl);
      const double down = (fr - hz) / (fr - fc);
      const double w    = std::max(0.0, std::min(up, down)) * enorm;
      mel_filters_[(size_t)m * kNBins + k] = (float)w;
    }
  }
}

int CosyVoice3MelExtractor::Compute(const float *pcm_16k, int n,
                                    std::vector<float> &out) const {
  if (!pcm_16k || n <= 0) { out.clear(); return 0; }
  const int pad = kNFft / 2;
  std::vector<float> padded((size_t)(n + 2 * pad));
  for (int i = 0; i < pad; ++i) {
    int src = pad - i; if (src >= n) src = n - 1;
    padded[i] = pcm_16k[src];
  }
  std::memcpy(padded.data() + pad, pcm_16k, (size_t)n * sizeof(float));
  for (int i = 0; i < pad; ++i) {
    int src = n - 2 - i; if (src < 0) src = 0;
    padded[(size_t)pad + n + i] = pcm_16k[src];
  }

  const int n_padded = (int)padded.size();
  if (n_padded < kNFft) { out.clear(); return 0; }
  // V3 matches Whisper's `log_mel_spectrogram` which drops the last STFT
  // frame (`stft[..., :-1]`). T_out = floor(n_samples / hop) so the centered
  // STFT result is `floor((n + 2*pad - n_fft)/hop) + 1` and we drop the last.
  const int n_frames_full = (n_padded - kNFft) / kHop + 1;
  if (n_frames_full <= 1) { out.clear(); return 0; }
  const int n_frames = n_frames_full - 1;

  // Output layout: mel slow, frame fast (matches whisper_mel and the ggml
  // layout ne[0]=T, ne[1]=128).
  out.assign((size_t)kNMels * n_frames, 0.f);

  std::vector<double> buf(kNFft);
  std::vector<double> power(kNBins);
  // DEBUG: optionally dump first-frame STFT power + filterbank for offline diff.
  FILE *dbg_pow = nullptr;
  FILE *dbg_fb  = nullptr;
  if (const char *p = std::getenv("RS_CV3_DUMP_MEL_DEBUG")) {
    std::string path = p;
    dbg_pow = std::fopen((path + ".pow").c_str(), "wb");
    dbg_fb  = std::fopen((path + ".fb").c_str(),  "wb");
  }
  for (int t = 0; t < n_frames; ++t) {
    const int off = t * kHop;
    for (int j = 0; j < kNFft; ++j)
      buf[j] = (double)padded[off + j] * hann_window_[j];
    real_power_spectrum(buf.data(), kNFft,
                        cos_tab_.data(), sin_tab_.data(),
                        power.data());
    if (dbg_pow && t < 5) {
      std::vector<float> pf(kNBins);
      for (int k = 0; k < kNBins; ++k) pf[k] = (float)power[k];
      std::fwrite(pf.data(), sizeof(float), kNBins, dbg_pow);
    }
    for (int m = 0; m < kNMels; ++m) {
      const float *row = &mel_filters_[(size_t)m * kNBins];
      double acc = 0.0;
      for (int k = 0; k < kNBins; ++k) acc += (double)row[k] * power[k];
      // log10(max(mel, 1e-10)) — matches upstream `log_mel_spectrogram`.
      out[(size_t)m * n_frames + t] = (float)std::log10(std::max(acc, 1e-10));
    }
  }
  if (dbg_pow) std::fclose(dbg_pow);
  if (dbg_fb) {
    std::fwrite(mel_filters_.data(), sizeof(float),
                mel_filters_.size(), dbg_fb);
    std::fclose(dbg_fb);
  }

  // Whisper-style global normalization: clamp(x, max-8), shift+4, scale/4.
  float mmax = -1e30f;
  for (float v : out)
    if (v > mmax) mmax = v;
  const float floor_v = mmax - 8.0f;
  for (float &v : out) {
    if (v < floor_v) v = floor_v;
    v = (v + 4.0f) / 4.0f;
  }
  return n_frames;
}

// =====================================================================
// 80-mel @ 24 kHz extractor for the Flow's `prompt_feat` input.
// =====================================================================

namespace {

constexpr int k24Sr     = 24000;
constexpr int k24NFft   = 1920;
constexpr int k24Hop    = 480;
constexpr int k24NMels  = 80;
constexpr int k24NBins  = k24NFft / 2 + 1;

struct Mel80Tables {
  std::vector<double> hann;
  std::vector<float>  mel_fb;        // [k24NMels, k24NBins]
  std::vector<double> cos_tab;       // length k24NFft
  std::vector<double> sin_tab;       // length k24NFft
};

const Mel80Tables &tables_80() {
  static Mel80Tables T = []() {
    Mel80Tables t;
    t.hann.resize(k24NFft);
    for (int i = 0; i < k24NFft; ++i)
      t.hann[i] = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / (double)(k24NFft - 1));
    t.cos_tab.resize(k24NFft);
    t.sin_tab.resize(k24NFft);
    for (int i = 0; i < k24NFft; ++i) {
      t.cos_tab[i] = std::cos(2.0 * M_PI * i / (double)k24NFft);
      t.sin_tab[i] = std::sin(2.0 * M_PI * i / (double)k24NFft);
    }

    t.mel_fb.assign((size_t)k24NMels * k24NBins, 0.f);
    const double mel_lo = hz_to_mel_slaney(0.0);
    const double mel_hi = hz_to_mel_slaney((double)k24Sr / 2.0);
    std::vector<double> mel_pts(k24NMels + 2);
    for (int i = 0; i < k24NMels + 2; ++i)
      mel_pts[i] = mel_lo + (mel_hi - mel_lo) * i / (double)(k24NMels + 1);
    std::vector<double> hz_pts(k24NMels + 2);
    for (int i = 0; i < k24NMels + 2; ++i)
      hz_pts[i] = mel_to_hz_slaney(mel_pts[i]);
    const double fft_bin_hz = (double)k24Sr / (double)k24NFft;
    for (int m = 0; m < k24NMels; ++m) {
      const double fl = hz_pts[m], fc = hz_pts[m + 1], fr = hz_pts[m + 2];
      const double enorm = 2.0 / (fr - fl);
      for (int k = 0; k < k24NBins; ++k) {
        const double hz   = fft_bin_hz * k;
        const double up   = (hz - fl) / (fc - fl);
        const double down = (fr - hz) / (fr - fc);
        const double w    = std::max(0.0, std::min(up, down)) * enorm;
        t.mel_fb[(size_t)m * k24NBins + k] = (float)w;
      }
    }
    return t;
  }();
  return T;
}

} // namespace

int compute_log_mel_80_24k(const float *pcm_24k, int n_samples,
                           std::vector<float> &out) {
  if (!pcm_24k || n_samples <= 0) { out.clear(); return 0; }
  const auto &T = tables_80();
  const int pad = k24NFft / 2;
  std::vector<float> padded((size_t)(n_samples + 2 * pad));
  for (int i = 0; i < pad; ++i) {
    int src = pad - i; if (src >= n_samples) src = n_samples - 1;
    padded[i] = pcm_24k[src];
  }
  std::memcpy(padded.data() + pad, pcm_24k,
              (size_t)n_samples * sizeof(float));
  for (int i = 0; i < pad; ++i) {
    int src = n_samples - 2 - i; if (src < 0) src = 0;
    padded[(size_t)pad + n_samples + i] = pcm_24k[src];
  }
  const int n_padded = (int)padded.size();
  if (n_padded < k24NFft) { out.clear(); return 0; }
  const int n_frames = (n_padded - k24NFft) / k24Hop + 1;

  // Output layout: T-fast, mel-slow → out[t * 80 + m].
  out.assign((size_t)k24NMels * n_frames, 0.f);
  std::vector<double> buf(k24NFft);
  std::vector<double> power(k24NBins);
  for (int t = 0; t < n_frames; ++t) {
    const int off = t * k24Hop;
    for (int j = 0; j < k24NFft; ++j)
      buf[j] = (double)padded[off + j] * T.hann[j];
    real_power_spectrum(buf.data(), k24NFft,
                        T.cos_tab.data(), T.sin_tab.data(),
                        power.data());
    for (int m = 0; m < k24NMels; ++m) {
      const float *row = &T.mel_fb[(size_t)m * k24NBins];
      double acc = 0.0;
      for (int k = 0; k < k24NBins; ++k) acc += (double)row[k] * power[k];
      // natural log (FunAudio convention for `prompt_feat`)
      out[(size_t)t * k24NMels + m] = (float)std::log(std::max(acc, 1e-5));
    }
  }
  return n_frames;
}
