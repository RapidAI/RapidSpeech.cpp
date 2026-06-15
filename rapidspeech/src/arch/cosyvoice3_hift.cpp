#include "cosyvoice3_hift.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

// =====================================================================
// File-local helpers.
// =====================================================================

namespace {

constexpr float kPi = 3.14159265358979323846f;

// ne[0] = T, ne[1] = C. Insert dim-1 axis at position 1.
ggml_tensor *unsqueeze1(ggml_context *ctx, ggml_tensor *x) {
  return ggml_view_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2],
                      x->nb[1], x->nb[1], x->nb[2], 0);
}
ggml_tensor *unsqueeze0(ggml_context *ctx, ggml_tensor *x) {
  return ggml_view_4d(ctx, x, 1, x->ne[0], x->ne[1], x->ne[2],
                      x->nb[0], x->nb[1], x->nb[2], 0);
}

// Linear(W [in, out], bias[out]): y = W·x + b. Bias optional.
ggml_tensor *linear(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                    ggml_tensor *x) {
  ggml_tensor *y = ggml_mul_mat(ctx, w, x);
  if (b) y = ggml_add(ctx, y, b);
  return y;
}

// Plain Conv1d (group=1) on layout [T, C_in, B] with weight [K, C_in, C_out].
ggml_tensor *conv1d(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                    ggml_tensor *x, int stride, int dilation) {
  ggml_tensor *im = ggml_im2col(ctx, w, x, stride, 0, 0, 0, dilation, 0,
                                false, GGML_TYPE_F16);
  ggml_tensor *wm = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
  ggml_tensor *y  = ggml_mul_mat(ctx, im, wm);
  y = ggml_reshape_3d(ctx, y, im->ne[1], w->ne[2], x->ne[2]);
  if (b) y = ggml_add(ctx, y, unsqueeze0(ctx, b));
  return y;
}

// CausalConv1d helper. Pads left or right by the causal amount, then plain
// stride-1 conv (with the supplied dilation). Matches ref1's
// `CausalConv1d::causal_padding()`:
//   pad = (k·d − d)/2·2 + (k+1)%2
enum class CausalSide { Left, Right };
ggml_tensor *causal_conv1d(ggml_context *ctx, ggml_tensor *w, ggml_tensor *b,
                           ggml_tensor *x, int dilation, CausalSide side) {
  const int k    = (int)w->ne[0];
  const int half = (k * dilation - dilation) / 2 * 2;
  const int odd  = (k + 1) % 2;
  const int pad  = half + odd;
  if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
  if (side == CausalSide::Left) {
    x = ggml_pad_ext(ctx, x, pad, 0, 0, 0, 0, 0, 0, 0);
  } else {
    x = ggml_pad_ext(ctx, x, 0, pad, 0, 0, 0, 0, 0, 0);
  }
  return conv1d(ctx, w, b, x, 1, dilation);
}

// Strided down-sample causal conv (used by source_downs[0..N-2]): left-pad
// (stride-1) then conv at stride=stride. Matches ref1
// `CausalConv1dDownSample`.
ggml_tensor *causal_downsample_conv1d(ggml_context *ctx, ggml_tensor *w,
                                      ggml_tensor *b, ggml_tensor *x,
                                      int stride) {
  if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
  x = ggml_pad_ext(ctx, x, stride - 1, 0, 0, 0, 0, 0, 0, 0);
  return conv1d(ctx, w, b, x, stride, 1);
}

// Upsample-then-conv (ref1 CausalConv1dUpsample). Nearest-upsample by rate
// along T (ne[0]), left-pad (k-1), then plain conv stride=1.
ggml_tensor *causal_upsample_conv1d(ggml_context *ctx, ggml_tensor *w,
                                    ggml_tensor *b, ggml_tensor *x,
                                    int rate) {
  // x is [T, C_in, B]. interpolate along ne[0] only.
  x = ggml_interpolate(ctx, x,
                       x->ne[0] * (int64_t)rate, x->ne[1], x->ne[2], 1,
                       GGML_SCALE_MODE_NEAREST);
  const int k = (int)w->ne[0];
  x = ggml_pad_ext(ctx, x, k - 1, 0, 0, 0, 0, 0, 0, 0);
  return conv1d(ctx, w, b, x, 1, 1);
}

// Snake activation:  x + sin(alpha·x)^2 / alpha
ggml_tensor *snake(ggml_context *ctx, ggml_tensor *alpha, ggml_tensor *x) {
  ggml_tensor *a = unsqueeze0(ctx, alpha);
  ggml_tensor *s = ggml_sin(ctx, ggml_mul(ctx, x, a));
  s = ggml_mul(ctx, s, s);
  return ggml_add(ctx, x, ggml_div(ctx, s, a));
}

} // namespace

// =====================================================================
// Load
// =====================================================================

bool CosyVoice3HiFTModel::Load(gguf_context *ctx_gguf,
                               ggml_context *gguf_data,
                               ggml_backend_t backend) {
  if (!ctx_gguf || !gguf_data) {
    RS_LOG_ERR("HiFT: invalid arguments");
    return false;
  }
  (void)backend;  // hift weights are already on the backend buffer

  if (!LoadHparams(ctx_gguf)) return false;

  std::map<std::string, ggml_tensor *> tensors;
  const int n = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }
  if (!LoadTensors(tensors)) return false;

  // Hann window + deterministic rand_ini precomputed once.
  PrecomputeWindowAndRandIni(0xC05A3ULL);

  RS_LOG_INFO("HiFT loaded: sr=%d n_fft=%d hop=%d rates=[%d,%d,%d] "
              "num_kernels=%d scale_factor=%d resblocks=%zu",
              sample_rate_, n_fft_, hop_len_,
              upsample_rates_[0], upsample_rates_[1], upsample_rates_[2],
              num_kernels_, scale_factor_, resblocks_.size());
  return true;
}

bool CosyVoice3HiFTModel::LoadHparams(gguf_context *ctx_gguf) {
  auto u32 = [&](const char *k, int dflt) {
    int id = gguf_find_key(ctx_gguf, k);
    return id >= 0 ? (int)gguf_get_val_u32(ctx_gguf, id) : dflt;
  };
  auto f32 = [&](const char *k, float dflt) {
    int id = gguf_find_key(ctx_gguf, k);
    return id >= 0 ? gguf_get_val_f32(ctx_gguf, id) : dflt;
  };
  sample_rate_         = u32("cosyvoice3.hift.sample_rate", 24000);
  n_fft_               = u32("cosyvoice3.hift.n_fft",       16);
  hop_len_             = u32("cosyvoice3.hift.hop_len",     4);
  nb_harmonics_        = u32("cosyvoice3.hift.nb_harmonics", 8);
  nsf_alpha_           = f32("cosyvoice3.hift.nsf_alpha",   0.1f);
  nsf_sigma_           = f32("cosyvoice3.hift.nsf_sigma",   0.003f);
  nsf_voiced_threshold_ = u32("cosyvoice3.hift.nsf_voiced_threshold", 10);
  lrelu_slope_         = f32("cosyvoice3.hift.lrelu_slope", 0.1f);
  audio_limit_         = f32("cosyvoice3.hift.audio_limit", 0.99f);
  num_kernels_         = u32("cosyvoice3.hift.num_kernels", 3);

  // upsample_rates is a uint32 array — fetch via GGUF metadata accessors.
  int id = gguf_find_key(ctx_gguf, "cosyvoice3.hift.upsample_rates");
  if (id >= 0) {
    int n_arr = (int)gguf_get_arr_n(ctx_gguf, id);
    upsample_rates_.resize(n_arr);
    for (int i = 0; i < n_arr; ++i) {
      const void *d = gguf_get_arr_data(ctx_gguf, id);
      upsample_rates_[i] =
          static_cast<const uint32_t *>(d)[i];
    }
  } else {
    upsample_rates_ = {8, 5, 3};
  }
  scale_factor_ = hop_len_;
  for (int r : upsample_rates_) scale_factor_ *= r;
  return true;
}

bool CosyVoice3HiFTModel::LoadResBlock(
    const std::map<std::string, ggml_tensor *> &tensors,
    const std::string &prefix, ResBlockW &out) const {
  bool ok = true;
  for (int i = 0; i < 3; ++i) {
    auto &s = out.subs[i];
    s.dilation = (i == 0) ? 1 : (i == 1) ? 3 : 5;
    char buf[256];
    auto fmt = [&](const char *fmtstr) {
      snprintf(buf, sizeof(buf), fmtstr, prefix.c_str(), i);
      return std::string(buf);
    };
    auto it = tensors.find(fmt("%s.activations1.%d.alpha"));
    if (it == tensors.end()) { ok = false; break; } s.a1 = it->second;
    it = tensors.find(fmt("%s.convs1.%d.weight"));
    if (it == tensors.end()) { ok = false; break; } s.c1w = it->second;
    it = tensors.find(fmt("%s.convs1.%d.bias"));
    if (it == tensors.end()) { ok = false; break; } s.c1b = it->second;
    it = tensors.find(fmt("%s.activations2.%d.alpha"));
    if (it == tensors.end()) { ok = false; break; } s.a2 = it->second;
    it = tensors.find(fmt("%s.convs2.%d.weight"));
    if (it == tensors.end()) { ok = false; break; } s.c2w = it->second;
    it = tensors.find(fmt("%s.convs2.%d.bias"));
    if (it == tensors.end()) { ok = false; break; } s.c2b = it->second;
  }
  return ok;
}

bool CosyVoice3HiFTModel::LoadTensors(
    const std::map<std::string, ggml_tensor *> &tensors) {
  auto get = [&](const char *name, ggml_tensor **dst, bool required = true) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      if (required) RS_LOG_ERR("HiFT: missing tensor %s", name);
      return !required;
    }
    *dst = it->second;
    return true;
  };
  bool ok = true;

  // F0 condnet: indices 0, 2, 4, 6, 8.
  const int idxs[5] = {0, 2, 4, 6, 8};
  char buf[256];
  for (int i = 0; i < 5; ++i) {
    snprintf(buf, sizeof(buf), "hift.f0_predictor.condnet.%d.weight", idxs[i]);
    ok &= get(buf, &condnet_w_[i]);
    snprintf(buf, sizeof(buf), "hift.f0_predictor.condnet.%d.bias", idxs[i]);
    ok &= get(buf, &condnet_b_[i]);
  }
  ok &= get("hift.f0_predictor.classifier.weight", &classifier_w_);
  ok &= get("hift.f0_predictor.classifier.bias",   &classifier_b_);

  ok &= get("hift.m_source.l_linear.weight", &m_source_lin_w_);
  ok &= get("hift.m_source.l_linear.bias",   &m_source_lin_b_);

  ok &= get("hift.conv_pre.weight",  &conv_pre_w_);
  ok &= get("hift.conv_pre.bias",    &conv_pre_b_);
  ok &= get("hift.conv_post.weight", &conv_post_w_);
  ok &= get("hift.conv_post.bias",   &conv_post_b_);

  // Upsample stages.
  const int n_stages = (int)upsample_rates_.size();
  ups_.resize(n_stages);
  source_downs_.resize(n_stages);
  source_resblocks_.resize(n_stages);
  for (int i = 0; i < n_stages; ++i) {
    auto &u = ups_[i];
    u.rate = upsample_rates_[i];
    snprintf(buf, sizeof(buf), "hift.ups.%d.weight", i);
    ok &= get(buf, &u.w);
    snprintf(buf, sizeof(buf), "hift.ups.%d.bias", i);
    ok &= get(buf, &u.b);
    u.kernel = u.w ? (int)u.w->ne[0] : 0;

    auto &sd = source_downs_[i];
    snprintf(buf, sizeof(buf), "hift.source_downs.%d.weight", i);
    ok &= get(buf, &sd.w);
    snprintf(buf, sizeof(buf), "hift.source_downs.%d.bias", i);
    ok &= get(buf, &sd.b);
    sd.kernel = sd.w ? (int)sd.w->ne[0] : 0;
    // Stride is the product of upsample rates AFTER index i. The last stage
    // is stride=1 (plain causal conv).
    int s = 1;
    for (int j = i + 1; j < n_stages; ++j) s *= upsample_rates_[j];
    sd.stride = s;

    snprintf(buf, sizeof(buf), "hift.source_resblocks.%d", i);
    if (!LoadResBlock(tensors, buf, source_resblocks_[i])) {
      RS_LOG_ERR("HiFT: failed to load %s", buf);
      ok = false;
    }
  }

  // 3 stages × num_kernels resblocks.
  const int n_res = n_stages * num_kernels_;
  resblocks_.resize(n_res);
  for (int i = 0; i < n_res; ++i) {
    snprintf(buf, sizeof(buf), "hift.resblocks.%d", i);
    if (!LoadResBlock(tensors, buf, resblocks_[i])) {
      RS_LOG_ERR("HiFT: failed to load %s", buf);
      ok = false;
    }
  }
  return ok;
}

void CosyVoice3HiFTModel::PrecomputeWindowAndRandIni(uint64_t seed) {
  hann_window_.assign(n_fft_, 0.f);
  for (int i = 0; i < n_fft_; ++i)
    hann_window_[i] = 0.5f * (1.f - std::cos(2.f * kPi * (float)i / n_fft_));

  // rand_ini[0] = 0; rand_ini[1..nb_harmonics] = uniform[0,1) seeded.
  rand_ini_.assign(nb_harmonics_ + 1, 0.f);
  std::mt19937 rng((uint32_t)seed);
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 1; i <= nb_harmonics_; ++i) rand_ini_[i] = dist(rng);
}

// =====================================================================
// F0 graph: speech_feat [80, T_mel, 1] → f0 [T_mel].
// Mirrors cosyvoice-graph.cpp:602 CausalConvRNNF0Predictor::build_cgraph.
// =====================================================================

ggml_tensor *CosyVoice3HiFTModel::BuildF0Graph(ggml_context *ctx,
                                               ggml_tensor *speech_feat) const {
  // speech_feat arrives as (mel_dim, T_mel, 1) — channels in ne[0]. Conv1d
  // helpers expect (T, C, B) layout (T in ne[0]); permute first.
  ggml_tensor *x = ggml_permute(ctx, speech_feat, 1, 0, 2, 3);
  x = ggml_cont(ctx, x);                          // (T_mel, mel_dim, 1, 1)

  // condnet_0 is right-causal (k=4); the rest are left-causal (k=3).
  static constexpr CausalSide sides[5] = {
      CausalSide::Right, CausalSide::Left, CausalSide::Left,
      CausalSide::Left,  CausalSide::Left};
  for (int i = 0; i < 5; ++i) {
    x = causal_conv1d(ctx, condnet_w_[i], condnet_b_[i], x, 1, sides[i]);
    x = ggml_elu(ctx, x);
  }
  // After condnets x is (T_mel, 512, 1). Linear needs channels in ne[0], so
  // permute back: (512, T_mel, 1).
  x = ggml_permute(ctx, x, 1, 0, 2, 3);
  x = ggml_cont(ctx, x);
  x = linear(ctx, classifier_w_, classifier_b_, x);   // (1, T_mel, 1)
  // Squeeze the channel axis.
  x = ggml_reshape_2d(ctx, x, x->ne[1], x->ne[2]);    // (T_mel, 1)
  return ggml_abs(ctx, x);
}

// =====================================================================
// CPU source path. Inputs:
//   f0_host[T_mel]   — predicted F0 (positive)
// Output:
//   out_sine_merge[T_audio]  where T_audio = T_mel * scale_factor_
//
// Mirrors `SineGen2 + SourceModuleHnNSF` from cosyvoice-graph.cpp:622-673.
// All math is single-channel-friendly and avoids any ggml fiddly bits.
// =====================================================================

void CosyVoice3HiFTModel::RunSourceCpu(const float *f0_host, int T_mel,
                                       std::vector<float> &out_sine_merge,
                                       std::mt19937 &rng) const {
  const int n_h     = nb_harmonics_ + 1;        // 9
  const int sf      = scale_factor_;            // 480
  const int T_audio = T_mel * sf;

  // 1) Nearest-upsample f0 from T_mel to T_audio.
  std::vector<float> f0_up((size_t)T_audio);
  for (int i = 0; i < T_audio; ++i)
    f0_up[i] = f0_host[i / sf];

  // 2) rad_values: for each harmonic h (1..n_h), phase increment per sample is
  //    h · f0 / sample_rate. The accumulated phase starts at rand_ini[h-1] and
  //    accumulates over time. To match torch's implementation:
  //      rad = ((h · f0) / sr) mod 1   (cumulative phase resets removed via mod)
  //      rad += rand_ini[h-1]
  //      rad → bilinear downsample by sf  (so that the cumsum below operates
  //                                          on a T_mel-scale phase grid)
  //      phase = cumsum(rad) · (2π · sf)
  //      phase → nearest upsample by sf back to T_audio
  //      sine = sin(phase) · sine_amp
  //
  //    Notes:
  //    - For voiced regions, this produces a steady harmonic with continuous
  //      phase. For unvoiced regions f0 ≤ voiced_threshold and the sine is
  //      masked out via uv.
  //    - The interleaved upsample/downsample around the cumsum is the
  //      FunAudio-NSF recipe — it keeps the phase accumulator stable for very
  //      short audio while letting the per-sample sine evaluate at audio rate.
  std::vector<float> rad_audio((size_t)n_h * T_audio);
  for (int h = 0; h < n_h; ++h) {
    const float harmonic_idx = (float)(h + 1);
    for (int t = 0; t < T_audio; ++t) {
      float v = harmonic_idx * f0_up[t] / (float)sample_rate_;
      v -= std::floor(v);
      rad_audio[(size_t)h * T_audio + t] = v;
    }
    // PT adds rand_ini only at t=0 (initial phase offset, not a frequency shift).
    rad_audio[(size_t)h * T_audio + 0] += rand_ini_[h];
  }
  // Bilinear downsample T_audio → T_mel along time axis.
  std::vector<float> rad_mel((size_t)n_h * T_mel);
  for (int h = 0; h < n_h; ++h) {
    for (int t = 0; t < T_mel; ++t) {
      // Map index t (in [0, T_mel)) to the source [0, T_audio) range with
      // align_corners=false. Standard ggml bilinear uses idx_src =
      // (t + 0.5) · sf − 0.5.
      float fsrc = (t + 0.5f) * (float)sf - 0.5f;
      int   i0   = (int)std::floor(fsrc);
      float fr   = fsrc - i0;
      int   i1   = i0 + 1;
      i0 = std::max(0, std::min(T_audio - 1, i0));
      i1 = std::max(0, std::min(T_audio - 1, i1));
      rad_mel[(size_t)h * T_mel + t] =
          rad_audio[(size_t)h * T_audio + i0] * (1.f - fr) +
          rad_audio[(size_t)h * T_audio + i1] * fr;
    }
  }
  // Cumulative sum along time per harmonic.
  std::vector<float> phase_mel((size_t)n_h * T_mel);
  for (int h = 0; h < n_h; ++h) {
    float acc = 0.f;
    for (int t = 0; t < T_mel; ++t) {
      acc += rad_mel[(size_t)h * T_mel + t];
      phase_mel[(size_t)h * T_mel + t] = acc * 2.f * kPi * (float)sf;
    }
  }
  // Nearest upsample phase from T_mel back to T_audio.
  std::vector<float> phase_audio((size_t)n_h * T_audio);
  for (int h = 0; h < n_h; ++h) {
    for (int t = 0; t < T_audio; ++t) {
      phase_audio[(size_t)h * T_audio + t] =
          phase_mel[(size_t)h * T_mel + (t / sf)];
    }
  }

  // 3) Voiced mask uv.
  std::vector<float> uv((size_t)T_audio);
  constexpr float eps = 1e-7f;
  for (int t = 0; t < T_audio; ++t) {
    float v = f0_up[t] - (float)nsf_voiced_threshold_;
    float pos = std::max(0.f, v);
    float denom = std::fabs(v) + eps;
    uv[t] = pos / denom;
  }

  // 4) Noise: deterministic Gaussian standard noise (mu=0, sigma=1). The
  //    actual amplitude shaping happens below via noise_amp.
  std::vector<float> noise((size_t)n_h * T_audio);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto &x : noise) x = nd(rng);

  // 5) sine_waves(h, t) = sin(phase_audio) · sine_amp · uv(t)
  //                       + noise(h, t) · noise_amp(t)
  //    noise_amp = uv·(noise_std − sine_amp/3) + (sine_amp/3)
  std::vector<float> sine_waves((size_t)n_h * T_audio);
  for (int t = 0; t < T_audio; ++t) {
    float u    = uv[t];
    float namp = u * (nsf_sigma_ - nsf_alpha_ / 3.f) + nsf_alpha_ / 3.f;
    for (int h = 0; h < n_h; ++h) {
      float s = std::sin(phase_audio[(size_t)h * T_audio + t]) * nsf_alpha_;
      s *= u;
      s += noise[(size_t)h * T_audio + t] * namp;
      sine_waves[(size_t)h * T_audio + t] = s;
    }
  }

  // 6) Linear(9 → 1) + tanh — applied per time step. The Linear weight is
  //    m_source_lin_w_, shape [9, 1] in PyTorch (in=9, out=1). In our GGUF
  //    we stored it as-is (no transpose), so the weight tensor data is a
  //    9-vector and bias is a single scalar.
  // Read m_source.l_linear weights into host memory. The weight tensor is
  // stored as F16 by the converter (2-D + no force_f32); the bias is F32.
  std::vector<float> w(n_h, 0.f), b(1, 0.f);
  if (m_source_lin_w_->type == GGML_TYPE_F16) {
    std::vector<ggml_fp16_t> wf16((size_t)n_h);
    ggml_backend_tensor_get(m_source_lin_w_, wf16.data(), 0,
                            (size_t)n_h * sizeof(ggml_fp16_t));
    for (int i = 0; i < n_h; ++i)
      w[i] = ggml_fp16_to_fp32(wf16[i]);
  } else {
    ggml_backend_tensor_get(m_source_lin_w_, w.data(), 0,
                            (size_t)n_h * sizeof(float));
  }
  ggml_backend_tensor_get(m_source_lin_b_, b.data(), 0, sizeof(float));

  out_sine_merge.assign((size_t)T_audio, 0.f);
  for (int t = 0; t < T_audio; ++t) {
    float acc = b[0];
    for (int h = 0; h < n_h; ++h)
      acc += sine_waves[(size_t)h * T_audio + t] * w[h];
    out_sine_merge[t] = std::tanh(acc);
  }
}

// =====================================================================
// CPU STFT (n_fft=16, hop=4, periodic Hann, center=true with reflection pad).
//
//   T_stft = ⌊(T + 2·pad − n_fft)/hop⌋ + 1   where pad = n_fft/2
//   For T = T_audio = T_mel · 480, hop = 4:  T_stft = T_audio/4 + 1 = T_mel·120+1.
//
// Output layout (matches the (T_stft, 18) reshape ref1 produces):
//   real bin m at (t, m),       m ∈ [0..8]
//   imag bin m at (t, 9 + m),   m ∈ [0..8]
// Two contiguous halves — NOT interleaved — so that the conv_post output can
// be split via two views in the main graph.
// =====================================================================

void CosyVoice3HiFTModel::StftCpu(const float *signal, int n,
                                  std::vector<float> &out_real,
                                  std::vector<float> &out_imag,
                                  int &T_stft) const {
  const int N    = n_fft_;
  const int hop  = hop_len_;
  const int pad  = N / 2;
  const int Nbin = N / 2 + 1;
  // Reflection-pad the signal.
  std::vector<float> padded((size_t)(n + 2 * pad));
  for (int i = 0; i < pad; ++i) padded[i] = signal[pad - i];
  std::memcpy(padded.data() + pad, signal, (size_t)n * sizeof(float));
  for (int i = 0; i < pad; ++i)
    padded[(size_t)pad + n + i] = signal[n - 2 - i];
  T_stft = (int)((padded.size() - N) / hop) + 1;
  out_real.assign((size_t)T_stft * Nbin, 0.f);
  out_imag.assign((size_t)T_stft * Nbin, 0.f);
  // Naive O(N²) DFT per frame — fine for N=16.
  for (int t = 0; t < T_stft; ++t) {
    const float *frame = padded.data() + (size_t)t * hop;
    for (int k = 0; k < Nbin; ++k) {
      float re = 0.f, im = 0.f;
      const float w = 2.f * kPi * (float)k / N;
      for (int n2 = 0; n2 < N; ++n2) {
        float v = frame[n2] * hann_window_[n2];
        re += v * std::cos(w * n2);
        im -= v * std::sin(w * n2);
      }
      out_real[(size_t)t * Nbin + k] = re;
      out_imag[(size_t)t * Nbin + k] = im;
    }
  }
}

// CPU iSTFT (overlap-add with Hann window). Matches torch.istft(center=True).
void CosyVoice3HiFTModel::IstftCpu(const float *mag, const float *phase,
                                   int T_stft,
                                   std::vector<float> &out_pcm) const {
  const int N    = n_fft_;
  const int hop  = hop_len_;
  const int Nbin = N / 2 + 1;
  const int pad  = N / 2;
  const int T_padded = (T_stft - 1) * hop + N;
  std::vector<float> out_buf((size_t)T_padded, 0.f);
  std::vector<float> wsum((size_t)T_padded, 0.f);

  for (int t = 0; t < T_stft; ++t) {
    // Reconstruct one frame by inverse DFT (size N, Hermitian-symmetric).
    std::vector<float> frame((size_t)N, 0.f);
    for (int n2 = 0; n2 < N; ++n2) {
      float acc = 0.f;
      for (int k = 0; k < Nbin; ++k) {
        float m  = mag[(size_t)t * Nbin + k];
        float ph = phase[(size_t)t * Nbin + k];
        float re = m * std::cos(ph);
        float im = m * std::sin(ph);
        float w  = 2.f * kPi * (float)k / N;
        float v  = re * std::cos(w * n2) - im * std::sin(w * n2);
        // Mirror the upper bins (k=1..N/2-1) onto the negative-frequency side
        // — already accounted for by Hermitian symmetry of (mag, phase).
        if (k > 0 && k < Nbin - 1) v *= 2.f;
        acc += v;
      }
      frame[n2] = acc / N * hann_window_[n2];
    }
    // Overlap-add.
    for (int n2 = 0; n2 < N; ++n2) {
      out_buf[(size_t)t * hop + n2] += frame[n2];
      wsum[(size_t)t * hop + n2]    += hann_window_[n2] * hann_window_[n2];
    }
  }
  // Normalise by window sum.
  for (size_t i = 0; i < out_buf.size(); ++i)
    if (wsum[i] > 1e-8f) out_buf[i] /= wsum[i];
  // Trim n_fft/2 from head and tail (center=True padding).
  const int T_out = T_padded - 2 * pad;
  out_pcm.assign((size_t)T_out, 0.f);
  std::memcpy(out_pcm.data(), out_buf.data() + pad,
              (size_t)T_out * sizeof(float));
}

// =====================================================================
// ResBlock graph: x → (Snake → Conv k=3 d=di → Snake → Conv k=3 d=1) + x.
// Three dilations (1, 3, 5), output is sum after each sub-block residual.
// =====================================================================

ggml_tensor *CosyVoice3HiFTModel::BuildResBlock(ggml_context *ctx,
                                                const ResBlockW &rb,
                                                ggml_tensor *x) const {
  for (int i = 0; i < 3; ++i) {
    const auto &s = rb.subs[i];
    ggml_tensor *xt = snake(ctx, s.a1, x);
    xt = causal_conv1d(ctx, s.c1w, s.c1b, xt, s.dilation, CausalSide::Left);
    xt = snake(ctx, s.a2, xt);
    xt = causal_conv1d(ctx, s.c2w, s.c2b, xt, 1, CausalSide::Left);
    x = ggml_add(ctx, x, xt);
  }
  return x;
}

// =====================================================================
// Main upsample graph. Mirrors cosyvoice-graph.cpp:710.
//
// Inputs:
//   speech_feat  [80, T_mel, 1]    (Flow output, float32)
//   s_stft       [18, T_stft, 1]   (CPU STFT of sine_merge; first 9 chans
//                                    real, next 9 imag)
// Output:
//   conv_post    [18, T_post, 1]   where T_post = T_mel · ∏rates
// =====================================================================

ggml_tensor *CosyVoice3HiFTModel::BuildMainGraph(ggml_context *ctx,
                                                 ggml_tensor *speech_feat,
                                                 ggml_tensor *s_stft) const {
  // Both inputs arrive in (C, T, B) layout — permute to (T, C, B) for conv.
  ggml_tensor *x_in = ggml_permute(ctx, speech_feat, 1, 0, 2, 3);
  x_in = ggml_cont(ctx, x_in);
  ggml_tensor *s_in = ggml_permute(ctx, s_stft, 1, 0, 2, 3);
  s_in = ggml_cont(ctx, s_in);

  // conv_pre is right-causal (k=5).
  ggml_tensor *x = causal_conv1d(ctx, conv_pre_w_, conv_pre_b_,
                                 x_in, 1, CausalSide::Right);

  const int n_stages = (int)ups_.size();
  for (int i = 0; i < n_stages; ++i) {
    x = ggml_leaky_relu(ctx, x, lrelu_slope_, true);
    x = causal_upsample_conv1d(ctx, ups_[i].w, ups_[i].b, x, ups_[i].rate);

    // Final stage: reflective pad of 1 on the left.
    if (i == n_stages - 1) {
      x = ggml_pad_reflect_1d(ctx, x, 1, 0);
    }

    // Source-side contribution: source_downs[i](s_stft) → source_resblocks[i].
    ggml_tensor *si;
    if (source_downs_[i].stride > 1) {
      si = causal_downsample_conv1d(ctx, source_downs_[i].w,
                                    source_downs_[i].b, s_in,
                                    source_downs_[i].stride);
    } else {
      si = causal_conv1d(ctx, source_downs_[i].w, source_downs_[i].b,
                         s_in, 1, CausalSide::Left);
    }
    si = BuildResBlock(ctx, source_resblocks_[i], si);
    x = ggml_add(ctx, x, si);

    // num_kernels ResBlocks per stage; result is the mean across them.
    ggml_tensor *acc = BuildResBlock(ctx, resblocks_[i * num_kernels_], x);
    for (int j = 1; j < num_kernels_; ++j) {
      ggml_tensor *r = BuildResBlock(ctx, resblocks_[i * num_kernels_ + j], x);
      acc = ggml_add(ctx, acc, r);
    }
    x = ggml_scale(ctx, acc, 1.f / (float)num_kernels_);
  }

  x = ggml_leaky_relu(ctx, x, 0.01f, true);
  x = causal_conv1d(ctx, conv_post_w_, conv_post_b_, x, 1, CausalSide::Left);
  return x;
}

// =====================================================================
// RunHiFT — orchestrates the 2-graph + 3-CPU-pass pipeline.
// =====================================================================

bool CosyVoice3HiFTModel::RunHiFT(CosyVoice3State &state,
                                  ggml_backend_sched_t sched) {
  if (state.hift_done) return true;
  if (state.mel_output.empty()) {
    RS_LOG_ERR("HiFT: empty mel_output (call RunFlow first)");
    return false;
  }
  const int mel_dim = 80;
  const int T_mel = (int)(state.mel_output.size() / mel_dim);
  GGML_ASSERT(T_mel * mel_dim == (int)state.mel_output.size());

  state.audio_output.clear();
  if (!RunHiFTCore(state, sched,
                   state.mel_output.data(), T_mel,
                   /*cache_source=*/nullptr, /*cache_source_n=*/0,
                   state.audio_output, /*out_source=*/nullptr)) {
    return false;
  }
  state.hift_done = true;
  RS_LOG_INFO("HiFT: produced %zu samples @ %d Hz", state.audio_output.size(),
              sample_rate_);
  return true;
}

bool CosyVoice3HiFTModel::RunHiFTStreaming(CosyVoice3State &state,
                                           ggml_backend_sched_t sched,
                                           const float *mel_chunk,
                                           int mel_chunk_T,
                                           bool finalize,
                                           std::vector<float> &out_pcm) {
  if (!mel_chunk || mel_chunk_T <= 0) {
    RS_LOG_ERR("HiFT-stream: empty mel chunk");
    return false;
  }
  const int mel_dim = 80;
  const int src_cache = source_cache_len();
  auto &cache = state.hift_stream;

  // Build the mel buffer fed to the HiFT graph: cached 8 frames (if primed)
  // followed by the new chunk.
  std::vector<float> mel_full;
  if (cache.primed) {
    if ((int)cache.mel.size() != kMelCacheLen * mel_dim) {
      RS_LOG_ERR("HiFT-stream: cache.mel size %zu != %d",
                 cache.mel.size(), kMelCacheLen * mel_dim);
      return false;
    }
    mel_full.resize((kMelCacheLen + mel_chunk_T) * (size_t)mel_dim);
    std::memcpy(mel_full.data(), cache.mel.data(),
                cache.mel.size() * sizeof(float));
    std::memcpy(mel_full.data() + cache.mel.size(),
                mel_chunk,
                (size_t)mel_chunk_T * mel_dim * sizeof(float));
  } else {
    mel_full.assign(mel_chunk,
                    mel_chunk + (size_t)mel_chunk_T * mel_dim);
  }
  const int T_full = (int)(mel_full.size() / mel_dim);

  std::vector<float> pcm;
  std::vector<float> source;
  if (!RunHiFTCore(state, sched, mel_full.data(), T_full,
                   cache.primed ? cache.source.data() : nullptr,
                   cache.primed ? src_cache : 0,
                   pcm, &source)) {
    return false;
  }

  // Hamming crossfade between the regenerated head of this chunk and the
  // deferred speech tail from the previous chunk (upstream fade_in_out).
  if (cache.primed) {
    if ((int)cache.speech.size() != src_cache) {
      RS_LOG_ERR("HiFT-stream: cache.speech size %zu != %d",
                 cache.speech.size(), src_cache);
      return false;
    }
    if ((int)pcm.size() < src_cache) {
      RS_LOG_ERR("HiFT-stream: pcm size %zu < src_cache=%d",
                 pcm.size(), src_cache);
      return false;
    }
    const auto &win = HammingWindow(2 * src_cache);
    for (int i = 0; i < src_cache; ++i) {
      pcm[i] = pcm[i] * win[i] + cache.speech[i] * win[src_cache + i];
    }
  }

  if (!finalize) {
    if ((int)pcm.size() < src_cache) {
      RS_LOG_ERR("HiFT-stream: pcm shorter than src_cache");
      return false;
    }
    // Save tails for the next chunk.
    cache.mel.assign(
        mel_full.end() - (size_t)kMelCacheLen * mel_dim, mel_full.end());
    cache.source.assign(source.end() - src_cache, source.end());
    cache.speech.assign(pcm.end() - src_cache, pcm.end());
    cache.primed = true;
    out_pcm.assign(pcm.begin(), pcm.end() - src_cache);
  } else {
    out_pcm = std::move(pcm);
    cache = {};
  }
  return true;
}

const std::vector<float> &CosyVoice3HiFTModel::HammingWindow(int N) const {
  if ((int)hamming_window_cached_.size() != N) {
    hamming_window_cached_.assign((size_t)N, 0.f);
    // np.hamming: w[n] = 0.54 - 0.46 * cos(2π n / (N-1))
    const double denom = (N > 1) ? (double)(N - 1) : 1.0;
    for (int n = 0; n < N; ++n) {
      hamming_window_cached_[n] =
          (float)(0.54 - 0.46 * std::cos(2.0 * M_PI * n / denom));
    }
  }
  return hamming_window_cached_;
}

// =====================================================================
// RunHiFTCore — shared mel → 24 kHz PCM pipeline used by both offline
// `RunHiFT` and the streaming `RunHiFTStreaming`.
// =====================================================================
bool CosyVoice3HiFTModel::RunHiFTCore(CosyVoice3State &state,
                                     ggml_backend_sched_t sched,
                                     const float *mel_ptr, int T_mel,
                                     const float *cache_source,
                                     int cache_source_n,
                                     std::vector<float> &out_pcm,
                                     std::vector<float> *out_source) {
  if (!mel_ptr || T_mel <= 0) {
    RS_LOG_ERR("HiFT-core: empty mel input");
    return false;
  }
  const int mel_dim = 80;

  constexpr int HIFT_MAX_NODES = 8192;
  constexpr size_t HIFT_CTX_BYTES =
      HIFT_MAX_NODES * (sizeof(ggml_tensor) + 256);

  // -----------------------------------------------------------------------
  // (1) F0 graph.
  // -----------------------------------------------------------------------
  std::vector<float> h_f0((size_t)T_mel, 0.f);
  {
    ggml_init_params ip = { HIFT_CTX_BYTES, nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, HIFT_MAX_NODES, false);

    ggml_tensor *sf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mel_dim, T_mel, 1);
    ggml_set_input(sf);
    ggml_set_name(sf, "speech_feat");

    // state.mel_output is row-major [T_mel, mel_dim]; the ggml tensor expects
    // ne[0]=mel_dim, ne[1]=T_mel — i.e. mel_dim is fastest. That matches the
    // row-major layout exactly, so we can memcpy directly.
    ggml_tensor *f0 = BuildF0Graph(ctx, sf);
    ggml_set_output(f0);
    ggml_build_forward_expand(gf, f0);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
      RS_LOG_ERR("HiFT: failed to alloc F0 graph");
      ggml_free(ctx);
      return false;
    }
    ggml_backend_tensor_set(sf, mel_ptr, 0,
                            (size_t)T_mel * mel_dim * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("HiFT: F0 compute failed");
      ggml_free(ctx);
      return false;
    }
    ggml_backend_tensor_get(f0, h_f0.data(), 0,
                            h_f0.size() * sizeof(float));
    ggml_free(ctx);
    ggml_backend_sched_reset(sched);
  }
  if (const char *p = std::getenv("RS_CV3_DUMP_HIFT_F0")) {
    FILE *f = std::fopen(p, "wb");
    if (f) {
      std::fwrite(h_f0.data(), sizeof(float), h_f0.size(), f);
      std::fclose(f);
      RS_LOG_INFO("HiFT: dumped f0 [%d] -> %s", T_mel, p);
    }
  }
  {
    float mn = +1e9f, mx = -1e9f, sum = 0.f;
    for (float v : h_f0) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
    RS_LOG_INFO("HiFT: f0 range [%.3f,%.3f] mean=%.3f", mn, mx,
                sum / std::max((size_t)1, h_f0.size()));
  }

  // -----------------------------------------------------------------------
  // (2) CPU source path → sine_merge [T_audio].
  // -----------------------------------------------------------------------
  std::vector<float> sine_merge;
  RunSourceCpu(h_f0.data(), T_mel, sine_merge, state.rng);
  const int T_audio = (int)sine_merge.size();

  // Streaming continuity: splice the cached NSF source over the front of
  // the freshly generated sine_merge so the noise / phase don't jump at
  // the chunk boundary (the hamming crossfade in the caller smooths the
  // residual splice).
  if (cache_source && cache_source_n > 0) {
    const int n = std::min(cache_source_n, T_audio);
    std::memcpy(sine_merge.data(), cache_source, (size_t)n * sizeof(float));
  }
  if (out_source) {
    *out_source = sine_merge;  // caller-side cache of the post-splice source.
  }

  // (3) CPU STFT of sine_merge.
  std::vector<float> s_real, s_imag;
  int T_stft = 0;
  StftCpu(sine_merge.data(), T_audio, s_real, s_imag, T_stft);

  // Build s_stft as [18, T_stft, 1] with real then imag halves along the
  // 18-channel dim (so ggml ne[0]=18, ne[1]=T_stft).
  const int Nbin = n_fft_ / 2 + 1;
  std::vector<float> s_stft_host((size_t)18 * T_stft, 0.f);
  for (int t = 0; t < T_stft; ++t) {
    for (int k = 0; k < Nbin; ++k) {
      s_stft_host[(size_t)t * 18 + k]        = s_real[(size_t)t * Nbin + k];
      s_stft_host[(size_t)t * 18 + Nbin + k] = s_imag[(size_t)t * Nbin + k];
    }
  }
  if (const char *p = std::getenv("RS_CV3_DUMP_HIFT_SINE")) {
    FILE *f = std::fopen(p, "wb");
    if (f) {
      std::fwrite(sine_merge.data(), sizeof(float), sine_merge.size(), f);
      std::fclose(f);
      RS_LOG_INFO("HiFT: dumped sine_merge [%d] -> %s",
                  (int)sine_merge.size(), p);
    }
  }
  if (const char *p = std::getenv("RS_CV3_DUMP_HIFT_SSTFT")) {
    FILE *f = std::fopen(p, "wb");
    if (f) {
      std::fwrite(s_stft_host.data(), sizeof(float), s_stft_host.size(), f);
      std::fclose(f);
      RS_LOG_INFO("HiFT: dumped s_stft [%d, 18] -> %s", T_stft, p);
    }
  }
  if (const char *p = std::getenv("RS_CV3_LOAD_HIFT_SSTFT")) {
    FILE *f = std::fopen(p, "rb");
    if (f) {
      std::fread(s_stft_host.data(), sizeof(float), s_stft_host.size(), f);
      std::fclose(f);
      RS_LOG_INFO("HiFT: overrode s_stft [%d, 18] from %s", T_stft, p);
    }
  }

  // -----------------------------------------------------------------------
  // (4) Main upsample graph.
  // -----------------------------------------------------------------------
  std::vector<float> conv_post_out;
  int T_post = 0;
  {
    ggml_init_params ip = { HIFT_CTX_BYTES, nullptr, true };
    ggml_context *ctx = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, HIFT_MAX_NODES, false);

    ggml_tensor *sf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mel_dim, T_mel, 1);
    ggml_set_input(sf); ggml_set_name(sf, "speech_feat");
    ggml_tensor *st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 18, T_stft, 1);
    ggml_set_input(st); ggml_set_name(st, "s_stft");

    ggml_tensor *out = BuildMainGraph(ctx, sf, st);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
      RS_LOG_ERR("HiFT: failed to alloc main graph");
      ggml_free(ctx);
      return false;
    }
    ggml_backend_tensor_set(sf, mel_ptr, 0,
                            (size_t)T_mel * mel_dim * sizeof(float));
    ggml_backend_tensor_set(st, s_stft_host.data(), 0,
                            s_stft_host.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
      RS_LOG_ERR("HiFT: main compute failed");
      ggml_free(ctx);
      return false;
    }
    // After the layout fixes, BuildMainGraph returns x in (T, C, B) layout,
    // so T sits in ne[0] and C=18 in ne[1].
    T_post = (int)out->ne[0];
    conv_post_out.assign((size_t)18 * T_post, 0.f);
    ggml_backend_tensor_get(out, conv_post_out.data(), 0,
                            conv_post_out.size() * sizeof(float));
    ggml_free(ctx);
    ggml_backend_sched_reset(sched);
  }
  if (const char *p = std::getenv("RS_CV3_DUMP_HIFT_CONVPOST")) {
    FILE *f = std::fopen(p, "wb");
    if (f) {
      std::fwrite(conv_post_out.data(), sizeof(float),
                  conv_post_out.size(), f);
      std::fclose(f);
      RS_LOG_INFO("HiFT: dumped conv_post_out [18, T_post=%d] -> %s", T_post, p);
    }
  }

  // -----------------------------------------------------------------------
  // (5) Mag/phase split → CPU iSTFT.
  // -----------------------------------------------------------------------
  // conv_post_out is the contiguous host copy of the BuildMainGraph output
  // tensor, which has shape (T_post=ne[0], 18=ne[1], 1). Since ne[0] is the
  // fastest dim, the memory layout is [channel-slow, T-fast]:
  //     conv_post_out[c * T_post + t] = value at channel c, time t.
  std::vector<float> mag((size_t)Nbin * T_post);
  std::vector<float> phase((size_t)Nbin * T_post);
  float dbg_lm_min = +1e9f, dbg_lm_max = -1e9f;
  float dbg_p_min  = +1e9f, dbg_p_max  = -1e9f;
  for (int t = 0; t < T_post; ++t) {
    for (int k = 0; k < Nbin; ++k) {
      float lm = conv_post_out[(size_t)k * T_post + t];
      float m = std::exp(lm);
      m = std::max(0.f, std::min(100.f, m));
      // PT does `phase = torch.sin(raw)` BEFORE the iSTFT call. Even though
      // its comment claims "sin is redundancy", omitting it leaks unbounded
      // network outputs into the iSTFT phase angle (cos/sin of large numbers
      // → uniform-looking spectra → output noise).
      float p_raw = conv_post_out[(size_t)(Nbin + k) * T_post + t];
      float p = std::sin(p_raw);
      mag[(size_t)t * Nbin + k]   = m;
      phase[(size_t)t * Nbin + k] = p;
      dbg_lm_min = std::min(dbg_lm_min, lm);
      dbg_lm_max = std::max(dbg_lm_max, lm);
      dbg_p_min  = std::min(dbg_p_min, p_raw);
      dbg_p_max  = std::max(dbg_p_max, p_raw);
    }
  }
  RS_LOG_INFO("HiFT: log_mag range [%.3f,%.3f], phase_raw range [%.3f,%.3f]",
              dbg_lm_min, dbg_lm_max, dbg_p_min, dbg_p_max);
  std::vector<float> pcm;
  IstftCpu(mag.data(), phase.data(), T_post, pcm);

  // Clamp final waveform.
  for (auto &v : pcm) {
    v = std::max(-audio_limit_, std::min(audio_limit_, v));
  }

  out_pcm = std::move(pcm);
  return true;
}
