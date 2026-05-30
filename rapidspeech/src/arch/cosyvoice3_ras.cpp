#include "cosyvoice3_ras.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace {

// Softmax with optional temperature, written into `out` (length n_vocab).
void softmax_temp(const float *logits, int32_t n_vocab, float temp,
                  std::vector<float> &out) {
  out.resize((size_t)n_vocab);
  const float t = std::max(temp, 1e-6f);
  float max_l = -INFINITY;
  for (int32_t i = 0; i < n_vocab; ++i) {
    const float v = logits[i] / t;
    if (v > max_l) max_l = v;
  }
  double sum = 0.0;
  for (int32_t i = 0; i < n_vocab; ++i) {
    const float v = std::exp((logits[i] / t) - max_l);
    out[(size_t)i] = v;
    sum += v;
  }
  const float inv = sum > 0 ? (float)(1.0 / sum) : 0.0f;
  for (int32_t i = 0; i < n_vocab; ++i) out[(size_t)i] *= inv;
}

int32_t multinomial(const std::vector<float> &probs,
                    const std::vector<int32_t> &idx,
                    std::mt19937 &rng) {
  // probs/idx are aligned; probs need not sum to 1 — re-normalise.
  double sum = 0.0;
  for (float p : probs) sum += p;
  if (sum <= 0.0) return idx.empty() ? 0 : idx.front();
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const float r = u(rng) * (float)sum;
  float acc = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) {
    acc += probs[i];
    if (r <= acc) return idx[i];
  }
  return idx.back();
}

} // namespace

int32_t cosyvoice3_sample_ras(const float *logits, int32_t n_vocab,
                              const std::vector<int32_t> &history,
                              const ras_params &p, std::mt19937 &rng) {
  if (n_vocab <= 0) return 0;

  // Greedy path for deterministic debugging.
  if (p.temperature <= 0.0f) {
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
      if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return best;
  }

  std::vector<float> probs;
  softmax_temp(logits, n_vocab, p.temperature, probs);

  // Sort indices by probability descending.
  std::vector<int32_t> idx((size_t)n_vocab);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(
      idx.begin(),
      idx.begin() + std::min<int32_t>(p.top_k > 0 ? p.top_k : n_vocab, n_vocab),
      idx.end(), [&](int32_t a, int32_t b) { return probs[a] > probs[b]; });

  const int32_t k = std::max(1, std::min(p.top_k > 0 ? p.top_k : n_vocab,
                                          n_vocab));
  std::vector<int32_t> top_idx(idx.begin(), idx.begin() + k);
  std::vector<float>   top_p_v(k);
  for (int32_t i = 0; i < k; ++i) top_p_v[i] = probs[top_idx[i]];

  // Nucleus / top-p truncation (cumulative on already-descending probs).
  if (p.top_p > 0.0f && p.top_p < 1.0f) {
    double cum = 0.0;
    int32_t keep = k;
    for (int32_t i = 0; i < k; ++i) {
      cum += top_p_v[i];
      if (cum >= p.top_p) { keep = i + 1; break; }
    }
    top_idx.resize(keep);
    top_p_v.resize(keep);
  }

  const int32_t cand = multinomial(top_p_v, top_idx, rng);

  // Repetition fallback: if the candidate appears too often in the recent
  // `win_size` history, resample from the FULL post-temperature distribution.
  if (p.win_size > 0 && p.tau_r > 0.0f && !history.empty()) {
    const size_t W = (size_t)p.win_size;
    const size_t start = history.size() > W ? history.size() - W : 0;
    int32_t hits = 0;
    for (size_t i = start; i < history.size(); ++i) {
      if (history[i] == cand) ++hits;
    }
    const float thresh = (float)p.win_size * p.tau_r;
    if ((float)hits >= thresh) {
      std::vector<int32_t> all_idx((size_t)n_vocab);
      std::iota(all_idx.begin(), all_idx.end(), 0);
      return multinomial(probs, all_idx, rng);
    }
  }

  return cand;
}
