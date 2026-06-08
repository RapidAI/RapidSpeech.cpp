#include "kws_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace rs {

namespace {

// log(exp(a) + exp(b)) numerically stable; treats -inf as identity.
inline float logsumexp(float a, float b) {
  if (a == -std::numeric_limits<float>::infinity()) return b;
  if (b == -std::numeric_limits<float>::infinity()) return a;
  float hi = std::max(a, b), lo = std::min(a, b);
  return hi + std::log1p(std::exp(lo - hi));
}

} // namespace

std::vector<KWSMatch> KWSDecoder::Decode(const float *log_probs, int T,
                                         int V) const {
  if (cfg_.beam_size > 1) return DecodeBeam(log_probs, T, V);
  return DecodeGreedy(log_probs, T, V);
}

std::vector<KWSMatch> KWSDecoder::DecodeGreedy(const float *log_probs, int T,
                                               int V) const {
  std::vector<KWSMatch> hits;
  if (!graph_ || !log_probs || T <= 0 || V <= 0) return hits;

  // 1. Greedy argmax with CTC collapse. Skip prefix frames + ignore_ids +
  //    punct_ids.
  struct Emitted {
    int32_t id;
    float prob;
    int32_t frame;
  };
  std::vector<Emitted> emitted;
  emitted.reserve(T);

  int32_t last_id = -1;
  const int t_start = std::min(cfg_.skip_prefix_frames, T);
  for (int t = t_start; t < T; ++t) {
    const float *row = log_probs + static_cast<size_t>(t) * V;
    int best = 0;
    float best_lp = row[0];
    for (int v = 1; v < V; ++v) {
      if (row[v] > best_lp) {
        best_lp = row[v];
        best = v;
      }
    }

    if (best == cfg_.blank_id) {
      last_id = -1;
      continue;
    }
    if (ignore_ids_.count(best) || punct_ids_.count(best)) {
      last_id = -1;
      continue;
    }
    if (best == last_id) {
      continue;
    }

    emitted.push_back({best, std::exp(best_lp), t});
    last_id = best;
  }

  // 2. Walk the AC trie; per-emit trailing-blank counts let us gate by
  //    num_trailing_blanks if the user asked for it.
  std::vector<int> tail_blanks(emitted.size(), 0);
  {
    int cur_blank_run = 0;
    int next_emit = static_cast<int>(emitted.size()) - 1;
    for (int t = T - 1; t >= t_start && next_emit >= 0; --t) {
      const float *row = log_probs + static_cast<size_t>(t) * V;
      int best = 0;
      float best_lp = row[0];
      for (int v = 1; v < V; ++v) {
        if (row[v] > best_lp) { best_lp = row[v]; best = v; }
      }
      if (t == emitted[next_emit].frame) {
        tail_blanks[next_emit] = cur_blank_run;
        --next_emit;
        cur_blank_run = 0;
      } else if (best == cfg_.blank_id) {
        ++cur_blank_run;
      } else {
        cur_blank_run = 0;
      }
    }
  }

  const ContextState *state = graph_->Root();
  for (size_t i = 0; i < emitted.size(); ++i) {
    auto [boost, next_state, matched] =
        graph_->ForwardOneStep(state, emitted[i].id, /*strict_mode=*/true);
    (void)boost;
    state = next_state;

    if (!matched) continue;

    int level = matched->level;
    if (level <= 0 || static_cast<int>(i) + 1 < level) continue;

    int start = static_cast<int>(i) + 1 - level;
    float sum_prob = 0.0f;
    for (int k = start; k <= static_cast<int>(i); ++k) {
      sum_prob += emitted[k].prob;
    }
    float avg_prob = sum_prob / static_cast<float>(level);

    if (cfg_.num_trailing_blanks > 0 &&
        tail_blanks[i] < cfg_.num_trailing_blanks) {
      continue;
    }
    if (avg_prob < matched->ac_threshold) continue;

    KWSMatch m;
    m.phrase = matched->phrase.empty() ? std::string() : matched->phrase;
    m.avg_prob = avg_prob;
    m.first_frame = emitted[start].frame;
    m.last_frame = emitted[i].frame;
    hits.push_back(std::move(m));

    state = graph_->Root();
  }

  return hits;
}

// ---------------------------------------------------------------------------
// V2: CTC beam search × ContextGraph
//
// Each hypothesis carries a trie pointer; boosting goes into beam log-prob,
// punct_ids/ignore_ids are masked out of the candidate set. Matches are
// reported as soon as IsMatched + ac_threshold passes; the matched hyp's
// trie state is rewound to root so the same span isn't re-reported.
// ---------------------------------------------------------------------------
std::vector<KWSMatch> KWSDecoder::DecodeBeam(const float *log_probs, int T,
                                             int V) const {
  std::vector<KWSMatch> hits;
  if (!graph_ || !log_probs || T <= 0 || V <= 0) return hits;

  struct Hyp {
    std::vector<int32_t> ys;     // CTC-collapsed token id stream
    std::vector<int32_t> ts;     // frame each token in `ys` was first emitted
    std::vector<float> probs;    // prob of each token in `ys` at its emit frame
    int32_t last_id = -1;        // most recent non-blank token id; -1 after a blank
    const ContextState *state = nullptr;
    float log_prob = 0.0f;
  };

  std::vector<Hyp> beams;
  beams.reserve(cfg_.beam_size * 4);
  {
    Hyp init;
    init.state = graph_->Root();
    beams.push_back(std::move(init));
  }

  const int t_start = std::min(cfg_.skip_prefix_frames, T);
  const int top_k = std::max(1, cfg_.top_k_per_frame);

  // Pre-allocate scratch for top-K extraction per frame.
  std::vector<std::pair<float, int>> ranked;
  ranked.reserve(64);

  // Dedup matches within this window (same phrase + similar end frame). The
  // outer rs_kws layer applies cross-window debounce on top.
  auto already_hit = [&](const std::string &phrase, int last_frame) {
    for (const auto &h : hits) {
      if (h.phrase == phrase && std::abs(h.last_frame - last_frame) < 4) {
        return true;
      }
    }
    return false;
  };

  for (int t = t_start; t < T; ++t) {
    const float *row = log_probs + static_cast<size_t>(t) * V;
    const float p_blank = row[cfg_.blank_id];

    // Top-K non-blank, non-masked candidates this frame.
    ranked.clear();
    for (int v = 0; v < V; ++v) {
      if (v == cfg_.blank_id) continue;
      if (ignore_ids_.count(v) || punct_ids_.count(v)) continue;
      ranked.emplace_back(row[v], v);
    }
    int kk = std::min(top_k, static_cast<int>(ranked.size()));
    if (kk == 0) {
      // Only blank survives; just accumulate p_blank.
      for (auto &b : beams) {
        b.log_prob += p_blank;
        b.last_id = -1;
      }
      continue;
    }
    std::partial_sort(ranked.begin(), ranked.begin() + kk, ranked.end(),
                      [](const auto &a, const auto &b) { return a.first > b.first; });

    // Expand every current hyp by (blank | top-K tokens).
    std::vector<Hyp> next;
    next.reserve(beams.size() * (kk + 1));

    for (const auto &h : beams) {
      // 1) extend with blank: prefix unchanged, last_id reset
      {
        Hyp nh = h;
        nh.log_prob += p_blank;
        nh.last_id = -1;
        next.push_back(std::move(nh));
      }

      // 2) extend with each top-K token
      for (int k = 0; k < kk; ++k) {
        const float p_v = ranked[k].first;
        const int v = ranked[k].second;

        if (v == h.last_id) {
          // CTC repeat collapse: prefix and trie unchanged
          Hyp nh = h;
          nh.log_prob += p_v;
          // last_id stays = v
          next.push_back(std::move(nh));
        } else {
          // New non-blank symbol: extend prefix, walk trie
          Hyp nh = h;
          auto [boost, new_state, matched] =
              graph_->ForwardOneStep(nh.state, v, /*strict_mode=*/true);
          nh.state = new_state;
          nh.ys.push_back(v);
          nh.ts.push_back(t);
          nh.probs.push_back(std::exp(p_v));
          nh.last_id = v;
          nh.log_prob += p_v + boost;
          (void)matched; // checked below after pruning
          next.push_back(std::move(nh));
        }
      }
    }

    // Prune to top beam_size by log_prob.
    if (static_cast<int>(next.size()) > cfg_.beam_size) {
      std::partial_sort(next.begin(), next.begin() + cfg_.beam_size,
                        next.end(),
                        [](const Hyp &a, const Hyp &b) {
                          return a.log_prob > b.log_prob;
                        });
      next.resize(cfg_.beam_size);
    }

    // Check matches in surviving hyps and emit / rewind.
    for (auto &h : next) {
      auto [is_match, matched] = graph_->IsMatched(h.state);
      if (!is_match || !matched) continue;
      int level = matched->level;
      if (level <= 0 || static_cast<int>(h.ys.size()) < level) continue;

      float sum_prob = 0.0f;
      int start = static_cast<int>(h.ys.size()) - level;
      for (int k = start; k < static_cast<int>(h.ys.size()); ++k) {
        sum_prob += h.probs[k];
      }
      float avg_prob = sum_prob / static_cast<float>(level);
      if (avg_prob < matched->ac_threshold) continue;

      std::string phrase =
          matched->phrase.empty() ? std::string() : matched->phrase;
      int last_frame = h.ts.back();

      if (!already_hit(phrase, last_frame)) {
        KWSMatch m;
        m.phrase = std::move(phrase);
        m.avg_prob = avg_prob;
        m.first_frame = h.ts[start];
        m.last_frame = last_frame;
        hits.push_back(std::move(m));
      }

      // Rewind this hyp so we don't keep matching the same span every frame.
      h.state = graph_->Root();
    }

    beams = std::move(next);
  }

  // logsumexp left unused intentionally — top-K + max-prune is sufficient for
  // KWS and avoids hash-of-prefix overhead. Keep helper around in case we
  // re-enable strict prefix-merge later.
  (void)logsumexp;
  return hits;
}

} // namespace rs
