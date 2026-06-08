#include "rs_kws.h"

#include "ggml-backend.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace rs {

RSKws::RSKws(std::shared_ptr<SenseVoiceModel> model,
             ggml_backend_sched_t sched, ContextGraphPtr graph,
             std::unordered_set<int32_t> ignore_ids,
             std::unordered_set<int32_t> punct_ids, RSKwsConfig cfg)
    : model_(std::move(model)),
      sched_(sched),
      graph_(std::move(graph)),
      cfg_(cfg) {
  state_ = model_->CreateState();
  model_->SetKWSMode(*state_, true);

  STFTConfig sf;
  const auto &meta = model_->GetMeta();
  sf.sample_rate = meta.audio_sample_rate;
  sf.n_mels = meta.n_mels;
  sf.window_type = meta.window_type;
  sf.use_lfr = true;
  sf.lfr_m = 7;
  sf.lfr_n = 6;
  audio_proc_ = std::make_unique<AudioProcessor>(sf);

  decoder_ = std::make_unique<KWSDecoder>(graph_, cfg_.decoder,
                                          std::move(ignore_ids),
                                          std::move(punct_ids));

  window_samples_ = cfg_.window_ms * cfg_.sample_rate / 1000;
  hop_samples_ = cfg_.hop_ms * cfg_.sample_rate / 1000;
}

RSKws::~RSKws() = default;

void RSKws::PushAudio(const float *pcm, size_t n) {
  if (!pcm || n == 0) return;
  ring_.insert(ring_.end(), pcm, pcm + n);
}

int RSKws::Poll(const KWSCallback &cb) {
  int n_windows = 0;
  while (static_cast<int>(ring_.size()) >= window_samples_) {
    // Snapshot the leading `window_samples_` of the ring as a contiguous buffer.
    std::vector<float> win;
    win.reserve(window_samples_);
    auto it = ring_.begin();
    for (int i = 0; i < window_samples_; ++i) win.push_back(*it++);

    // 1. Mel/fbank
    std::vector<float> feats;
    audio_proc_->Compute(win, feats);
    if (feats.empty()) {
      // Drop one hop and continue.
      for (int i = 0; i < hop_samples_ && !ring_.empty(); ++i) {
        ring_.pop_front();
        ++consumed_samples_;
      }
      continue;
    }

    // 2. Encode + CTC (KWS mode → state holds [T, V] log_probs)
    ggml_backend_sched_reset(sched_);
    if (!model_->Encode(feats, *state_, sched_)) {
      RS_LOG_ERR("rs_kws: encode failed");
      break;
    }
    ggml_backend_sched_reset(sched_);
    if (!model_->Decode(*state_, sched_)) {
      RS_LOG_ERR("rs_kws: decode failed");
      break;
    }
    int T = 0, V = 0;
    const float *lp = model_->GetCTCLogits(*state_, &T, &V);
    if (!lp || T <= 0 || V <= 0) {
      RS_LOG_ERR("rs_kws: empty logits");
      break;
    }

    if (const char *dbg = std::getenv("RS_KWS_DEBUG"); dbg && dbg[0] == '1') {
      const auto &id2tok = model_->GetIdToToken();
      std::string dump;
      int last = -1;
      int t_start = std::min(cfg_.decoder.skip_prefix_frames, T);
      for (int t = t_start; t < T; ++t) {
        const float *row = lp + static_cast<size_t>(t) * V;
        int best = 0;
        float best_lp = row[0];
        for (int v = 1; v < V; ++v) {
          if (row[v] > best_lp) { best_lp = row[v]; best = v; }
        }
        if (best == 0 || best == last) { last = best; continue; }
        last = best;
        auto it = id2tok.find(best);
        dump += "[";
        dump += std::to_string(best);
        dump += ":";
        dump += (it == id2tok.end() ? "?" : it->second);
        dump += "] ";
      }
      RS_LOG_INFO("rs_kws DBG win_t=%.2fs T=%d  greedy=%s",
                  consumed_samples_ / double(cfg_.sample_rate), T, dump.c_str());
    }

    // 3. Match keywords in this window.
    auto hits = decoder_->Decode(lp, T, V);

    double win_start_s =
        static_cast<double>(consumed_samples_) / cfg_.sample_rate;
    auto now = std::chrono::steady_clock::now();
    for (auto &h : hits) {
      const std::string &key = h.phrase;
      auto debounce_it = last_hit_.find(key);
      if (debounce_it != last_hit_.end()) {
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now - debounce_it->second)
                      .count();
        if (dt < cfg_.debounce_ms) continue;
      }
      last_hit_[key] = now;

      KWSHit out;
      out.phrase = std::move(h.phrase);
      out.avg_prob = h.avg_prob;
      out.time_s = win_start_s; // window-start time; finer alignment is V2
      if (cb) cb(out);
    }

    // 4. Advance the ring by one hop.
    for (int i = 0; i < hop_samples_ && !ring_.empty(); ++i) {
      ring_.pop_front();
      ++consumed_samples_;
    }
    ++n_windows;
  }
  return n_windows;
}

void RSKws::Reset() {
  ring_.clear();
  consumed_samples_ = 0;
  last_hit_.clear();
  state_ = model_->CreateState();
  model_->SetKWSMode(*state_, true);
}

} // namespace rs
