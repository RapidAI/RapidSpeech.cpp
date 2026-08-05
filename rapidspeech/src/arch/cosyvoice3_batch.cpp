// cosyvoice3_batch.cpp — TTS continuous-batching scheduler.
//
// All ggml compute runs on one worker thread (the shared sched is not
// thread-safe). Concurrency comes from token-level interleaving: each loop
// iteration runs ONE unit of work — a prefill, one AR step of one slot
// (round-robin), or one finished slot's Flow+HiFT — so N in-flight requests
// all make progress and a new request never waits behind a whole utterance.

#include "cosyvoice3_batch.h"
#include "utils/rs_log.h"

#include <algorithm>

#include <algorithm>

// ── ctor/dtor ────────────────────────────────────────────────────────────────

TtsBatchScheduler::TtsBatchScheduler(CosyVoice3LMModel *model,
                                     ggml_backend_sched_t sched,
                                     int n_slots)
    : model_(model), sched_(sched), n_slots_(n_slots) {
    pool_.resize((size_t)n_slots_);
    for (int i = 0; i < n_slots_; ++i) {
        pool_[(size_t)i] = std::make_unique<TtsSlot>();
        pool_[(size_t)i]->id = i;
    }
}

TtsBatchScheduler::TtsBatchScheduler(CosyVoice3LMModel *model,
                                     ggml_backend_sched_t sched_lm,
                                     std::vector<ggml_backend_sched_t> scheds_voc,
                                     int n_slots)
    : model_(model), sched_(sched_lm), scheds_voc_(std::move(scheds_voc)),
      dual_worker_(!scheds_voc_.empty()), n_slots_(n_slots) {
    pool_.resize((size_t)n_slots_);
    for (int i = 0; i < n_slots_; ++i) {
        pool_[(size_t)i] = std::make_unique<TtsSlot>();
        pool_[(size_t)i]->id = i;
    }
}

TtsBatchScheduler::~TtsBatchScheduler() { Stop(); }

// ── public API ───────────────────────────────────────────────────────────────

void TtsBatchScheduler::Start() {
    // Allocate the shared batched KV cache: [kv_dim, n_kv_max * n_slots] per
    // layer. n_kv_max covers the longest prompt + max speech tokens any one
    // request can produce (1500 speech + ~500 prompt headroom).
    const int n_kv_max = 2048;
    if (!model_->InitBatchKV(n_slots_, n_kv_max, sched_)) {
        RS_LOG_ERR("TtsBatchScheduler: InitBatchKV failed - batching disabled");
        return;
    }
    stop_ = false;
    worker_ = std::thread(dual_worker_ ? &TtsBatchScheduler::RunLoopLM
                                       : &TtsBatchScheduler::RunLoop, this);
    if (dual_worker_) {
        for (size_t vi = 0; vi < scheds_voc_.size(); ++vi)
            workers_voc_.emplace_back(&TtsBatchScheduler::RunLoopVoc, this, (int)vi);
        RS_LOG_INFO("TtsBatchScheduler: dual-worker (LM || Flow/HiFT x %zu)",
                    scheds_voc_.size());
    }
}

void TtsBatchScheduler::Stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    voc_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    for (auto &t : workers_voc_) if (t.joinable()) t.join();
    // Reject anything still queued so callers don't block forever.
    std::lock_guard<std::mutex> lk(mu_);
    while (!pending_.empty()) {
        pending_.front().promise.set_value({{}, "server shutting down"});
        pending_.pop_front();
    }
    // Fail any in-flight slots.
    for (auto &s : pool_) {
        if (s->phase != TtsSlot::Phase::IDLE && !s->error.empty()) {
            // already errored
        } else if (s->phase != TtsSlot::Phase::IDLE) {
            s->error = "server shutting down";
            FinishSlot(*s);
        }
    }
    flow_queue_.clear();
}

std::future<std::pair<std::vector<float>, std::string>>
TtsBatchScheduler::Submit(const std::string &text, const std::string &voice,
                          const std::string &language, int32_t seed,
                          uint64_t session_id) {
    PendingReq req;
    req.text = text;
    req.voice = voice;
    req.language = language;
    req.seed = seed;
    req.session_id = session_id;
    auto fut = req.promise.get_future();
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.push_back(std::move(req));
    }
    cv_.notify_one();
    return fut;
}

// ── True streaming first sentence (DecodeStream on vocoder sched) ──────────
// Runs LM (token-by-token) + incremental Flow/HiFT chunks directly on
// sched_voc_ so the first audio chunk arrives in ~350 ms (≈25 tokens × ~8 ms
// LM + ~100 ms Flow chunk + ~50 ms HiFT chunk), concurrent with the main LM
// thread's batched decode on sched_. The Voc thread is paused during this
// call (voc_sched_mu_) — only one streaming request at a time, but subsequent
// sentences use the normal batch path where the two workers overlap freely.
bool TtsBatchScheduler::StreamFirstSentence(const std::string &text,
                                            const std::string &voice,
                                            const std::string &language,
                                            int32_t seed, ChunkCB cb) {
    if (!dual_worker_ || scheds_voc_.empty()) return false;
    std::lock_guard<std::mutex> lk(voc_sched_mu_);

    model_->SetSeed((uint64_t)seed);
    auto st = std::static_pointer_cast<CosyVoice3State>(model_->CreateState());
    if (!model_->PushText(*st, text.c_str(), language.c_str(), voice.c_str()))
        return false;
    return model_->DecodeStream(*st, scheds_voc_[0], std::move(cb));
}

int TtsBatchScheduler::n_active() {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0;
    for (auto &s : pool_)
        if (s->phase != TtsSlot::Phase::IDLE) n++;
    return n;
}

int TtsBatchScheduler::n_queued() {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)pending_.size();
}

// ── internals ────────────────────────────────────────────────────────────────

void TtsBatchScheduler::AssignPending() {
    // Admit at most ONE pending request per call: prefill is serial (one graph
    // shape), and the new slot takes physical slot_index = n_decode_ (the next
    // contiguous position at the end of the DECODING block).
    //
    // Session-fair admission: scan pending_ for a request whose session_id
    // has NO active slot yet, and admit that one first. This ensures every
    // concurrent WS client's early sentences all begin together instead of
    // one client (whose sentences arrived slightly earlier) monopolising all
    // slots while the others queue in FIFO order.
    for (auto &slot : pool_) {
        if (slot->phase != TtsSlot::Phase::IDLE) continue;
        if (pending_.empty()) break;

        // Find a pending request from a "new" session (none of its sentences
        // are active yet). Fall back to FIFO front if all pending sessions
        // already have at least one active slot.
        std::unordered_set<uint64_t> active_sids;
        for (auto &s : pool_)
            if (s->phase != TtsSlot::Phase::IDLE && s->session_id)
                active_sids.insert(s->session_id);

        size_t pick = 0;
        for (size_t j = 0; j < pending_.size(); ++j) {
            if (pending_[j].session_id &&
                active_sids.count(pending_[j].session_id) == 0) {
                pick = j;
                break;
            }
        }
        // Swap the chosen request to the front so we can pop it.
        if (pick != 0) std::swap(pending_[pick], pending_[0]);
        auto &req = pending_.front();
        slot->session_id = req.session_id;
        // The first sentence of a session runs Flow+HiFT inline on the LM
        // thread (zero wakeup latency for the first audio chunk); later
        // sentences go to the voc thread for GPU concurrency.
        slot->first_of_session = (req.session_id &&
                                  active_sids.count(req.session_id) == 0);
        slot->text     = std::move(req.text);
        slot->voice    = std::move(req.voice);
        slot->language = std::move(req.language);
        slot->seed     = req.seed;
        slot->promise  = std::move(req.promise);
        slot->error.clear();
        slot->state.reset();
        slot->lm.reset();
        slot->lm.slot_index = n_decode_;  // next contiguous physical KV slice
        slot->phase = TtsSlot::Phase::PREFILLING;
        pending_.pop_front();
        break;  // one admission per call (prefill is serial)
    }
}

void TtsBatchScheduler::FinishSlot(TtsSlot &slot) {
    // KV is the shared batched buffer (freed once by the model); the slot's
    // slice is re-zeroed when the slot is reused for a new prefill.
    if (slot.error.empty() && slot.state) {
        slot.promise.set_value({std::move(slot.state->audio_output), ""});
    } else {
        slot.promise.set_value({{}, slot.error.empty() ? "unknown error"
                                                       : slot.error});
    }
    slot.state.reset();
    std::lock_guard<std::mutex> lk(mu_);
    slot.phase = TtsSlot::Phase::IDLE;
}

void TtsBatchScheduler::RunLoop() {
    while (!stop_) {
        // ── assign pending → idle slots, or sleep if nothing to do ────────
        {
            std::unique_lock<std::mutex> lk(mu_);
            AssignPending();
            bool any_active = false;
            for (auto &s : pool_)
                if (s->phase != TtsSlot::Phase::IDLE) { any_active = true; break; }
            if (!any_active) {
                cv_.wait(lk, [&] { return stop_.load() || !pending_.empty(); });
                if (stop_) break;
                AssignPending();
            }
        }

        // ── 1. Prefill takes priority: admit new requests quickly ─────────
        TtsSlot *pf = nullptr;
        for (auto &s : pool_)
            if (s->phase == TtsSlot::Phase::PREFILLING) { pf = s.get(); break; }
        if (pf) {
            // SetSeed must precede CreateState — the state's rng is seeded
            // from the model's seed_ inside CreateState.
            model_->SetSeed((uint64_t)pf->seed);
            pf->state = std::static_pointer_cast<CosyVoice3State>(
                model_->CreateState());
            CosyVoice3State &st = *pf->state;
            bool ok = model_->PushText(st, pf->text.c_str(),
                                       pf->language.c_str(),
                                       pf->voice.c_str()) &&
                      model_->RunLM_Prefill(st, pf->lm, sched_);
            if (!ok) {
                pf->error = "LM prefill failed";
                pf->phase = TtsSlot::Phase::FAILED;
            } else if (pf->lm.done) {
                pf->phase = TtsSlot::Phase::FLOW;  // stop on first sample
            } else {
                pf->phase = TtsSlot::Phase::DECODING;
                n_decode_++;  // extends the contiguous DECODING block
            }
            continue;
        }

        // ── 2. Batched AR step: run ALL DECODING slots through one forward ──
        // This is the vLLM-style win: N sequences' current tokens share one
        // QKV-projection + FFN matmul and one ne3-batched flash attention,
        // instead of N separate forwards. Slots at different n_cached are
        // handled together (per-seq positions/mask/write-rows). Slots are
        // sorted by slot_index so ne3 slice i == physical slot i (contiguous).
        {
            std::vector<TtsSlot *> dec;
            dec.reserve(n_slots_);
            for (auto &s : pool_)
                if (s->phase == TtsSlot::Phase::DECODING) dec.push_back(s.get());
            if (!dec.empty()) {
                std::sort(dec.begin(), dec.end(),
                          [](TtsSlot *a, TtsSlot *b) {
                              return a->lm.slot_index < b->lm.slot_index;
                          });
                const int n = (int)dec.size();
                std::vector<CosyVoice3State *> states(n);
                std::vector<CosyVoice3LMSlot *> slots(n);
                for (int i = 0; i < n; ++i) {
                    states[i] = dec[i]->state.get();
                    slots[i] = &dec[i]->lm;
                }
                bool ok = model_->RunLM_Step_Batch(states.data(), slots.data(),
                                                   n, sched_);
                // Mark finished / errored slots out of DECODING.
                for (int i = 0; i < n; ++i) {
                    if (!ok) {
                        dec[i]->error = "LM step failed";
                        dec[i]->phase = TtsSlot::Phase::FAILED;
                    } else if (dec[i]->lm.done) {
                        dec[i]->phase = TtsSlot::Phase::FLOW;
                    }
                }
                // Rebuild a contiguous physical block 0..k-1 from the slots
                // still DECODING, copying KV where a slot's physical position
                // changed (fills gaps left by finished slots). Handles multiple
                // simultaneous finishes correctly.
                {
                    std::vector<TtsSlot *> rem;
                    for (auto &s : pool_)
                        if (s->phase == TtsSlot::Phase::DECODING) rem.push_back(s.get());
                    std::sort(rem.begin(), rem.end(),
                              [](TtsSlot *a, TtsSlot *b) {
                                  return a->lm.slot_index < b->lm.slot_index;
                              });
                    for (int i = 0; i < (int)rem.size(); ++i) {
                        if (rem[i]->lm.slot_index != i) {
                            model_->CopyBatchKVSlice(rem[i]->lm.slot_index, i);
                            rem[i]->lm.slot_index = i;
                        }
                    }
                    n_decode_ = (int)rem.size();
                }
            }
        }

        // ── 3. Batched Flow: all FLOW slots through one pipeline ───────────
        // N slots' speech tokens -> mel in one CFG-batched (2N) DiT forward
        // (per-slot pad mask). They then move to HIFT for batched vocoding.
        //
        // Coalescing: while other slots are still DECODING they will EOS soon
        // and can join the Flow batch, so we HOLD the ready FLOW slots for a
        // few LM steps to accumulate a larger batch (Flow/HiFT is the ~50%
        // per-request cost and batches N-for-1). Flush when: no slot is still
        // decoding (nothing more will arrive), the FLOW group is already full,
        // or we've waited kMaxFlowWait steps (bounds added latency to a few
        // decode steps ~tens of ms, so single/short requests don't regress).
        {
            std::vector<TtsSlot *> fl;
            for (auto &s : pool_)
                if (s->phase == TtsSlot::Phase::FLOW) fl.push_back(s.get());

            // Hold the FLOW group while more work could still join the batch:
            // slots still DECODING/PREFILLING, or requests queued (their prefill
            // + decode will finish soon). Flush when nothing more is coming, the
            // group is full, or we've waited kMaxFlowWait steps.
            int n_inflight = 0, n_pend = 0;
            for (auto &s : pool_)
                if (s->phase == TtsSlot::Phase::DECODING ||
                    s->phase == TtsSlot::Phase::PREFILLING) n_inflight++;
            { std::lock_guard<std::mutex> lk(mu_); n_pend = (int)pending_.size(); }

            constexpr int kMaxFlowWait = 40;  // LM steps (~upper bound on hold)
            bool flush = !fl.empty() &&
                         ((n_inflight == 0 && n_pend == 0) ||
                          (int)fl.size() >= n_slots_ ||
                          flow_wait_ >= kMaxFlowWait);
            if (!fl.empty() && !flush) {
                flow_wait_++;  // hold this round, let in-flight work catch up
            } else if (flush) {
                flow_wait_ = 0;
                const int n = (int)fl.size();
                std::vector<CosyVoice3State *> st(n);
                for (int i = 0; i < n; ++i) st[i] = fl[i]->state.get();
                bool ok = model_->RunFlow_Batch(st.data(), n, sched_);
                for (int i = 0; i < n; ++i) {
                    if (!ok) {
                        fl[i]->error = "Flow failed";
                        fl[i]->phase = TtsSlot::Phase::FAILED;
                    } else {
                        fl[i]->phase = TtsSlot::Phase::HIFT;
                    }
                }
            }
        }

        // ── 3b. Batched HiFT: all HIFT slots through one vocoder forward ───
        {
            std::vector<TtsSlot *> hf;
            for (auto &s : pool_)
                if (s->phase == TtsSlot::Phase::HIFT) hf.push_back(s.get());
            if (!hf.empty()) {
                const int n = (int)hf.size();
                std::vector<CosyVoice3State *> st(n);
                for (int i = 0; i < n; ++i) st[i] = hf[i]->state.get();
                bool ok = model_->RunHiFT_Batch(st.data(), n, sched_);
                for (int i = 0; i < n; ++i) {
                    if (!ok) {
                        hf[i]->error = "HiFT failed";
                        hf[i]->phase = TtsSlot::Phase::FAILED;
                    } else {
                        hf[i]->phase = TtsSlot::Phase::DONE;
                    }
                }
            }
        }

        // ── 4. Resolve DONE / FAILED slots ────────────────────────────────
        for (auto &s : pool_) {
            if (s->phase == TtsSlot::Phase::DONE ||
                s->phase == TtsSlot::Phase::FAILED) {
                FinishSlot(*s);
                cv_.notify_all();
            }
        }
    }

    // Drain: fail any in-flight slots on shutdown.
    for (auto &s : pool_) {
        if (s->phase != TtsSlot::Phase::IDLE) {
            s->error = "server shutting down";
            FinishSlot(*s);
        }
    }
}

// ── dual-worker: LM thread (prefill + batched decode) ──────────────────────
// Hands finished (FLOW) slots to the voc thread via flow_queue_. Runs on
// sched_ (owns the batched KV). Flow/HiFT run concurrently on scheds_voc_[voc_idx].
void TtsBatchScheduler::RunLoopLM() {
    auto handoff = [&](TtsSlot *s) {
        std::lock_guard<std::mutex> lk(mu_);
        flow_queue_.push_back(s);
        voc_cv_.notify_one();
    };
    while (!stop_) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            AssignPending();
            bool any_active = false;
            for (auto &s : pool_)
                if (s->phase != TtsSlot::Phase::IDLE) { any_active = true; break; }
            if (!any_active) {
                cv_.wait(lk, [&] { return stop_.load() || !pending_.empty(); });
                if (stop_) break;
                AssignPending();
            }
        }

        // 1. Prefill (one slot).
        TtsSlot *pf = nullptr;
        for (auto &s : pool_)
            if (s->phase == TtsSlot::Phase::PREFILLING) { pf = s.get(); break; }
        if (pf) {
            model_->SetSeed((uint64_t)pf->seed);
            pf->state = std::static_pointer_cast<CosyVoice3State>(
                model_->CreateState());
            CosyVoice3State &st = *pf->state;
            bool ok = model_->PushText(st, pf->text.c_str(),
                                       pf->language.c_str(),
                                       pf->voice.c_str()) &&
                      model_->RunLM_Prefill(st, pf->lm, sched_);
            if (!ok) {
                pf->error = "LM prefill failed";
                pf->phase = TtsSlot::Phase::FAILED;
                FinishSlot(*pf);
            } else if (pf->lm.done) {
                pf->phase = TtsSlot::Phase::FLOW;
                handoff(pf);
            } else {
                pf->phase = TtsSlot::Phase::DECODING;
                n_decode_++;
            }
            continue;
        }

        // 2. Batched decode step.
        std::vector<TtsSlot *> dec;
        dec.reserve(n_slots_);
        for (auto &s : pool_)
            if (s->phase == TtsSlot::Phase::DECODING) dec.push_back(s.get());
        if (!dec.empty()) {
            std::sort(dec.begin(), dec.end(),
                      [](TtsSlot *a, TtsSlot *b) {
                          return a->lm.slot_index < b->lm.slot_index;
                      });
            const int n = (int)dec.size();
            std::vector<CosyVoice3State *> states(n);
            std::vector<CosyVoice3LMSlot *> slots(n);
            for (int i = 0; i < n; ++i) {
                states[i] = dec[i]->state.get();
                slots[i] = &dec[i]->lm;
            }
            bool ok = model_->RunLM_Step_Batch(states.data(), slots.data(),
                                               n, sched_);
            std::vector<TtsSlot *> to_flow;
            for (int i = 0; i < n; ++i) {
                if (!ok) {
                    dec[i]->error = "LM step failed";
                    dec[i]->phase = TtsSlot::Phase::FAILED;
                    FinishSlot(*dec[i]);
                } else if (dec[i]->lm.done) {
                    dec[i]->phase = TtsSlot::Phase::FLOW;
                    to_flow.push_back(dec[i]);
                }
            }
            // Compact the DECODING block.
            {
                std::vector<TtsSlot *> rem;
                for (auto &s : pool_)
                    if (s->phase == TtsSlot::Phase::DECODING) rem.push_back(s.get());
                std::sort(rem.begin(), rem.end(),
                          [](TtsSlot *a, TtsSlot *b) {
                              return a->lm.slot_index < b->lm.slot_index;
                          });
                for (int i = 0; i < (int)rem.size(); ++i) {
                    if (rem[i]->lm.slot_index != i) {
                        model_->CopyBatchKVSlice(rem[i]->lm.slot_index, i);
                        rem[i]->lm.slot_index = i;
                    }
                }
                n_decode_ = (int)rem.size();
            }
            // Hand finished slots: first sentence of a session runs Flow+HiFT
            // inline on the LM thread (zero wakeup latency, same as single-worker
            // path); subsequent sentences go to the voc thread so Flow/HiFT overlap
            // with LM decode on the independent CUDA stream.
            std::vector<TtsSlot *> to_voc;
            for (auto *s : to_flow) {
                if (dual_worker_ && s->first_of_session) {
                    // Do first sentence's Flow+HiFT on this (LM) thread with sched_.
                    bool fok = model_->RunFlow(*s->state, sched_) &&
                               model_->RunHiFT(*s->state, sched_);
                    if (!fok) {
                        s->error = "Flow/HiFT inline failed";
                        s->phase = TtsSlot::Phase::FAILED;
                        FinishSlot(*s);
                    } else {
                        s->phase = TtsSlot::Phase::DONE;
                        FinishSlot(*s);
                        cv_.notify_all();
                    }
                } else {
                    to_voc.push_back(s);
                    s->first_of_session = false;
                }
            }
            if (!to_voc.empty()) {
                std::lock_guard<std::mutex> lk(mu_);
                for (auto *s : to_voc) flow_queue_.push_back(s);
                voc_cv_.notify_one();
            }
        } else {
            // Nothing to decode this round; brief yield to avoid spinning.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

// ── dual-worker: vocoder thread (Flow_Batch + HiFT_Batch) ──────────────────
// Drains flow_queue_, batching whatever is ready (natural coalescing - no
// hold, so Flow/HiFT overlap with LM decode on the independent CUDA stream).
void TtsBatchScheduler::RunLoopVoc(int voc_idx) {
    while (!stop_) {
        std::vector<TtsSlot *> fl;
        {
            std::unique_lock<std::mutex> lk(mu_);
            voc_cv_.wait(lk, [&] { return stop_.load() || !flow_queue_.empty(); });
            if (stop_) break;
            fl.swap(flow_queue_);  // take all ready slots
        }
        if (fl.empty()) continue;

        const int n = (int)fl.size();
        std::vector<CosyVoice3State *> st(n);
        for (int i = 0; i < n; ++i) st[i] = fl[i]->state.get();

        bool ok = model_->RunFlow_Batch(st.data(), n, scheds_voc_[voc_idx]);
        std::vector<TtsSlot *> hf;
        for (int i = 0; i < n; ++i) {
            if (!ok) {
                fl[i]->error = "Flow failed";
                fl[i]->phase = TtsSlot::Phase::FAILED;
            } else {
                fl[i]->phase = TtsSlot::Phase::HIFT;
                hf.push_back(fl[i]);
            }
        }
        if (!hf.empty()) {
            const int m = (int)hf.size();
            std::vector<CosyVoice3State *> st2(m);
            for (int i = 0; i < m; ++i) st2[i] = hf[i]->state.get();
            bool ok2 = model_->RunHiFT_Batch(st2.data(), m, scheds_voc_[voc_idx]);
            for (int i = 0; i < m; ++i) {
                if (!ok2) {
                    hf[i]->error = "HiFT failed";
                    hf[i]->phase = TtsSlot::Phase::FAILED;
                } else {
                    hf[i]->phase = TtsSlot::Phase::DONE;
                }
            }
        }
        for (auto *s : fl) {
            FinishSlot(*s);
            cv_.notify_all();
        }
    }
}
