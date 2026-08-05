#pragma once
// cosyvoice3_batch.h — TTS continuous-batching scheduler for CosyVoice3.
//
// Slot lifecycle: IDLE → PREFILLING → DECODING → FLOW → HIFT → DONE/ERROR
//
// Concurrency model (single shared ggml sched — not thread-safe, so all
// compute runs on one worker thread; concurrency comes from interleaving):
//   - LM AR decode is the long pole (one graph per token). The scheduler
//     round-robins RunLM_Step across all DECODING slots, so N requests
//     progress together instead of queueing serially ("model busy").
//   - Prefill for a new request happens between steps — a new request only
//     waits for the current token, not for a whole utterance.
//   - Flow + HiFT run when a slot's LM finishes, between other slots' steps.
//
// Usage (server side):
//   TtsBatchScheduler ts(model, backend_sched, /*n_slots=*/4);
//   ts.Start();
//   auto fut = ts.Submit(text, voice, lang, seed);
//   auto [pcm, err] = fut.get();   // err empty on success

#include "cosyvoice3.h"
#include "ggml-backend.h"

#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// ── per-request slot ────────────────────────────────────────────────────────

struct TtsSlot {
    enum class Phase { IDLE, PREFILLING, DECODING, FLOW, HIFT, DONE, ERROR };

    int id = -1;
    Phase phase = Phase::IDLE;

    std::shared_ptr<CosyVoice3State> state;  // fresh per request
    CosyVoice3LMSlot lm;                     // KV cache + AR bookkeeping

    // request params
    std::string text;
    std::string voice;
    std::string language;
    int32_t seed = 42;
    uint64_t session_id = 0;   // WS connection cookie (0 = one-off HTTP)
    bool first_of_session = false;  // first req of this session → LM thread does Flow+HiFT inline (no thread hop)

    // result
    std::string error;
    std::promise<std::pair<std::vector<float>, std::string>> promise;
};

// ── scheduler ───────────────────────────────────────────────────────────────

class RS_CV3_EXPORT TtsBatchScheduler {
public:
    // model: the loaded CosyVoice3LMModel (must outlive this object).
    // sched: the ggml backend scheduler (shared with the model).
    // n_slots: max concurrent requests (each holds an n_kv_max-wide f32 KV
    //          cache on the GPU: ~24 layers × 2 × kv_dim × n_kv_max floats).
    TtsBatchScheduler(CosyVoice3LMModel *model,
                      ggml_backend_sched_t sched,
                      int n_slots = 4);
    // sched_voc: independent scheduler(s) (separate CUDA backend/stream) for
    // the Flow/HiFT worker thread(s) -> LM || vocoder GPU concurrency. nullptr
    // or empty => single-worker (Metal/CPU): Flow/HiFT run on the LM thread.
    TtsBatchScheduler(CosyVoice3LMModel *model,
                      ggml_backend_sched_t sched_lm,
                      std::vector<ggml_backend_sched_t> scheds_voc,
                      int n_slots = 4);
    ~TtsBatchScheduler();

    void Start();
    void Stop();

    // Submit a TTS request. Returns a future resolving to {pcm, error};
    // error is empty on success. Never rejects — requests queue for a slot.
    std::future<std::pair<std::vector<float>, std::string>>
    Submit(const std::string &text, const std::string &voice,
           const std::string &language, int32_t seed,
           uint64_t session_id = 0);

    int n_slots() const { return n_slots_; }
    int n_active();   // slots not IDLE (for /health reporting)
    int n_queued();   // requests waiting for a slot
    bool CanStreamFirstSentence() const {
        return dual_worker_ && !scheds_voc_.empty();
    }

    // Stream the first sentence of a WS connection directly via DecodeStream
    // on the vocoder thread's sched (sub-500ms first-chunk latency, concurrent
    // with LM batched decode on the main sched). Must NOT be called while the
    // voc thread is busy with a Flow/HiFT batch (voc_sched_mu_ protects this).
    using ChunkCB = CosyVoice3LMModel::AudioChunkCallback;
    bool StreamFirstSentence(const std::string &text, const std::string &voice,
                             const std::string &language, int32_t seed,
                             ChunkCB chunk_cb);

private:
    void RunLoop();          // single-worker (LM + Flow/HiFT on one thread)
    void RunLoopLM();        // dual-worker: prefill + decode, hands FLOW slots to voc
    void RunLoopVoc(int voc_idx);       // dual-worker: Flow_Batch + HiFT_Batch
    void AssignPending();               // pending queue → idle slots (mu_ held)
    void FinishSlot(TtsSlot &slot);     // resolve promise, return slot to IDLE

    CosyVoice3LMModel *model_;
    ggml_backend_sched_t sched_;       // LM worker (prefill + decode, owns KV)
    std::vector<ggml_backend_sched_t> scheds_voc_;  // Flow/HiFT worker(s)
    bool dual_worker_ = false;
    int n_slots_;
    int n_decode_ = 0;  // DECODING slots occupy contiguous physical 0..n_decode_-1
    int flow_wait_ = 0; // LM steps a FLOW slot has waited to batch (see RunLoop)

    std::vector<std::unique_ptr<TtsSlot>> pool_;

    struct PendingReq {
        std::string text, voice, language;
        int32_t seed = 42;
        uint64_t session_id = 0;
        std::promise<std::pair<std::vector<float>, std::string>> promise;
    };
    std::deque<PendingReq> pending_;

    // Dual-worker handoff: LM thread pushes FLOW slots here, voc thread drains.
    std::vector<TtsSlot *> flow_queue_;

    std::mutex mu_;               // guards pending_ and slot phase transitions
    std::mutex voc_sched_mu_;      // serializes StreamFirstSentence vs voc thread(s)
    std::condition_variable cv_;
    std::condition_variable voc_cv_;  // signals flow_queue_ not empty
    std::thread worker_;
    std::vector<std::thread> workers_voc_;
    std::atomic<bool> stop_{false};
};
