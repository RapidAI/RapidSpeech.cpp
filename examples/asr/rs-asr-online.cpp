/**
 * rs-asr-online.cpp — Online (streaming) ASR with VAD endpoint detection
 *
 * Architecture (producer-consumer):
 *   Thread 1 — Audio + VAD (producer)
 *     Mic / WAV → 512-sample chunks → Silero VAD → speech segments
 *       → push to thread-safe bounded queue (non-blocking)
 *
 *   Thread 2 — ASR (consumer)
 *     Pop segments from queue → ASR inference → timestamped output
 *
 *   The queue decouples audio collection from inference.  If ASR falls
 *   behind and the queue exceeds 15 s of accumulated audio, a real-time
 *   warning is emitted.  The queue is hard-limited to 60 s; beyond that
 *   the oldest segments are dropped (with a warning).
 *
 * Usage:
 *   rs-asr-online -m <asr.gguf> -v <vad.gguf> -w <audio.wav> [options]
 *   rs-asr-online -m <asr.gguf> -v <vad.gguf> --mic [options]
 *
 * Options:
 *   -m, --model <path>       ASR model path  (required)
 *   -v, --vad <path>         Silero-VAD GGUF path  (required)
 *   -w, --wav <path>         WAV file to process (simulate streaming)
 *       --mic                Use microphone input (live mode)
 *       --mic-device <n>     Audio device index (default: auto)
 *       --mic-chunk-ms <ms>  Mic read chunk size (default: 32)
 *   -t, --threads <n>        CPU threads (default: 4)
 *       --gpu <true|false>   GPU on/off (default: true)
 *       --vad-threshold <f>  Speech probability threshold (default: 0.5)
 *       --silence-ms <ms>    Silence duration to trigger decode (default: 600)
 *       --speech-pad-ms <ms> Pre-roll padding before speech start (default: 200)
 *       --preroll-ms <ms>    Audio kept during silence to avoid onset clip (default: 500)
 *       --prob-smooth <f>    VAD prob EMA alpha, 0=disable (default: 0.3)
 *   -h, --help               Show help
 */

#include "arch/silero_vad.h"
#include "arch/fireredvad.h"
#include "rapidspeech.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#ifdef RS_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef RS_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <deque>
#include <string>
#include <thread>
#include <vector>

// Global stop flag for signal handler
static std::atomic<bool> g_stop_flag{false};
static void signal_handler(int) { g_stop_flag.store(true); }

// ─────────────────────────────────────────────────────
// Logging
//   LOG_INFO  — always shown, kept terse
//   LOG_DEBUG — only shown when --verbose / -V is set (or RS_VERBOSE=1)
//   LOG_ERROR — always shown to stderr
// ─────────────────────────────────────────────────────
static bool g_verbose = false;

#define LOG_INFO(fmt, ...) std::printf("[online] " fmt "\n", ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...)                                                    \
  do {                                                                         \
    if (g_verbose)                                                             \
      std::printf("[debug] " fmt "\n", ##__VA_ARGS__);                         \
  } while (0)
#define LOG_ERROR(fmt, ...)                                                    \
  std::fprintf(stderr, "[online] ERROR: " fmt "\n", ##__VA_ARGS__)

// ─────────────────────────────────────────────────────
// Terminal color helpers (ANSI escape codes)
// ─────────────────────────────────────────────────────
namespace Color {
static const char *Reset = "\033[0m";
static const char *Bold = "\033[1m";
static const char *Dim = "\033[2m";
static const char *Green = "\033[32m";
static const char *Yellow = "\033[33m";
static const char *Cyan = "\033[36m";
static const char *Grey = "\033[90m";

static bool enabled = true;

const char *c(const char *code) { return enabled ? code : ""; }
} // namespace Color

// ─────────────────────────────────────────────────────
// Thread-safe speech segment queue (producer-consumer)
//
// VAD thread pushes completed speech segments; ASR worker
// thread pops and transcribes them.  The queue is bounded
// by total audio duration to prevent unbounded memory growth
// when ASR cannot keep up with real-time audio.
// ─────────────────────────────────────────────────────
class SpeechSegmentQueue {
public:
  struct Segment {
    std::vector<float> pcm;
    float start_s, end_s;
    int index;
    bool is_partial = false;
  };

  SpeechSegmentQueue(float max_dur_s = 60.0f, float warn_dur_s = 15.0f)
      : max_queue_dur_s_(max_dur_s), warn_queue_dur_s_(warn_dur_s) {}

  // Push a completed speech segment (non-blocking).
  // If the queue is full the oldest segment is dropped.
  void push(Segment seg) {
    float seg_dur = seg.end_s - seg.start_s;
    std::lock_guard<std::mutex> lock(mtx_);

    // Keep only the newest partial for a segment.  If the final segment is
    // ready, any queued partial for the same segment is now stale.
    for (auto it = segments_.begin(); it != segments_.end();) {
      if (it->index == seg.index && (it->is_partial || !seg.is_partial)) {
        queue_dur_s_ -= (it->end_s - it->start_s);
        it = segments_.erase(it);
      } else {
        ++it;
      }
    }

    while (queue_dur_s_ + seg_dur > max_queue_dur_s_ && !segments_.empty()) {
      auto &oldest = segments_.front();
      float oldest_dur = oldest.end_s - oldest.start_s;
      queue_dur_s_ -= oldest_dur;
      dropped_++;
      segments_.pop_front();
    }

    queue_dur_s_ += seg_dur;
    segments_.push_back(std::move(seg));
    total_queued_++;

    // Signal ASR thread
    cv_.notify_one();
  }

  // Pop a segment.  Blocks until a segment is available OR the
  // queue is marked done AND empty.  Returns false only when the
  // producer has finished and the queue is drained.
  bool pop(Segment &seg) {
    std::unique_lock<std::mutex> lock(mtx_);
    while (segments_.empty() && !done_) {
      cv_.wait(lock);
    }
    if (segments_.empty())
      return false; // done_ is true and nothing left
    seg = std::move(segments_.front());
    segments_.pop_front();
    queue_dur_s_ -= (seg.end_s - seg.start_s);
    return true;
  }

  // Non-blocking pop — returns false if queue is empty.
  bool try_pop(Segment &seg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (segments_.empty())
      return false;
    seg = std::move(segments_.front());
    segments_.pop_front();
    queue_dur_s_ -= (seg.end_s - seg.start_s);
    return true;
  }

  // Push a segment to the front (used when putting back an unmerged segment).
  void push_front(Segment seg) {
    std::lock_guard<std::mutex> lock(mtx_);
    float seg_dur = seg.end_s - seg.start_s;
    queue_dur_s_ += seg_dur;
    segments_.push_front(std::move(seg));
  }

  // Signal that no more segments will be pushed.
  void set_done() {
    std::lock_guard<std::mutex> lock(mtx_);
    done_ = true;
    cv_.notify_all();
  }

  // Current queue depth in seconds of audio.
  float duration() {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_dur_s_;
  }

  // Check whether the queue is above the warning threshold.
  // Returns true once per threshold crossing (resets after ack).
  bool check_and_clear_slow_warning() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_dur_s_ >= warn_queue_dur_s_ && !warned_) {
      warned_ = true;
      return true;
    }
    if (queue_dur_s_ < warn_queue_dur_s_ / 2.0f) {
      warned_ = false; // recovered — re-arm for next time
    }
    return false;
  }

  int total_queued() const { return total_queued_.load(); }
  int dropped() const { return dropped_.load(); }

private:
  std::deque<Segment> segments_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool done_ = false;
  bool warned_ = false;
  float queue_dur_s_ = 0.0f;
  float max_queue_dur_s_;
  float warn_queue_dur_s_;
  std::atomic<int> total_queued_{0};
  std::atomic<int> dropped_{0};
};

// ─────────────────────────────────────────────────────
// Thread-safe audio ring buffer (mic → VAD decoupling)
//
// The mic capture thread writes raw PCM continuously;
// the main/VAD thread reads in fixed-size chunks.
// A fixed-capacity ring buffer prevents unbounded growth
// and alerts when the reader falls behind.
// ─────────────────────────────────────────────────────
class AudioRingBuffer {
public:
  AudioRingBuffer(int capacity_samples)
      : capacity_(capacity_samples), buf_(capacity_samples) {}

  // Write samples (non-blocking).  Drops oldest data if full.
  void write(const float *data, size_t n) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < n; ++i) {
      if (count_ >= capacity_) {
        read_pos_ = (read_pos_ + 1) % capacity_;
        count_--;
        overruns_++;
      }
      buf_[write_pos_] = data[i];
      write_pos_ = (write_pos_ + 1) % capacity_;
      count_++;
    }
    cv_.notify_one();
  }

  // Read exactly n samples (blocks until available or stopped).
  // Returns actual number read (may be < n if stopped).
  size_t read(float *out, size_t n) {
    std::unique_lock<std::mutex> lock(mtx_);
    size_t total = 0;
    while (total < n && !stopped_) {
      if (count_ == 0) {
        cv_.wait(lock);
        continue;
      }
      size_t to_read = std::min(n - total, count_);
      for (size_t i = 0; i < to_read; ++i) {
        out[total + i] = buf_[read_pos_];
        read_pos_ = (read_pos_ + 1) % capacity_;
      }
      count_ -= to_read;
      total += to_read;
    }
    return total;
  }

  // Signal that capture has stopped; unblocks any waiting reader.
  void stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    stopped_ = true;
    cv_.notify_all();
  }

  size_t available() const { return count_; }
  size_t overruns() const { return overruns_; }

private:
  const int capacity_;
  std::vector<float> buf_;
  int read_pos_ = 0;
  int write_pos_ = 0;
  size_t count_ = 0;
  size_t overruns_ = 0;
  bool stopped_ = false;
  std::mutex mtx_;
  std::condition_variable cv_;
};

// ─────────────────────────────────────────────────────
// Format timestamp as [MM:SS.mmm]
// ─────────────────────────────────────────────────────
static std::string format_timestamp(float seconds) {
  int total_ms = (int)(seconds * 1000 + 0.5f);
  if (total_ms < 0)
    total_ms = 0;
  int h = total_ms / 3600000;
  int m = (total_ms % 3600000) / 60000;
  int s = (total_ms % 60000) / 1000;
  int ms = total_ms % 1000;
  char buf[32];
  if (h > 0) {
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", h, m, s, ms);
  } else {
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", m, s, ms);
  }
  return std::string(buf);
}

// ─────────────────────────────────────────────────────
// Utility: VAD backend (lightweight)
// ─────────────────────────────────────────────────────
struct VadBackend {
  ggml_backend_t backend = nullptr;
  ggml_backend_sched_t sched = nullptr;

  bool init() {
    backend = ggml_backend_cpu_init();
    if (!backend) {
      LOG_ERROR("Failed to init CPU backend for VAD");
      return false;
    }
    LOG_INFO("VAD backend: CPU");
    ggml_backend_t backends[1] = {backend};
    ggml_backend_buffer_type_t buft[1] = {
        ggml_backend_get_default_buffer_type(backend)};
    sched = ggml_backend_sched_new(backends, buft, 1, 512, false, false);
    return sched != nullptr;
  }

  ~VadBackend() {
    if (sched)
      ggml_backend_sched_free(sched);
    if (backend)
      ggml_backend_free(backend);
  }
};

// ─────────────────────────────────────────────────────
// Load a GGUF for VAD weights
// ─────────────────────────────────────────────────────
static ggml_context *load_gguf_weights(const char *path,
                                       gguf_context **out_gguf) {
  ggml_context *ctx = nullptr;
  gguf_init_params p = {false, &ctx};
  *out_gguf = gguf_init_from_file(path, p);
  if (!*out_gguf) {
    LOG_ERROR("Failed to open VAD GGUF: %s", path);
    return nullptr;
  }
  return ctx;
}

// Read `general.architecture` from a loaded gguf_context.
// Returns an empty string if the key is absent.
static std::string read_gguf_arch(gguf_context *gguf) {
  if (!gguf) return "";
  int idx = gguf_find_key(gguf, "general.architecture");
  if (idx < 0) return "";
  const char *s = gguf_get_val_str(gguf, idx);
  return s ? std::string(s) : std::string();
}

// ─────────────────────────────────────────────────────
// Cross-platform microphone capture via miniaudio
// ─────────────────────────────────────────────────────
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

struct MicCapture {
  ma_device device;
  AudioRingBuffer *ring = nullptr;
  std::atomic<bool> running{false};

  static void data_callback(ma_device *pDevice, void *, const void *pInput,
                            ma_uint32 frameCount) {
    auto *self = static_cast<MicCapture *>(pDevice->pUserData);
    if (self->ring && pInput) {
      self->ring->write(static_cast<const float *>(pInput),
                        frameCount * pDevice->capture.channels);
    }
  }

  bool start(int sample_rate, AudioRingBuffer *rb) {
    ring = rb;

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format    = ma_format_f32;
    config.capture.channels  = 1;
    config.sampleRate        = (ma_uint32)sample_rate;
    config.dataCallback      = data_callback;
    config.pUserData         = this;

    if (ma_device_init(nullptr, &config, &device) != MA_SUCCESS) {
      LOG_ERROR("Failed to initialize miniaudio capture device");
      return false;
    }

    if (ma_device_start(&device) != MA_SUCCESS) {
      LOG_ERROR("Failed to start miniaudio capture device");
      ma_device_uninit(&device);
      return false;
    }

    running = true;
    LOG_INFO("Microphone capture started @ %d Hz (miniaudio)", sample_rate);
    return true;
  }

  void stop() {
    running = false;
    if (ma_device_is_started(&device))
      ma_device_stop(&device);
    ma_device_uninit(&device);
  }

  ~MicCapture() { stop(); }
};

// ─────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────
struct OnlineArgs {
  const char *asr_model = nullptr;
  const char *vad_model = nullptr;
  const char *wav_path = nullptr;
  bool use_mic = false;
  int mic_device = -1;
  int mic_chunk_ms = 32;
  int n_threads = 4;
  bool use_gpu = true;
  float vad_threshold = 0.5f;
  int silence_ms = 600;
  int speech_pad_ms = 200;
  int preroll_ms = 500;
  int partial_ms = 1000;
  int max_segment_ms = 0; // 0 = disabled; force-flush long segments above this
  float prob_smooth = 0.3f;
  bool two_pass = false;
  bool ctc_precheck = false;
};

static bool parse_bool(const std::string &v) {
  return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

static bool parse_args(int argc, char **argv, OnlineArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model" || a == "--asr-model") && i + 1 < argc) {
      args.asr_model = argv[++i];
    } else if ((a == "-v" || a == "--vad" || a == "--vad-model") &&
               i + 1 < argc) {
      args.vad_model = argv[++i];
    } else if ((a == "-w" || a == "--wav") && i + 1 < argc) {
      args.wav_path = argv[++i];
    } else if (a == "--mic") {
      args.use_mic = true;
    } else if (a == "--mic-device" && i + 1 < argc) {
      args.mic_device = std::stoi(argv[++i]);
    } else if (a == "--mic-chunk-ms" && i + 1 < argc) {
      args.mic_chunk_ms = std::stoi(argv[++i]);
    } else if ((a == "-t" || a == "--threads") && i + 1 < argc) {
      args.n_threads = std::stoi(argv[++i]);
    } else if (a == "--gpu" && i + 1 < argc) {
      args.use_gpu = parse_bool(argv[++i]);
    } else if (a == "--vad-threshold" && i + 1 < argc) {
      args.vad_threshold = std::stof(argv[++i]);
    } else if (a == "--silence-ms" && i + 1 < argc) {
      args.silence_ms = std::stoi(argv[++i]);
    } else if (a == "--speech-pad-ms" && i + 1 < argc) {
      args.speech_pad_ms = std::stoi(argv[++i]);
    } else if (a == "--preroll-ms" && i + 1 < argc) {
      args.preroll_ms = std::stoi(argv[++i]);
    } else if (a == "--partial-ms" && i + 1 < argc) {
      args.partial_ms = std::stoi(argv[++i]);
    } else if (a == "--max-segment-ms" && i + 1 < argc) {
      args.max_segment_ms = std::stoi(argv[++i]);
    } else if (a == "--prob-smooth" && i + 1 < argc) {
      args.prob_smooth = std::stof(argv[++i]);
    } else if (a == "--two-pass") {
      args.two_pass = true;
    } else if (a == "--ctc-precheck") {
      args.ctc_precheck = true;
    } else if (a == "-V" || a == "--verbose") {
      g_verbose = true;
    } else if (a == "-h" || a == "--help") {
      std::cout
          << "Usage: rs-asr-online -m <asr.gguf> -v <vad.gguf> [options]\n"
             "\n"
             "Input source (choose one):\n"
             "  -w, --wav <path>         WAV file (simulate streaming)\n"
             "      --mic                Use microphone (live mode)\n"
             "\n"
             "Required:\n"
             "  -m, --model <path>       ASR model path\n"
             "  -v, --vad <path>         Silero-VAD model path\n"
             "\n"
             "Options:\n"
             "  -t, --threads <n>        CPU threads (default: 4)\n"
             "      --gpu <true|false>   GPU acceleration (default: true)\n"
             "      --vad-threshold <f>  VAD threshold (default: 0.5)\n"
             "      --silence-ms <ms>    Silence to end segment (default: "
             "600)\n"
             "      --speech-pad-ms <ms> Pre-roll padding before speech start "
             "(default: 200)\n"
             "      --preroll-ms <ms>    Audio buffer kept during silence to "
             "avoid onset clipping (default: 500)\n"
             "      --partial-ms <ms>    2-pass mode: interval for CTC "
             "partial updates (default: 1000, 0=disable)\n"
             "      --max-segment-ms <ms> Force-finalize a segment after this "
             "much continuous speech (default: 0=disable; try 10000 for "
             "newscasts to avoid long-segment stalls)\n"
             "      --prob-smooth <f>    EMA alpha for VAD prob smoothing "
             "(0=disable, default: 0.3)\n"
             "      --mic-chunk-ms <ms>  Mic read chunk size (default: 32)\n"
             "      --two-pass           2-pass mode: CTC fast pass + LLM "
             "rescore (FunASRNano)\n"
             "      --ctc-precheck       CTC pre-check before LLM to skip "
             "silence (reduces hallucination)\n"
             "  -V, --verbose            Show debug logs and per-segment "
             "stats (also: RS_VERBOSE=1)\n"
             "  -h, --help               Show this help\n"
          << std::endl;
      return false;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      return false;
    }
  }

  if (!args.asr_model || !args.vad_model) {
    std::cerr << "Error: --model and --vad are required.\n";
    return false;
  }
  if (!args.wav_path && !args.use_mic) {
    std::cerr << "Error: specify --wav <file> or --mic for input source.\n";
    return false;
  }
  if (args.wav_path && args.use_mic) {
    std::cerr << "Error: --wav and --mic are mutually exclusive.\n";
    return false;
  }
  return true;
}

// ─────────────────────────────────────────────────────
// VAD streaming state (updated per VAD frame)
// ─────────────────────────────────────────────────────
struct StreamState {
  std::vector<float> speech_buf;
  bool in_speech = false;
  int silence_frames = 0;
  int speech_start_offset = 0;
  int total_samples_read = 0;
  int current_segment_index = 0;
  int last_partial_samples = 0;

  // --- Pre-roll buffer: retains recent audio during silence so that ---
  // speech onset isn't truncated.  Works as a sliding window that is
  // flushed into speech_buf when VAD triggers.
  std::deque<float> preroll_buf;
  int preroll_max_samples = 8000; // 500 ms @ 16 kHz (overridden by CLI)

  // --- EMA-smoothed VAD probability ---
  float smoothed_prob = 0.0f;
  float prob_smooth_alpha = 0.3f; // 0 = disable smoothing

  // --- Adaptive silence threshold ---
  float adaptive_silence_ms = 600.0f; // current effective threshold
  float base_silence_ms = 600.0f;     // user-configured baseline
  std::deque<float> recent_seg_durs;  // last few segment durations (seconds)

  // --- Soft endpoint search around max_segment_ms cap ---
  // Track the lowest-VAD-probability frame within the search window before
  // the hard cap so we can cut at a relative dip instead of mid-word.
  int soft_cut_best_offset = -1; // sample offset within speech_buf, -1 = none
  float soft_cut_best_prob = 1.0f;
};

// ─────────────────────────────────────────────────────
// ASR segment processing — called from worker thread
// ─────────────────────────────────────────────────────
static bool process_segment(rs_context_t *asr_ctx,
                            const std::vector<float> &pcm, float start_s,
                            float end_s, int seg_index, bool two_pass,
                            bool is_partial) {
  if (pcm.empty())
    return false;

  float seg_dur = (float)pcm.size() / 16000;
  auto t0 = std::chrono::steady_clock::now();

  // Pad short segments (< 1s) with trailing zeros so the ASR processor
  // has enough audio to run inference (Process() requires >= chunk_size_samples).
  const int MIN_ASR_SAMPLES = 16000; // 1 second at 16 kHz
  std::vector<float> pcm_padded;
  const float *pcm_ptr = pcm.data();
  int32_t pcm_len = (int32_t)pcm.size();
  if (pcm_len < MIN_ASR_SAMPLES) {
    pcm_padded = pcm;
    pcm_padded.resize(MIN_ASR_SAMPLES, 0.f);
    pcm_ptr = pcm_padded.data();
    pcm_len = MIN_ASR_SAMPLES;
  }

  rs_error_t err = rs_push_audio(asr_ctx, pcm_ptr, pcm_len);
  if (err != RS_OK) {
    LOG_ERROR("rs_push_audio failed for seg #%d", seg_index);
    rs_reset(asr_ctx);
    return false;
  }

  // ── First pass: CTC decode (fast, no LLM) ──
  if (two_pass || is_partial)
    rs_set_use_llm(asr_ctx, false);

  int32_t status = rs_process(asr_ctx);
  const char *text = nullptr;
  if (status > 0)
    text = rs_get_text_output(asr_ctx);

  // ANSI: \r returns to col 0, \033[K clears to end of line — used to
  // wipe any prior streaming/partial/VAD progress line cleanly.
  const char *CLEAR_LINE = Color::enabled ? "\r\033[K" : "\r";

  if (is_partial) {
    // Streaming partial result — single overwriting line, indented + dim.
    std::cout << CLEAR_LINE << "  " << Color::c(Color::Grey) << "["
              << format_timestamp(end_s) << "] " << Color::c(Color::Reset)
              << Color::c(Color::Dim)
              << (text && text[0] ? text : "(listening)") << " …"
              << Color::c(Color::Reset) << std::flush;
  } else if (two_pass) {
    // Interim (CTC) result — dimmed, overwrites the partial line.
    std::cout << CLEAR_LINE << "  " << Color::c(Color::Grey) << "["
              << format_timestamp(start_s) << " → " << format_timestamp(end_s)
              << "] " << Color::c(Color::Reset) << Color::c(Color::Dim)
              << (text && text[0] ? text : "(silence)")
              << Color::c(Color::Reset) << std::flush;

    // ── Second pass: LLM rescore ──
    auto t_llm0 = std::chrono::steady_clock::now();
    rs_set_use_llm(asr_ctx, true);
    int32_t status2 = rs_redecode(asr_ctx);
    const char *final_text = nullptr;
    if (status2 > 0)
      final_text = rs_get_text_output(asr_ctx);

    auto t1 = std::chrono::steady_clock::now();
    float elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
        1e6f;
    float llm_elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t_llm0)
            .count() /
        1e6f;
    float rtf = seg_dur > 0 ? elapsed / seg_dur : 0.f;

    // Final result — overwrite interim with bold text.
    std::cout << CLEAR_LINE << Color::c(Color::Cyan) << "["
              << format_timestamp(start_s) << " → " << format_timestamp(end_s)
              << "] " << Color::c(Color::Reset) << Color::c(Color::Bold)
              << (final_text && final_text[0] ? final_text : "(no speech)")
              << Color::c(Color::Reset) << "\n";

    if (g_verbose) {
      std::cout << Color::c(Color::Grey) << "  seg #" << seg_index << " · "
                << std::fixed << std::setprecision(2) << seg_dur << "s · total "
                << elapsed << "s (LLM " << std::setprecision(2) << llm_elapsed
                << "s) · RTF " << std::setprecision(3) << rtf
                << Color::c(Color::Reset) << "\n";
    }
    std::cout.flush();
  } else {
    auto t1 = std::chrono::steady_clock::now();
    float elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
        1e6f;
    float rtf = seg_dur > 0 ? elapsed / seg_dur : 0.f;

    if (text && text[0]) {
      std::cout << CLEAR_LINE << Color::c(Color::Cyan) << "["
                << format_timestamp(start_s) << " → "
                << format_timestamp(end_s) << "] " << Color::c(Color::Reset)
                << Color::c(Color::Bold) << text << Color::c(Color::Reset)
                << "\n";
    } else if (g_verbose) {
      // Suppress empty-segment noise unless verbose.
      std::cout << CLEAR_LINE << Color::c(Color::Grey) << "["
                << format_timestamp(start_s) << " → "
                << format_timestamp(end_s) << "] " << Color::c(Color::Dim)
                << "(no speech recognized)" << Color::c(Color::Reset) << "\n";
    } else {
      // Quietly wipe any leftover partial/VAD line.
      std::cout << CLEAR_LINE;
    }

    if (g_verbose) {
      std::cout << Color::c(Color::Grey) << "  seg #" << seg_index << " · "
                << std::fixed << std::setprecision(2) << seg_dur << "s · ASR "
                << elapsed << "s · RTF " << std::setprecision(3) << rtf
                << Color::c(Color::Reset) << "\n";
    }
    std::cout.flush();
  }

  rs_reset(asr_ctx);
  return true;
}

// ─────────────────────────────────────────────────────
// ASR worker thread — owns the ASR context so that GPU
// resources (Metal / CUDA) are created on this thread.
// Pops segments from the queue and transcribes them.
// ─────────────────────────────────────────────────────
static void asr_worker(const std::string &model_path, int n_threads,
                       bool use_gpu, bool two_pass, bool ctc_precheck,
                       SpeechSegmentQueue &queue,
                       std::atomic<int> &seg_processed,
                       std::atomic<bool> &worker_done,
                       std::atomic<bool> &worker_ready) {
  rs_init_params_t asr_params = rs_default_params();
  asr_params.model_path = model_path.c_str();
  asr_params.n_threads = n_threads;
  asr_params.use_gpu = use_gpu;

  rs_context_t *asr_ctx = rs_init_from_file(asr_params);
  if (!asr_ctx) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("ASR worker: init failed: %s", err.message);
    worker_done = true;
    return;
  }
  worker_ready = true;
  rs_set_ctc_precheck(asr_ctx, ctc_precheck);

  // Merge consecutive short segments that are close together.
  // Short utterances (< 1.2s) with small gaps (< 0.5s) are often
  // VAD fragmentation artifacts — merging them gives ASR enough
  // context to produce a result.
  const float SHORT_SEG_S = 1.2f;
  const float MERGE_GAP_S = 0.5f;
  const int SAMPLE_RATE = 16000;

  SpeechSegmentQueue::Segment seg;
  while (queue.pop(seg)) {
    float seg_dur = seg.end_s - seg.start_s;
    if (seg_dur < SHORT_SEG_S) {
      // Try to merge with upcoming segments
      SpeechSegmentQueue::Segment next;
      while (queue.try_pop(next)) {
        float gap = next.start_s - seg.end_s;
        if (gap <= MERGE_GAP_S) {
          // Fill gap with zeros and append
          int gap_samples = (int)(gap * SAMPLE_RATE);
          seg.pcm.insert(seg.pcm.end(), gap_samples, 0.f);
          seg.pcm.insert(seg.pcm.end(), next.pcm.begin(), next.pcm.end());
          seg.end_s = next.end_s;
          seg_dur = seg.end_s - seg.start_s;
          if (seg_dur >= SHORT_SEG_S)
            break; // enough audio now
        } else {
          // Gap too large — put back and stop merging
          queue.push_front(std::move(next));
          break;
        }
      }
    }

    process_segment(asr_ctx, seg.pcm, seg.start_s, seg.end_s, seg.index,
                    two_pass, seg.is_partial);
    if (!seg.is_partial)
      seg_processed++;

    if (queue.check_and_clear_slow_warning()) {
      std::cerr << Color::c(Color::Yellow)
                << "\n⚠  ASR is falling behind — real-time transcription may "
                   "not be achievable.\n"
                << Color::c(Color::Reset) << std::flush;
    }
  }

  rs_free(asr_ctx);
  rs_clear_error();
  worker_done = true;
}

// ─────────────────────────────────────────────────────
// FireRedVAD streaming state.
//
// FireRedVAD's postprocessor (4-state DFSMN state machine) emits
// `is_speech_end` events with absolute (start_frame, end_frame) at
// 100 Hz. We keep a rolling PCM buffer (with global sample offset)
// and slice out segments by frame -> sample mapping.
// ─────────────────────────────────────────────────────
struct FireredStreamState {
  std::deque<float> audio_buf;          // rolling PCM
  int64_t audio_buf_start_sample = 0;   // global sample index of audio_buf[0]
  int64_t total_samples_read = 0;
  // Two-pass partial streaming state.
  // current_seg_idx != 0 while a speech segment is in progress (between
  // is_speech_start and is_speech_end). last_partial_sample is the global
  // sample index at which the most recent partial was emitted.
  int current_seg_idx = 0;
  int64_t last_partial_sample = 0;
};

static bool process_vad_chunk_firered(
    const float *chunk, int chunk_samples, FireRedVadModel &vad_model,
    FireRedVadState &vad_state, FireredStreamState &state,
    const OnlineArgs &args, SpeechSegmentQueue &queue,
    std::atomic<int> &seg_counter) {
  const int SAMPLE_RATE = 16000;
  const int SAMPLES_PER_FRAME = 160;  // 10 ms shift @ 16 kHz

  // Append to rolling PCM buffer.
  state.audio_buf.insert(state.audio_buf.end(), chunk, chunk + chunk_samples);
  state.total_samples_read += chunk_samples;

  std::vector<float> pcm_chunk(chunk, chunk + chunk_samples);
  auto results = vad_model.DetectStreamingChunk(pcm_chunk, vad_state);

  // Per-frame progress in verbose mode.
  if (g_verbose && !results.empty()) {
    const auto &last = results.back();
    float pos_s = (float)state.total_samples_read / SAMPLE_RATE;
    if (last.is_speech) {
      std::cout << "\r" << Color::c(Color::Green) << "["
                << format_timestamp(pos_s) << "] "
                << "▌SPEECH  p=" << std::fixed << std::setprecision(3)
                << last.smoothed_prob << Color::c(Color::Reset) << "  "
                << std::flush;
    } else {
      std::cout << "\r" << Color::c(Color::Grey) << "["
                << format_timestamp(pos_s) << "] "
                << "·silence p=" << std::fixed << std::setprecision(3)
                << last.smoothed_prob << Color::c(Color::Reset) << "  "
                << std::flush;
    }
  }

  // Apply explicit speech_pad_ms (rewind start) and trailing pad (extend end)
  // for consistency with the Silero path's CLI options.
  const int pad_pre_samples = (args.speech_pad_ms * SAMPLE_RATE) / 1000;

  int64_t latest_extracted_end_sample = 0;
  for (const auto &r : results) {
    if (r.is_speech_start) {
      // New segment begins — assign a stable index that all partials and the
      // final will share. The queue dedups partials when a final with the
      // same index arrives.
      state.current_seg_idx = ++seg_counter;
      // Anchor the partial cadence to the segment's true start so the first
      // partial fires after partial_ms of speech, not immediately.
      if (r.speech_start_frame > 0) {
        state.last_partial_sample =
            (int64_t)(r.speech_start_frame - 1) * SAMPLES_PER_FRAME;
      } else {
        state.last_partial_sample = state.total_samples_read;
      }
    }

    if (!r.is_speech_end) continue;
    if (r.speech_start_frame <= 0 || r.speech_end_frame <= 0) continue;

    int64_t start_sample =
        (int64_t)(r.speech_start_frame - 1) * SAMPLES_PER_FRAME;
    int64_t end_sample = (int64_t)r.speech_end_frame * SAMPLES_PER_FRAME;
    start_sample = std::max<int64_t>(0, start_sample - pad_pre_samples);

    int64_t buf_start = start_sample - state.audio_buf_start_sample;
    int64_t buf_end = end_sample - state.audio_buf_start_sample;
    if (buf_start < 0) buf_start = 0;
    if (buf_end > (int64_t)state.audio_buf.size())
      buf_end = (int64_t)state.audio_buf.size();
    if (buf_end <= buf_start) continue;

    std::vector<float> seg_pcm;
    seg_pcm.reserve((size_t)(buf_end - buf_start));
    auto it_begin = state.audio_buf.begin() + buf_start;
    auto it_end = state.audio_buf.begin() + buf_end;
    seg_pcm.insert(seg_pcm.end(), it_begin, it_end);

    // Reuse the index assigned at is_speech_start so any prior partials are
    // superseded by this final. If we somehow missed a start event, mint one.
    int idx = state.current_seg_idx != 0 ? state.current_seg_idx : ++seg_counter;
    state.current_seg_idx = 0;
    state.last_partial_sample = 0;
    float start_s = (float)start_sample / SAMPLE_RATE;
    float end_s = (float)end_sample / SAMPLE_RATE;
    queue.push({std::move(seg_pcm), start_s, end_s, idx, false});

    latest_extracted_end_sample =
        std::max(latest_extracted_end_sample, end_sample);
  }

  // ---- Partial emission for --two-pass mode ----
  // After processing the chunk's events, if we're still inside an active
  // segment and partial_ms worth of audio has accumulated since the last
  // partial, emit a CTC-fast partial covering [speech_start .. now].
  if (args.two_pass && args.partial_ms > 0 && state.current_seg_idx != 0 &&
      vad_state.last_speech_start_frame > 0) {
    const int64_t partial_interval_samples =
        (int64_t)args.partial_ms * SAMPLE_RATE / 1000;
    int64_t now_sample = state.total_samples_read;
    if (partial_interval_samples > 0 &&
        now_sample - state.last_partial_sample >= partial_interval_samples) {
      constexpr float PARTIAL_BACKOFF_S = 5.0f;
      if (queue.duration() < PARTIAL_BACKOFF_S) {
        int64_t start_sample =
            (int64_t)(vad_state.last_speech_start_frame - 1) * SAMPLES_PER_FRAME;
        start_sample = std::max<int64_t>(0, start_sample - pad_pre_samples);

        int64_t buf_start = start_sample - state.audio_buf_start_sample;
        int64_t buf_end = now_sample - state.audio_buf_start_sample;
        if (buf_start < 0) buf_start = 0;
        if (buf_end > (int64_t)state.audio_buf.size())
          buf_end = (int64_t)state.audio_buf.size();
        if (buf_end > buf_start) {
          std::vector<float> seg_pcm(state.audio_buf.begin() + buf_start,
                                     state.audio_buf.begin() + buf_end);
          float start_s = (float)start_sample / SAMPLE_RATE;
          float end_s = (float)now_sample / SAMPLE_RATE;
          queue.push({std::move(seg_pcm), start_s, end_s,
                      state.current_seg_idx, true});
          state.last_partial_sample = now_sample;
        }
      }
    }
  }

  // Trim the rolling buffer. When in active speech we must keep audio back
  // to the current segment's start (potentially up to max_speech_frame = 20 s
  // in default config). When idle we only need a small head-of-segment pad.
  int64_t keep_from_sample = state.total_samples_read;
  if (vad_state.last_speech_start_frame > 0 &&
      (vad_state.fsm_state == VadFsmState::SPEECH ||
       vad_state.fsm_state == VadFsmState::POSSIBLE_SPEECH ||
       vad_state.fsm_state == VadFsmState::POSSIBLE_SILENCE)) {
    int64_t cur_seg_start =
        (int64_t)(vad_state.last_speech_start_frame - 1) * SAMPLES_PER_FRAME;
    keep_from_sample = std::min(keep_from_sample, cur_seg_start);
  } else {
    // Idle: keep a generous head pad (covers pad_start_frame + smoothing).
    const int64_t IDLE_KEEP_SAMPLES = SAMPLE_RATE; // 1 s
    keep_from_sample =
        std::max<int64_t>(0, state.total_samples_read - IDLE_KEEP_SAMPLES);
  }
  // Also account for speech_pad_ms preroll.
  keep_from_sample =
      std::max<int64_t>(0, keep_from_sample - pad_pre_samples);
  // Never drop audio we just emitted past, redundantly safe.
  keep_from_sample =
      std::max(keep_from_sample, latest_extracted_end_sample);

  if (keep_from_sample > state.audio_buf_start_sample) {
    int64_t to_drop = keep_from_sample - state.audio_buf_start_sample;
    if (to_drop > (int64_t)state.audio_buf.size())
      to_drop = (int64_t)state.audio_buf.size();
    state.audio_buf.erase(state.audio_buf.begin(),
                          state.audio_buf.begin() + (size_t)to_drop);
    state.audio_buf_start_sample += to_drop;
  }

  return true;
}

// Emit any unfinished trailing speech at EOF. Mirrors the "Flush remaining
// speech" behavior of the Silero path.
static void flush_firered_tail(FireRedVadModel & /*vad_model*/,
                               FireRedVadState &vad_state,
                               FireredStreamState &state,
                               SpeechSegmentQueue &queue,
                               std::atomic<int> &seg_counter,
                               const OnlineArgs &args) {
  const int SAMPLE_RATE = 16000;
  const int SAMPLES_PER_FRAME = 160;

  const bool in_speech =
      (vad_state.fsm_state == VadFsmState::SPEECH ||
       vad_state.fsm_state == VadFsmState::POSSIBLE_SPEECH ||
       vad_state.fsm_state == VadFsmState::POSSIBLE_SILENCE);
  if (!in_speech || vad_state.last_speech_start_frame <= 0) return;

  const int pad_pre_samples = (args.speech_pad_ms * SAMPLE_RATE) / 1000;
  int64_t start_sample =
      (int64_t)(vad_state.last_speech_start_frame - 1) * SAMPLES_PER_FRAME;
  start_sample = std::max<int64_t>(0, start_sample - pad_pre_samples);
  int64_t end_sample = state.total_samples_read;

  int64_t buf_start = start_sample - state.audio_buf_start_sample;
  int64_t buf_end = end_sample - state.audio_buf_start_sample;
  if (buf_start < 0) buf_start = 0;
  if (buf_end > (int64_t)state.audio_buf.size())
    buf_end = (int64_t)state.audio_buf.size();
  if (buf_end <= buf_start) return;

  std::vector<float> seg_pcm(state.audio_buf.begin() + buf_start,
                             state.audio_buf.begin() + buf_end);
  int idx = state.current_seg_idx != 0 ? state.current_seg_idx : ++seg_counter;
  state.current_seg_idx = 0;
  state.last_partial_sample = 0;
  float start_s = (float)start_sample / SAMPLE_RATE;
  float end_s = (float)end_sample / SAMPLE_RATE;
  queue.push({std::move(seg_pcm), start_s, end_s, idx, false});
}

// ─────────────────────────────────────────────────────
// Process a single VAD chunk — pushes completed speech
// segments to the queue (does NOT run ASR inline).
//
// Improvements over the baseline:
//   A. Pre-roll buffer — audio during silence is kept in a sliding
//      window so speech onset isn't clipped.
//   B. Adaptive silence threshold — dynamically adjusted based on
//      recent segment durations to match the speaker's pace.
//   C. EMA probability smoothing — reduces jitter / fragmentation
//      from single-frame probability spikes or dips.
// ─────────────────────────────────────────────────────
static bool
process_vad_chunk(const float *chunk, int chunk_samples,
                  SileroVadModel &vad_model, SileroVadState &vad_state,
                  ggml_backend_sched_t vad_sched, StreamState &state,
                  const OnlineArgs &args,
                  SpeechSegmentQueue &queue, std::atomic<int> &seg_counter) {
  const int SAMPLE_RATE = 16000;
  const int VAD_WINDOW = 512;

  std::vector<float> vad_in(chunk, chunk + chunk_samples);
  if (chunk_samples < VAD_WINDOW)
    vad_in.insert(vad_in.end(), VAD_WINDOW - chunk_samples, 0.f);

  ggml_backend_sched_reset(vad_sched);
  bool vad_ok = vad_model.Encode(vad_in, vad_state, vad_sched);
  float prob = vad_ok ? vad_model.GetSpeechProbability(vad_state) : 0.f;

  // --- C. EMA probability smoothing ---
  if (args.prob_smooth > 0.0f) {
    state.smoothed_prob = args.prob_smooth * prob +
                          (1.0f - args.prob_smooth) * state.smoothed_prob;
  } else {
    state.smoothed_prob = prob;
  }

  // Use smoothed probability for threshold decisions.
  // Hysteresis: lower exit threshold once in speech to avoid splitting
  // on brief probability dips (matches Silero VADIterator logic).
  float enter_threshold = args.vad_threshold;
  float exit_threshold = std::max(0.01f, args.vad_threshold - 0.15f);
  bool is_speech = state.in_speech
                       ? (state.smoothed_prob >= exit_threshold)
                       : (state.smoothed_prob >= enter_threshold);

  // --- B. Adaptive silence threshold ---
  // Recompute SILENCE_FRAMES each call so the adaptive value takes effect.
  int silence_frames_limit =
      (int)(state.adaptive_silence_ms * SAMPLE_RATE) / (1000 * VAD_WINDOW);
  if (silence_frames_limit < 1)
    silence_frames_limit = 1;

  // Per-frame progress indicator — only shown in verbose mode.
  // Otherwise we keep the terminal quiet between segments so that
  // partial / final transcripts remain visually clean.
  if (g_verbose) {
    float pos_s = (float)state.total_samples_read / SAMPLE_RATE;
    if (is_speech) {
      std::cout << "\r" << Color::c(Color::Green) << "["
                << format_timestamp(pos_s) << "] "
                << "▌SPEECH  p=" << std::fixed << std::setprecision(3)
                << state.smoothed_prob << Color::c(Color::Reset) << "  "
                << std::flush;
    } else {
      std::cout << "\r" << Color::c(Color::Grey) << "["
                << format_timestamp(pos_s) << "] "
                << "·silence p=" << std::fixed << std::setprecision(3)
                << state.smoothed_prob << Color::c(Color::Reset) << "  "
                << std::flush;
    }
  }

  // --- A. Pre-roll buffer management ---
  // During silence, keep audio in a sliding window.
  if (!state.in_speech) {
    state.preroll_buf.insert(state.preroll_buf.end(), chunk,
                             chunk + chunk_samples);
    // Trim to max size
    while ((int)state.preroll_buf.size() > state.preroll_max_samples) {
      int overshoot = (int)state.preroll_buf.size() - state.preroll_max_samples;
      int to_drop = std::min(overshoot, chunk_samples);
      for (int i = 0; i < to_drop; ++i)
        state.preroll_buf.pop_front();
    }
  }

  if (is_speech) {
    state.silence_frames = 0;
    if (!state.in_speech) {
      state.in_speech = true;
      if (args.two_pass) {
        state.current_segment_index = ++seg_counter;
        state.last_partial_samples = 0;
      }

      // Flush pre-roll buffer into speech_buf so the segment starts
      // with audio from *before* the VAD trigger point.
      int preroll_samples = (int)state.preroll_buf.size();
      if (preroll_samples > 0) {
        state.speech_buf.clear();
        state.speech_buf.insert(state.speech_buf.end(),
                                state.preroll_buf.begin(),
                                state.preroll_buf.end());
        state.preroll_buf.clear();
      } else {
        state.speech_buf.clear();
      }

      // speech_start_offset accounts for the pre-roll audio we just prepended.
      state.speech_start_offset = std::max(
          0, state.total_samples_read - chunk_samples - preroll_samples);

      // Also apply the explicit speech_pad_ms: rewind the start offset
      // further so the timestamp includes the configured padding.
      int extra_pad_samples = (args.speech_pad_ms * SAMPLE_RATE) / 1000;
      state.speech_start_offset =
          std::max(0, state.speech_start_offset - extra_pad_samples);

      // Fresh segment — reset soft-cut tracking.
      state.soft_cut_best_offset = -1;
      state.soft_cut_best_prob = 1.0f;
    }
    state.speech_buf.insert(state.speech_buf.end(), chunk,
                            chunk + chunk_samples);

    if (args.two_pass && args.partial_ms > 0) {
      const int partial_interval_samples =
          (args.partial_ms * SAMPLE_RATE) / 1000;
      if (partial_interval_samples > 0 &&
          (int)state.speech_buf.size() - state.last_partial_samples >=
              partial_interval_samples) {
        // Throttle partials when the ASR queue is backed up.  Partials are
        // best-effort and re-decoding the full speech_buf on every push
        // starves the worker when speech is fast and continuous.  We do NOT
        // advance last_partial_samples on skip, so as soon as the queue
        // drains we push the most up-to-date partial available.
        constexpr float PARTIAL_BACKOFF_S = 5.0f;
        if (queue.duration() < PARTIAL_BACKOFF_S) {
          float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
          float end_s =
              (float)(state.total_samples_read + chunk_samples) / SAMPLE_RATE;
          queue.push({state.speech_buf, start_s, end_s,
                      state.current_segment_index, true});
          state.last_partial_samples = (int)state.speech_buf.size();
        } else {
          LOG_DEBUG("partial skipped: queue depth %.1fs >= %.1fs",
                    queue.duration(), PARTIAL_BACKOFF_S);
        }
      }
    }
  } else {
    if (state.in_speech) {
      state.speech_buf.insert(state.speech_buf.end(), chunk,
                              chunk + chunk_samples);
      ++state.silence_frames;

      if (state.silence_frames >= silence_frames_limit) {
        int idx = args.two_pass ? state.current_segment_index : ++seg_counter;
        float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
        float end_s = (float)state.total_samples_read / SAMPLE_RATE;
        float seg_dur = end_s - start_s;

        // --- B. Update adaptive silence threshold ---
        // Short utterances → lower threshold (faster response).
        // Long utterances  → raise threshold (avoid premature cuts).
        const float ADAPT_ALPHA = 0.15f;
        if (seg_dur < 1.0f) {
          state.adaptive_silence_ms =
              std::max(300.0f, state.adaptive_silence_ms * (1.0f - ADAPT_ALPHA) +
                                   300.0f * ADAPT_ALPHA);
        } else if (seg_dur > 5.0f) {
          state.adaptive_silence_ms =
              std::min(1200.0f, state.adaptive_silence_ms * (1.0f - ADAPT_ALPHA) +
                                    1200.0f * ADAPT_ALPHA);
        } else {
          // Drift back toward the user-configured baseline.
          state.adaptive_silence_ms =
              state.adaptive_silence_ms * (1.0f - ADAPT_ALPHA * 0.5f) +
              state.base_silence_ms * ADAPT_ALPHA * 0.5f;
        }

        queue.push({std::move(state.speech_buf), start_s, end_s, idx, false});

        state.in_speech = false;
        state.current_segment_index = 0;
        state.last_partial_samples = 0;
        state.speech_buf.clear();
        state.silence_frames = 0;
      }
    }
  }

  // --- D. Soft max-segment cap ---
  // Placed AFTER the is_speech / silence branches so it sees the smoothed
  // prob from both paths — especially the deep dips that VAD classifies as
  // brief silence-within-speech (prob < exit_threshold).  In long continuous
  // speech (newscasters), partials re-decode the whole buffer and the final
  // CTC+LLM rescore on a 30s+ buffer stalls the worker, so once the segment
  // enters the search window approaching the cap we hunt for a VAD prob dip
  // (an inter-word gap) and cut there instead of chopping mid-word.
  if (state.in_speech && args.max_segment_ms > 0) {
    const int max_samples = (args.max_segment_ms * SAMPLE_RATE) / 1000;
    int search_ms = args.max_segment_ms / 4; // 25% of max
    if (search_ms < 1000) search_ms = 1000;
    if (search_ms > 2000) search_ms = 2000;
    const int search_samples = (search_ms * SAMPLE_RATE) / 1000;
    const int search_start = std::max(0, max_samples - search_samples);
    const float soft_threshold =
        std::max(0.05f, args.vad_threshold - 0.2f);
    const int seg_size = (int)state.speech_buf.size();

    if (seg_size >= search_start) {
      // Track the lowest-prob frame seen in the search window.
      if (state.smoothed_prob < state.soft_cut_best_prob) {
        state.soft_cut_best_prob = state.smoothed_prob;
        state.soft_cut_best_offset = seg_size; // end of just-inserted chunk
      }

      int cut_at = -1;
      const char *cut_reason = nullptr;
      bool is_dip_cut = false;
      if (state.smoothed_prob < soft_threshold) {
        cut_at = seg_size;
        cut_reason = "dip";
        is_dip_cut = true;
      } else if (seg_size >= max_samples) {
        cut_at = (state.soft_cut_best_offset > 0)
                     ? state.soft_cut_best_offset
                     : seg_size;
        cut_reason = (cut_at == seg_size) ? "cap" : "best";
      }

      if (cut_at > 0) {
        int idx =
            args.two_pass ? state.current_segment_index : ++seg_counter;
        float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
        float end_s =
            (float)(state.speech_start_offset + cut_at) / SAMPLE_RATE;

        std::vector<float> final_buf(state.speech_buf.begin(),
                                     state.speech_buf.begin() + cut_at);
        std::vector<float> tail(state.speech_buf.begin() + cut_at,
                                state.speech_buf.end());

        LOG_DEBUG("max-seg cut at %.2fs/%.2fs (%s, prob=%.3f, tail=%dms)",
                  (float)cut_at / SAMPLE_RATE,
                  (float)seg_size / SAMPLE_RATE, cut_reason,
                  state.soft_cut_best_prob,
                  (int)tail.size() * 1000 / SAMPLE_RATE);

        queue.push({std::move(final_buf), start_s, end_s, idx, false});

        if (is_dip_cut) {
          // Clean break — treat the dip as a natural endpoint.  Drop the
          // (likely empty) tail and let the normal in_speech entry path
          // start a new segment with preroll when speech resumes.
          state.speech_buf.clear();
          state.in_speech = false;
          state.current_segment_index = 0;
        } else {
          // Contiguous cap/best cut — keep the tail as the start of the
          // next segment (no preroll, no silence transition).
          state.speech_buf = std::move(tail);
          state.speech_start_offset += cut_at;
          if (args.two_pass) {
            state.current_segment_index = ++seg_counter;
          }
        }
        state.silence_frames = 0;
        state.last_partial_samples = 0;
        state.soft_cut_best_offset = -1;
        state.soft_cut_best_prob = 1.0f;
      }
    }
  }

  state.total_samples_read += chunk_samples;
  return true;
}

// ─────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────
int main(int argc, char **argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif

  OnlineArgs args;
  if (!parse_args(argc, argv, args))
    return 1;

  // Allow enabling verbose mode via env var too (e.g. RS_VERBOSE=1).
  if (!g_verbose) {
    const char *v = std::getenv("RS_VERBOSE");
    if (v && (*v == '1' || *v == 't' || *v == 'T' || *v == 'y' || *v == 'Y'))
      g_verbose = true;
  }

  // Silence framework + ggml chatter unless --verbose is set.  Without this,
  // model loading prints dozens of internal lines that visually drown out the
  // partial/final transcripts we want to highlight.
  if (!g_verbose) {
    rs_log_set_level(RSLogLevel::RS_LOG_LEVEL_WARN);
    ggml_log_set([](enum ggml_log_level, const char *, void *) {}, nullptr);
  }

  const int SAMPLE_RATE = 16000;
  const int VAD_WINDOW = 512;

  // Detect terminal color support
  const char *term = std::getenv("TERM");
  Color::enabled = (term != nullptr && std::string(term) != "dumb");

  // ── 1. Load VAD model ──────────────────────────────────
  LOG_INFO("Loading VAD model: %s", args.vad_model);

  VadBackend vad_backend;
  if (!vad_backend.init()) {
    LOG_ERROR("Failed to init VAD backend");
    return 1;
  }

  gguf_context *vad_gguf = nullptr;
  ggml_context *vad_data = load_gguf_weights(args.vad_model, &vad_gguf);
  if (!vad_data)
    return 1;

  // Detect VAD architecture from GGUF metadata.
  // Silero VAD: "silero-vad"  ·  FireRedVAD: "firered-vad"
  const std::string vad_arch = read_gguf_arch(vad_gguf);
  const bool use_firered =
      (vad_arch == "firered-vad" || vad_arch == "firered_vad");
  LOG_INFO("VAD arch: %s",
           vad_arch.empty() ? "<unknown, defaulting to silero-vad>"
                            : vad_arch.c_str());

  std::shared_ptr<SileroVadModel> silero_model;
  std::shared_ptr<SileroVadState> silero_state;
  std::shared_ptr<FireRedVadModel> firered_model;
  std::shared_ptr<FireRedVadState> firered_state;

  if (use_firered) {
    firered_model = std::make_shared<FireRedVadModel>();
    if (!firered_model->LoadDirect(vad_data, vad_gguf, vad_backend.backend)) {
      LOG_ERROR("Failed to load FireRedVAD weights");
      return 1;
    }
    firered_model->SetThreshold(args.vad_threshold);
    LOG_INFO("VAD threshold set to: %.2f", args.vad_threshold);
    firered_state = std::dynamic_pointer_cast<FireRedVadState>(
        firered_model->CreateState());
    if (!firered_state) {
      LOG_ERROR("Failed to create FireRedVAD state");
      return 1;
    }
  } else {
    silero_model = std::make_shared<SileroVadModel>();
    if (!silero_model->LoadDirect(vad_data, vad_gguf, vad_backend.backend)) {
      LOG_ERROR("Failed to load Silero VAD weights");
      return 1;
    }
    silero_model->SetThreshold(args.vad_threshold);
    LOG_INFO("VAD threshold set to: %.2f", args.vad_threshold);
    silero_state = std::dynamic_pointer_cast<SileroVadState>(
        silero_model->CreateState());
    if (!silero_state) {
      LOG_ERROR("Failed to create VAD state");
      return 1;
    }
  }

  // ── VAD warmup: run a few silent frames to initialize internal state ──
  LOG_INFO("Warming up VAD with silent frames...");
  if (use_firered) {
    // FireRedVAD: 320ms of silence at 16kHz = 5120 samples (32 frames).
    std::vector<float> warmup_silent(5120, 0.0f);
    firered_model->DetectStreamingChunk(warmup_silent, *firered_state);
    firered_model->Reset(*firered_state);
  } else {
    std::vector<float> warmup_silent(VAD_WINDOW, 0.0f);
    for (int i = 0; i < 3; ++i) {
      silero_model->Encode(warmup_silent, *silero_state, vad_backend.sched);
    }
  }
  LOG_INFO("VAD warmup complete.");

  // ── 2. Start ASR worker (creates ASR context on its own thread) ──
  SpeechSegmentQueue seg_queue(/*max_dur_s=*/60.0f, /*warn_dur_s=*/15.0f);
  std::atomic<int> seg_counter{0};
  std::atomic<int> seg_processed{0};
  std::atomic<bool> worker_done{false};
  std::atomic<bool> worker_ready{false};

  std::string model_path = args.asr_model;
  std::thread worker(asr_worker, model_path, args.n_threads, args.use_gpu,
                     args.two_pass, args.ctc_precheck,
                     std::ref(seg_queue), std::ref(seg_processed),
                     std::ref(worker_done), std::ref(worker_ready));

  // Wait for worker to finish init or fail (max 15 s for model load)
  {
    int wait_ms = 0;
    while (!worker_ready && !worker_done && wait_ms < 15000) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      wait_ms += 100;
    }
    if (worker_done) {
      LOG_ERROR("ASR worker failed to initialize");
      worker.join();
      return 1;
    }
    if (!worker_ready) {
      LOG_ERROR("ASR worker init timed out");
      seg_queue.set_done();
      worker.join();
      return 1;
    }
  }

  // ── 3. Streaming loop ──────────────────────────────────
  StreamState state;
  state.preroll_max_samples = (args.preroll_ms * SAMPLE_RATE) / 1000;
  state.prob_smooth_alpha = args.prob_smooth;
  state.base_silence_ms = (float)args.silence_ms;
  state.adaptive_silence_ms = (float)args.silence_ms;
  FireredStreamState firered_stream_state;
  auto t_start = std::chrono::steady_clock::now();

  // Dispatch: per-chunk step + EOF flush. Both VAD backends share the
  // mic/wav streaming loops via these closures.
  std::function<void(const float *, int)> vad_step;
  std::function<void()> vad_flush;
  std::function<int64_t()> vad_total_samples;

  if (use_firered) {
    vad_step = [&](const float *chunk, int n) {
      process_vad_chunk_firered(chunk, n, *firered_model, *firered_state,
                                firered_stream_state, args, seg_queue,
                                seg_counter);
    };
    vad_flush = [&]() {
      flush_firered_tail(*firered_model, *firered_state, firered_stream_state,
                         seg_queue, seg_counter, args);
    };
    vad_total_samples = [&]() {
      return firered_stream_state.total_samples_read;
    };
  } else {
    vad_step = [&](const float *chunk, int n) {
      process_vad_chunk(chunk, n, *silero_model, *silero_state,
                        vad_backend.sched, state, args, seg_queue, seg_counter);
    };
    vad_flush = [&]() {
      if (state.speech_buf.empty()) return;
      int idx = ++seg_counter;
      float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
      float end_s = (float)state.total_samples_read / SAMPLE_RATE;
      if (g_verbose) {
        std::cout << "\n[EOF] Flushing remaining speech ("
                  << state.speech_buf.size() << " samples)";
      }
      seg_queue.push({std::move(state.speech_buf), start_s, end_s, idx});
    };
    vad_total_samples = [&]() {
      return (int64_t)state.total_samples_read;
    };
  }

  LOG_INFO("─────────────────────────────────────────────────────────");
  char max_seg_buf[32];
  if (args.max_segment_ms > 0)
    snprintf(max_seg_buf, sizeof(max_seg_buf), "  max-seg=%dms",
             args.max_segment_ms);
  else
    max_seg_buf[0] = '\0';
  if (args.use_mic) {
    LOG_INFO("Live mode: microphone   vad=%.2f  silence=%dms%s%s",
             args.vad_threshold, args.silence_ms, max_seg_buf,
             args.two_pass ? "  [2-pass]" : "");
  } else {
    LOG_INFO("File mode: %s   vad=%.2f  silence=%dms%s%s", args.wav_path,
             args.vad_threshold, args.silence_ms, max_seg_buf,
             args.two_pass ? "  [2-pass]" : "");
  }
  LOG_INFO("─────────────────────────────────────────────────────────");

  if (args.use_mic) {
    // ── Microphone mode (miniaudio) ───────────────────
    // miniaudio captures on its own internal thread and
    // writes to the ring buffer via data_callback.
    // The main thread reads from the ring buffer → VAD.
    AudioRingBuffer audio_ring(160000);

    MicCapture mic;
    if (!mic.start(SAMPLE_RATE, &audio_ring)) {
      seg_queue.set_done();
      worker.join();
      return 1;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::cout << Color::c(Color::Green)
              << "\n🎤 Listening... (press Ctrl+C to stop)\n"
              << Color::c(Color::Reset) << std::flush;

    // VAD loop — read from ring buffer in VAD_WINDOW-sized chunks
    std::vector<float> vad_buf(VAD_WINDOW);
    while (!g_stop_flag && mic.running) {
      size_t n = audio_ring.read(vad_buf.data(), VAD_WINDOW);
      if (n == 0)
        break;

      vad_step(vad_buf.data(), (int)n);

      // Warn if ring buffer is overrunning (VAD too slow)
      static int overrun_check = 0;
      if (++overrun_check >= 100) { // ~ every 3.2 s
        overrun_check = 0;
        size_t ov = audio_ring.overruns();
        if (ov > 0) {
          std::cerr << Color::c(Color::Yellow)
                    << "\n⚠  Audio ring buffer overrun (" << ov
                    << " samples dropped) — VAD may be too slow.\n"
                    << Color::c(Color::Reset) << std::flush;
        }
      }
    }

    mic.stop();
    audio_ring.stop();

    // Flush remaining speech (backend-specific)
    vad_flush();

    // Wait for ASR worker to drain the queue
    LOG_DEBUG("Waiting for ASR to finish... (queue depth: %.1fs)",
              seg_queue.duration());
    seg_queue.set_done();
    worker.join();

  } else {
    // ── WAV file mode (simulate streaming) ──────────────
    std::vector<float> pcm_all;
    int wav_sr = SAMPLE_RATE;

    LOG_INFO("Loading WAV: %s", args.wav_path);
    if (!load_wav_file(args.wav_path, pcm_all, &wav_sr)) {
      LOG_ERROR("Failed to load WAV");
      seg_queue.set_done();
      worker.join();

      return 1;
    }
    LOG_INFO("Loaded %zu samples @ %d Hz (%.2f s)", pcm_all.size(), wav_sr,
             (float)pcm_all.size() / wav_sr);

    // load_wav_file already normalizes to [-1, 1]

    if (wav_sr != SAMPLE_RATE) {
      LOG_ERROR("WAV sample rate %d != expected %d", wav_sr, SAMPLE_RATE);
      seg_queue.set_done();
      worker.join();

      return 1;
    }

    // Feed VAD frames at real-time pace (32 ms per 512-sample frame).
    // This simulates streaming and allows ASR to keep up.
    const int n_total = (int)pcm_all.size();
    const int FRAME_MS = VAD_WINDOW * 1000 / SAMPLE_RATE; // 32 ms
    auto last_frame_time = std::chrono::steady_clock::now();

    for (int offset = 0; offset < n_total; offset += VAD_WINDOW) {
      int end = std::min(offset + VAD_WINDOW, n_total);
      int frame_len = end - offset;

      vad_step(pcm_all.data() + offset, frame_len);

      // Pace to real-time
      auto now = std::chrono::steady_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - last_frame_time)
                         .count();
      if (elapsed < FRAME_MS) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(FRAME_MS - elapsed));
      }
      last_frame_time = std::chrono::steady_clock::now();
    }

    // Flush remaining speech (backend-specific)
    vad_flush();

    // Wait for ASR worker to drain the queue
    if (g_verbose) {
      std::cout << "\r" << Color::c(Color::Grey)
                << "Waiting for ASR to finish... "
                << "(queue: " << seg_counter.load() << " segments, "
                << seg_processed.load() << " done)" << Color::c(Color::Reset)
                << std::endl;
    }
    seg_queue.set_done();
    worker.join();
  }

  auto t_end = std::chrono::steady_clock::now();
  float wall_s =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count() /
      1e6f;
  float audio_s = (float)vad_total_samples() / SAMPLE_RATE;

  int dropped = seg_queue.dropped();
  std::cout << "\n─────────────────────────────────────────────────────────\n";
  LOG_INFO("Done. %d/%d segments · %.2fs audio · %.2fs wall · RTF %.3f",
           seg_processed.load(), seg_counter.load(), audio_s, wall_s,
           audio_s > 0 ? wall_s / audio_s : 0.f);
  if (dropped > 0) {
    LOG_INFO("⚠  %d segment(s) dropped (ASR too slow for real-time).",
             dropped);
  }

  rs_clear_error();
  return 0;
}
