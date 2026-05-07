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
 *   -h, --help               Show help
 */

#include "arch/silero_vad.h"
#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
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
// ─────────────────────────────────────────────────────
#define LOG_INFO(fmt, ...) std::printf("[online] " fmt "\n", ##__VA_ARGS__)
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
  };

  SpeechSegmentQueue(float max_dur_s = 60.0f, float warn_dur_s = 15.0f)
      : max_queue_dur_s_(max_dur_s), warn_queue_dur_s_(warn_dur_s) {}

  // Push a completed speech segment (non-blocking).
  // If the queue is full the oldest segment is dropped.
  void push(Segment seg) {
    float seg_dur = seg.end_s - seg.start_s;
    std::lock_guard<std::mutex> lock(mtx_);

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

// ─────────────────────────────────────────────────────
// Cross-platform microphone capture via miniaudio
// ─────────────────────────────────────────────────────
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
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
  bool two_pass = false;
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
    } else if (a == "--two-pass") {
      args.two_pass = true;
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
             "      --mic-chunk-ms <ms>  Mic read chunk size (default: 32)\n"
             "      --two-pass           2-pass mode: CTC fast pass + LLM "
             "rescore (FunASRNano)\n"
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
};

// ─────────────────────────────────────────────────────
// ASR segment processing — called from worker thread
// ─────────────────────────────────────────────────────
static bool process_segment(rs_context_t *asr_ctx,
                            const std::vector<float> &pcm, float start_s,
                            float end_s, int seg_index, bool two_pass) {
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
  if (two_pass)
    rs_set_use_llm(asr_ctx, false);

  int32_t status = rs_process(asr_ctx);
  const char *text = nullptr;
  if (status > 0)
    text = rs_get_text_output(asr_ctx);

  if (two_pass) {
    // Print interim (CTC) result — dimmed
    std::cout << "\n"
              << Color::c(Color::Grey) << "[" << format_timestamp(start_s)
              << " → " << format_timestamp(end_s) << "]"
              << Color::c(Color::Reset) << "  " << Color::c(Color::Dim)
              << "1st: " << (text && text[0] ? text : "(silence)")
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

    // Final result — bold, overwrites the interim line visually
    std::cout << "\r"
              << Color::c(Color::Cyan) << "[" << format_timestamp(start_s)
              << " → " << format_timestamp(end_s) << "]"
              << Color::c(Color::Reset) << "  " << Color::c(Color::Bold)
              << (final_text && final_text[0] ? final_text : "(no speech)")
              << Color::c(Color::Reset) << "\n";

    std::cout << Color::c(Color::Grey) << "  seg #" << seg_index << " | "
              << std::fixed << std::setprecision(2) << seg_dur << "s"
              << " | total: " << elapsed << "s"
              << " (LLM: " << std::setprecision(2) << llm_elapsed << "s)"
              << " | RTF: " << std::setprecision(3) << rtf
              << Color::c(Color::Reset) << std::endl;
  } else {
    auto t1 = std::chrono::steady_clock::now();
    float elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
        1e6f;
    float rtf = seg_dur > 0 ? elapsed / seg_dur : 0.f;

    if (text && text[0]) {
      std::cout << "\n"
                << Color::c(Color::Cyan) << "[" << format_timestamp(start_s)
                << " → " << format_timestamp(end_s) << "]"
                << Color::c(Color::Reset) << "  " << Color::c(Color::Bold)
                << text << Color::c(Color::Reset) << "\n";
    } else {
      std::cout << "\n"
                << Color::c(Color::Grey) << "[" << format_timestamp(start_s)
                << " → " << format_timestamp(end_s) << "]"
                << Color::c(Color::Reset) << "  " << Color::c(Color::Dim)
                << "(no speech recognized)" << Color::c(Color::Reset) << "\n";
    }

    std::cout << Color::c(Color::Grey) << "  seg #" << seg_index << " | "
              << std::fixed << std::setprecision(2) << seg_dur << "s"
              << " | ASR: " << elapsed << "s"
              << " | RTF: " << std::setprecision(3) << rtf
              << Color::c(Color::Reset) << std::endl;
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
                       bool use_gpu, bool two_pass,
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
                    two_pass);
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
// Process a single VAD chunk — pushes completed speech
// segments to the queue (does NOT run ASR inline).
// ─────────────────────────────────────────────────────
static bool
process_vad_chunk(const float *chunk, int chunk_samples,
                  SileroVadModel &vad_model, SileroVadState &vad_state,
                  ggml_backend_sched_t vad_sched, StreamState &state,
                  const OnlineArgs &args,
                  SpeechSegmentQueue &queue, std::atomic<int> &seg_counter) {
  const int SAMPLE_RATE = 16000;
  const int VAD_WINDOW = 512;
  const int SILENCE_FRAMES =
      (args.silence_ms * SAMPLE_RATE) / (1000 * VAD_WINDOW);
  const int SPEECH_PAD_SAMPLES =
      (args.speech_pad_ms * SAMPLE_RATE) / 1000;

  std::vector<float> vad_in(chunk, chunk + chunk_samples);
  if (chunk_samples < VAD_WINDOW)
    vad_in.insert(vad_in.end(), VAD_WINDOW - chunk_samples, 0.f);

  ggml_backend_sched_reset(vad_sched);
  bool vad_ok = vad_model.Encode(vad_in, vad_state, vad_sched);
  float prob = vad_ok ? vad_model.GetSpeechProbability(vad_state) : 0.f;

  // Hysteresis: use a lower exit threshold once in speech to avoid
  // splitting on brief probability dips (matches Silero VADIterator logic).
  float enter_threshold = args.vad_threshold;
  float exit_threshold = std::max(0.01f, args.vad_threshold - 0.15f);
  bool is_speech = state.in_speech ? (prob >= exit_threshold)
                                   : (prob >= enter_threshold);

  // Progress indicator
  float pos_s = (float)state.total_samples_read / SAMPLE_RATE;
  if (is_speech) {
    std::cout << "\r" << Color::c(Color::Green) << "["
              << format_timestamp(pos_s) << "] "
              << "▌SPEECH  p=" << std::fixed << std::setprecision(3) << prob
              << Color::c(Color::Reset) << "  " << std::flush;
  } else {
    std::cout << "\r" << Color::c(Color::Grey) << "[" << format_timestamp(pos_s)
              << "] "
              << "·silence p=" << std::fixed << std::setprecision(3) << prob
              << Color::c(Color::Reset) << "  " << std::flush;
  }

  if (is_speech) {
    state.silence_frames = 0;
    if (!state.in_speech) {
      state.in_speech = true;
      state.speech_start_offset =
          std::max(0, state.total_samples_read - chunk_samples - SPEECH_PAD_SAMPLES);
      state.speech_buf.clear();
      // Add pre-roll context (matching offline VAD behavior)
      // Note: pre-roll samples are already in the PCM stream; we record the
      // earlier start time so timestamps are correct. The actual audio before
      // chunk_samples was already processed (and discarded during silence).
    }
    state.speech_buf.insert(state.speech_buf.end(), chunk,
                            chunk + chunk_samples);
  } else {
    if (state.in_speech) {
      state.speech_buf.insert(state.speech_buf.end(), chunk,
                              chunk + chunk_samples);
      ++state.silence_frames;

      if (state.silence_frames >= SILENCE_FRAMES) {
        int idx = ++seg_counter;
        float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
        float end_s = (float)state.total_samples_read / SAMPLE_RATE;

        queue.push({std::move(state.speech_buf), start_s, end_s, idx});

        state.in_speech = false;
        state.speech_buf.clear();
        state.silence_frames = 0;
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
  OnlineArgs args;
  if (!parse_args(argc, argv, args))
    return 1;

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

  auto vad_model = std::make_shared<SileroVadModel>();
  if (!vad_model->LoadDirect(vad_data, vad_gguf, vad_backend.backend)) {
    LOG_ERROR("Failed to load Silero VAD weights");
    return 1;
  }

  // Override threshold from command line
  vad_model->SetThreshold(args.vad_threshold);
  LOG_INFO("VAD threshold set to: %.2f", args.vad_threshold);

  auto vad_state =
      std::dynamic_pointer_cast<SileroVadState>(vad_model->CreateState());
  if (!vad_state) {
    LOG_ERROR("Failed to create VAD state");
    return 1;
  }

  // ── VAD warmup: run a few silent frames to initialize LSTM state ──
  LOG_INFO("Warming up VAD with silent frames...");
  std::vector<float> warmup_silent(VAD_WINDOW, 0.0f);
  for (int i = 0; i < 3; ++i) {
    vad_model->Encode(warmup_silent, *vad_state, vad_backend.sched);
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
                     args.two_pass,
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
  auto t_start = std::chrono::steady_clock::now();

  LOG_INFO("─────────────────────────────────────────────────────────");
  if (args.use_mic) {
    LOG_INFO("Live mode: microphone input (VAD threshold=%.2f, silence=%dms)%s",
             args.vad_threshold, args.silence_ms,
             args.two_pass ? "  [2-pass]" : "");
  } else {
    LOG_INFO("File mode: %s (VAD threshold=%.2f, silence=%dms)%s",
             args.wav_path, args.vad_threshold, args.silence_ms,
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

      process_vad_chunk(vad_buf.data(), (int)n, *vad_model, *vad_state,
                        vad_backend.sched, state, args, seg_queue, seg_counter);

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

    // Flush remaining speech
    if (!state.speech_buf.empty()) {
      int idx = ++seg_counter;
      float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
      float end_s = (float)state.total_samples_read / SAMPLE_RATE;
      std::cout << "\n[EOF] Flushing remaining speech ("
                << state.speech_buf.size() << " samples)";
      seg_queue.push({std::move(state.speech_buf), start_s, end_s, idx});
    }

    // Wait for ASR worker to drain the queue
    LOG_INFO("Waiting for ASR to finish... (queue depth: %.1fs)",
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

    for (auto &s : pcm_all)
      s /= 32768.f;

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

      process_vad_chunk(pcm_all.data() + offset, frame_len, *vad_model,
                        *vad_state, vad_backend.sched, state, args,
                        seg_queue, seg_counter);

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

    // Flush remaining speech
    if (!state.speech_buf.empty()) {
      int idx = ++seg_counter;
      float start_s = (float)state.speech_start_offset / SAMPLE_RATE;
      float end_s = (float)n_total / SAMPLE_RATE;
      std::cout << "\n[EOF] Flushing remaining speech ("
                << state.speech_buf.size() << " samples)";
      seg_queue.push({std::move(state.speech_buf), start_s, end_s, idx});
    }

    // Wait for ASR worker to drain the queue
    std::cout << "\r" << Color::c(Color::Grey)
              << "Waiting for ASR to finish... "
              << "(queue: " << seg_counter.load() << " segments, "
              << seg_processed.load() << " done)" << Color::c(Color::Reset)
              << std::endl;
    seg_queue.set_done();
    worker.join();
  }

  auto t_end = std::chrono::steady_clock::now();
  float wall_s =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count() /
      1e6f;
  float audio_s = (float)state.total_samples_read / SAMPLE_RATE;

  int dropped = seg_queue.dropped();
  std::cout << "\n─────────────────────────────────────────────────────────\n";
  LOG_INFO("Done.  Segments: %d/%d  Audio: %.2fs  Wall: %.2fs  Overall-RTF: "
           "%.3f",
           seg_processed.load(), seg_counter.load(), audio_s, wall_s,
           audio_s > 0 ? wall_s / audio_s : 0.f);
  if (dropped > 0) {
    LOG_INFO("⚠  %d segment(s) dropped (ASR too slow for real-time).",
             dropped);
  }

  rs_clear_error();
  return 0;
}
