/**
 * rs-asr-offline.cpp — Offline ASR with VAD-based segmentation & timestamps
 *
 * Architecture:
 *   WAV file
 *     → Silero VAD segmentation (detect speech segments)
 *       → Per-segment ASR inference (FunASR-Nano / SenseVoice)
 *         → Timestamped transcription output
 *
 * Usage:
 *   rs-asr-offline -m <asr.gguf> -w <audio.wav> [options]
 *   rs-asr-offline -m <asr.gguf> -w <audio.wav> -v <vad.gguf> [options]
 *
 * Options:
 *   -m, --model <path>       ASR model path (required)
 *   -w, --wav <path>         WAV file path (required)
 *   -v, --vad <path>         Silero-VAD GGUF path (optional, enables VAD
 * segmentation) -t, --threads <n>        CPU threads (default: 4)
 *       --gpu <true|false>   Enable GPU acceleration (default: true)
 *       --vad-threshold <f>  VAD speech probability threshold (default: 0.5)
 *       --silence-ms <ms>    Silence duration to split segments (default: 600)
 *       --speech-pad-ms <ms> Pre-roll padding before speech start (default: 200)
 *       --preroll-ms <ms>    Extra audio before speech onset to include (default: 0)
 *       --prob-smooth <f>    VAD prob EMA alpha, 0=disable (default: 0.3)
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
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────
// Logging
// ─────────────────────────────────────────────────────
#define LOG_INFO(fmt, ...) std::printf("[offline] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...)                                                    \
  std::fprintf(stderr, "[offline] ERROR: " fmt "\n", ##__VA_ARGS__)

// ─────────────────────────────────────────────────────
// Utility: VAD backend (lightweight, CPU-only for VAD)
// ─────────────────────────────────────────────────────
struct VadBackend {
  ggml_backend_t backend = nullptr;
  ggml_backend_sched_t sched = nullptr;

  bool init() {
    // VAD is tiny (~1.2MB), CPU is always faster than GPU scheduling overhead
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
// Load GGUF weights for VAD
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
// Argument parsing
// ─────────────────────────────────────────────────────
struct OfflineArgs {
  const char *model_path = nullptr;
  const char *wav_path = nullptr;
  const char *vad_path = nullptr;
  int n_threads = 4;
  bool use_gpu = true;
  float vad_threshold = 0.5f;
  int silence_ms = 600;
  int speech_pad_ms = 200;
  int preroll_ms = 0;
  float prob_smooth = 0.3f;
  float max_segment_s = 30.0f;
};

static bool parse_bool(const std::string &v) {
  return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

static void print_usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " -m <model.gguf> -w <audio.wav> [options]\n\n"
      << "Options:\n"
      << "  -m, --model <path>       ASR model path (required)\n"
      << "  -w, --wav <path>         WAV audio file path (required)\n"
      << "  -v, --vad <path>         Silero-VAD GGUF path (optional, enables "
         "VAD segmentation)\n"
      << "  -t, --threads <n>        CPU threads (default: 4)\n"
      << "      --gpu <true|false>   Enable GPU acceleration (default: true)\n"
      << "      --vad-threshold <f>  VAD speech probability threshold "
         "(default: 0.5)\n"
      << "      --silence-ms <ms>    Silence duration to split segments "
         "(default: 600)\n"
      << "      --speech-pad-ms <ms> Pre-roll padding before speech start "
         "(default: 200)\n"
      << "      --preroll-ms <ms>    Extra audio before speech onset "
         "(default: 0)\n"
      << "      --prob-smooth <f>    VAD prob EMA alpha, 0=disable "
         "(default: 0.3)\n"
      << "      --max-segment-s <s>  Max segment length for ASR input "
         "(default: 30.0)\n"
      << "  -h, --help               Show this help\n"
      << std::endl;
}

static bool parse_args(int argc, char **argv, OfflineArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model") && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if ((a == "-w" || a == "--wav") && i + 1 < argc) {
      args.wav_path = argv[++i];
    } else if ((a == "-v" || a == "--vad") && i + 1 < argc) {
      args.vad_path = argv[++i];
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
    } else if (a == "--prob-smooth" && i + 1 < argc) {
      args.prob_smooth = std::stof(argv[++i]);
    } else if (a == "--max-segment-s" && i + 1 < argc) {
      args.max_segment_s = std::stof(argv[++i]);
    } else if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      return false;
    }
  }
  if (!args.model_path) {
    std::cerr << "Error: --model is required\n";
    return false;
  }
  if (!args.wav_path) {
    std::cerr << "Error: --wav is required\n";
    return false;
  }
  return true;
}

// ─────────────────────────────────────────────────────
// Speech segment info
// ─────────────────────────────────────────────────────
struct SpeechSegment {
  float start_s;          // start time in seconds
  float end_s;            // end time in seconds
  std::vector<float> pcm; // audio samples
};

// ─────────────────────────────────────────────────────
// Format timestamp as [HH:MM:SS.mmm]
// ─────────────────────────────────────────────────────
static std::string format_timestamp(float seconds) {
  int total_ms = (int)(seconds * 1000 + 0.5f);
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
// VAD segmentation: detect speech segments from PCM
//
// Improvements over the baseline:
//   A. Pre-roll buffer — offline variant: directly indexes into the full
//      PCM array to include audio before the VAD trigger point.  An extra
//      preroll_ms window is also included when requested.
//   B. Adaptive silence threshold — dynamically adjusted based on recent
//      segment durations to match the speaker's pace.
//   C. EMA probability smoothing — reduces jitter / fragmentation from
//      single-frame spikes or dips.
//   + Hysteresis — lower exit threshold once in speech (matches Silero
//      VADIterator logic, already used in the online CLI).
// ─────────────────────────────────────────────────────
static std::vector<SpeechSegment>
vad_segment(const std::vector<float> &pcm, SileroVadModel &vad_model,
            SileroVadState &vad_state, ggml_backend_sched_t vad_sched,
            float vad_threshold, int silence_ms, int speech_pad_ms,
            int preroll_ms, float prob_smooth_alpha) {
  const int SAMPLE_RATE = 16000;
  const int VAD_WINDOW = 512;

  // ── Initialise adaptive threshold ──────────────────────
  float adaptive_silence_ms = (float)silence_ms;
  const float base_silence_ms = (float)silence_ms;
  const float ADAPT_ALPHA = 0.15f;

  const int SPEECH_PAD_SAMPLES = (speech_pad_ms * SAMPLE_RATE) / 1000;
  const int PREROLL_SAMPLES = (preroll_ms * SAMPLE_RATE) / 1000;

  // Hysteresis thresholds
  float enter_threshold = vad_threshold;
  float exit_threshold = std::max(0.01f, vad_threshold - 0.15f);

  std::vector<SpeechSegment> segments;
  std::vector<float> speech_buf;
  bool in_speech = false;
  int silence_frames = 0;
  int speech_start_offset = 0;
  float smoothed_prob = 0.0f;

  const int n_total = (int)pcm.size();

  for (int offset = 0; offset < n_total; offset += VAD_WINDOW) {
    int end = std::min(offset + VAD_WINDOW, n_total);
    int frame_len = end - offset;

    std::vector<float> vad_in(pcm.begin() + offset, pcm.begin() + end);
    if (frame_len < VAD_WINDOW) {
      vad_in.insert(vad_in.end(), VAD_WINDOW - frame_len, 0.f);
    }

    ggml_backend_sched_reset(vad_sched);
    bool vad_ok = vad_model.Encode(vad_in, vad_state, vad_sched);
    float prob = vad_ok ? vad_model.GetSpeechProbability(vad_state) : 0.f;

    // --- C. EMA probability smoothing ---
    if (prob_smooth_alpha > 0.0f) {
      smoothed_prob = prob_smooth_alpha * prob +
                      (1.0f - prob_smooth_alpha) * smoothed_prob;
    } else {
      smoothed_prob = prob;
    }

    // Hysteresis-aware speech decision
    bool is_speech = in_speech ? (smoothed_prob >= exit_threshold)
                               : (smoothed_prob >= enter_threshold);

    // --- B. Adaptive silence threshold ---
    int silence_frames_limit =
        (int)(adaptive_silence_ms * SAMPLE_RATE) / (1000 * VAD_WINDOW);
    if (silence_frames_limit < 1)
      silence_frames_limit = 1;

    // Progress indicator
    float pos_s = (float)offset / SAMPLE_RATE;
    float total_s = (float)n_total / SAMPLE_RATE;
    int pct = (int)(pos_s / total_s * 100);
    std::cout << "\r  VAD scanning: [" << std::string(pct / 5, '=')
              << std::string(20 - pct / 5, ' ') << "] " << std::fixed
              << std::setprecision(1) << pos_s << "s / " << total_s << "s  "
              << "p=" << std::setprecision(3) << smoothed_prob
              << " thr=" << std::setprecision(0)
              << adaptive_silence_ms << "ms"
              << (is_speech ? " ▌SPEECH" : " ·silence") << "  " << std::flush;

    if (is_speech) {
      silence_frames = 0;
      if (!in_speech) {
        in_speech = true;
        // --- A. Pre-roll: read audio before trigger from full PCM buffer ---
        speech_start_offset =
            std::max(0, offset - SPEECH_PAD_SAMPLES - PREROLL_SAMPLES);
        speech_buf.clear();
        speech_buf.insert(speech_buf.end(),
                          pcm.begin() + speech_start_offset,
                          pcm.begin() + offset);
      }
      speech_buf.insert(speech_buf.end(), pcm.begin() + offset,
                        pcm.begin() + end);
    } else {
      if (in_speech) {
        speech_buf.insert(speech_buf.end(), pcm.begin() + offset,
                          pcm.begin() + end);
        ++silence_frames;

        if (silence_frames >= silence_frames_limit) {
          SpeechSegment seg;
          seg.start_s = (float)speech_start_offset / SAMPLE_RATE;
          seg.end_s = (float)(offset + VAD_WINDOW) / SAMPLE_RATE;
          seg.pcm = std::move(speech_buf);
          float seg_dur = seg.end_s - seg.start_s;

          // --- B. Update adaptive silence threshold ---
          if (seg_dur < 1.0f) {
            adaptive_silence_ms =
                std::max(300.0f,
                         adaptive_silence_ms * (1.0f - ADAPT_ALPHA) +
                             300.0f * ADAPT_ALPHA);
          } else if (seg_dur > 5.0f) {
            adaptive_silence_ms =
                std::min(1200.0f,
                         adaptive_silence_ms * (1.0f - ADAPT_ALPHA) +
                             1200.0f * ADAPT_ALPHA);
          } else {
            adaptive_silence_ms =
                adaptive_silence_ms * (1.0f - ADAPT_ALPHA * 0.5f) +
                base_silence_ms * ADAPT_ALPHA * 0.5f;
          }

          segments.push_back(std::move(seg));

          in_speech = false;
          speech_buf.clear();
          silence_frames = 0;
        }
      }
    }
  }

  // Flush remaining speech
  if (in_speech && !speech_buf.empty()) {
    SpeechSegment seg;
    seg.start_s = (float)speech_start_offset / SAMPLE_RATE;
    seg.end_s = (float)n_total / SAMPLE_RATE;
    seg.pcm = std::move(speech_buf);
    segments.push_back(std::move(seg));
  }

  std::cout << "\r  VAD scanning: [====================] done.           \n"
            << std::flush;
  return segments;
}

// ─────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────
int main(int argc, char **argv) {
  OfflineArgs args;
  if (!parse_args(argc, argv, args))
    return 1;

  const int SAMPLE_RATE = 16000;

  // ── 1. Load ASR model ──────────────────────────────────
  LOG_INFO("RapidSpeech.cpp v%s", rs_get_version());
  LOG_INFO("ASR model: %s", args.model_path);
  LOG_INFO("Threads: %d  GPU: %s", args.n_threads, args.use_gpu ? "ON" : "OFF");

  rs_init_params_t asr_params = rs_default_params();
  asr_params.model_path = args.model_path;
  asr_params.n_threads = args.n_threads;
  asr_params.use_gpu = args.use_gpu;

  rs_context_t *asr_ctx = rs_init_from_file(asr_params);
  if (!asr_ctx) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("ASR init failed: %s", err.message);
    return 1;
  }

  rs_model_meta_t meta = rs_get_model_meta(asr_ctx);
  LOG_INFO("Architecture: %s  SampleRate: %d  Vocab: %d", meta.arch_name,
           meta.audio_sample_rate, meta.vocab_size);
  LOG_INFO("Backend: %s", rs_get_backend_name(asr_ctx));

  // ── 2. Load audio ──────────────────────────────────────
  LOG_INFO("Loading WAV: %s", args.wav_path);

  std::vector<float> pcm;
  int wav_sr = SAMPLE_RATE;
  if (!load_wav_file(args.wav_path, pcm, &wav_sr)) {
    LOG_ERROR("Failed to load WAV file");
    rs_free(asr_ctx);
    return 1;
  }

  float audio_dur = (float)pcm.size() / wav_sr;
  LOG_INFO("Loaded %zu samples @ %d Hz (%.2f s)", pcm.size(), wav_sr,
           audio_dur);

  if (wav_sr != meta.audio_sample_rate) {
    LOG_INFO("Resampling %d Hz -> %d Hz", wav_sr, meta.audio_sample_rate);
    std::vector<float> resampled;
    if (!resample_pcm(pcm, wav_sr, resampled, meta.audio_sample_rate)) {
      LOG_ERROR("Resampling failed (%d -> %d)", wav_sr,
                meta.audio_sample_rate);
      rs_free(asr_ctx);
      return 1;
    }
    pcm = std::move(resampled);
    wav_sr = meta.audio_sample_rate;
    audio_dur = (float)pcm.size() / wav_sr;
    LOG_INFO("Resampled: %zu samples @ %d Hz (%.2f s)", pcm.size(), wav_sr,
             audio_dur);
  }

  // load_wav_file already normalizes to [-1, 1], use as-is
  std::vector<float> pcm_normalized = pcm;

  // ── 3. VAD segmentation (if VAD model provided) ───────
  std::vector<SpeechSegment> segments;

  if (args.vad_path) {
    LOG_INFO("VAD model: %s (threshold=%.2f, silence=%dms, smooth=%.2f)",
             args.vad_path, args.vad_threshold, args.silence_ms,
             args.prob_smooth);

    VadBackend vad_backend;
    if (!vad_backend.init()) {
      LOG_ERROR("Failed to init VAD backend");
      rs_free(asr_ctx);
      return 1;
    }

    gguf_context *vad_gguf = nullptr;
    ggml_context *vad_data = load_gguf_weights(args.vad_path, &vad_gguf);
    if (!vad_data) {
      rs_free(asr_ctx);
      return 1;
    }

    auto vad_model = std::make_shared<SileroVadModel>();
    if (!vad_model->LoadDirect(vad_data, vad_gguf, vad_backend.backend)) {
      LOG_ERROR("Failed to load Silero VAD weights");
      rs_free(asr_ctx);
      return 1;
    }

    auto vad_state =
        std::dynamic_pointer_cast<SileroVadState>(vad_model->CreateState());
    if (!vad_state) {
      LOG_ERROR("Failed to create VAD state");
      rs_free(asr_ctx);
      return 1;
    }

    LOG_INFO("Running VAD segmentation...");
    segments =
        vad_segment(pcm_normalized, *vad_model, *vad_state, vad_backend.sched,
                    args.vad_threshold, args.silence_ms, args.speech_pad_ms,
                    args.preroll_ms, args.prob_smooth);

    LOG_INFO("Detected %zu speech segment(s)", segments.size());
  } else {
    // No VAD model: treat entire audio as one segment
    SpeechSegment seg;
    seg.start_s = 0.f;
    seg.end_s = audio_dur;
    seg.pcm = std::move(pcm_normalized);
    segments.push_back(std::move(seg));
    LOG_INFO("No VAD model provided, processing entire audio as one segment");
  }

  // ── 4. Filter silent segments & split long segments ────
  const int SAMPLE_RATE_F = 16000;
  const float SILENCE_RMS_THRESHOLD = 0.0005f; // below this = completely silent
  const int MAX_SEGMENT_SAMPLES = (int)(args.max_segment_s * SAMPLE_RATE_F);

  std::vector<SpeechSegment> filtered;
  int skipped_silent = 0;
  int split_count = 0;

  for (auto &seg : segments) {
    // Check RMS energy — skip completely silent segments
    double sum_sq = 0.0;
    for (float s : seg.pcm)
      sum_sq += (double)s * (double)s;
    float rms = std::sqrt((float)(sum_sq / (double)seg.pcm.size()));

    if (rms < SILENCE_RMS_THRESHOLD) {
      skipped_silent++;
      continue;
    }

    // Split long segments
    int seg_samples = (int)seg.pcm.size();
    if (seg_samples <= MAX_SEGMENT_SAMPLES) {
      filtered.push_back(std::move(seg));
    } else {
      int n_splits = (seg_samples + MAX_SEGMENT_SAMPLES - 1) / MAX_SEGMENT_SAMPLES;
      float seg_dur = seg.end_s - seg.start_s;
      float split_dur = seg_dur / (float)n_splits;

      for (int s = 0; s < n_splits; s++) {
        int sub_start = s * MAX_SEGMENT_SAMPLES;
        int sub_end = std::min(sub_start + MAX_SEGMENT_SAMPLES, seg_samples);
        int sub_len = sub_end - sub_start;

        SpeechSegment sub;
        sub.start_s = seg.start_s + (float)s * split_dur;
        sub.end_s = seg.start_s + (float)(s + 1) * split_dur;
        sub.pcm.assign(seg.pcm.begin() + sub_start, seg.pcm.begin() + sub_end);
        filtered.push_back(std::move(sub));
        split_count++;
      }
    }
  }

  if (skipped_silent > 0)
    LOG_INFO("Skipped %d silent segment(s)", skipped_silent);
  if (split_count > 0)
    LOG_INFO("Split %d long segment(s) (max %.1fs per segment)", split_count, args.max_segment_s);

  // ── 5. ASR inference per segment ───────────────────────
  if (filtered.empty()) {
    LOG_INFO("No speech detected in audio");
    rs_free(asr_ctx);
    return 0;
  }

  LOG_INFO("─────────────────────────────────────────────────────────");
  LOG_INFO("Starting ASR inference (%zu segment(s))", filtered.size());
  LOG_INFO("─────────────────────────────────────────────────────────");

  auto t_total_start = std::chrono::steady_clock::now();
  int total_segments = 0;

  for (size_t i = 0; i < filtered.size(); ++i) {
    const auto &seg = filtered[i];
    float seg_dur = seg.end_s - seg.start_s;

    auto t0 = std::chrono::steady_clock::now();

    rs_error_t err =
        rs_push_audio(asr_ctx, seg.pcm.data(), (int32_t)seg.pcm.size());
    if (err != RS_OK) {
      LOG_ERROR("rs_push_audio failed for segment %zu", i + 1);
      continue;
    }

    int32_t status = rs_process(asr_ctx);
    const char *text = nullptr;
    if (status > 0) {
      text = rs_get_text_output(asr_ctx);
    }

    auto t1 = std::chrono::steady_clock::now();
    float elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
        1e6f;
    float rtf = seg_dur > 0 ? elapsed / seg_dur : 0.f;

    // Timestamped output
    if (text && text[0]) {
      std::cout << "\n"
                << "[" << format_timestamp(seg.start_s) << " → "
                << format_timestamp(seg.end_s) << "]  " << text << "\n";
      total_segments++;
    } else {
      std::cout << "\n"
                << "[" << format_timestamp(seg.start_s) << " → "
                << format_timestamp(seg.end_s) << "]  (no speech recognized)\n";
    }

    // Segment stats (dimmed)
    std::cout << "  " << std::fixed << std::setprecision(2) << "seg #"
              << (i + 1) << " | " << seg_dur << "s | "
              << "ASR: " << elapsed << "s | "
              << "RTF: " << std::setprecision(3) << rtf << "\n"
              << std::flush;

    // Reset for next segment (ignore error if not implemented)
    rs_reset(asr_ctx);
  }

  auto t_total_end = std::chrono::steady_clock::now();
  float wall_s = std::chrono::duration_cast<std::chrono::microseconds>(
                     t_total_end - t_total_start)
                     .count() /
                 1e6f;

  std::cout << "\n─────────────────────────────────────────────────────────\n";
  LOG_INFO(
      "Done.  Segments: %d/%zu  Audio: %.2fs  Wall: %.2fs  Overall-RTF: %.3f",
      total_segments, filtered.size(), audio_dur, wall_s,
      audio_dur > 0 ? wall_s / audio_dur : 0.f);

  // ── Cleanup ────────────────────────────────────────────
  rs_free(asr_ctx);
  rs_clear_error();
  return 0;
}
