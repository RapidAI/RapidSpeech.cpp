/**
 * rs-asr-offline.cpp — Offline ASR with VAD-based segmentation & timestamps
 *
 * Architecture:
 *   WAV file
 *     → VAD segmentation (Silero or FireRed, auto-detected from GGUF arch)
 *       → Per-segment ASR inference (FunASR-Nano / SenseVoice)
 *       → Per-segment CAMPPlus speaker embedding + clustering (optional)
 *         → Timestamped transcription output with speaker labels
 *
 * Usage:
 *   rs-asr-offline -m <asr.gguf> -w <audio.wav> [options]
 *   rs-asr-offline -m <asr.gguf> -w <audio.wav> -v <vad.gguf> [options]
 *   rs-asr-offline -m <asr.gguf> -w <audio.wav> -v <vad.gguf> \
 *                  --speaker <campplus.gguf> [options]
 *
 * Options:
 *   -m, --model <path>       ASR model path (required)
 *   -w, --wav <path>         WAV file path (required)
 *   -v, --vad <path>         VAD GGUF path — Silero or FireRed
 *                            (optional, auto-detected from general.architecture)
 *   -s, --speaker <path>     CAMPPlus speaker model GGUF — enables speaker
 *                            diarization, each segment gets a speaker label
 *       --spk-cluster <m>    Clustering method: spectral | ahc
 *                            (default: spectral — auto speaker count via
 *                            max eigengap, no threshold needed)
 *       --spk-threshold <f>  AHC cosine similarity threshold; higher → more
 *                            speakers (ahc only, default: 0.5)
 *       --num-speakers <n>   Fixed (oracle) number of speakers, skips auto
 *                            estimation (default: auto)
 *       --max-speakers <n>   Max speakers for spectral auto estimation
 *                            (default: 10)
 *   -t, --threads <n>        CPU threads (default: 4)
 *       --gpu <true|false>   Enable GPU acceleration (default: true)
 *       --vad-threshold <f>  VAD speech probability threshold (default: 0.5)
 *       --silence-ms <ms>    Silence duration to split segments. Silero: live
 *                            endpoint trigger. FireRed: post-merge adjacent
 *                            segments whose gap is shorter (default: 600)
 *       --speech-pad-ms <ms> Pre-roll padding before speech start (default: 200)
 *       --preroll-ms <ms>    Extra audio before speech onset to include (default: 0)
 *       --prob-smooth <f>    VAD prob EMA alpha, 0=disable (Silero only,
 *                            default: 0.3)
 *   -h, --help               Show help
 */

#include "arch/silero_vad.h"
#include "arch/fireredvad.h"
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
#include <limits>
#include <random>
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

// Read `general.architecture` from a loaded gguf_context (empty if absent).
static std::string read_gguf_arch(gguf_context *gguf) {
  if (!gguf) return "";
  int idx = gguf_find_key(gguf, "general.architecture");
  if (idx < 0) return "";
  const char *s = gguf_get_val_str(gguf, idx);
  return s ? std::string(s) : std::string();
}

// ─────────────────────────────────────────────────────
// Argument parsing
// ─────────────────────────────────────────────────────
struct OfflineArgs {
  const char *model_path = nullptr;
  const char *wav_path = nullptr;
  const char *vad_path = nullptr;
  const char *speaker_path = nullptr;
  int n_threads = 4;
  bool use_gpu = true;
  float vad_threshold = 0.5f;
  int silence_ms = 600;
  int speech_pad_ms = 200;
  int preroll_ms = 0;
  float prob_smooth = 0.3f;
  float max_segment_s = 30.0f;
  std::string spk_cluster = "spectral"; // spectral | ahc
  float spk_threshold = 0.5f;           // ahc only
  int num_speakers = 0;                 // 0 = auto
  int max_speakers = 10;                // spectral auto estimation cap
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
      << "  -v, --vad <path>         VAD GGUF path — Silero or FireRed\n"
      << "                           (optional, auto-detected from "
         "general.architecture)\n"
      << "  -s, --speaker <path>     CAMPPlus speaker model GGUF — enables\n"
         "                           speaker diarization per segment\n"
      << "      --spk-cluster <m>    Clustering method: spectral | ahc\n"
         "                           (default: spectral — auto speaker count\n"
         "                           via max eigengap, no threshold needed)\n"
      << "      --spk-threshold <f>  AHC cosine similarity threshold; higher\n"
         "                           → more speakers (ahc only, default: 0.5)\n"
      << "      --num-speakers <n>   Fixed (oracle) number of speakers, skips\n"
         "                           auto estimation (default: auto)\n"
      << "      --max-speakers <n>   Max speakers for spectral auto\n"
         "                           estimation (default: 10)\n"
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
    } else if ((a == "-s" || a == "--speaker") && i + 1 < argc) {
      args.speaker_path = argv[++i];
    } else if (a == "--spk-cluster" && i + 1 < argc) {
      args.spk_cluster = argv[++i];
      if (args.spk_cluster != "spectral" && args.spk_cluster != "ahc") {
        std::cerr << "Error: --spk-cluster must be 'spectral' or 'ahc'\n";
        return false;
      }
    } else if (a == "--spk-threshold" && i + 1 < argc) {
      args.spk_threshold = std::stof(argv[++i]);
    } else if (a == "--num-speakers" && i + 1 < argc) {
      args.num_speakers = std::stoi(argv[++i]);
    } else if (a == "--max-speakers" && i + 1 < argc) {
      args.max_speakers = std::stoi(argv[++i]);
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
// FireRed VAD segmentation
//
// FireRedVAD's built-in DFSMN state-machine postprocessor handles smoothing
// and speech/silence gating internally. Its baked-in `min_silence_frame`
// (default 200 ms) is much shorter than the Silero path's `silence_ms`,
// so adjacent timestamps are post-merged here whenever the inter-segment
// gap is < silence_ms — preserving the silence audio between them.
// Padding (speech_pad_ms / preroll_ms) is applied on the merged timestamps
// for parity with the Silero path's CLI options.
// ─────────────────────────────────────────────────────
static std::vector<SpeechSegment>
firered_segment(const std::vector<float> &pcm, FireRedVadModel &vad_model,
                int silence_ms, int speech_pad_ms, int preroll_ms) {
  const int SAMPLE_RATE = 16000;
  std::vector<SpeechSegment> segments;
  if (pcm.empty()) return segments;

  std::cout << "  VAD scanning (FireRed)... " << std::flush;
  auto full = vad_model.DetectFull(pcm);
  const size_t raw_count = full.timestamps.size();

  // Merge adjacent timestamps whose gap is shorter than silence_ms.
  // FireRed's internal min_silence_frame (~200 ms) over-segments otherwise.
  std::vector<std::pair<float, float>> merged;
  merged.reserve(raw_count);
  const float silence_s = (float)silence_ms / 1000.0f;
  for (const auto &ts : full.timestamps) {
    if (!merged.empty() && ts.first - merged.back().second < silence_s) {
      merged.back().second = ts.second;
    } else {
      merged.push_back(ts);
    }
  }
  std::cout << "done. " << raw_count << " raw → " << merged.size()
            << " merged segment(s) (silence_ms=" << silence_ms << ")\n"
            << std::flush;

  const int pad_pre_samples =
      ((speech_pad_ms + preroll_ms) * SAMPLE_RATE) / 1000;
  const int n_total = (int)pcm.size();

  for (const auto &ts : merged) {
    int start_sample = (int)(ts.first * SAMPLE_RATE) - pad_pre_samples;
    int end_sample = (int)(ts.second * SAMPLE_RATE);
    start_sample = std::max(0, start_sample);
    end_sample = std::min(n_total, end_sample);
    if (end_sample <= start_sample) continue;

    SpeechSegment seg;
    seg.start_s = (float)start_sample / SAMPLE_RATE;
    seg.end_s = (float)end_sample / SAMPLE_RATE;
    seg.pcm.assign(pcm.begin() + start_sample, pcm.begin() + end_sample);
    segments.push_back(std::move(seg));
  }
  return segments;
}

// ─────────────────────────────────────────────────────
// Speaker clustering: agglomerative hierarchical clustering with
// average linkage on cosine similarity.
//
// num_speakers > 0 — merge until exactly that many clusters remain.
// num_speakers = 0 — merge while the best pair's average similarity
//                    is >= threshold (auto speaker count).
// Returns one label per embedding; labels are renumbered in order of
// first appearance (segment order = time order).
// ─────────────────────────────────────────────────────
static std::vector<int>
cluster_speaker_embeddings(const std::vector<std::vector<float>> &embs,
                           int dim, float threshold, int num_speakers) {
  const int n = (int)embs.size();
  std::vector<int> labels(n, 0);
  if (n <= 1)
    return labels;

  std::vector<float> sim((size_t)n * n, 1.0f);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      float c = rs_speaker_cosine(embs[i].data(), embs[j].data(), dim);
      sim[(size_t)i * n + j] = c;
      sim[(size_t)j * n + i] = c;
    }
  }

  std::vector<std::vector<int>> clusters(n);
  for (int i = 0; i < n; ++i)
    clusters[i] = {i};

  auto avg_link = [&](const std::vector<int> &a, const std::vector<int> &b) {
    double s = 0.0;
    for (int i : a)
      for (int j : b)
        s += sim[(size_t)i * n + j];
    return (float)(s / ((double)a.size() * (double)b.size()));
  };

  while ((int)clusters.size() > 1) {
    float best = -2.0f;
    int bi = -1, bj = -1;
    for (size_t i = 0; i < clusters.size(); ++i) {
      for (size_t j = i + 1; j < clusters.size(); ++j) {
        float s = avg_link(clusters[i], clusters[j]);
        if (s > best) {
          best = s;
          bi = (int)i;
          bj = (int)j;
        }
      }
    }
    bool do_merge = (num_speakers > 0)
                        ? ((int)clusters.size() > num_speakers)
                        : (best >= threshold);
    if (!do_merge)
      break;
    clusters[bi].insert(clusters[bi].end(), clusters[bj].begin(),
                        clusters[bj].end());
    clusters.erase(clusters.begin() + bj);
  }

  std::sort(clusters.begin(), clusters.end(),
            [](const std::vector<int> &a, const std::vector<int> &b) {
              return *std::min_element(a.begin(), a.end()) <
                     *std::min_element(b.begin(), b.end());
            });
  for (size_t k = 0; k < clusters.size(); ++k)
    for (int idx : clusters[k])
      labels[idx] = (int)k;
  return labels;
}

// ─────────────────────────────────────────────────────
// Spectral clustering — port of 3D-Speaker's SpectralCluster
// (speakerlab/process/cluster.py, itself adapted from speechbrain).
//
// Pipeline: cosine affinity → p-pruning (keep top-p per row) →
// symmetrize → unnormalized Laplacian → eigendecomposition →
// speaker count = max eigengap among smallest eigenvalues (or oracle)
// → k-means on the first num_spk eigenvectors.
//
// No similarity threshold needed; speaker count is auto-estimated.
// ─────────────────────────────────────────────────────

// Cyclic Jacobi eigendecomposition of a symmetric matrix A (n×n,
// row-major; destroyed). Outputs eigenvalues ascending in `w` and
// eigenvectors as columns of `V` (V[i*n+k] = i-th component of the
// k-th eigenvector). O(n^3) per sweep — fine for n = segment count.
static void jacobi_eigh(std::vector<double> &A, int n, std::vector<double> &w,
                        std::vector<double> &V) {
  V.assign((size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i)
    V[(size_t)i * n + i] = 1.0;

  double frob2 = 0.0;
  for (size_t i = 0; i < A.size(); ++i)
    frob2 += A[i] * A[i];
  const double tol = 1e-12 * std::max(1.0, frob2);

  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q)
        off += A[(size_t)p * n + q] * A[(size_t)p * n + q];
    if (off < tol)
      break;

    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = A[(size_t)p * n + q];
        if (std::fabs(apq) < 1e-300)
          continue;
        const double app = A[(size_t)p * n + p];
        const double aqq = A[(size_t)q * n + q];
        const double theta = (aqq - app) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;

        // A ← Jᵀ·A·J applied as two passes (columns, then rows).
        for (int k = 0; k < n; ++k) {
          const double akp = A[(size_t)k * n + p];
          const double akq = A[(size_t)k * n + q];
          A[(size_t)k * n + p] = c * akp - s * akq;
          A[(size_t)k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const double apk = A[(size_t)p * n + k];
          const double aqk = A[(size_t)q * n + k];
          A[(size_t)p * n + k] = c * apk - s * aqk;
          A[(size_t)q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          const double vkp = V[(size_t)k * n + p];
          const double vkq = V[(size_t)k * n + q];
          V[(size_t)k * n + p] = c * vkp - s * vkq;
          V[(size_t)k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  w.resize(n);
  for (int i = 0; i < n; ++i)
    w[i] = A[(size_t)i * n + i];

  std::vector<int> order(n);
  for (int i = 0; i < n; ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return w[a] < w[b]; });

  std::vector<double> w_sorted(n);
  std::vector<double> V_sorted((size_t)n * n);
  for (int k = 0; k < n; ++k) {
    w_sorted[k] = w[order[k]];
    for (int i = 0; i < n; ++i)
      V_sorted[(size_t)i * n + k] = V[(size_t)i * n + order[k]];
  }
  w = std::move(w_sorted);
  V = std::move(V_sorted);
}

// k-means with k-means++ seeding on row-major X (n × d).
// Runs n_init restarts, keeps the labeling with the lowest inertia.
static std::vector<int> kmeans_cluster(const std::vector<double> &X, int n,
                                       int d, int k) {
  std::vector<int> best_labels(n, 0);
  if (k <= 1 || n <= 1)
    return best_labels;
  k = std::min(k, n);

  std::mt19937 rng(42); // deterministic CLI output
  const int n_init = 10;
  const int max_iter = 100;
  double best_inertia = std::numeric_limits<double>::max();

  auto dist2 = [&](const double *a, const double *b) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      const double diff = a[j] - b[j];
      s += diff * diff;
    }
    return s;
  };

  std::vector<double> centers((size_t)k * d);
  std::vector<double> min_d2(n);
  std::vector<int> labels(n);
  std::vector<int> counts(k);

  for (int init = 0; init < n_init; ++init) {
    // k-means++ seeding
    std::uniform_int_distribution<int> uni(0, n - 1);
    int first = uni(rng);
    std::copy(X.begin() + (size_t)first * d, X.begin() + (size_t)(first + 1) * d,
              centers.begin());
    for (int i = 0; i < n; ++i)
      min_d2[i] = dist2(&X[(size_t)i * d], &centers[0]);

    for (int c = 1; c < k; ++c) {
      double total = 0.0;
      for (int i = 0; i < n; ++i)
        total += min_d2[i];
      int chosen;
      if (total <= 0.0) {
        chosen = uni(rng);
      } else {
        std::uniform_real_distribution<double> ur(0.0, total);
        double r = ur(rng), acc = 0.0;
        chosen = n - 1;
        for (int i = 0; i < n; ++i) {
          acc += min_d2[i];
          if (acc >= r) {
            chosen = i;
            break;
          }
        }
      }
      std::copy(X.begin() + (size_t)chosen * d,
                X.begin() + (size_t)(chosen + 1) * d,
                centers.begin() + (size_t)c * d);
      for (int i = 0; i < n; ++i)
        min_d2[i] =
            std::min(min_d2[i], dist2(&X[(size_t)i * d], &centers[(size_t)c * d]));
    }

    // Lloyd iterations
    std::fill(labels.begin(), labels.end(), -1);
    for (int iter = 0; iter < max_iter; ++iter) {
      bool changed = false;
      for (int i = 0; i < n; ++i) {
        int best_c = 0;
        double best_d = std::numeric_limits<double>::max();
        for (int c = 0; c < k; ++c) {
          const double dd = dist2(&X[(size_t)i * d], &centers[(size_t)c * d]);
          if (dd < best_d) {
            best_d = dd;
            best_c = c;
          }
        }
        if (labels[i] != best_c) {
          labels[i] = best_c;
          changed = true;
        }
      }
      if (!changed)
        break;

      std::fill(centers.begin(), centers.end(), 0.0);
      std::fill(counts.begin(), counts.end(), 0);
      for (int i = 0; i < n; ++i) {
        counts[labels[i]]++;
        for (int j = 0; j < d; ++j)
          centers[(size_t)labels[i] * d + j] += X[(size_t)i * d + j];
      }
      for (int c = 0; c < k; ++c) {
        if (counts[c] > 0) {
          for (int j = 0; j < d; ++j)
            centers[(size_t)c * d + j] /= (double)counts[c];
        } else {
          // Empty cluster: re-seed from the point farthest from its center.
          int far_i = 0;
          double far_d = -1.0;
          for (int i = 0; i < n; ++i) {
            const double dd =
                dist2(&X[(size_t)i * d], &centers[(size_t)labels[i] * d]);
            if (dd > far_d) {
              far_d = dd;
              far_i = i;
            }
          }
          std::copy(X.begin() + (size_t)far_i * d,
                    X.begin() + (size_t)(far_i + 1) * d,
                    centers.begin() + (size_t)c * d);
        }
      }
    }

    double inertia = 0.0;
    for (int i = 0; i < n; ++i)
      inertia += dist2(&X[(size_t)i * d], &centers[(size_t)labels[i] * d]);
    if (inertia < best_inertia) {
      best_inertia = inertia;
      best_labels = labels;
    }
  }
  return best_labels;
}

// Renumber arbitrary cluster ids so the first-seen segment gets label 0
// (segment order = time order → "Speaker 1" appears first in output).
static void renumber_labels(std::vector<int> &labels) {
  std::vector<int> remap;
  for (int &l : labels) {
    int found = -1;
    for (size_t i = 0; i < remap.size(); ++i) {
      if (remap[i] == l) {
        found = (int)i;
        break;
      }
    }
    if (found < 0) {
      remap.push_back(l);
      found = (int)remap.size() - 1;
    }
    l = found;
  }
}

struct SpectralClusterOpts {
  int min_num_spks = 1;
  int max_num_spks = 10;
  float pval = 0.02f;
  int min_pnum = 6;
  int oracle_num = 0; // 0 = auto via max eigengap
};

static std::vector<int>
spectral_cluster_embeddings(const std::vector<std::vector<float>> &embs,
                            int dim, const SpectralClusterOpts &opts) {
  const int n = (int)embs.size();
  std::vector<int> labels(n, 0);
  if (n <= 1)
    return labels;

  // 1. Cosine affinity matrix
  std::vector<double> M((size_t)n * n, 1.0);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double c =
          (double)rs_speaker_cosine(embs[i].data(), embs[j].data(), dim);
      M[(size_t)i * n + j] = c;
      M[(size_t)j * n + i] = c;
    }
  }

  // 2. p-pruning: per row, zero out the n_elems smallest similarities,
  //    keeping at least min_pnum entries (mirrors cluster.py p_pruning).
  int n_elems = (int)((1.0f - opts.pval) * (float)n);
  n_elems = std::min(n_elems, n - opts.min_pnum);
  if (n_elems > 0) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j)
        idx[j] = j;
      double *row = &M[(size_t)i * n];
      std::sort(idx.begin(), idx.end(),
                [row](int a, int b) { return row[a] < row[b]; });
      for (int e = 0; e < n_elems; ++e)
        row[idx[e]] = 0.0;
    }
  }

  // 3. Symmetrize: 0.5 * (A + Aᵀ)
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double v = 0.5 * (M[(size_t)i * n + j] + M[(size_t)j * n + i]);
      M[(size_t)i * n + j] = v;
      M[(size_t)j * n + i] = v;
    }
  }

  // 4. Unnormalized Laplacian: L = D − M (diagonal of M zeroed,
  //    D_i = Σ_j |M_ij|)
  for (int i = 0; i < n; ++i)
    M[(size_t)i * n + i] = 0.0;
  std::vector<double> L((size_t)n * n);
  for (int i = 0; i < n; ++i) {
    double deg = 0.0;
    for (int j = 0; j < n; ++j)
      deg += std::fabs(M[(size_t)i * n + j]);
    for (int j = 0; j < n; ++j)
      L[(size_t)i * n + j] = -M[(size_t)i * n + j];
    L[(size_t)i * n + i] = deg;
  }

  // 5. Eigendecomposition (ascending)
  std::vector<double> w, V;
  jacobi_eigh(L, n, w, V);

  // 6. Speaker count: oracle, or max eigengap among the smallest
  //    eigenvalues — argmax over gaps of w[min_spks-1 .. max_spks].
  int num_spk;
  if (opts.oracle_num > 0) {
    num_spk = std::min(opts.oracle_num, n);
  } else {
    const int lo = opts.min_num_spks - 1;
    const int hi = std::min(opts.max_num_spks + 1, n); // exclusive
    num_spk = opts.min_num_spks;
    double best_gap = -std::numeric_limits<double>::max();
    for (int i = lo; i + 1 < hi; ++i) {
      const double gap = w[i + 1] - w[i];
      if (gap > best_gap) {
        best_gap = gap;
        num_spk = (i - lo) + opts.min_num_spks;
      }
    }
    num_spk = std::max(1, std::min(num_spk, n));
  }

  // 7. Spectral embeddings: first num_spk eigenvector columns (n × k)
  std::vector<double> spec_emb((size_t)n * num_spk);
  for (int i = 0; i < n; ++i)
    for (int c = 0; c < num_spk; ++c)
      spec_emb[(size_t)i * num_spk + c] = V[(size_t)i * n + c];

  // 8. k-means
  labels = kmeans_cluster(spec_emb, n, num_spk, num_spk);
  renumber_labels(labels);
  return labels;
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

    // Auto-detect VAD architecture from GGUF metadata.
    // Silero VAD: "silero-vad"   FireRedVAD: "firered-vad" / "firered_vad"
    const std::string vad_arch = read_gguf_arch(vad_gguf);
    const bool use_firered =
        (vad_arch == "firered-vad" || vad_arch == "firered_vad");
    LOG_INFO("VAD model: %s", args.vad_path);
    LOG_INFO("VAD arch: %s",
             vad_arch.empty() ? "<unknown, defaulting to silero-vad>"
                              : vad_arch.c_str());

    if (use_firered) {
      LOG_INFO("VAD params: threshold=%.2f, silence=%dms, speech-pad=%dms, "
               "preroll=%dms",
               args.vad_threshold, args.silence_ms, args.speech_pad_ms,
               args.preroll_ms);

      auto vad_model = std::make_shared<FireRedVadModel>();
      if (!vad_model->LoadDirect(vad_data, vad_gguf, vad_backend.backend)) {
        LOG_ERROR("Failed to load FireRed VAD weights");
        rs_free(asr_ctx);
        return 1;
      }
      vad_model->SetThreshold(args.vad_threshold);

      LOG_INFO("Running VAD segmentation...");
      segments = firered_segment(pcm_normalized, *vad_model, args.silence_ms,
                                 args.speech_pad_ms, args.preroll_ms);
    } else {
      LOG_INFO("VAD params: threshold=%.2f, silence=%dms, smooth=%.2f",
               args.vad_threshold, args.silence_ms, args.prob_smooth);

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
      segments = vad_segment(pcm_normalized, *vad_model, *vad_state,
                             vad_backend.sched, args.vad_threshold,
                             args.silence_ms, args.speech_pad_ms,
                             args.preroll_ms, args.prob_smooth);
    }

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

  // ── 5. Speaker diarization (if CAMPPlus model provided) ─
  if (filtered.empty()) {
    LOG_INFO("No speech detected in audio");
    rs_free(asr_ctx);
    return 0;
  }

  std::vector<int> spk_labels; // empty = diarization disabled
  int n_speakers = 0;

  if (args.speaker_path) {
    LOG_INFO("Speaker model: %s", args.speaker_path);

    rs_speaker_t *sp = rs_speaker_init_from_file(args.speaker_path,
                                                 args.n_threads, args.use_gpu);
    if (!sp) {
      rs_error_info_t err = rs_get_last_error();
      LOG_ERROR("Speaker model init failed: %s", err.message);
      rs_free(asr_ctx);
      return 1;
    }

    const int spk_dim = rs_speaker_dim(sp);
    if (args.num_speakers > 0) {
      LOG_INFO("Speaker embedding: dim=%d  cluster=%s  num-speakers=%d",
               spk_dim, args.spk_cluster.c_str(), args.num_speakers);
    } else if (args.spk_cluster == "ahc") {
      LOG_INFO("Speaker embedding: dim=%d  cluster=ahc  threshold=%.2f",
               spk_dim, args.spk_threshold);
    } else {
      LOG_INFO("Speaker embedding: dim=%d  cluster=spectral  max-speakers=%d",
               spk_dim, args.max_speakers);
    }

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(filtered.size());
    bool embed_ok = true;

    auto t_spk0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < filtered.size(); ++i) {
      std::cout << "\r  Speaker embedding: " << (i + 1) << "/"
                << filtered.size() << std::flush;
      std::vector<float> emb((size_t)spk_dim, 0.0f);
      rs_error_t rc = rs_speaker_embed(sp, filtered[i].pcm.data(),
                                       (int32_t)filtered[i].pcm.size(),
                                       emb.data(), spk_dim);
      if (rc != RS_OK) {
        rs_error_info_t err = rs_get_last_error();
        std::cout << "\n";
        LOG_ERROR("Speaker embed failed for segment %zu: %s", i + 1,
                  err.message);
        embed_ok = false;
        break;
      }
      embeddings.push_back(std::move(emb));
    }
    rs_speaker_free(sp);

    if (embed_ok) {
      auto t_spk1 = std::chrono::steady_clock::now();
      float spk_s = std::chrono::duration_cast<std::chrono::microseconds>(
                        t_spk1 - t_spk0)
                        .count() /
                    1e6f;
      std::cout << "\r  Speaker embedding: " << filtered.size() << "/"
                << filtered.size() << " done (" << std::fixed
                << std::setprecision(2) << spk_s << "s)\n";

      if (args.spk_cluster == "spectral") {
        SpectralClusterOpts sc;
        sc.max_num_spks = std::max(1, args.max_speakers);
        sc.oracle_num = args.num_speakers;
        spk_labels = spectral_cluster_embeddings(embeddings, spk_dim, sc);
      } else {
        spk_labels = cluster_speaker_embeddings(
            embeddings, spk_dim, args.spk_threshold, args.num_speakers);
      }
      for (int l : spk_labels)
        n_speakers = std::max(n_speakers, l + 1);
      LOG_INFO("Speaker diarization: %d speaker(s) across %zu segment(s)",
               n_speakers, filtered.size());
    } else {
      LOG_INFO("Speaker diarization disabled (embedding failed)");
    }
  }

  // ── 6. ASR inference per segment ───────────────────────

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

    // Timestamped output (with speaker label when diarization is on)
    std::string spk_tag;
    if (i < spk_labels.size()) {
      spk_tag = "[Speaker " + std::to_string(spk_labels[i] + 1) + "]  ";
    }
    if (text && text[0]) {
      std::cout << "\n"
                << "[" << format_timestamp(seg.start_s) << " → "
                << format_timestamp(seg.end_s) << "]  " << spk_tag << text
                << "\n";
      total_segments++;
    } else {
      std::cout << "\n"
                << "[" << format_timestamp(seg.start_s) << " → "
                << format_timestamp(seg.end_s) << "]  " << spk_tag
                << "(no speech recognized)\n";
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
