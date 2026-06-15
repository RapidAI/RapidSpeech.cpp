/**
 * rs-tts-online.cpp — Streaming Text-to-Speech (CosyVoice3).
 *
 * Drives the chunk-level streaming pipeline (see
 * `CosyVoice3LMModel::DecodeStream`): the LM emits speech tokens, and every
 * `token_hop_len` tokens (25 → 100, doubling) trigger one Flow + HiFT chunk
 * that comes out as PCM. First-chunk latency is roughly (LM × hop +
 * Flow + HiFT) instead of the full-utterance cost.
 *
 * Usage:
 *   rs-tts-online -m <cv3.gguf> -t "text" [options]
 *
 * Options mirror rs-tts-offline; output is either a streaming WAV (default,
 * 24 kHz 16-bit, header patched on close) or raw f32 PCM on stdout (`-o -`).
 */

#include "rapidspeech.h"
#include "utils/rs_wav.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#define LOG_INFO(fmt, ...)  std::fprintf(stderr, "[tts-online] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) std::fprintf(stderr, "[tts-online] ERROR: " fmt "\n", ##__VA_ARGS__)

struct TtsArgs {
  const char *model_path = nullptr;
  const char *text = nullptr;
  const char *output_path = "output.wav";   // "-" means stdout (raw f32 LE)
  const char *instruct = "male";
  const char *language = "English";
  const char *ref_path = nullptr;
  const char *ref_text = nullptr;
  const char *speech_tokenizer_path = nullptr;
  const char *campplus_path = nullptr;
  const char *voice_path = nullptr;
  const char *save_voice_path = nullptr;
  int seed = 42;
  int n_threads = 4;
  bool use_gpu = true;
  bool play = false;       // live playback via miniaudio
  bool no_wav = false;     // skip WAV writing (e.g. play-only)
  int  prebuffer_ms = 2500; // queue this many ms before unmuting playback.
                            // Sized for RTF up to ~1.4 on a few-second
                            // utterance (M1 + q4_k_m). Drop it on faster
                            // hardware/smaller quants for lower start lag.
};

static bool parse_bool(const std::string &v) {
  return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

static void print_usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " -m <cv3.gguf> -t \"text\" [options]\n\n"
      << "Options:\n"
      << "  -m, --model <path>       CosyVoice3 model path (required)\n"
      << "  -t, --text <text>        Text to synthesize (required)\n"
      << "  -o, --output <path>      Output WAV file (default: output.wav).\n"
      << "                           Pass `-` to stream raw float32 LE 24 kHz\n"
      << "                           mono PCM on stdout (pipeable to e.g.\n"
      << "                           `play -t raw -r 24000 -e float -b 32 -c 1 -`).\n"
      << "      --ref <path>         Reference audio WAV (voice cloning)\n"
      << "      --ref-text <text>    Transcript of the reference audio\n"
      << "      --speech-tokenizer <path>\n"
      << "                          CosyVoice3 speech tokenizer GGUF\n"
      << "      --campplus <path>    CosyVoice3 CAM++ speaker embedding GGUF\n"
      << "      --voice <path>       CosyVoice3 pre-baked voice GGUF\n"
      << "      --save-voice <path>  Bake the resolved voice tuple here\n"
      << "      --instruct <text>    Voice description (default: male)\n"
      << "      --lang <lang>        Target language (default: English)\n"
      << "      --seed <n>           Random seed (default: 42)\n"
      << "      --threads <n>        CPU threads (default: 4)\n"
      << "      --gpu <true|false>   Enable GPU acceleration (default: true)\n"
      << "      --play               Live playback via miniaudio (default OFF)\n"
      << "      --no-wav             Skip WAV output (useful with --play)\n"
      << "      --prebuffer-ms <n>   Audio to queue before unmuting playback\n"
      << "                           (default: 2500; lower = lower latency,\n"
      << "                            higher = safer when RTF > 1)\n"
      << "  -h, --help               Show this help\n";
}

static bool parse_args(int argc, char **argv, TtsArgs &args) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-m" || a == "--model") && i + 1 < argc) {
      args.model_path = argv[++i];
    } else if ((a == "-t" || a == "--text") && i + 1 < argc) {
      args.text = argv[++i];
    } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
      args.output_path = argv[++i];
    } else if (a == "--ref" && i + 1 < argc) {
      args.ref_path = argv[++i];
    } else if (a == "--ref-text" && i + 1 < argc) {
      args.ref_text = argv[++i];
    } else if (a == "--speech-tokenizer" && i + 1 < argc) {
      args.speech_tokenizer_path = argv[++i];
    } else if (a == "--campplus" && i + 1 < argc) {
      args.campplus_path = argv[++i];
    } else if (a == "--voice" && i + 1 < argc) {
      args.voice_path = argv[++i];
    } else if (a == "--save-voice" && i + 1 < argc) {
      args.save_voice_path = argv[++i];
    } else if (a == "--instruct" && i + 1 < argc) {
      args.instruct = argv[++i];
    } else if (a == "--lang" && i + 1 < argc) {
      args.language = argv[++i];
    } else if (a == "--seed" && i + 1 < argc) {
      args.seed = std::stoi(argv[++i]);
    } else if (a == "--threads" && i + 1 < argc) {
      args.n_threads = std::stoi(argv[++i]);
    } else if (a == "--gpu" && i + 1 < argc) {
      args.use_gpu = parse_bool(argv[++i]);
    } else if (a == "--play") {
      args.play = true;
    } else if (a == "--no-wav") {
      args.no_wav = true;
    } else if (a == "--prebuffer-ms" && i + 1 < argc) {
      args.prebuffer_ms = std::stoi(argv[++i]);
    } else if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return false;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      return false;
    }
  }
  if (!args.model_path) { std::cerr << "Error: --model is required\n"; return false; }
  if (!args.text)       { std::cerr << "Error: --text is required\n"; return false; }
  return true;
}

// Streaming WAV writer: opens with a placeholder header, appends int16 PCM
// per chunk, patches `data` and `RIFF` sizes on close.
class StreamingWavWriter {
public:
  StreamingWavWriter() = default;
  ~StreamingWavWriter() { Close(); }

  bool Open(const char *path, int sample_rate) {
    sample_rate_ = sample_rate;
    file_.open(path, std::ios::binary);
    if (!file_) return false;
    WriteHeader(/*data_bytes=*/0);
    return true;
  }

  void WriteChunk(const float *pcm, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      float c = std::max(-1.0f, std::min(1.0f, pcm[i]));
      int16_t s = (int16_t)std::lrintf(c * 32767.0f);
      file_.write(reinterpret_cast<const char *>(&s), sizeof(int16_t));
    }
    samples_written_ += n;
  }

  void Close() {
    if (!file_.is_open()) return;
    // Patch sizes.
    const int bytes_per_sample = 2;
    const int data_bytes = (int)(samples_written_ * bytes_per_sample);
    const int chunk_size = 36 + data_bytes;
    file_.seekp(4, std::ios::beg);
    file_.write(reinterpret_cast<const char *>(&chunk_size), 4);
    file_.seekp(40, std::ios::beg);
    file_.write(reinterpret_cast<const char *>(&data_bytes), 4);
    file_.close();
  }

  size_t samples() const { return samples_written_; }

private:
  void WriteHeader(int data_bytes) {
    const short num_channels = 1;
    const short bits_per_sample = 16;
    const int byte_rate = sample_rate_ * num_channels * bits_per_sample / 8;
    const short block_align = num_channels * bits_per_sample / 8;
    const int chunk_size = 36 + data_bytes;
    const int subchunk1_size = 16;
    const short audio_format = 1;
    file_.write("RIFF", 4);
    file_.write(reinterpret_cast<const char *>(&chunk_size), 4);
    file_.write("WAVE", 4);
    file_.write("fmt ", 4);
    file_.write(reinterpret_cast<const char *>(&subchunk1_size), 4);
    file_.write(reinterpret_cast<const char *>(&audio_format), 2);
    file_.write(reinterpret_cast<const char *>(&num_channels), 2);
    file_.write(reinterpret_cast<const char *>(&sample_rate_), 4);
    file_.write(reinterpret_cast<const char *>(&byte_rate), 4);
    file_.write(reinterpret_cast<const char *>(&block_align), 2);
    file_.write(reinterpret_cast<const char *>(&bits_per_sample), 2);
    file_.write("data", 4);
    file_.write(reinterpret_cast<const char *>(&data_bytes), 4);
  }

  std::ofstream file_;
  int sample_rate_ = 24000;
  size_t samples_written_ = 0;
};

// ─────────────────────────────────────────────────────
// Live playback: a non-blocking PCM queue + miniaudio playback device.
// The miniaudio data callback runs on a realtime audio thread and MUST
// NOT block — on underrun we zero-fill the requested frames.
//
// The queue stores producer chunks as-is (deque of vectors). Drain
// advances a read offset within the front chunk and pops finished
// chunks, so we never memmove or reallocate on the audio thread — the
// lock-held work is just a memcpy of exactly what the device asked for.
// ─────────────────────────────────────────────────────
class PcmPlaybackQueue {
public:
  // Push a chunk; never blocks the producer. We std::move into the
  // deque so there's no copy under the lock.
  void Push(const float *pcm, size_t n) {
    std::vector<float> chunk(pcm, pcm + n);
    {
      std::lock_guard<std::mutex> lk(mu_);
      total_pushed_      += n;
      buffered_samples_  += n;
      chunks_.emplace_back(std::move(chunk));
    }
    cv_.notify_all();
  }

  // Audio-thread side: read up to n samples; zero-fill the rest.
  // Returns how many real samples were drained (the rest were zeros).
  size_t Drain(float *out, size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    size_t written = 0;
    while (written < n && !chunks_.empty()) {
      const auto &front = chunks_.front();
      size_t avail = front.size() - read_offset_;
      size_t take  = std::min(avail, n - written);
      std::memcpy(out + written,
                  front.data() + read_offset_,
                  take * sizeof(float));
      read_offset_      += take;
      written           += take;
      buffered_samples_ -= take;
      if (read_offset_ >= front.size()) {
        chunks_.pop_front();
        read_offset_ = 0;
      }
    }
    if (written < n) {
      std::memset(out + written, 0, (n - written) * sizeof(float));
      if (primed_ && !done_) underruns_++;
    }
    total_drained_ += written;
    if (chunks_.empty()) cv_.notify_all();
    return written;
  }

  // Mark the device as live — only after this do we count underruns.
  void MarkPrimed() {
    std::lock_guard<std::mutex> lk(mu_);
    primed_ = true;
  }

  // Producer signals end-of-stream. Consumer can then drain remaining
  // samples; WaitDrained blocks until the buffer is empty.
  void SignalDone() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      done_ = true;
    }
    cv_.notify_all();
  }

  void WaitDrained() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return chunks_.empty(); });
  }

  size_t buffered_samples() const {
    std::lock_guard<std::mutex> lk(mu_);
    return buffered_samples_;
  }

  size_t pushed()    const { return total_pushed_; }
  size_t drained()   const { return total_drained_; }
  size_t underruns() const { return underruns_; }

private:
  std::deque<std::vector<float>> chunks_;
  size_t read_offset_ = 0;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  bool primed_ = false;
  bool done_   = false;
  size_t buffered_samples_ = 0;
  size_t total_pushed_     = 0;
  size_t total_drained_    = 0;
  size_t underruns_        = 0;
};

class PcmPlayback {
public:
  bool Start(int sample_rate, PcmPlaybackQueue *queue) {
    queue_ = queue;
    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format   = ma_format_f32;
    cfg.playback.channels = 1;
    cfg.sampleRate        = (ma_uint32)sample_rate;
    cfg.dataCallback      = &PcmPlayback::DataCallback;
    cfg.pUserData         = this;
    if (ma_device_init(nullptr, &cfg, &device_) != MA_SUCCESS) {
      LOG_ERROR("Failed to initialize miniaudio playback device");
      return false;
    }
    if (ma_device_start(&device_) != MA_SUCCESS) {
      LOG_ERROR("Failed to start miniaudio playback device");
      ma_device_uninit(&device_);
      return false;
    }
    running_ = true;
    LOG_INFO("Playback started @ %d Hz (miniaudio)", sample_rate);
    return true;
  }

  void Stop() {
    if (!running_) return;
    if (ma_device_is_started(&device_)) ma_device_stop(&device_);
    ma_device_uninit(&device_);
    running_ = false;
  }

  ~PcmPlayback() { Stop(); }

private:
  static void DataCallback(ma_device *dev, void *out, const void *,
                           ma_uint32 frame_count) {
    auto *self = static_cast<PcmPlayback *>(dev->pUserData);
    if (!self || !self->queue_ || !out) return;
    self->queue_->Drain(static_cast<float *>(out),
                        frame_count * dev->playback.channels);
  }

  ma_device device_{};
  PcmPlaybackQueue *queue_ = nullptr;
  bool running_ = false;
};

int main(int argc, char **argv) {
  TtsArgs args;
  if (!parse_args(argc, argv, args)) return 1;

  LOG_INFO("RapidSpeech.cpp v%s", rs_get_version());
  LOG_INFO("TTS model: %s", args.model_path);
  LOG_INFO("Threads: %d  GPU: %s", args.n_threads, args.use_gpu ? "ON" : "OFF");

  auto set_env_var = [](const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, /*overwrite=*/1);
#endif
  };
  if (args.speech_tokenizer_path && args.speech_tokenizer_path[0])
    set_env_var("RS_CV3_SPEECH_TOKENIZER_PATH", args.speech_tokenizer_path);
  if (args.campplus_path && args.campplus_path[0])
    set_env_var("RS_CV3_CAMPPLUS_PATH", args.campplus_path);
  if (args.voice_path && args.voice_path[0])
    set_env_var("RS_CV3_VOICE_PATH", args.voice_path);
  if (args.save_voice_path && args.save_voice_path[0])
    set_env_var("RS_CV3_SAVE_VOICE_PATH", args.save_voice_path);

  rs_init_params_t tts_params = rs_default_params();
  tts_params.model_path = args.model_path;
  tts_params.n_threads = args.n_threads;
  tts_params.use_gpu = args.use_gpu;
  tts_params.task_type = RS_TASK_TTS_ONLINE;

  rs_context_t *ctx = rs_init_from_file(tts_params);
  if (!ctx) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("TTS init failed: %s", err.message);
    return 1;
  }

  rs_model_meta_t meta = rs_get_model_meta(ctx);
  const std::string arch = meta.arch_name;
  if (arch != "cosyvoice3" && arch != "cosyvoice3-llm") {
    LOG_ERROR("rs-tts-online only supports CosyVoice3 (got arch=%s).",
              arch.c_str());
    rs_free(ctx);
    return 1;
  }
  LOG_INFO("Architecture: %s  SampleRate: %d", meta.arch_name,
           meta.audio_sample_rate);
  LOG_INFO("Backend: %s", rs_get_backend_name(ctx));

  rs_set_tts_params(ctx, args.instruct, args.language, args.seed);
  LOG_INFO("Instruct: %s  Language: %s  Seed: %d", args.instruct, args.language,
           args.seed);

  auto t0 = std::chrono::steady_clock::now();

  if (args.ref_path && args.ref_path[0]) {
    std::vector<float> ref_pcm; int ref_sr = 0;
    if (!load_wav_file(args.ref_path, ref_pcm, &ref_sr)) {
      LOG_ERROR("Failed to load reference WAV: %s", args.ref_path);
      rs_free(ctx); return 1;
    }
    LOG_INFO("Reference audio: %zu samples @ %d Hz", ref_pcm.size(), ref_sr);
    if (rs_push_reference_audio(ctx, ref_pcm.data(), (int32_t)ref_pcm.size(),
                                ref_sr) != 0) {
      rs_error_info_t err = rs_get_last_error();
      LOG_ERROR("rs_push_reference_audio failed: %s", err.message);
      rs_free(ctx); return 1;
    }
    if (args.ref_text && args.ref_text[0]) {
      if (rs_push_reference_text(ctx, args.ref_text) != RS_OK) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("rs_push_reference_text failed: %s", err.message);
        rs_free(ctx); return 1;
      }
      LOG_INFO("Reference text: \"%s\"", args.ref_text);
    }
  }

  LOG_INFO("Synthesizing: \"%s\"", args.text);
  if (rs_push_text(ctx, args.text) != RS_OK) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("rs_push_text failed: %s", err.message);
    rs_free(ctx); return 1;
  }

  // Output sinks. Three independent destinations the same chunk can fan out to:
  //   1) live playback queue (--play)
  //   2) streaming WAV writer (default, unless --no-wav or `-o -`)
  //   3) raw f32 LE on stdout (`-o -`)
  const bool stdout_raw = (std::string(args.output_path) == "-");
  const bool want_wav   = !stdout_raw && !args.no_wav;
  StreamingWavWriter wav;
  if (want_wav) {
    if (!wav.Open(args.output_path, meta.audio_sample_rate)) {
      LOG_ERROR("Cannot open output WAV: %s", args.output_path);
      rs_free(ctx); return 1;
    }
  } else if (stdout_raw) {
#if defined(_WIN32)
    _setmode(_fileno(stdout), _O_BINARY);
#endif
  }

  PcmPlaybackQueue play_queue;
  PcmPlayback player;
  bool playback_started = false;
  const size_t prebuffer_samples =
      args.play ? (size_t)((double)meta.audio_sample_rate *
                           std::max(0, args.prebuffer_ms) / 1000.0)
                : 0;
  auto try_start_playback = [&](bool force) {
    if (!args.play || playback_started) return true;
    if (!force && play_queue.buffered_samples() < prebuffer_samples) return true;
    if (!player.Start(meta.audio_sample_rate, &play_queue)) {
      LOG_ERROR("Live playback requested but device init failed");
      return false;
    }
    play_queue.MarkPrimed();
    playback_started = true;
    LOG_INFO("Playback unmuted after %zu samples (%.0f ms) buffered",
             play_queue.buffered_samples(),
             1000.0 * play_queue.buffered_samples() /
                 meta.audio_sample_rate);
    return true;
  };

  int chunk_idx = 0;
  size_t total_samples = 0;
  double first_chunk_ms = -1.0;
  int ret;
  while ((ret = rs_process(ctx)) >= 0) {
    float *chunk = nullptr;
    int n = rs_get_audio_output(ctx, &chunk);
    if (n > 0 && chunk) {
      if (chunk_idx == 0) {
        auto now = std::chrono::steady_clock::now();
        first_chunk_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(now - t0)
                .count() / 1e3;
        LOG_INFO("first-chunk latency: %.0f ms (%d samples)",
                 first_chunk_ms, n);
      }
      if (stdout_raw) {
        std::fwrite(chunk, sizeof(float), (size_t)n, stdout);
        std::fflush(stdout);
      } else if (want_wav) {
        wav.WriteChunk(chunk, (size_t)n);
      }
      if (args.play) {
        play_queue.Push(chunk, (size_t)n);
        if (!try_start_playback(/*force=*/false)) {
          if (want_wav) wav.Close();
          rs_free(ctx); return 1;
        }
      }
      total_samples += (size_t)n;
      ++chunk_idx;
    }
    if (ret == 0) break;
  }
  // Producer done — start playback now even if prebuffer wasn't reached
  // (short utterance whose total audio is shorter than prebuffer_ms).
  if (args.play && !playback_started) {
    if (!try_start_playback(/*force=*/true)) {
      if (want_wav) wav.Close();
      rs_free(ctx); return 1;
    }
  }
  if (ret < 0) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("TTS streaming failed: %s", err.message);
    if (args.play) {
      play_queue.SignalDone();
      if (playback_started) player.Stop();
    }
    if (want_wav) wav.Close();
    rs_free(ctx); return 1;
  }

  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1e3;
  double audio_dur = (double)total_samples / meta.audio_sample_rate;
  double rtf = audio_dur > 0 ? (elapsed_ms / 1e3) / audio_dur : 0.0;

  if (want_wav) wav.Close();

  // Block on playback drain so we don't tear the device down mid-buffer.
  if (args.play) {
    play_queue.SignalDone();
    play_queue.WaitDrained();
    player.Stop();
    LOG_INFO("Playback: pushed %zu drained %zu underruns %zu",
             play_queue.pushed(), play_queue.drained(), play_queue.underruns());
  }

  LOG_INFO("Generated %zu samples (%.2f s) in %d chunks, %.0f ms total, "
           "first-chunk %.0f ms, RTF %.3f",
           total_samples, audio_dur, chunk_idx, elapsed_ms, first_chunk_ms,
           rtf);
  if (want_wav) LOG_INFO("Wrote: %s", args.output_path);

  rs_free(ctx);
  rs_clear_error();
  return 0;
}
