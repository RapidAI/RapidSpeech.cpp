/**
 * rs-tts-offline.cpp — Offline Text-to-Speech (OmniVoice / OpenVoice2 /
 *                       CosyVoice3-LLM Phase 2)
 *
 * Usage:
 *   rs-tts-offline -m <tts.gguf> -t "text" [options]
 *
 * Options:
 *   -m, --model <path>       TTS model path (required)
 *   -t, --text <text>        Text to synthesize (required)
 *   -o, --output <path>      Output WAV file path (default: output.wav). For
 *                            models that emit speech tokens instead of audio
 *                            (e.g. cosyvoice3-llm Phase 2), the int32 LE token
 *                            sequence is written here.
 *       --instruct <text>    Voice description (default: male)
 *       --lang <lang>        Target language (default: English)
 *       --seed <n>           Random seed (default: 42)
 *       --n-steps <n>        Diffusion steps (1-128, default: 32)
 *       --threads <n>        CPU threads (default: 4)
 *       --gpu <true|false>   Enable GPU acceleration (default: true)
 *       --dump-step0-logits <path>
 *                            Debug: write the first speech-token logits
 *                            (float32) to this path. CosyVoice3-LLM only.
 *   -h, --help               Show help
 */

#include "rapidspeech.h"
#include "utils/rs_wav.h"
#include "../common/rs_cli_utf8.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define LOG_INFO(fmt, ...) std::printf("[tts] " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) std::fprintf(stderr, "[tts] ERROR: " fmt "\n", ##__VA_ARGS__)

struct TtsArgs {
  const char *model_path = nullptr;
  const char *text = nullptr;
  const char *output_path = "output.wav";
  const char *instruct = "male";
  const char *language = "English";
  const char *ref_path = nullptr;       // reference audio for voice cloning
  const char *ref_text = nullptr;       // transcript of reference audio
  const char *bert_path = nullptr;      // ZH BERT (1024-dim) GGUF, OpenVoice2 ZH
  const char *mbert_path = nullptr;     // multilingual BERT (768-dim) GGUF
  const char *dump_step0 = nullptr;     // CosyVoice3-LLM step-0 logits dump
  const char *speech_tokenizer_path = nullptr; // CosyVoice3 speech tokenizer GGUF
  const char *campplus_path = nullptr;         // CosyVoice3 CAM++ speaker GGUF
  const char *prompt_tokens_bin = nullptr;     // CosyVoice3 prompt-tokens int32 LE blob
  const char *voice_path = nullptr;            // CosyVoice3 pre-baked voice GGUF
  const char *save_voice_path = nullptr;       // CosyVoice3 voice bake output GGUF
  const char *bigvgan_gguf = nullptr;          // IndexTTS-2 BigVGAN GGUF
  const char *gpt_test_dir = nullptr;          // IndexTTS-2 GPT prefill smoke
  const char *w2v_bert_test_dir = nullptr;     // IndexTTS-2 W2V-BERT smoke
  const char *s2mel_test_dir = nullptr;        // IndexTTS-2 S2Mel smoke
  const char *conformer_test_dir = nullptr;    // IndexTTS-2 Conformer smoke
  const char *sc_hidden_npy = nullptr;         // IndexTTS-2 pre-computed sc_hidden
  // IndexTTS-2 emotion control.
  const char *emo_audio_path = nullptr;        // mode 1: emotion reference WAV
  const char *emo_vector_csv = nullptr;        // mode 2: "a,b,...,h" (8 floats)
  const char *emo_text = nullptr;              // mode 3: emotion description
  const char *qwen_emo_gguf = nullptr;         // mode 3: Qwen3 emotion GGUF
  const char *emo_test_dir = nullptr;          // emovec smoke dump dir
  float emo_alpha = 1.0f;                       // emo_weight (0..1)
  bool emo_random = false;                      // random prototype (vector path)
  int emo_mode = -1;                            // -1 = auto from provided args
  int s2mel_steps = 0;                          // IndexTTS-2 CFM steps (0 = default)
  float peak_norm = 0.95f;                       // peak-normalize target (0 = off)
  float gain = 1.0f;                             // extra linear gain
  int seed = 42;
  int n_steps = 32;
  bool n_steps_explicit = false;
  int n_threads = 4;
  bool use_gpu = true;
  bool phonemes_input = false;                 // Kokoro: --phonemes (treat -t as IPA)
  float length_scale = 1.0f;                   // Kokoro: --length-scale
};

static bool parse_bool(const std::string &v) {
  return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

static void print_usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " -m <model.gguf> -t \"text\" [options]\n\n"
      << "Options:\n"
      << "  -m, --model <path>       TTS model path (required)\n"
      << "  -t, --text <text>        Text to synthesize (required)\n"
      << "  -o, --output <path>      Output WAV file path (default: output.wav)\n"
      << "      --ref <path>         Reference audio for voice cloning (WAV)\n"
	      << "      --ref-text <text>    Transcript of the reference audio\n"
      << "      --bert <path>        ZH BERT GGUF (1024-dim, OpenVoice2 ZH only)\n"
      << "      --mbert <path>       Multilingual BERT GGUF (768-dim)\n"
      << "      --speech-tokenizer <path>\n"
      << "                          CosyVoice3 speech tokenizer GGUF (required for --ref)\n"
      << "      --campplus <path>    CosyVoice3 CAM++ speaker embedding GGUF (required for --ref)\n"
      << "      --prompt-tokens-bin <path>\n"
      << "                          CosyVoice3 prompt-tokens int32 LE blob (debug override)\n"
      << "      --voice <path>       CosyVoice3 pre-baked voice GGUF (replaces --ref, no tokenizer/CAM++ needed)\n"
      << "      --save-voice <path>  CosyVoice3: bake the resolved voice tuple to this GGUF for later reuse\n"
      << "      --phonemes           Kokoro: treat -t as already-IPA phonemes (bypass G2P)\n"
      << "      --length-scale <f>   Kokoro: duration multiplier (default 1.0; <1 faster, >1 slower)\n"
      << "                          Kokoro ZH G2P env vars (optional):\n"
      << "                            RS_KOKORO_JIEBA_DICT_DIR (default third_party/cppjieba/dict)\n"
      << "                            RS_KOKORO_ZH_DATA_DIR    (default rapidspeech/data/kokoro_zh)\n"
      << "                            RS_KOKORO_DUMP_G2P=1     log G2P bopomofo to stderr\n"
      << "      --instruct <text>    Voice description (default: male)\n"
      << "      --lang <lang>        Target language (default: English)\n"
      << "      --seed <n>           Random seed (default: 42)\n"
      << "      --n-steps <n>        Diffusion steps 1-128 (default: 32)\n"
      << "      --threads <n>        CPU threads (default: 4)\n"
      << "      --gpu <true|false>   Enable GPU acceleration (default: true)\n"
      << "      --dump-step0-logits <path>\n"
      << "                          Debug: dump first speech-token logits "
         "(CosyVoice3-LLM only)\n"
      << "      --bigvgan-gguf <path>\n"
      << "                          IndexTTS-2: BigVGAN-v2 GGUF path\n"
      << "      --gpt-test-dir <dir>\n"
      << "                          IndexTTS-2: GPT prefill smoke dir\n"
      << "      --w2v-bert-test-dir <dir>\n"
      << "                          IndexTTS-2: W2V-BERT smoke dir\n"
      << "      --s2mel-test-dir <dir>\n"
      << "                          IndexTTS-2: S2Mel smoke dir\n"
      << "      --conformer-test-dir <dir>\n"
      << "                          IndexTTS-2: Conformer smoke dir\n"
      << "      --sc-hidden-npy <path>\n"
      << "                          IndexTTS-2: pre-computed sc_hidden.npy\n"
      << "      --emo-audio <path>   IndexTTS-2: emotion reference WAV (mode 1)\n"
      << "      --emo-vector <csv>   IndexTTS-2: 8 floats happy,angry,sad,afraid,\n"
      << "                          disgusted,melancholic,surprised,calm (mode 2)\n"
      << "      --emo-text <text>    IndexTTS-2: emotion description (mode 3, Qwen)\n"
      << "      --qwen-emo-gguf <path>\n"
      << "                          IndexTTS-2: Qwen3 emotion classifier GGUF (mode 3)\n"
      << "      --emo-alpha <f>      IndexTTS-2: emotion weight 0..1 (default 1.0)\n"
      << "      --emo-random         IndexTTS-2: random prototype in vector path\n"
      << "      --emo-test-dir <dir> IndexTTS-2: dump base_vec/emovec for validation\n"
      << "      --s2mel-steps <n>    IndexTTS-2: CFM diffusion steps (default 6; lower=faster)\n"
      << "      --peak <v>           Peak-normalize output to |v| (default 0.95; 0=off)\n"
      << "      --no-normalize       Disable peak normalization (keep raw amplitude)\n"
      << "      --gain <x>           Extra linear gain applied to output (default 1.0)\n"
      << "  -h, --help               Show this help\n"
      << std::endl;
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
    } else if (a == "--bert" && i + 1 < argc) {
      args.bert_path = argv[++i];
    } else if (a == "--mbert" && i + 1 < argc) {
      args.mbert_path = argv[++i];
    } else if (a == "--speech-tokenizer" && i + 1 < argc) {
      args.speech_tokenizer_path = argv[++i];
    } else if (a == "--campplus" && i + 1 < argc) {
      args.campplus_path = argv[++i];
    } else if (a == "--prompt-tokens-bin" && i + 1 < argc) {
      args.prompt_tokens_bin = argv[++i];
    } else if (a == "--voice" && i + 1 < argc) {
      args.voice_path = argv[++i];
    } else if (a == "--save-voice" && i + 1 < argc) {
      args.save_voice_path = argv[++i];
    } else if (a == "--phonemes") {
      args.phonemes_input = true;
    } else if (a == "--length-scale" && i + 1 < argc) {
      args.length_scale = std::stof(argv[++i]);
    } else if (a == "--instruct" && i + 1 < argc) {
      args.instruct = argv[++i];
    } else if (a == "--lang" && i + 1 < argc) {
      args.language = argv[++i];
    } else if (a == "--seed" && i + 1 < argc) {
      args.seed = std::stoi(argv[++i]);
    } else if (a == "--n-steps" && i + 1 < argc) {
      args.n_steps = std::stoi(argv[++i]);
      args.n_steps_explicit = true;
    } else if (a == "--threads" && i + 1 < argc) {
      args.n_threads = std::stoi(argv[++i]);
    } else if (a == "--gpu" && i + 1 < argc) {
      args.use_gpu = parse_bool(argv[++i]);
    } else if (a == "--dump-step0-logits" && i + 1 < argc) {
      args.dump_step0 = argv[++i];
    } else if (a == "--bigvgan-gguf" && i + 1 < argc) {
      args.bigvgan_gguf = argv[++i];
    } else if (a == "--gpt-test-dir" && i + 1 < argc) {
      args.gpt_test_dir = argv[++i];
    } else if (a == "--w2v-bert-test-dir" && i + 1 < argc) {
      args.w2v_bert_test_dir = argv[++i];
    } else if (a == "--s2mel-test-dir" && i + 1 < argc) {
      args.s2mel_test_dir = argv[++i];
    } else if (a == "--conformer-test-dir" && i + 1 < argc) {
      args.conformer_test_dir = argv[++i];
    } else if (a == "--sc-hidden-npy" && i + 1 < argc) {
      args.sc_hidden_npy = argv[++i];
    } else if (a == "--emo-audio" && i + 1 < argc) {
      args.emo_audio_path = argv[++i];
    } else if (a == "--emo-vector" && i + 1 < argc) {
      args.emo_vector_csv = argv[++i];
    } else if (a == "--emo-text" && i + 1 < argc) {
      args.emo_text = argv[++i];
    } else if (a == "--qwen-emo-gguf" && i + 1 < argc) {
      args.qwen_emo_gguf = argv[++i];
    } else if (a == "--emo-alpha" && i + 1 < argc) {
      args.emo_alpha = std::stof(argv[++i]);
    } else if (a == "--emo-random") {
      args.emo_random = true;
    } else if (a == "--emo-mode" && i + 1 < argc) {
      args.emo_mode = std::stoi(argv[++i]);
    } else if (a == "--emo-test-dir" && i + 1 < argc) {
      args.emo_test_dir = argv[++i];
    } else if (a == "--s2mel-steps" && i + 1 < argc) {
      args.s2mel_steps = std::stoi(argv[++i]);
    } else if (a == "--no-normalize") {
      args.peak_norm = 0.0f;
    } else if (a == "--peak" && i + 1 < argc) {
      args.peak_norm = std::stof(argv[++i]);
    } else if (a == "--gain" && i + 1 < argc) {
      args.gain = std::stof(argv[++i]);
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
  if (!args.text) {
    std::cerr << "Error: --text is required\n";
    return false;
  }
  return true;
}

static bool write_wav_file(const char *filename, const std::vector<float> &pcm,
                           int sample_rate) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return false;

  int num_samples = (int)pcm.size();
  int num_channels = 1;
  int bits_per_sample = 16;
  int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
  int block_align = num_channels * bits_per_sample / 8;
  int data_size = num_samples * block_align;
  int chunk_size = 36 + data_size;

  // RIFF header
  file.write("RIFF", 4);
  file.write(reinterpret_cast<const char *>(&chunk_size), 4);
  file.write("WAVE", 4);

  // fmt chunk
  file.write("fmt ", 4);
  int subchunk1_size = 16;
  short audio_format = 1; // PCM
  file.write(reinterpret_cast<const char *>(&subchunk1_size), 4);
  file.write(reinterpret_cast<const char *>(&audio_format), 2);
  file.write(reinterpret_cast<const char *>(&num_channels), 2);
  file.write(reinterpret_cast<const char *>(&sample_rate), 4);
  file.write(reinterpret_cast<const char *>(&byte_rate), 4);
  file.write(reinterpret_cast<const char *>(&block_align), 2);
  file.write(reinterpret_cast<const char *>(&bits_per_sample), 2);

  // data chunk
  file.write("data", 4);
  file.write(reinterpret_cast<const char *>(&data_size), 4);

  // Convert float [-1,1] to int16
  for (float s : pcm) {
    float clipped = std::max(-1.0f, std::min(1.0f, s));
    int16_t sample = static_cast<int16_t>(clipped * 32767.0f);
    file.write(reinterpret_cast<const char *>(&sample), sizeof(int16_t));
  }

  return true;
}

int main(int argc, char **argv) {
  rs::cli::Utf8Args utf8_args(argc, argv);

  TtsArgs args;
  if (!parse_args(utf8_args.argc(), utf8_args.argv(), args)) return 1;

  // Init TTS model
  LOG_INFO("RapidSpeech.cpp v%s", rs_get_version());
  LOG_INFO("TTS model: %s", args.model_path);
  LOG_INFO("Threads: %d  GPU: %s", args.n_threads, args.use_gpu ? "ON" : "OFF");

  // BERT paths are consumed by OpenVoice2Model::Load via env vars; setting them
  // here keeps the C API stable while still exposing a CLI flag to the user.
  auto set_env_var = [](const char *name, const char *value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, /*overwrite=*/1);
#endif
  };
  if (args.bert_path && args.bert_path[0]) {
    set_env_var("RS_ZH_BERT_PATH", args.bert_path);
    LOG_INFO("ZH BERT: %s", args.bert_path);
  }
  if (args.mbert_path && args.mbert_path[0]) {
    set_env_var("RS_MBERT_PATH", args.mbert_path);
    LOG_INFO("mBERT:   %s", args.mbert_path);
  }
  if (args.speech_tokenizer_path && args.speech_tokenizer_path[0]) {
    set_env_var("RS_CV3_SPEECH_TOKENIZER_PATH", args.speech_tokenizer_path);
    LOG_INFO("CV3 speech tokenizer: %s", args.speech_tokenizer_path);
  }
  if (args.campplus_path && args.campplus_path[0]) {
    set_env_var("RS_CV3_CAMPPLUS_PATH", args.campplus_path);
    LOG_INFO("CV3 CAM++: %s", args.campplus_path);
  }
  if (args.prompt_tokens_bin && args.prompt_tokens_bin[0]) {
    set_env_var("RS_CV3_PROMPT_TOKENS_BIN", args.prompt_tokens_bin);
    LOG_INFO("CV3 prompt-tokens bin: %s", args.prompt_tokens_bin);
  }
  if (args.voice_path && args.voice_path[0]) {
    set_env_var("RS_CV3_VOICE_PATH", args.voice_path);
    set_env_var("RS_KOKORO_VOICE_PATH", args.voice_path);
    LOG_INFO("Voice (reuse): %s", args.voice_path);
  }
  if (args.save_voice_path && args.save_voice_path[0]) {
    set_env_var("RS_CV3_SAVE_VOICE_PATH", args.save_voice_path);
    LOG_INFO("CV3 voice (bake to): %s", args.save_voice_path);
  }
  if (args.dump_step0 && args.dump_step0[0]) {
    // CosyVoice3-LLM reads this env var inside Decode (Phase-2 debug hook).
    set_env_var("RS_CV3_DUMP_STEP0_LOGITS", args.dump_step0);
    LOG_INFO("Step-0 logits dump: %s", args.dump_step0);
  }
  if (args.bigvgan_gguf && args.bigvgan_gguf[0]) {
    set_env_var("RS_INDEXTTS2_BIGVGAN_GGUF", args.bigvgan_gguf);
    LOG_INFO("BigVGAN GGUF: %s", args.bigvgan_gguf);
  }
  if (args.gpt_test_dir && args.gpt_test_dir[0]) {
    set_env_var("RS_INDEXTTS2_GPT_TEST_DIR", args.gpt_test_dir);
    LOG_INFO("GPT test dir: %s", args.gpt_test_dir);
  }
  if (args.w2v_bert_test_dir && args.w2v_bert_test_dir[0]) {
    set_env_var("RS_INDEXTTS2_W2V_BERT_TEST_DIR", args.w2v_bert_test_dir);
    LOG_INFO("W2V-BERT test dir: %s", args.w2v_bert_test_dir);
  }
  if (args.s2mel_test_dir && args.s2mel_test_dir[0]) {
    set_env_var("RS_INDEXTTS2_S2MEL_TEST_DIR", args.s2mel_test_dir);
    LOG_INFO("S2Mel test dir: %s", args.s2mel_test_dir);
  }
  if (args.conformer_test_dir && args.conformer_test_dir[0]) {
    set_env_var("RS_INDEXTTS2_CONFORMER_TEST_DIR", args.conformer_test_dir);
    LOG_INFO("Conformer test dir: %s", args.conformer_test_dir);
  }
  if (args.sc_hidden_npy && args.sc_hidden_npy[0]) {
    set_env_var("RS_INDEXTTS2_SC_HIDDEN_NPY", args.sc_hidden_npy);
    LOG_INFO("sc_hidden npy: %s", args.sc_hidden_npy);
  }
  if (args.qwen_emo_gguf && args.qwen_emo_gguf[0]) {
    set_env_var("RS_INDEXTTS2_QWEN_EMO_GGUF", args.qwen_emo_gguf);
    LOG_INFO("Qwen emotion GGUF: %s", args.qwen_emo_gguf);
  }
  if (args.emo_test_dir && args.emo_test_dir[0]) {
    set_env_var("RS_INDEXTTS2_EMO_TEST_DIR", args.emo_test_dir);
    LOG_INFO("Emo test dir: %s", args.emo_test_dir);
  }
  if (args.s2mel_steps > 0) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", args.s2mel_steps);
    set_env_var("RS_INDEXTTS2_S2MEL_NSTEPS", buf);
    LOG_INFO("S2Mel CFM steps: %d", args.s2mel_steps);
  }

  rs_init_params_t tts_params = rs_default_params();
  tts_params.model_path = args.model_path;
  tts_params.n_threads = args.n_threads;
  tts_params.use_gpu = args.use_gpu;
  tts_params.task_type = RS_TASK_TTS_OFFLINE;

  rs_context_t *tts_ctx = rs_init_from_file(tts_params);
  if (!tts_ctx) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("TTS init failed: %s", err.message);
    return 1;
  }

  rs_model_meta_t meta = rs_get_model_meta(tts_ctx);
  LOG_INFO("Architecture: %s  SampleRate: %d", meta.arch_name,
           meta.audio_sample_rate);
  LOG_INFO("Backend: %s", rs_get_backend_name(tts_ctx));

  // Set TTS params. For Kokoro with --phonemes, swap language to "ipa" so
  // KokoroModel::PushText takes the bypass-G2P branch. Also propagate
  // --length-scale via env var (read inside Kokoro engine).
  const char *effective_lang = args.language;
  if (args.phonemes_input) {
    effective_lang = "ipa";
    LOG_INFO("Phonemes mode: treating -t as already-IPA input");
  }
  if (args.length_scale != 1.0f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6f", args.length_scale);
    set_env_var("RS_KOKORO_LENGTH_SCALE", buf);
    LOG_INFO("Kokoro length_scale: %.3f", args.length_scale);
  }
  rs_set_tts_params(tts_ctx, args.instruct, effective_lang, args.seed);
  int effective_steps = args.n_steps;
  if (!args.n_steps_explicit && std::strcmp(meta.arch_name, "voxcpm2") == 0) {
    effective_steps = 8;
  }
  rs_set_tts_diffusion_steps(tts_ctx, effective_steps);
  LOG_INFO("Instruct: %s  Language: %s  Seed: %d  Steps: %d", args.instruct, effective_lang, args.seed, effective_steps);

  // Request timing starts before any per-input work. This includes optional
  // reference voice preparation, text ingestion, synthesis, and output fetch.
  auto t0 = std::chrono::steady_clock::now();

  // Push reference audio for voice cloning (optional)
  if (args.ref_path && args.ref_path[0]) {
    std::vector<float> ref_pcm;
    int ref_sr = 0;
    if (!load_wav_file(args.ref_path, ref_pcm, &ref_sr)) {
      LOG_ERROR("Failed to load reference WAV: %s", args.ref_path);
      rs_free(tts_ctx);
      return 1;
    }
    LOG_INFO("Reference audio: %zu samples @ %d Hz", ref_pcm.size(), ref_sr);
    if (rs_push_reference_audio(tts_ctx, ref_pcm.data(), (int32_t)ref_pcm.size(),
                                ref_sr) != 0) {
      rs_error_info_t err = rs_get_last_error();
      LOG_ERROR("rs_push_reference_audio failed: %s", err.message);
      rs_free(tts_ctx);
      return 1;
    }
    if (args.ref_text && args.ref_text[0]) {
      if (rs_push_reference_text(tts_ctx, args.ref_text) != RS_OK) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("rs_push_reference_text failed: %s", err.message);
        rs_free(tts_ctx);
        return 1;
      }
      LOG_INFO("Reference text: \"%s\"", args.ref_text);
    }
  }

  // IndexTTS-2 emotion control (optional). Determine the mode from the provided
  // flags unless --emo-mode forces it:
  //   --emo-text   → mode 3 (Qwen text), --emo-vector → mode 2, --emo-audio →
  //   mode 1, otherwise mode 0 (follow speaker).
  {
    int mode = args.emo_mode;
    if (mode < 0) {
      if (args.emo_text && args.emo_text[0]) mode = 3;
      else if (args.emo_vector_csv && args.emo_vector_csv[0]) mode = 2;
      else if (args.emo_audio_path && args.emo_audio_path[0]) mode = 1;
      else mode = 0;
    }
    if (mode == 1 && args.emo_audio_path && args.emo_audio_path[0]) {
      std::vector<float> emo_pcm;
      int emo_sr = 0;
      if (!load_wav_file(args.emo_audio_path, emo_pcm, &emo_sr)) {
        LOG_ERROR("Failed to load emotion WAV: %s", args.emo_audio_path);
        rs_free(tts_ctx);
        return 1;
      }
      LOG_INFO("Emotion audio: %zu samples @ %d Hz", emo_pcm.size(), emo_sr);
      if (rs_push_emotion_audio(tts_ctx, emo_pcm.data(),
                                (int32_t)emo_pcm.size(), emo_sr) != RS_OK) {
        rs_error_info_t err = rs_get_last_error();
        LOG_ERROR("rs_push_emotion_audio failed: %s", err.message);
        rs_free(tts_ctx);
        return 1;
      }
    }
    float vec[8] = {0};
    const float *vecp = nullptr;
    if (mode == 2 && args.emo_vector_csv && args.emo_vector_csv[0]) {
      int n = 0;
      std::string buf;
      for (const char *p = args.emo_vector_csv;; ++p) {
        if (*p == ',' || *p == '\0') {
          if (!buf.empty() && n < 8) vec[n++] = std::stof(buf);
          buf.clear();
          if (*p == '\0') break;
        } else if (*p != ' ') {
          buf.push_back(*p);
        }
      }
      vecp = vec;
      LOG_INFO("Emotion vector: [%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f]",
               vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]);
    }
    rs_emotion_mode_t emode = (rs_emotion_mode_t)mode;
    if (rs_set_emotion(tts_ctx, emode, args.emo_alpha, vecp, args.emo_random,
                       /*apply_bias=*/(mode == 2), args.emo_text) != RS_OK) {
      rs_error_info_t err = rs_get_last_error();
      LOG_ERROR("rs_set_emotion failed: %s", err.message);
      rs_free(tts_ctx);
      return 1;
    }
    LOG_INFO("Emotion mode: %d  alpha: %.3f  random: %s", mode, args.emo_alpha,
             args.emo_random ? "yes" : "no");
    if (mode == 3 && args.emo_text)
      LOG_INFO("Emotion text: \"%s\"", args.emo_text);
  }

  // Push text
  LOG_INFO("Synthesizing: \"%s\"", args.text);

  if (rs_push_text(tts_ctx, args.text) != RS_OK) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("rs_push_text failed: %s", err.message);
    rs_free(tts_ctx);
    return 1;
  }

  // Streaming synthesis loop
  std::vector<float> all_pcm;
  int ret;
  int chunks = 0;
  while ((ret = rs_process(tts_ctx)) >= 0) {
    float *chunk = nullptr;
    int n = rs_get_audio_output(tts_ctx, &chunk);
    if (n > 0 && chunk) {
      all_pcm.insert(all_pcm.end(), chunk, chunk + n);
      chunks++;
    }
    if (ret == 0) break;
  }

  if (ret < 0) {
    rs_error_info_t err = rs_get_last_error();
    LOG_ERROR("TTS inference failed: %s", err.message);
    rs_free(tts_ctx);
    return 1;
  }

  auto t1 = std::chrono::steady_clock::now();
  float elapsed_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
      1e3f;

  // ----- Token-emitting models (e.g. CosyVoice3-LLM Phase 2) ----------------
  // These don't return PCM; they expose a CSV of int32 speech-token ids via
  // rs_get_text_output. Persist them as a packed little-endian int32 blob so
  // downstream tooling (flow + HiFT integration, validation harnesses) can
  // consume them.
  if (all_pcm.empty()) {
    const std::string arch = meta.arch_name;
    if (arch == "cosyvoice3-llm") {
      const char *csv = rs_get_text_output(tts_ctx);
      std::vector<int32_t> ids;
      if (csv && *csv) {
        std::string buf;
        for (const char *p = csv; ; ++p) {
          if (*p == ',' || *p == '\0') {
            if (!buf.empty()) ids.push_back((int32_t)std::atoi(buf.c_str()));
            buf.clear();
            if (*p == '\0') break;
          } else {
            buf.push_back(*p);
          }
        }
      }
      std::ofstream out(args.output_path, std::ios::binary);
      if (!out) {
        LOG_ERROR("Cannot write tokens to %s", args.output_path);
        rs_free(tts_ctx);
        return 1;
      }
      if (!ids.empty()) {
        out.write(reinterpret_cast<const char *>(ids.data()),
                  (std::streamsize)(ids.size() * sizeof(int32_t)));
      }
      LOG_INFO("Generated %zu speech tokens in %.0f ms → %s", ids.size(),
               elapsed_ms, args.output_path);
      rs_free(tts_ctx);
      rs_clear_error();
      return 0;
    }
    LOG_ERROR("No audio generated");
    rs_free(tts_ctx);
    return 1;
  }

  float audio_dur = (float)all_pcm.size() / meta.audio_sample_rate;
  float rtf = audio_dur > 0 ? (elapsed_ms / 1e3f) / audio_dur : 0.f;

  LOG_INFO("Generated %zu samples (%.2f s) in %d chunks, %.0f ms, RTF: %.3f",
           all_pcm.size(), audio_dur, chunks, elapsed_ms, rtf);

  // Post-gain + peak normalization. IndexTTS-2 / BigVGAN output is often
  // low-amplitude (matches PyTorch level); normalize so the WAV is audible.
  if (args.gain != 1.0f) {
    for (float &v : all_pcm) v *= args.gain;
  }
  if (args.peak_norm > 0.0f && !all_pcm.empty()) {
    float peak = 0.0f;
    for (float v : all_pcm) peak = std::max(peak, std::fabs(v));
    if (peak > 1e-6f) {
      const float scale = args.peak_norm / peak;
      for (float &v : all_pcm) v *= scale;
      LOG_INFO("Peak-normalized: peak %.4f → %.2f (gain %.1fx)", peak,
               args.peak_norm, scale);
    }
  }

  // Write WAV
  if (write_wav_file(args.output_path, all_pcm, meta.audio_sample_rate)) {
    LOG_INFO("Wrote: %s", args.output_path);
  } else {
    LOG_ERROR("Failed to write WAV: %s", args.output_path);
    rs_free(tts_ctx);
    return 1;
  }

  rs_free(tts_ctx);
  rs_clear_error();
  return 0;
}
