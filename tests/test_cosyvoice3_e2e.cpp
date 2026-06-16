// tests/test_cosyvoice3_e2e.cpp
//
// End-to-end smoke test for the CosyVoice3 unified pipeline. Loads a unified
// GGUF (default voice baked or zero-emb fallback), pushes a fixed sentence
// through the public C API, and verifies that the resulting waveform passes
// basic shape + sanity checks.
//
// The test is best-effort: when the GGUF path is missing (CI without the
// 1.7 GB checkpoint) it skips with status 77. Set
//   RS_CV3_E2E_MODEL=/path/to/cv3-unified-f16.gguf
// to enable it.

#include "rapidspeech.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int skip(const char *why) {
  std::fprintf(stderr, "[test_cosyvoice3_e2e] SKIP: %s\n", why);
  return 77;  // GNU automake convention
}

int main() {
  const char *model_path = std::getenv("RS_CV3_E2E_MODEL");
  if (!model_path || !*model_path) {
    return skip("RS_CV3_E2E_MODEL is unset; pass a path to "
                "cv3-unified-f16.gguf to run.");
  }

  rs_init_params_t params = rs_default_params();
  params.model_path = model_path;
  params.n_threads  = 4;
  params.use_gpu    = false;

  rs_context_t *ctx = rs_init_from_file(params);
  if (!ctx) {
    std::fprintf(stderr, "FAIL: rs_init_from_file(%s)\n", model_path);
    return 1;
  }

  rs_set_tts_params(ctx, /*instruct=*/nullptr, /*language=*/"English",
                    /*seed=*/0xC05A3);

  const char *text = "Hello, RapidSpeech.";
  if (rs_push_text(ctx, text) != RS_OK) {
    std::fprintf(stderr, "FAIL: rs_push_text\n");
    rs_free(ctx);
    return 1;
  }

  // Drive the pipeline to completion.
  std::vector<float> all_pcm;
  while (true) {
    int32_t status = rs_process(ctx);
    if (status < 0) {
      std::fprintf(stderr, "FAIL: rs_process status=%d\n", status);
      rs_free(ctx);
      return 1;
    }
    float *p = nullptr;
    int32_t n = rs_get_audio_output(ctx, &p);
    if (n > 0 && p) all_pcm.insert(all_pcm.end(), p, p + n);
    if (status == 0) break;
  }
  bool any_pcm = !all_pcm.empty();

  if (!any_pcm || all_pcm.empty()) {
    // Legacy `cosyvoice3-llm` GGUF path produces tokens only — accept that
    // as "skipped" rather than fail.
    const char *transcript = rs_get_text_output(ctx);
    if (transcript && *transcript) {
      std::fprintf(stderr, "[test_cosyvoice3_e2e] LLM-only GGUF detected; "
                   "transcript has %zu chars. Skipping audio checks.\n",
                   std::strlen(transcript));
      rs_free(ctx);
      return skip("LLM-only GGUF (no Flow+HiFT weights). To exercise the "
                  "full pipeline, point RS_CV3_E2E_MODEL at the unified "
                  "GGUF produced by convert_cosyvoice3_to_gguf.py.");
    }
    std::fprintf(stderr, "FAIL: no audio AND no transcript\n");
    rs_free(ctx);
    return 1;
  }

  // Basic sanity: 24 kHz × at least 0.25 s; bounded amplitude.
  const float sr = 24000.0f;
  const float min_dur_s = 0.25f;
  if ((float)all_pcm.size() < min_dur_s * sr) {
    std::fprintf(stderr, "FAIL: audio too short: %zu samples (~%.2f s)\n",
                 all_pcm.size(), all_pcm.size() / sr);
    rs_free(ctx);
    return 1;
  }
  double sumsq = 0.0;
  float  amax  = 0.0f;
  int    bad   = 0;
  for (float v : all_pcm) {
    if (!std::isfinite(v)) { ++bad; continue; }
    if (std::fabs(v) > amax) amax = std::fabs(v);
    sumsq += (double)v * v;
  }
  const double rms = std::sqrt(sumsq / (double)all_pcm.size());
  std::fprintf(stderr, "[test_cosyvoice3_e2e] %zu samples (%.2f s), "
               "RMS=%.4f max=%.3f bad=%d\n",
               all_pcm.size(), all_pcm.size() / sr, rms, amax, bad);
  if (bad != 0) {
    std::fprintf(stderr, "FAIL: %d non-finite samples\n", bad);
    rs_free(ctx);
    return 1;
  }
  if (amax > 1.01f) {
    std::fprintf(stderr, "FAIL: audio_limit exceeded (max=%.3f)\n", amax);
    rs_free(ctx);
    return 1;
  }
  // RMS bounds intentionally loose — the zero-embedding fallback produces
  // garbage but the pipeline should still emit something audible.
  if (rms < 1e-5 || rms > 0.95) {
    std::fprintf(stderr, "FAIL: RMS out of range [1e-5, 0.95]: %.4f\n", rms);
    rs_free(ctx);
    return 1;
  }

  std::fprintf(stderr, "[test_cosyvoice3_e2e] PASS\n");
  rs_free(ctx);
  return 0;
}
