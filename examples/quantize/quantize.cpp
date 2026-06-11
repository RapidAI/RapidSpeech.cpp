#include "common_quantize.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <string.h>
#define strcasecmp _stricmp
#endif

#define RS_QUANTIZE_LOG_INFO(fmt, ...)                                         \
  std::printf("[RapidSpeech] " fmt "\n", ##__VA_ARGS__)
#define RS_QUANTIZE_LOG_ERROR(fmt, ...)                                        \
  std::fprintf(stderr, "[RapidSpeech] ERROR: " fmt "\n", ##__VA_ARGS__)

// Map an ftype string (e.g. "q4_k") to its ggml_type for the per-tensor
// override flags.  Returns GGML_TYPE_COUNT on unknown input.
static ggml_type parse_qtype_name(const char *name) {
  // First try common quantized types directly by name.
  for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
    const char *n = ggml_type_name((ggml_type)t);
    if (n && strcasecmp(n, name) == 0) {
      return (ggml_type)t;
    }
  }
  // Fall back to ftype parser: "q4_k" → MOSTLY_Q4_K → Q4_K
  ggml_ftype ft = ggml_parse_ftype(name);
  if (ft == GGML_FTYPE_UNKNOWN) return GGML_TYPE_COUNT;
  // Map the few RS-custom ftypes back to their base type.
  switch (ft) {
  case GGML_FTYPE_MOSTLY_Q2_K_M:    return GGML_TYPE_Q2_K;
  case GGML_FTYPE_MOSTLY_Q3_K_M:    return GGML_TYPE_Q3_K;
  case GGML_FTYPE_MOSTLY_Q4_K_M:    return GGML_TYPE_Q4_K;
  case GGML_FTYPE_MOSTLY_Q5_K_M:    return GGML_TYPE_Q5_K;
  case GGML_FTYPE_MOSTLY_IQ1_S_M:   return GGML_TYPE_IQ1_S;
  case GGML_FTYPE_MOSTLY_IQ2_XXS_M: return GGML_TYPE_IQ2_XXS;
  default:                          return ggml_ftype_to_ggml_type(ft);
  }
}

// quantize a model
static bool rs_model_quantize(const std::string &fname_inp,
                              const std::string &fname_out,
                              const rs_quantize_options &cli_opts) {

  RS_QUANTIZE_LOG_INFO("Loading model from '%s'", fname_inp.c_str());

  struct ggml_context *ctx = NULL;
  struct gguf_init_params gguf_params = {
      /*.no_alloc = */ false,
      /*.ctx      = */ &ctx,
  };

  struct gguf_context *gguf_ctx =
      gguf_init_from_file(fname_inp.c_str(), gguf_params);
  if (!gguf_ctx) {
    RS_QUANTIZE_LOG_ERROR("Failed to open GGUF file: %s", fname_inp.c_str());
    return false;
  }

  // Print GGUF metadata
  {
    RS_QUANTIZE_LOG_INFO("GGUF version:      %d", gguf_get_version(gguf_ctx));
    RS_QUANTIZE_LOG_INFO("GGUF alignment:   %zu", gguf_get_alignment(gguf_ctx));
    RS_QUANTIZE_LOG_INFO("GGUF data offset: %zu",
                         gguf_get_data_offset(gguf_ctx));

    const int n_kv = gguf_get_n_kv(gguf_ctx);
    RS_QUANTIZE_LOG_INFO("GGUF n_kv: %d", n_kv);

    for (int i = 0; i < n_kv; ++i) {
      const char *key = gguf_get_key(gguf_ctx, i);
      RS_QUANTIZE_LOG_INFO("  kv[%d]: %s", i, key);
    }
  }

  const int n_tensors = gguf_get_n_tensors(gguf_ctx);
  RS_QUANTIZE_LOG_INFO("Number of tensors: %d", n_tensors);

  // Regexes of tensor names to NOT quantize (keep original precision)
  // For Q2_K_M / Q3_K_M / Q4_K_M / Q5_K_M and the IQ_M variants:
  // embed/output/ctc_lo are bumped to higher-precision types by
  // rs_get_qtype_for_tensor, so they must NOT be in the skip list.
  ggml_ftype ftype = cli_opts.ftype;
  bool is_k_m =
      (ftype == GGML_FTYPE_MOSTLY_Q2_K_M ||
       ftype == GGML_FTYPE_MOSTLY_Q3_K_M ||
       ftype == GGML_FTYPE_MOSTLY_Q4_K_M ||
       ftype == GGML_FTYPE_MOSTLY_Q5_K_M ||
       ftype == GGML_FTYPE_MOSTLY_IQ1_S_M ||
       ftype == GGML_FTYPE_MOSTLY_IQ2_XXS_M);

  std::vector<std::string> to_skip = {
      // FSMN block weights — 1D convolution, quantization hurts accuracy
      "encoder.*.fsmn_block.weight",
      // Norm scale/bias — small 1D tensors, quantization not beneficial.
      // Pattern anchors at the end so it does NOT swallow projection layers
      // that happen to live under a `*_norm` parent (e.g. Flow DiT
      // `attn_norm.linear.weight`, which is a 1024x6144 AdaLN modulation
      // matrix — the largest weight family in the model).
      ".*norm\\.weight",
      ".*norm\\.bias",
      // VAD parameters (if present in model)
      "_model.stft.forward_basis_buffer.weight",
      "_model.encoder.*.reparam_conv.weight",
      "_model.encoder.*.reparam_conv.bias", "_model.decoder.rnn.weight_ih",
      "_model.decoder.rnn.weight_hh", "_model.decoder.decoder.2.weight",
      "_model.decoder.decoder.2.bias",
      // OmniVoice audio embedding/head tables — critical for MaskGIT token
      // generation; Q4_K quantization produces incorrect tokens (silence)
      "audio_embeddings.weight",
      "audio_heads.weight",
      // RVQ codec projection weights — quantization breaks the decode path,
      // causing constant (saturated) output from DAC vocoder
      "quantizer.*.project_in.weight",
      "quantizer.*.project_out.weight",
      // Vocoder and encoder 2D weights
      "fc.weight",
      "fc2.weight",
      // HuBERT semantic encoder weights — ggml graph uses dup/concat ops that
      // don't support quantized types; only used for voice cloning (~50 MB)
      "semantic_model.*.weight",
      // CosyVoice3 HiFT vocoder: previously skipped wholesale. Re-enabled
      // because the conv kernels are small but plentiful (~40 MB f16) and the
      // alignment-safe demote (Q4_0 block 32) means quantized HiFT is in
      // budget. If audio noise becomes a problem, re-skip with
      //   "hift\\..*\\.weight"
      // CosyVoice3 Flow ancillary tensors: token-embedding lookup, conv1d
      // pre/lookahead, time/positional/conditioning embeddings, output proj.
      // All small (<5 MB each) and precision-critical for the CFM trajectory.
      // NOTE: per-block AdaLN modulation (attn_norm.linear.weight) is *not*
      // skipped — it's 1024x6144 (24 MB/block x 22 blocks = 528 MB), the
      // largest single weight family in Flow DiT, and a plain matmul that
      // quantizes cleanly under K-quants.
      "flow\\.input_embedding\\.weight",
      "flow\\.spk_embed_affine_layer\\.weight",
      "flow\\.pre_lookahead_layer\\..*\\.weight",
      "flow\\.decoder\\.estimator\\.time_embed\\..*\\.weight",
      "flow\\.decoder\\.estimator\\.input_embed\\..*\\.weight",
      "flow\\.decoder\\.estimator\\.norm_out\\..*\\.weight",
      "flow\\.decoder\\.estimator\\.proj_out\\.weight",
      // CosyVoice3 baked voice tuple — kept at original F32 (host metadata).
      "cv3\\.default_voice\\..*"};

  // For non-mixed strategies (bare Q*_K / IQ*): default policy is to keep
  // embed/lm_head/ctc at source precision (F32/F16) because the GPU F16
  // matmul kernel loses precision on the large-vocab projection.  --pure
  // overrides this so the user can explore minimum size — embed/lm_head/ctc
  // then get quantized by the chosen ftype, and any GPU precision concern
  // is delegated to the per-op ggml_mul_mat_set_prec(GGML_PREC_F32) calls
  // in llm_graph.cpp / sensevoice_encoder.cpp / funasr-nano.cpp.
  if (!is_k_m && !cli_opts.pure) {
    to_skip.insert(to_skip.begin(), {".*embed_tokens\\.weight",
                                     ".*lm_head\\.weight",
                                     ".*ctc\\.ctc_lo\\.weight",
                                     ".*ctc_out_linear\\.weight"});
  }

  // Build the full options for the underlying quantize call.
  rs_quantize_options opts = cli_opts;
  opts.to_quant = {".*"};
  opts.to_skip = to_skip;

  if (!rapid_speech_ggml_quantize(ctx, gguf_ctx, fname_inp, fname_out, opts)) {
    RS_QUANTIZE_LOG_ERROR("Failed to quantize model '%s'", fname_inp.c_str());
    return false;
  }

  return true;
}

static void rs_print_usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <model-input.gguf> <model-output.gguf> <type> [options]\n\n",
               argv0);
  std::fprintf(stderr, "Options:\n");
  std::fprintf(stderr, "  --imatrix <file.dat>           Importance matrix for activation-aware quantization\n");
  std::fprintf(stderr, "  --pure                         Disable mixed-precision bumps; quantize every weight\n");
  std::fprintf(stderr, "                                 (including embed/lm_head/ctc) with the chosen ftype.\n");
  std::fprintf(stderr, "                                 Use for minimum-size experiments.\n");
  std::fprintf(stderr, "  --token-embedding-type <type>  Force embed_tokens.weight to this type (overrides\n");
  std::fprintf(stderr, "                                 both --pure and K_M / IQ_M bumps).\n");
  std::fprintf(stderr, "  --output-tensor-type <type>    Force lm_head / ctc_lo / ctc_out_linear to this type.\n");
  std::fprintf(stderr, "  --threads <n>                  Number of quantization threads (default 4).\n\n");
  std::fprintf(stderr, "Quantization types:\n");
  ggml_print_ftypes(stderr);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    rs_print_usage(argv[0]);
    return 1;
  }

  // needed to initialize f16 tables
  {
    struct ggml_init_params params = {0, NULL, false};
    struct ggml_context *ctx = ggml_init(params);
    ggml_free(ctx);
  }

  const std::string fname_inp = argv[1];
  const std::string fname_out = argv[2];

  const ggml_ftype ftype = ggml_parse_ftype(argv[3]);
  if (ftype == GGML_FTYPE_UNKNOWN) {
    RS_QUANTIZE_LOG_ERROR("Unknown quantization type: %s", argv[3]);
    ggml_print_ftypes(stderr);
    return 1;
  }

  rs_quantize_options cli_opts;
  cli_opts.ftype = ftype;
  cli_opts.nthread = 4;

  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--imatrix") == 0 && i + 1 < argc) {
      cli_opts.imatrix_file = argv[++i];
    } else if (strcmp(argv[i], "--pure") == 0) {
      cli_opts.pure = true;
    } else if (strcmp(argv[i], "--token-embedding-type") == 0 && i + 1 < argc) {
      ggml_type t = parse_qtype_name(argv[++i]);
      if (t == GGML_TYPE_COUNT) {
        RS_QUANTIZE_LOG_ERROR("Unknown --token-embedding-type: %s", argv[i]);
        return 1;
      }
      cli_opts.token_embedding_type = t;
    } else if (strcmp(argv[i], "--output-tensor-type") == 0 && i + 1 < argc) {
      ggml_type t = parse_qtype_name(argv[++i]);
      if (t == GGML_TYPE_COUNT) {
        RS_QUANTIZE_LOG_ERROR("Unknown --output-tensor-type: %s", argv[i]);
        return 1;
      }
      cli_opts.output_tensor_type = t;
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      cli_opts.nthread = atoi(argv[++i]);
      if (cli_opts.nthread < 1) cli_opts.nthread = 1;
    } else {
      RS_QUANTIZE_LOG_ERROR("Unknown option: %s", argv[i]);
      rs_print_usage(argv[0]);
      return 1;
    }
  }

  RS_QUANTIZE_LOG_INFO("Input:  %s", fname_inp.c_str());
  RS_QUANTIZE_LOG_INFO("Output: %s", fname_out.c_str());
  RS_QUANTIZE_LOG_INFO("Type:   %d (%s)", ftype,
                       ftype == GGML_FTYPE_MOSTLY_Q2_K_M    ? "Q2_K_M"
                       : ftype == GGML_FTYPE_MOSTLY_Q3_K_M  ? "Q3_K_M"
                       : ftype == GGML_FTYPE_MOSTLY_Q4_K_M  ? "Q4_K_M"
                       : ftype == GGML_FTYPE_MOSTLY_Q5_K_M  ? "Q5_K_M"
                       : ftype == GGML_FTYPE_MOSTLY_IQ1_S_M ? "IQ1_S_M"
                       : ftype == GGML_FTYPE_MOSTLY_IQ2_XXS_M ? "IQ2_XXS_M"
                       : ggml_type_name(ggml_ftype_to_ggml_type(ftype)));
  if (cli_opts.pure) {
    RS_QUANTIZE_LOG_INFO("--pure: embed/lm_head/ctc will be quantized too");
  }
  if (cli_opts.token_embedding_type != GGML_TYPE_COUNT) {
    RS_QUANTIZE_LOG_INFO("--token-embedding-type %s",
                         ggml_type_name(cli_opts.token_embedding_type));
  }
  if (cli_opts.output_tensor_type != GGML_TYPE_COUNT) {
    RS_QUANTIZE_LOG_INFO("--output-tensor-type   %s",
                         ggml_type_name(cli_opts.output_tensor_type));
  }
  if (!cli_opts.imatrix_file.empty()) {
    RS_QUANTIZE_LOG_INFO("IMatrix: %s", cli_opts.imatrix_file.c_str());
  }

  const int64_t t_main_start_us = ggml_time_us();

  int64_t t_quantize_us = 0;
  {
    const int64_t t_start_us = ggml_time_us();

    if (!rs_model_quantize(fname_inp, fname_out, cli_opts)) {
      RS_QUANTIZE_LOG_ERROR("Failed to quantize model from '%s'",
                            fname_inp.c_str());
      return 1;
    }

    t_quantize_us = ggml_time_us() - t_start_us;
  }

  // report timing
  {
    const int64_t t_main_end_us = ggml_time_us();

    std::printf("\n");
    std::printf("%s: quantize time = %8.2f ms\n", __func__,
                t_quantize_us / 1000.0f);
    std::printf("%s:    total time = %8.2f ms\n", __func__,
                (t_main_end_us - t_main_start_us) / 1000.0f);
  }

  return 0;
}
