#!/usr/bin/env python3
"""
Convert X-ASR (Zipformer2 transducer, icefall) pretrained.pt to GGUF.

Usage:
    python3 scripts/convert_xasr_to_gguf.py \
        --checkpoint /Volumes/z5_picture/resource/X-ASR/pretrained.pt \
        --tokens /Volumes/z5_picture/resource/X-ASR/tokens.txt \
        --output /Volumes/z5_picture/resource/X-ASR/xasr-f32.gguf \
        --out-type f32

Tensor naming (GGUF):
    encoder_embed.conv.{0,4,7}.{weight,bias}       Conv2d subsampling stack
    encoder_embed.convnext.{depthwise_conv,pointwise_conv1,pointwise_conv2}.{weight,bias}
    encoder_embed.out.{weight,bias}                Linear(out_width*128 -> enc_dim0)
    encoder_embed.out_norm.{bias,log_scale}        BiasNorm
    enc.{s}.layers.{l}.<module>                    Zipformer2EncoderLayer params;
                                                   wrapped stacks' ".encoder." is stripped
    enc.{s}.downsample.bias                        SimpleDownsample (ds>1 stacks)
    enc.{s}.out_combiner.bypass_scale              DownsampledZipformer2Encoder combiner
    enc.downsample_output.bias                     final output downsample (x2)
    decoder.embedding.weight, decoder.conv.weight  stateless predictor
    joiner.{encoder_proj,decoder_proj,output_linear}.{weight,bias}

Dropped: per-layer top-level `bypass_scale` (unused TODO param in icefall),
simple_am_proj/simple_lm_proj (pruned-RNNT training heads).

Linear weights stored PT C-order [out, in] -> ggml ne=[in, out]; mul_mat(W, x)
== F.linear(x, W).  Depthwise conv weights PT [C, 1, k] -> ggml ne=[k, 1, C]
(what ggml_conv_1d_dw expects).  decoder.conv is a grouped conv [512, 4, 2]
(groups=128), stored as-is.

dtype policy: --out-type f16 keeps all 2-D linear projection weights in f16;
everything else (biases, norms, bypass scales, conv kernels,
chunkwise_conv_scale, decoder.embedding, joiner.output_linear) stays f32.

Quantization: run rs-quantize on the f32 GGUF, e.g.
    ./build/rs-quantize xasr-f32.gguf xasr-q4_k_m.gguf q4_k_m
The quantizer (examples/quantize/quantize.cpp) skips the host-read tables
(decoder.embedding/conv), the ChunkCausal depthwise conv kernels, and the
encoder_embed convs; the 19 encoder-layer FF/attention projections quantize
cleanly. q8_0 is bit-identical to f32; q4_k_m is 99.5 MB (vs 585 MB f32).
"""
import argparse
import os
import sys

import numpy as np
import torch
from gguf import GGUFWriter

_scripts = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_scripts, "xasr_ref"))
from build_model import derive_config, load_state_dict  # noqa: E402

# streaming deployment constants from the official encoder ONNX export metadata
LEFT_CONTEXT_FRAMES = 96  # at 50 Hz encoder rate, per-stack = 96 // ds
PAD_LENGTH = 13           # T_chunk = decode_chunk_len + 13 fbank frames


def load_tokens(path):
    toks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            tok, idx = line.rsplit(" ", 1)
            toks[int(idx)] = tok
    return [toks[i] for i in range(len(toks))]


def map_name(key):
    """checkpoint key -> gguf tensor name, or None to drop."""
    if key.startswith(("simple_am_proj.", "simple_lm_proj.")):
        return None
    if key.startswith("encoder_embed.") or key.startswith(("decoder.", "joiner.")):
        return key
    if key == "encoder.downsample_output.bias":
        return "enc.downsample_output.bias"
    if key.startswith("encoder.encoders."):
        rest = key[len("encoder.encoders."):]
        s, rest = rest.split(".", 1)
        # DownsampledZipformer2Encoder wraps the real encoder under ".encoder."
        if rest.startswith("encoder."):
            rest = rest[len("encoder."):]
        # drop the unused per-layer top-level bypass_scale ("TODO: remove it"):
        # layers.{l}.bypass_scale (but keep layers.{l}.bypass.bypass_scale)
        parts = rest.split(".")
        if len(parts) == 3 and parts[0] == "layers" and parts[2] == "bypass_scale":
            return None
        return f"enc.{s}.{rest}"
    raise RuntimeError(f"unmapped checkpoint key: {key}")


def want_f16(name, tensor):
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return False
    # large-vocab matmuls stay f32 (degenerate logits in f16 on Metal/CUDA)
    if name.startswith(("decoder.embedding", "joiner.output_linear")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/Volumes/z5_picture/resource/X-ASR/deployed_weights",
                    help="deployed_weights dir (from extract_onnx_weights.py) or a .pt checkpoint")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--out-type", choices=["f32", "f16"], default="f32")
    args = ap.parse_args()

    if os.path.isdir(args.weights):
        from deployed_model import npz_to_state_dict
        sd = {k: v for k, v in npz_to_state_dict(args.weights).items()}
    else:
        sd = load_state_dict(args.weights)
    cfg = derive_config(sd)
    tokens = load_tokens(args.tokens)
    assert len(tokens) == cfg["vocab_size"], (len(tokens), cfg["vocab_size"])

    writer = GGUFWriter(args.output, "x-asr")
    writer.add_string("general.name", "x-asr-zh-en")

    writer.add_int32("xasr.num_stacks", cfg["num_stacks"])
    writer.add_array("xasr.num_layers", list(cfg["num_encoder_layers"]))
    writer.add_array("xasr.downsampling_factor", list(cfg["downsampling_factor"]))
    writer.add_array("xasr.encoder_dim", list(cfg["encoder_dim"]))
    writer.add_array("xasr.feedforward_dim", list(cfg["feedforward_dim"]))
    writer.add_array("xasr.num_heads", list(cfg["num_heads"]))
    writer.add_array("xasr.cnn_module_kernel", list(cfg["cnn_module_kernel"]))
    writer.add_array("xasr.query_head_dim", list(cfg["query_head_dim"]))
    writer.add_array("xasr.pos_head_dim", list(cfg["pos_head_dim"]))
    writer.add_array("xasr.value_head_dim", list(cfg["value_head_dim"]))
    writer.add_int32("xasr.pos_dim", cfg["pos_dim"])
    writer.add_int32("xasr.feature_dim", cfg["feature_dim"])
    writer.add_array("xasr.embed_layer_channels", list(cfg["layer_channels"]))
    writer.add_int32("xasr.output_downsampling_factor", 2)
    writer.add_int32("xasr.decoder_dim", cfg["decoder_dim"])
    writer.add_int32("xasr.joiner_dim", cfg["joiner_dim"])
    writer.add_int32("xasr.context_size", cfg["context_size"])
    writer.add_int32("xasr.vocab_size", cfg["vocab_size"])
    writer.add_int32("xasr.blank_id", 0)
    writer.add_int32("xasr.sos_eos_id", 1)
    writer.add_bool("xasr.causal", True)
    writer.add_int32("xasr.left_context_frames", LEFT_CONTEXT_FRAMES)
    writer.add_int32("xasr.pad_length", PAD_LENGTH)
    writer.add_int32("frontend.sample_rate", 16000)
    writer.add_int32("frontend.num_mels", cfg["feature_dim"])
    writer.add_string("frontend.window", "povey")

    writer.add_token_list(tokens)
    writer.add_int32("tokenizer.vocab_size", cfg["vocab_size"])

    n_f16 = n_f32 = 0
    total_bytes = 0
    for key in sd:
        name = map_name(key)
        if name is None:
            continue
        t = sd[key]
        if not torch.is_tensor(t):
            t = torch.from_numpy(np.ascontiguousarray(t))
        # SimpleDownsample: store softmax(bias) directly as `.weight` so the
        # C++ side does a plain weighted sum (softmax(log w) == w for the
        # ONNX-extracted weights, softmax(bias) for a raw checkpoint).
        if name.endswith("downsample.bias") or name == "enc.downsample_output.bias":
            t = torch.softmax(t.to(torch.float64), dim=0)
            name = name[: -len(".bias")] + ".weight"
        arr = t.detach().to(torch.float32).numpy()
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if args.out_type == "f16" and want_f16(name, arr):
            arr = arr.astype(np.float16)
            n_f16 += 1
        else:
            arr = arr.astype(np.float32)
            n_f32 += 1
        arr = np.ascontiguousarray(arr)
        total_bytes += arr.nbytes
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}: {n_f16} f16 + {n_f32} f32 tensors, "
          f"{total_bytes/1e6:.1f} MB tensor data")


if __name__ == "__main__":
    main()
