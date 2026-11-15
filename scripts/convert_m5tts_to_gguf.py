#!/usr/bin/env python3
"""Convert M5-TTS (ASLP-lab/M5-TTS-50M-VoiceClone) + Qwen 12Hz codec to GGUF.

Sources:
  --m5-dir      M5 checkpoint bundle (model.safetensors, config.json,
                conditioning.json, tokenizer/tokenizer.json)
  --codec-dir   Qwen3-TTS 12Hz audio tokenizer bundle (model.safetensors,
                config.json)
  --output      GGUF path (one file holds the LM, both codec halves, the
                tokenizer and the RoPE tables)

Tensor naming (GGUF):
    text_embeddings.weight / audio_embeddings.weight   embedding tables
    projection.weight / codebook0_head.weight / audio_head
    backbone.layers.{n}.<module>                       llama-30M backbone
    decoder.layers.{n}.<module>                        llama-20M MTP decoder
    backbone.inv_freq / decoder.inv_freq               llama3-rope-scaled table
    codec.<name>                                       Qwen codec decoder +
                                                       encoder (RVQ, SEANet,
                                                       transformers)
    tokenizer.ggml.tokens / tokenizer.ggml.merges      llama3 byte-level BPE

Linear weights are stored PT C-order [out, in] -> ggml ne=[in, out];
mul_mat(W, x) == F.linear(x, W).  Embedding tables stay [dim, vocab] so
ggml_get_rows works directly.  Conv1d kernels [out_c, in_c, k] -> ne=[k,
in_c, out_c]; depthwise [out_c, 1, k] -> ne=[k, 1, out_c]; ConvTranspose1d
[in_c, out_c, k] -> ne=[k, out_c, in_c].

dtype policy: --out-type f16 stores the 2-D transformer projections in f16
and everything else (embeddings, heads, conv kernels, norms, scales,
codebooks, SnakeBeta params) in f32.

Quantization: run rs-quantize on the f16 GGUF, e.g.
    ./build/rs-quantize m5tts-f16.gguf m5tts-q4_k_m.gguf q4_k_m
The quantizer skips the embedding tables, heads, conv kernels, codebooks
and per-channel parameters; the transformer projections quantize cleanly.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from gguf import GGUFWriter


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------

def read_all(path):
    """Load a safetensors file as tensors (bf16-safe via torch)."""
    from safetensors.torch import load_file
    return load_file(path)


def arr_f32(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().to(torch.float32).numpy()
    return np.ascontiguousarray(t.astype(np.float32))


# --------------------------------------------------------------------------
# llama3 RoPE (HF 'llama3' rope_type: frequency rescaling with
# factor=32, low_freq_factor=1, high_freq_factor=4 over the original
# 2048 positions) — precomputed here so the C++ side needs no formula.
# --------------------------------------------------------------------------

def llama3_inv_freq(head_dim, theta, factor, low_freq_factor, high_freq_factor,
                    original_max):
    half = head_dim // 2
    inv = []
    low_wavelen = original_max / low_freq_factor
    high_wavelen = original_max / high_freq_factor
    for i in range(half):
        # HF llama3: inv_freq = 1 / theta^(2i / head_dim)
        freq = theta ** (-2.0 * i / head_dim)
        wavelen = 2.0 * np.pi / freq
        if wavelen < high_wavelen:
            f = freq
        elif wavelen > low_wavelen:
            f = freq / factor
        else:
            smooth = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            f = (1.0 - smooth) * freq / factor + smooth * freq
        inv.append(f)
    return np.array(inv, dtype=np.float32)


def plain_inv_freq(head_dim, theta):
    half = head_dim // 2
    return np.array([theta ** (-2.0 * i / head_dim) for i in range(half)],
                    dtype=np.float32)


# --------------------------------------------------------------------------
# tensor policy
# --------------------------------------------------------------------------

F16_SUFFIX = (
    ".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
    ".mlp.fc1.weight", ".mlp.fc2.weight",
    "input_proj.weight", "output_proj.weight",
)


def want_f16(name, arr, out_type):
    if out_type != "f16":
        return False
    if arr.ndim != 2:
        return False
    for suf in F16_SUFFIX:
        if name.endswith(suf):
            # conv kernels are 3-D so this only matches true 2-D linears;
            # quantizer input/output_proj are Conv1d k=1 kernels — keep f32
            if "quantizer" in name:
                return False
            return True
    return False


# --------------------------------------------------------------------------
# main conversion
# --------------------------------------------------------------------------

def map_lm(key, cfg):
    """LM safetensors key -> gguf name."""
    if key in ("text_embeddings.weight", "audio_embeddings.weight",
               "audio_head", "projection.weight", "codebook0_head.weight"):
        return key
    if key.startswith(("backbone.", "decoder.")):
        return key
    return None


def map_codec(key):
    """Codec safetensors key -> gguf name (codec. prefix)."""
    name = key
    # unify codebook parameter names
    name = name.replace("._codebook.", ".codebook.")
    name = name.replace(".codebook.embed_sum", ".codebook.embedding_sum")
    name = name.replace(".codebook.initialized", ".codebook.initialized")
    return "codec." + name


SKIP_CODEC_KEYS = re.compile(r"\.codebook\.initialized$")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m5-dir", required=True,
                    help="M5 checkpoint directory (model.safetensors, config.json, conditioning.json, tokenizer/)")
    ap.add_argument("--codec-dir", required=True,
                    help="Qwen3-TTS 12Hz audio tokenizer directory (model.safetensors, config.json)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--out-type", choices=["f32", "f16"], default="f16")
    args = ap.parse_args()

    m5_dir = os.path.expanduser(args.m5_dir)
    codec_dir = os.path.expanduser(args.codec_dir)

    lm = read_all(os.path.join(m5_dir, "model.safetensors"))
    codec = read_all(os.path.join(codec_dir, "model.safetensors"))
    cfg = json.load(open(os.path.join(m5_dir, "config.json")))
    cond = json.load(open(os.path.join(m5_dir, "conditioning.json")))
    ccfg = json.load(open(os.path.join(codec_dir, "config.json")))
    dcfg = ccfg["decoder_config"]
    ecfg = ccfg["encoder_config"]

    writer = GGUFWriter(args.output, "m5-tts")
    writer.add_string("general.name", "m5-tts-50m-voiceclone")

    # ---- LM metadata ----
    writer.add_int32("m5tts.backbone_layers", 8)
    writer.add_int32("m5tts.backbone_heads", 8)
    writer.add_int32("m5tts.backbone_kv_heads", 2)
    writer.add_int32("m5tts.backbone_head_dim", 64)
    writer.add_int32("m5tts.backbone_ff", 2048)
    writer.add_int32("m5tts.decoder_layers", 5)
    writer.add_int32("m5tts.decoder_heads", 8)
    writer.add_int32("m5tts.decoder_kv_heads", 2)
    writer.add_int32("m5tts.decoder_head_dim", 64)
    writer.add_int32("m5tts.decoder_ff", 2048)
    writer.add_int32("m5tts.hidden_size", 512)
    writer.add_int32("m5tts.text_vocab", 129992)
    writer.add_int32("m5tts.audio_vocab", 2051)
    writer.add_int32("m5tts.codebooks", 16)
    writer.add_int32("m5tts.max_seq_len", 2048)
    writer.add_float32("m5tts.norm_eps", 1e-5)
    writer.add_float32("m5tts.temperature", 0.9)
    writer.add_int32("m5tts.topk", 10)
    writer.add_int32("m5tts.min_frames", 6)
    writer.add_int32("m5tts.sample_rate", 24000)
    writer.add_int32("m5tts.hop_length", 1920)
    writer.add_int32("m5tts.max_frames", 500)
    for tok, tid in cond["control_token_ids"].items():
        writer.add_int32("m5tts.special." + tok.strip("<|>"), tid)
    writer.add_int32("m5tts.bos_token_id", cond["bos_token_id"])
    writer.add_int32("m5tts.eos_token_id", cond["eos_token_id"])

    # ---- codec metadata (decoder transformer) ----
    writer.add_int32("codec.layers", dcfg["num_hidden_layers"])
    writer.add_int32("codec.heads", dcfg["num_attention_heads"])
    writer.add_int32("codec.kv_heads", dcfg["num_key_value_heads"])
    writer.add_int32("codec.head_dim", dcfg["head_dim"])
    writer.add_int32("codec.hidden_size", dcfg["hidden_size"])
    writer.add_int32("codec.ff_dim", dcfg["intermediate_size"])
    writer.add_int32("codec.latent_dim", dcfg["latent_dim"])
    writer.add_int32("codec.decoder_dim", dcfg["decoder_dim"])
    writer.add_int32("codec.codebook_dim", dcfg["codebook_dim"])
    writer.add_int32("codec.codebook_size", dcfg["codebook_size"])
    writer.add_int32("codec.semantic_codebook_size", dcfg["semantic_codebook_size"])
    writer.add_int32("codec.num_quantizers", dcfg["num_quantizers"])
    writer.add_int32("codec.window", dcfg["sliding_window"])
    writer.add_float32("codec.rope_theta", dcfg["rope_theta"])
    writer.add_float32("codec.norm_eps", dcfg["rms_norm_eps"])
    writer.add_array("codec.upsample_rates", list(dcfg["upsample_rates"]))
    writer.add_array("codec.upsampling_ratios", list(dcfg["upsampling_ratios"]))

    # ---- codec metadata (encoder / SEANet + transformer) ----
    writer.add_int32("codec.enc_layers", ecfg["num_hidden_layers"])
    writer.add_int32("codec.enc_heads", ecfg["num_attention_heads"])
    writer.add_int32("codec.enc_kv_heads", ecfg["num_key_value_heads"])
    writer.add_int32("codec.enc_head_dim", ecfg["head_dim"])
    writer.add_int32("codec.enc_hidden_size", ecfg["hidden_size"])
    writer.add_int32("codec.enc_ff_dim", ecfg["intermediate_size"])
    writer.add_int32("codec.enc_window", ecfg["sliding_window"])
    writer.add_float32("codec.enc_rope_theta", float(ecfg["rope_theta"]))
    writer.add_float32("codec.enc_norm_eps", ecfg["norm_eps"])
    writer.add_int32("codec.enc_num_filters", ecfg["num_filters"])
    writer.add_int32("codec.enc_kernel_size", ecfg["kernel_size"])
    writer.add_int32("codec.enc_compress", ecfg["compress"])
    writer.add_int32("codec.enc_valid_quantizers", ccfg["encoder_valid_num_quantizers"])
    writer.add_array("codec.enc_upsampling_ratios", list(ecfg["upsampling_ratios"]))

    # ---- tokenizer (llama3 byte-level BPE) ----
    tpath = os.path.join(m5_dir, "tokenizer", "tokenizer.json")
    if not os.path.exists(tpath):
        tpath = os.path.join(os.path.dirname(m5_dir), "tokenizer.json")
    tj = json.load(open(tpath))
    vocab = tj["model"]["vocab"]
    tokens = [None] * 129992
    for tok, tid in vocab.items():
        tokens[tid] = tok
    for added in tj["added_tokens"]:
        tokens[added["id"]] = added["content"]
    assert all(t is not None for t in tokens), "tokenizer vocab has holes"
    merges = tj["model"]["merges"]
    writer.add_token_list(tokens)
    writer.add_array("tokenizer.ggml.merges", ["\x01".join(pair) if isinstance(pair, list) else pair
                                               for pair in merges])
    writer.add_int32("tokenizer.vocab_size", 129992)
    byte2uni = {}
    n_bytes = 0
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = [chr(b) for b in bs]
    n = 0
    for b in range(256):
        if b not in bs:
            cs.append(chr(256 + n))
            n += 1
    for b, c in zip(range(256), cs):
        byte2uni[b] = c

    # ---- RoPE tables ----
    writer.add_tensor("backbone.inv_freq",
                      llama3_inv_freq(64, 500000.0, 32.0, 1.0, 4.0, 2048))
    writer.add_tensor("decoder.inv_freq",
                      llama3_inv_freq(64, 500000.0, 32.0, 1.0, 4.0, 2048))
    writer.add_tensor("codec.inv_freq",
                      plain_inv_freq(dcfg["head_dim"], float(dcfg["rope_theta"])))
    writer.add_tensor("codec.encoder.inv_freq",
                      plain_inv_freq(ecfg["head_dim"], float(ecfg["rope_theta"])))

    # ---- LM tensors ----
    n_f16 = n_f32 = 0
    total = 0
    for key in sorted(lm):
        name = map_lm(key, cfg)
        if name is None:
            print("skip LM:", key)
            continue
        arr = arr_f32(lm[key])
        if want_f16(name, arr, args.out_type):
            arr = arr.astype(np.float16)
            n_f16 += 1
        else:
            n_f32 += 1
        total += arr.nbytes
        writer.add_tensor(name, arr)

    # ---- codec tensors ----
    for key in sorted(codec):
        if SKIP_CODEC_KEYS.search(key):
            continue
        name = map_codec(key)
        arr = arr_f32(codec[key])
        if want_f16(name, arr, args.out_type):
            arr = arr.astype(np.float16)
            n_f16 += 1
        else:
            n_f32 += 1
        total += arr.nbytes
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}: {n_f16} f16 + {n_f32} f32 tensors, "
          f"{total / 1e6:.1f} MB tensor data")


if __name__ == "__main__":
    main()
