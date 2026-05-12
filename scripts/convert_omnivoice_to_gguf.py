#!/usr/bin/env python3
"""
Convert OmniVoice TTS model (k2-fsa/OmniVoice) to GGUF format.

OmniVoice architecture:
  Text Tokenizer (Qwen3 vocab) ──┐
                                 ├→ Bidirectional Transformer (Qwen3-0.6B)
  Acoustic Tokens (8 codebooks) ──┘   ↓
                                 8× Codebook-Specific Prediction Heads
                                     ↓  32-step iterative diffusion
                                 Audio Tokenizer Decoder (Higgs)
                                     ↓
                                 24 kHz waveform

Usage:
  python convert_omnivoice_to_gguf.py \
    --model-dir ./OmniVoice \
    --output omnivoice.gguf \
    [--f16] [--q8]

If the official model isn't available yet, this script supports converting
from a directory containing PyTorch checkpoint files.

Requirements:
  pip install torch transformers numpy
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

# OmniVoice defaults (Qwen3-0.6B + 8 codebooks)
DEFAULT_N_LAYER = 28
DEFAULT_N_EMBD = 1024
DEFAULT_N_HEAD = 16
DEFAULT_N_HEAD_KV = 4
DEFAULT_HEAD_DIM = 128
DEFAULT_N_CODEBOOKS = 8
DEFAULT_AUDIO_SR = 24000
DEFAULT_DIFF_STEPS = 32
DEFAULT_ROPE_THETA = 1000000.0
DEFAULT_TEXT_VOCAB_SIZE = 151936
DEFAULT_CODEBOOK_SIZE = 2048  # per codebook vocab size (Higgs tokenizer)


class GGUFWriter:
    """Minimal GGUF writer."""

    def __init__(self):
        self.kv_data = []
        self.tensors = []

    def add_string(self, key: str, value: str):
        self.kv_data.append(("string", key, value))

    def add_int32(self, key: str, value: int):
        self.kv_data.append(("int32", key, value))

    def add_float32(self, key: str, value: float):
        self.kv_data.append(("float32", key, value))

    def _write_string(self, f, s: str):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def add_tensor(self, name: str, data: np.ndarray, dtype=GGML_TYPE_F32):
        if dtype == GGML_TYPE_F16:
            data = data.astype(np.float16)
        elif dtype == GGML_TYPE_Q8_0:
            data = _quantize_q8_0(data.astype(np.float32))
        else:
            data = data.astype(np.float32)
        self.tensors.append((name, list(data.shape), dtype, data.tobytes()))

    def write(self, path: str):
        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv_data)))

            for kv in self.kv_data:
                if kv[0] == "string":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 8))
                    self._write_string(f, kv[2])
                elif kv[0] == "int32":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 4))
                    f.write(struct.pack("<i", kv[2]))
                elif kv[0] == "float32":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 6))
                    f.write(struct.pack("<f", kv[2]))

            data_offset = 0
            tensor_offsets = []
            for name, shape, dtype, data_bytes in self.tensors:
                self._write_string(f, name)
                n_dims = len(shape)
                f.write(struct.pack("<I", n_dims))
                for dim in shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", dtype))
                aligned = (data_offset + 31) & ~31
                f.write(struct.pack("<Q", aligned))
                tensor_offsets.append(aligned)
                data_offset = aligned + len(data_bytes)

            current_pos = f.tell()
            aligned_start = (current_pos + 31) & ~31
            f.write(b"\x00" * (aligned_start - current_pos))
            data_base = f.tell()

            for i, (name, shape, dtype, data_bytes) in enumerate(self.tensors):
                target_pos = data_base + tensor_offsets[i]
                current = f.tell()
                if current < target_pos:
                    f.write(b"\x00" * (target_pos - current))
                f.write(data_bytes)

        print(f"Written {path} ({len(self.tensors)} tensors)")


def _quantize_q8_0(data: np.ndarray) -> np.ndarray:
    """Q8_0 block quantization matching ggml format (32-value blocks with fp16 scale)."""
    data = data.astype(np.float32).reshape(-1)
    n = data.size
    block_size = 32
    n_blocks = (n + block_size - 1) // block_size
    out = np.zeros(n_blocks * (2 + block_size), dtype=np.uint8)
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = data[start:end]
        amax = np.max(np.abs(block)).astype(np.float32)
        if amax == 0:
            amax = np.float32(1.0)
        scale_f16 = np.float16(amax / 127.0)
        block_out = np.zeros(block_size, dtype=np.int8)
        pad = block_size - (end - start)
        for i, val in enumerate(block):
            block_out[i] = max(-127, min(127, int(round(val / float(scale_f16)))))
        off = b * (2 + block_size)
        out[off:off + 2] = np.frombuffer(scale_f16.tobytes(), dtype=np.uint8)
        out[off + 2:off + 2 + block_size] = block_out.view(np.uint8)
        if pad > 0:
            out[off + 2 + end - start:off + 2 + block_size] = 0
    return out


def _get_dtype(use_f16: bool, use_q8: bool):
    if use_q8:
        return GGML_TYPE_Q8_0
    elif use_f16:
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def _add_tensor_weighted(writer, name, data, dtype):
    """Add tensor with dtype selection for weights."""
    if dtype != GGML_TYPE_F32 and data.ndim >= 2:
        writer.add_tensor(name, data, dtype)
    else:
        writer.add_tensor(name, data, GGML_TYPE_F32)


def convert_hf_omnivoice(writer, model, dtype):
    """
    Convert HuggingFace OmniVoice model to GGUF tensors.

    Expected HF model structure:
      - model.embed_tokens.weight          (text embedding, shared with Qwen3)
      - model.layers.{i}.input_layernorm.weight
      - model.layers.{i}.self_attn.q_proj.weight / k_proj.weight / v_proj.weight / o_proj.weight
      - model.layers.{i}.post_attention_layernorm.weight
      - model.layers.{i}.mlp.gate_proj.weight / up_proj.weight / down_proj.weight
      - model.norm.weight                  (final layer norm)
      - codebook_heads.{c}.weight / bias    (per-codebook prediction heads)
      - acoustic_embeddings.{c}.weight      (per-codebook acoustic embeddings)
      - vocoder.decoder.conv1.weight / bias (Higgs decoder)
      - ...
    """
    converted = 0
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # --- Qwen3 backbone (via qwen3.blk naming) ---
    token_embd = state_dict.get("model.embed_tokens.weight")
    if token_embd is not None:
        writer.add_tensor("qwen3.model.embed_tokens.weight",
                          token_embd.numpy(), GGML_TYPE_F32)
        converted += 1

    output_norm = state_dict.get("model.norm.weight")
    if output_norm is not None:
        writer.add_tensor("output_norm.weight", output_norm.numpy(), GGML_TYPE_F32)
        converted += 1

    # Transformer layers
    for i in range(DEFAULT_N_LAYER):
        prefix_hf = f"model.layers.{i}."
        prefix_gguf = f"qwen3.blk.{i}."
        layer_mappings = [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("input_layernorm.bias", "input_layernorm.bias"),
            ("self_attn.q_proj.weight", "self_attn.q_proj.weight"),
            ("self_attn.q_proj.bias", "self_attn.q_proj.bias"),
            ("self_attn.k_proj.weight", "self_attn.k_proj.weight"),
            ("self_attn.k_proj.bias", "self_attn.k_proj.bias"),
            ("self_attn.v_proj.weight", "self_attn.v_proj.weight"),
            ("self_attn.v_proj.bias", "self_attn.v_proj.bias"),
            ("self_attn.o_proj.weight", "self_attn.o_proj.weight"),
            ("post_attention_layernorm.weight", "post_attention_layernorm.weight"),
            ("post_attention_layernorm.bias", "post_attention_layernorm.bias"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
        ]
        for hf_suffix, gguf_suffix in layer_mappings:
            tensor = state_dict.get(prefix_hf + hf_suffix)
            if tensor is not None:
                t_dtype = dtype if "weight" in hf_suffix and tensor.ndim >= 2 else GGML_TYPE_F32
                writer.add_tensor(prefix_gguf + gguf_suffix, tensor.numpy(), t_dtype)
                converted += 1

    # --- Codebook prediction heads ---
    for c in range(DEFAULT_N_CODEBOOKS):
        head_w = state_dict.get(f"codebook_heads.{c}.weight")
        if head_w is not None:
            writer.add_tensor(f"codebook_head.{c}.weight", head_w.numpy(), dtype)
            converted += 1
        head_b = state_dict.get(f"codebook_heads.{c}.bias")
        if head_b is not None:
            writer.add_tensor(f"codebook_head.{c}.bias", head_b.numpy(), GGML_TYPE_F32)
            converted += 1

    # --- Acoustic codebook embeddings ---
    for c in range(DEFAULT_N_CODEBOOKS):
        emb_w = state_dict.get(f"acoustic_embeddings.{c}.weight")
        if emb_w is not None:
            writer.add_tensor(f"acoustic_embd.{c}.weight", emb_w.numpy(), GGML_TYPE_F32)
            converted += 1

    # --- Vocoder (Higgs decoder) ---
    vocoder_mappings = [
        ("vocoder.decoder.conv1.weight", "vocoder.decoder.conv1.weight"),
        ("vocoder.decoder.conv1.bias", "vocoder.decoder.conv1.bias"),
        ("vocoder.decoder.conv2.weight", "vocoder.decoder.conv2.weight"),
        ("vocoder.decoder.conv2.bias", "vocoder.decoder.conv2.bias"),
        ("vocoder.decoder.conv3.weight", "vocoder.decoder.conv3.weight"),
        ("vocoder.decoder.conv3.bias", "vocoder.decoder.conv3.bias"),
        ("vocoder.decoder.conv_post.weight", "vocoder.decoder.conv_post.weight"),
        ("vocoder.decoder.conv_post.bias", "vocoder.decoder.conv_post.bias"),
    ]
    for hf_name, gguf_name in vocoder_mappings:
        tensor = state_dict.get(hf_name)
        if tensor is not None:
            t_dtype = dtype if "weight" in hf_name and tensor.ndim >= 2 else GGML_TYPE_F32
            writer.add_tensor(gguf_name, tensor.numpy(), t_dtype)
            converted += 1

    # --- Upsample weights (if present) ---
    for k, v in state_dict.items():
        if k.startswith("vocoder.decoder.upsamples."):
            t_dtype = dtype if v.ndim >= 2 else GGML_TYPE_F32
            writer.add_tensor(k, v.numpy(), t_dtype)
            converted += 1
        if k.startswith("vocoder.decoder.resblocks."):
            t_dtype = dtype if v.ndim >= 2 else GGML_TYPE_F32
            writer.add_tensor(k, v.numpy(), t_dtype)
            converted += 1

    return converted


def convert_pytorch_checkpoints(writer, model_dir: Path, dtype):
    """Convert from raw PyTorch checkpoint files."""
    converted = 0
    ckpt_files = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
    if not ckpt_files:
        ckpt_files = sorted(model_dir.glob("*.bin")) + sorted(model_dir.glob("*.safetensors"))

    for ckpt in ckpt_files:
        print(f"  Loading {ckpt.name} ...")
        if ckpt.suffix == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(str(ckpt))
        else:
            import torch
            state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=True)

        for name, tensor in state_dict.items():
            t = tensor.cpu().numpy() if hasattr(tensor, "numpy") else np.array(tensor)
            use_dtype = dtype if t.ndim >= 2 else GGML_TYPE_F32
            writer.add_tensor(name, t, use_dtype)
            converted += 1

    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert OmniVoice TTS to GGUF")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to HuggingFace model or checkpoint directory")
    parser.add_argument("--output", type=str, default="omnivoice.gguf",
                        help="Output GGUF file path")
    parser.add_argument("--f16", action="store_true",
                        help="Store large weights in float16")
    parser.add_argument("--q8", action="store_true",
                        help="Store large weights in Q8_0 quantized format")
    parser.add_argument("--hf", action="store_true",
                        help="Load as HuggingFace transformers model")
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-head-kv", type=int, default=DEFAULT_N_HEAD_KV)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--n-codebooks", type=int, default=DEFAULT_N_CODEBOOKS)
    parser.add_argument("--codebook-size", type=int, default=DEFAULT_CODEBOOK_SIZE)
    parser.add_argument("--audio-sr", type=int, default=DEFAULT_AUDIO_SR)
    parser.add_argument("--diff-steps", type=int, default=DEFAULT_DIFF_STEPS)
    parser.add_argument("--rope-theta", type=float, default=DEFAULT_ROPE_THETA)
    parser.add_argument("--text-vocab-size", type=int, default=DEFAULT_TEXT_VOCAB_SIZE)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    dtype = _get_dtype(args.f16, args.q8)
    dtype_name = {GGML_TYPE_F32: "F32", GGML_TYPE_F16: "F16", GGML_TYPE_Q8_0: "Q8_0"}[dtype]

    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}")
        sys.exit(1)

    writer = GGUFWriter()

    # --- Metadata ---
    writer.add_string("general.architecture", "OmniVoice")
    writer.add_string("general.name", "OmniVoice TTS")
    writer.add_string("general.description",
                      "OmniVoice: Omnilingual Zero-Shot TTS with Diffusion Language Models")

    # --- OmniVoice hyperparameters ---
    writer.add_int32("omnivoice.block_count", args.n_layer)
    writer.add_int32("omnivoice.embedding_length", args.n_embd)
    writer.add_int32("omnivoice.attention.head_count", args.n_head)
    writer.add_int32("omnivoice.attention.head_count_kv", args.n_head_kv)
    writer.add_int32("omnivoice.attention.key_length", args.head_dim)
    writer.add_int32("omnivoice.codebook_count", args.n_codebooks)
    writer.add_float32("omnivoice.rope.freq_base", args.rope_theta)
    writer.add_int32("omnivoice.audio_sample_rate", args.audio_sr)
    writer.add_int32("omnivoice.diffusion_steps", args.diff_steps)
    writer.add_float32("omnivoice.diffusion_tau", 0.1)

    # Codebook sizes
    for c in range(args.n_codebooks):
        writer.add_int32(f"omnivoice.codebook.{c}.size", args.codebook_size)

    # Tokenizer
    writer.add_int32("tokenizer.vocab_size", args.text_vocab_size)

    # --- Convert weights ---
    print(f"Model directory: {model_dir}")
    print(f"Load method: {'HuggingFace' if args.hf else 'raw checkpoints'}")

    if args.hf:
        try:
            import torch
            from transformers import AutoModel
        except ImportError:
            print("Error: for --hf mode, install transformers: pip install transformers torch")
            sys.exit(1)

        print(f"Loading HuggingFace model from {model_dir} ...")
        model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)
        n_converted = convert_hf_omnivoice(writer, model, dtype)
    else:
        n_converted = convert_pytorch_checkpoints(writer, model_dir, dtype)

    print(f"Converted {n_converted} tensors (dtype={dtype_name})")
    writer.write(str(output_path))
    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
