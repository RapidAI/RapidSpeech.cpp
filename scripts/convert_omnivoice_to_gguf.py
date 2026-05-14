#!/usr/bin/env python3
"""
Convert OmniVoice TTS model (k2-fsa/OmniVoice) to single GGUF file.

Merges both the LLM backbone (model.safetensors) and audio tokenizer
(audio_tokenizer/model.safetensors) into one GGUF file.

Usage:
  python convert_omnivoice_to_gguf.py \
    --model-dir ./OmniVoice \
    --output omnivoice.gguf \
    [--f16] [--q8]

Requirements:
  pip install safetensors numpy
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

# OmniVoice defaults (Qwen3-0.6B + 8 codebooks)
DEFAULT_N_LAYER = 28
DEFAULT_N_EMBD = 1024
DEFAULT_N_HEAD = 16
DEFAULT_N_HEAD_KV = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_N_CODEBOOKS = 8
DEFAULT_AUDIO_SR = 24000
DEFAULT_DIFF_STEPS = 32
DEFAULT_ROPE_THETA = 1000000.0
DEFAULT_TEXT_VOCAB_SIZE = 151936
DEFAULT_CODEBOOK_SIZE = 1025


class GGUFWriter:
    """Minimal GGUF writer."""

    def __init__(self):
        self.kv_data = []
        self.tensors = []

    def add_string(self, key: str, value: str):
        self.kv_data.append(("string", key, value))

    def add_int32(self, key: str, value: int):
        self.kv_data.append(("int32", key, value))

    def add_uint32(self, key: str, value: int):
        self.kv_data.append(("uint32", key, value))

    def add_float32(self, key: str, value: float):
        self.kv_data.append(("float32", key, value))

    def add_string_array(self, key: str, values: list):
        self.kv_data.append(("string_array", key, values))

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
                # ggml GGUF type codes: UINT32=4, INT32=5
                elif kv[0] == "uint32":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 4))  # GGUF_TYPE_UINT32
                    f.write(struct.pack("<I", kv[2]))
                elif kv[0] == "int32":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 5))  # GGUF_TYPE_INT32
                    f.write(struct.pack("<i", kv[2]))
                elif kv[0] == "float32":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 6))
                    f.write(struct.pack("<f", kv[2]))
                elif kv[0] == "string_array":
                    # GGUF_TYPE_ARRAY(9) with element type GGUF_TYPE_STRING(8)
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 9))  # type: array
                    f.write(struct.pack("<I", 8))  # element type: string
                    f.write(struct.pack("<Q", len(kv[2])))  # count
                    for s in kv[2]:
                        self._write_string(f, s)

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
        for i, val in enumerate(block):
            block_out[i] = max(-127, min(127, int(round(val / float(scale_f16)))))
        off = b * (2 + block_size)
        out[off:off + 2] = np.frombuffer(scale_f16.tobytes(), dtype=np.uint8)
        out[off + 2:off + 2 + block_size] = block_out.view(np.uint8)
        if end - start < block_size:
            out[off + 2 + end - start:off + 2 + block_size] = 0
    return out


def _get_dtype(use_f16: bool, use_q8: bool):
    if use_q8:
        return GGML_TYPE_Q8_0
    elif use_f16:
        return GGML_TYPE_F16
    return GGML_TYPE_F32


def _tensor_dtype(t: np.ndarray, default_dtype):
    """Use F32 for biases and effectively-1-D tensors (e.g. [C] or [1,C,1]),
       default_dtype for >=2-D weights."""
    # Count dimensions with size > 1 (effective dimensionality)
    eff_ndim = sum(1 for d in t.shape if d > 1)
    if eff_ndim >= 2:
        return default_dtype
    return GGML_TYPE_F32


# Tensor names in the tokenizer safetensors that are runtime state (not inference weights)
SKIP_TOKENIZER_TENSORS = {
    "cluster_size", "embed_avg", "inited",
}

# Pos-conv parametrization tensors that should be merged into a single weight
POS_CONV_ORIGINAL0 = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
POS_CONV_ORIGINAL1 = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
POS_CONV_WEIGHT = "semantic_model.encoder.pos_conv_embed.conv.weight"


def _should_skip_tokenizer_tensor(name: str) -> bool:
    """Filter out runtime state tensors from the tokenizer checkpoint."""
    for skip_kw in SKIP_TOKENIZER_TENSORS:
        if skip_kw in name:
            return True
    # Skip parametrization tensors (merged into single weight)
    if "parametrizations" in name:
        return True
    return False


def convert_from_safetensors(model_dir: Path, writer: GGUFWriter, dtype):
    """
    Load both model.safetensors and audio_tokenizer/model.safetensors,
    merge into a single GGUF with all tensors.
    """
    converted = 0

    # --- Load main model (LLM backbone + codebook heads + acoustic embeddings) ---
    main_path = model_dir / "model.safetensors"
    if not main_path.exists():
        print(f"Error: model.safetensors not found in {model_dir}")
        sys.exit(1)

    import safetensors.torch
    print(f"  Loading {main_path.name} ...")
    main_state = safetensors.torch.load_file(str(main_path), device="cpu")

    for name, tensor in main_state.items():
        t = tensor.cpu().numpy()
        # Reorder weight matrices for ggml memory layout (ne0 varies fastest).
        # Numpy C-order has last dim fastest; ggml has first dim fastest.
        # The raw bytes of PT tensors are already correct for ggml:
        #   PT [OC,IC,K] C-order: K fastest → ggml [K,IC,OC] reads K fastest ✓
        #   PT [out,in] C-order: in fastest → ggml [in,out] reads in fastest ✓
        # We just need to change the declared shape via ravel().reshape().
        if t.ndim == 2 and (
            name.startswith("llm.layers.") or
            name in ("llm.embed_tokens.weight", "llm.norm.weight", "lm_head.weight",
                      "audio_embeddings.weight", "audio_heads.weight")
        ):
            t = t.ravel().reshape(t.shape[1], t.shape[0])
        elif t.ndim == 3:
            t = t.ravel().reshape(t.shape[2], t.shape[1], t.shape[0])
        dt = _tensor_dtype(t, dtype)
        writer.add_tensor(name, t, dt)
        converted += 1

    # Load config.json for audio_codebook_weights metadata
    config_path = model_dir / "config.json"
    codebook_weights_info = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        codebook_weights_info = config.get("audio_codebook_weights", None)

    # --- Load audio tokenizer (codec: RVQ + DAC + HuBERT + SemanticEncoder) ---
    tokenizer_path = model_dir / "audio_tokenizer" / "model.safetensors"
    if tokenizer_path.exists():
        print(f"  Loading {tokenizer_path.name} from audio_tokenizer/ ...")
        tok_state = safetensors.torch.load_file(str(tokenizer_path), device="cpu")

        # Separate pos_conv parametrization tensors for merging
        pos_conv_0 = None
        pos_conv_1 = None

        for name, tensor in tok_state.items():
            if _should_skip_tokenizer_tensor(name):
                if name == POS_CONV_ORIGINAL0:
                    pos_conv_0 = tensor.cpu().numpy()
                elif name == POS_CONV_ORIGINAL1:
                    pos_conv_1 = tensor.cpu().numpy()
                continue

            t = tensor.cpu().numpy()
            # Reorder weight matrices for ggml memory layout (ne0 varies fastest).
            # PT C-order has last dim fastest; ggml has first dim fastest.
            # rav el().reshape() changes the declared shape while keeping
            # the same raw bytes (which already have the correct layout).
            if t.ndim == 2 and (name.endswith(".weight") or name.endswith(".embed")):
                t = t.ravel().reshape(t.shape[1], t.shape[0])
            elif t.ndim == 3:
                t = t.ravel().reshape(t.shape[2], t.shape[1], t.shape[0])
            dt = _tensor_dtype(t, dtype)
            writer.add_tensor(name, t, dt)
            converted += 1

        # Merge pos_conv parametrization: weight = original0 * original1
        if pos_conv_0 is not None and pos_conv_1 is not None:
            # original0: [1, 1, 128], original1: [768, 48, 128]
            # Broadcasting: [1,1,128] * [768,48,128] -> [768,48,128]
            # Result is PyTorch [OC, IC, K]; reorder to ggml [K, IC, OC]
            merged = pos_conv_0 * pos_conv_1
            merged = merged.ravel().reshape(merged.shape[2], merged.shape[1], merged.shape[0])
            dt = _tensor_dtype(merged, dtype)
            writer.add_tensor(POS_CONV_WEIGHT, merged, dt)
            converted += 1
            print(f"  Merged pos_conv_embed: {pos_conv_0.shape} * {pos_conv_1.shape} -> {merged.shape}")
        elif pos_conv_0 is not None or pos_conv_1 is not None:
            print("  WARNING: only one pos_conv parametrization tensor found, skipping merge")
    else:
        print(f"  WARNING: audio_tokenizer/model.safetensors not found in {model_dir}")
        print(f"  Only LLM backbone tensors will be included (no vocoder/codec).")

    return converted


def load_tokenizer_data(model_dir: Path, writer: GGUFWriter):
    """Load BPE tokenizer data from tokenizer.json and add as GGUF metadata."""
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"  WARNING: tokenizer.json not found, BPE tokenizer will not be available")
        return

    import json
    with open(tokenizer_path) as f:
        tok = json.load(f)

    model = tok.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])

    if not vocab:
        print(f"  WARNING: empty vocab in tokenizer.json")
        return

    # Build sorted token list (by ID)
    max_id = max(vocab.values())
    tokens = [""] * (max_id + 1)
    for token_str, tid in vocab.items():
        tokens[tid] = token_str

    # Read config to determine the actual model vocab size
    config_path = model_dir / "config.json"
    llm_vocab_size = len(tokens)
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        llm_config = cfg.get("llm_config", {})
        llm_vocab_size = llm_config.get("vocab_size", len(tokens))

    # Extend tokens array to cover full model vocab (includes added special tokens)
    if llm_vocab_size > len(tokens):
        tokens.extend([""] * (llm_vocab_size - len(tokens)))
        print(f"  Extended token list from {max_id+1} to {llm_vocab_size} entries")

    # Map extra special token strings to GGUF metadata key suffixes
    special_key_map = {
        "<|denoise|>": "denoise",
        "<|lang_start|>": "lang_start",
        "<|lang_end|>": "lang_end",
        "<|instruct_start|>": "instruct_start",
        "<|instruct_end|>": "instruct_end",
        "<|text_start|>": "text_start",
        "<|text_end|>": "text_end",
    }

    # Determine special token IDs and insert their strings into tokens array
    tok_cfg_path = model_dir / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        extra_specials = tok_cfg.get("extra_special_tokens", [])
        if extra_specials:
            try:
                from tokenizers import Tokenizer
                hf_tok = Tokenizer.from_file(str(tokenizer_path))
                for sp in extra_specials:
                    tid = hf_tok.token_to_id(sp)
                    if tid is not None and tid < len(tokens):
                        tokens[tid] = sp  # set the token string at its ID
                        key = special_key_map.get(sp, sp.strip("<>").replace("|", "_"))
                        writer.add_uint32(f"omnivoice.special.{key}", tid)
                        print(f"  Added omnivoice.special.{key}={tid}")
            except ImportError:
                print(f"  WARNING: tokenizers library not available, estimating special token IDs")
                base_vocab_size = max_id + 1
                for i, sp in enumerate(extra_specials):
                    tid = base_vocab_size + i
                    if tid < len(tokens):
                        tokens[tid] = sp
                    key = special_key_map.get(sp, sp.strip("<>").replace("|", "_"))
                    writer.add_uint32(f"omnivoice.special.{key}", tid)
                    print(f"  Estimated omnivoice.special.{key}={tid}")
    else:
        print(f"  WARNING: tokenizer_config.json not found, special tokens not added")

    # Now write tokenizer.ggml.tokens (after extending and adding special token strings)
    writer.add_string_array("tokenizer.ggml.tokens", tokens)
    print(f"  Added tokenizer.ggml.tokens ({len(tokens)} tokens)")

    # Add tokenizer.ggml.merges (convert from list format ["a","b"] to string "a b")
    merge_strings = []
    for m in merges:
        if isinstance(m, list):
            merge_strings.append(" ".join(m))
        else:
            merge_strings.append(str(m))
    writer.add_string_array("tokenizer.ggml.merges", merge_strings)
    print(f"  Added tokenizer.ggml.merges ({len(merge_strings)} merges)")

    # Update vocab size to match the actual model vocab
    writer.add_uint32("tokenizer.vocab_size", llm_vocab_size)


def main():
    parser = argparse.ArgumentParser(description="Convert OmniVoice TTS to GGUF")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to directory containing model.safetensors + audio_tokenizer/")
    parser.add_argument("--output", type=str, default="omnivoice.gguf",
                        help="Output GGUF file path")
    parser.add_argument("--f16", action="store_true",
                        help="Store large weights in float16")
    parser.add_argument("--q8", action="store_true",
                        help="Store large weights in Q8_0 quantized format")
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

    # Auto-detect vocab size from model config
    text_vocab_size = args.text_vocab_size
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        llm_cfg = cfg.get("llm_config", {})
        detected_vocab = llm_cfg.get("vocab_size", 0)
        if detected_vocab > 0:
            if args.text_vocab_size == DEFAULT_TEXT_VOCAB_SIZE:
                text_vocab_size = detected_vocab
                print(f"Auto-detected vocab_size={text_vocab_size} from config.json")
            elif args.text_vocab_size != detected_vocab:
                print(f"WARNING: --text-vocab-size={args.text_vocab_size} but config says {detected_vocab}")

    # OmniVoice LLM hyperparameters (keys expected by LoadLM in omnivoice.cpp)
    # Use uint32 to match gguf_get_val_u32() on C++ side
    writer.add_uint32("omnivoice-lm.block_count", args.n_layer)
    writer.add_uint32("omnivoice-lm.embedding_length", args.n_embd)
    writer.add_uint32("omnivoice-lm.attention.head_count", args.n_head)
    writer.add_uint32("omnivoice-lm.attention.head_count_kv", args.n_head_kv)
    writer.add_uint32("omnivoice-lm.attention.key_length", args.head_dim)
    writer.add_uint32("omnivoice-lm.feed_forward_length", 3072)
    writer.add_float32("omnivoice-lm.rope.freq_base", args.rope_theta)
    writer.add_float32("omnivoice-lm.attention.layer_norm_rms_epsilon", 1e-6)
    writer.add_uint32("omnivoice-lm.vocab_size", text_vocab_size)

    # Audio codec metadata
    writer.add_uint32("omnivoice.num_audio_codebook", args.n_codebooks)
    writer.add_uint32("omnivoice.audio_vocab_size", args.codebook_size)
    writer.add_uint32("omnivoice.audio_mask_id", args.codebook_size - 1)
    writer.add_uint32("omnivoice.audio_sample_rate", args.audio_sr)
    writer.add_uint32("omnivoice.diffusion_steps", args.diff_steps)
    writer.add_float32("omnivoice.diffusion_tau", 0.1)

    # Codec parameters
    writer.add_uint32("omnivoice.codebook_size", args.codebook_size)
    writer.add_uint32("omnivoice.codebook_dim", 64)

    # Codebook sizes (per-codebook)
    for c in range(args.n_codebooks):
        writer.add_uint32(f"omnivoice.codebook.{c}.size", args.codebook_size)

    # Load tokenizer data (BPE tokens + merges + special token IDs)
    # This also sets tokenizer.vocab_size so don't set it separately
    load_tokenizer_data(model_dir, writer)

    # --- Convert weights ---
    print(f"Model directory: {model_dir}")
    print(f"Output: {output_path}")

    n_converted = convert_from_safetensors(model_dir, writer, dtype)

    print(f"Converted {n_converted} tensors (dtype={dtype_name})")
    writer.write(str(output_path))
    print(f"\nDone! Output: {output_path}")


if __name__ == "__main__":
    main()
