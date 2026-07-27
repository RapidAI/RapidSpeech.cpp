#!/usr/bin/env python3
"""
Convert MOSS-Transcribe-Diarize (OpenMOSS-Team/MOSS-Transcribe-Diarize) to a
single GGUF file compatible with RapidSpeech.cpp's `MossTranscribeModel`.

Architecture: Whisper-Medium encoder (conv1d stem, 24 layers, d=1024)
              -> 4x temporal merge -> VQAdaptor (4096->1024, SiLU, LayerNorm)
              -> Qwen3-0.6B decoder (tied embeddings, no lm_head tensor).

Output layout:
  general.architecture = "MossTranscribeDiarize"
  mosstd.encoder.*  — Whisper encoder hparams
  mosstd.adaptor.*  — merge size / adaptor dims
  mosstd.audio.*    — special token ids + time-marker config
  qwen3.*           — LLM hparams (same keys as convert_qwen3_asr_to_gguf.py)
  tokenizer.*       — Qwen BPE vocab + merges
  Tensors:
    a.conv1d.{0,1}.{weight,bias}      — whisper_encoder.conv{1,2}
    a.position_embd.weight            — whisper_encoder.embed_positions.weight
    a.blk.<N>.{attn_{q,k,v,o},attn_norm,ffn_{up,down,norm}}.{weight,bias}
    a.post_norm.{weight,bias}         — whisper_encoder.layer_norm
    mm.a.mlp.{0,1}.{weight,bias}      — vq_adaptor.layers.{0,2}
    mm.a.norm.{weight,bias}           — vq_adaptor.layers.3 (LayerNorm)
    llm.model.embed_tokens.weight, llm.model.layers.N.*, llm.model.norm.weight
      (lm_head is tied to embed_tokens; the C++ loader aliases it at Load).

Memory-layout convention follows convert_qwen3_asr_to_gguf.py: PyTorch weights
keep their native C-order bytes; only the declared shape is reversed so ggml
sees ne0 as the fastest axis.

The safetensors file is read with a small pure-numpy parser (handles BF16),
so neither torch nor transformers is required.

Usage:
  python scripts/convert_moss_transcribe_to_gguf.py \
      --hf-dir /Volumes/z5_picture/resource/MOSS-Transcribe-Diarize \
      --output /Volumes/z5_picture/resource/moss-transcribe-diarize-f16.gguf \
      --f16

Requires: numpy, gguf.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter
except ImportError:
    print("Please install gguf: pip install gguf", file=sys.stderr)
    raise


# ----------------------------------------------------------------------------
# Pure-numpy safetensors reader (BF16-aware)
# ----------------------------------------------------------------------------

_ST_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8":  np.int8,
    "U8":  np.uint8,
    "BOOL": np.bool_,
}


def load_safetensors(path: Path) -> dict:
    """Read all tensors as float32 numpy arrays (BF16 upcast included)."""
    tensors = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            shape = meta["shape"]
            off0, off1 = meta["data_offsets"]
            f.seek(data_start + off0)
            raw = f.read(off1 - off0)
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            elif dtype in _ST_DTYPES:
                arr = np.frombuffer(raw, dtype=_ST_DTYPES[dtype]).astype(np.float32)
            else:
                sys.exit(f"Unsupported safetensors dtype {dtype} for {name}")
            tensors[name] = arr.reshape(shape)
    return tensors


# ----------------------------------------------------------------------------
# Tensor name mapping (HF -> GGUF)
# ----------------------------------------------------------------------------

def encoder_name_map(hf_name: str, n_enc_layers: int) -> str | None:
    n = hf_name
    if not n.startswith("model.whisper_encoder."):
        return None
    n = n[len("model.whisper_encoder."):]

    for i in (1, 2):
        if n == f"conv{i}.weight":
            return f"a.conv1d.{i - 1}.weight"
        if n == f"conv{i}.bias":
            return f"a.conv1d.{i - 1}.bias"

    if n == "embed_positions.weight":
        return "a.position_embd.weight"
    if n == "layer_norm.weight":
        return "a.post_norm.weight"
    if n == "layer_norm.bias":
        return "a.post_norm.bias"

    prefix = "layers."
    if n.startswith(prefix):
        rest = n[len(prefix):]
        idx_str, suffix = rest.split(".", 1)
        try:
            il = int(idx_str)
        except ValueError:
            return None
        if il < 0 or il >= n_enc_layers:
            return None
        base = f"a.blk.{il}."
        mapping = {
            "self_attn.q_proj.weight":     base + "attn_q.weight",
            "self_attn.q_proj.bias":       base + "attn_q.bias",
            "self_attn.k_proj.weight":     base + "attn_k.weight",
            # k_proj has no bias in Whisper
            "self_attn.v_proj.weight":     base + "attn_v.weight",
            "self_attn.v_proj.bias":       base + "attn_v.bias",
            "self_attn.out_proj.weight":   base + "attn_o.weight",
            "self_attn.out_proj.bias":     base + "attn_o.bias",
            "self_attn_layer_norm.weight": base + "attn_norm.weight",
            "self_attn_layer_norm.bias":   base + "attn_norm.bias",
            "fc1.weight":                  base + "ffn_up.weight",
            "fc1.bias":                    base + "ffn_up.bias",
            "fc2.weight":                  base + "ffn_down.weight",
            "fc2.bias":                    base + "ffn_down.bias",
            "final_layer_norm.weight":     base + "ffn_norm.weight",
            "final_layer_norm.bias":       base + "ffn_norm.bias",
        }
        return mapping.get(suffix)
    return None


def adaptor_name_map(hf_name: str) -> str | None:
    m = {
        "model.vq_adaptor.layers.0.weight": "mm.a.mlp.0.weight",
        "model.vq_adaptor.layers.0.bias":   "mm.a.mlp.0.bias",
        "model.vq_adaptor.layers.2.weight": "mm.a.mlp.1.weight",
        "model.vq_adaptor.layers.2.bias":   "mm.a.mlp.1.bias",
        "model.vq_adaptor.layers.3.weight": "mm.a.norm.weight",
        "model.vq_adaptor.layers.3.bias":   "mm.a.norm.bias",
    }
    return m.get(hf_name)


def llm_name_map(hf_name: str) -> str | None:
    prefix = "model.language_model."
    if hf_name.startswith(prefix):
        return "llm.model." + hf_name[len(prefix):]
    return None


# ----------------------------------------------------------------------------
# Tensor write helper (same policy as convert_qwen3_asr_to_gguf.py)
# ----------------------------------------------------------------------------

def _store(writer: GGUFWriter, name: str, t: np.ndarray, use_f16: bool):
    # NOTE: this gguf-py GGUFWriter already reverses the numpy shape into
    # ggml's ne order (ne0 fastest) when writing tensor info, so the PyTorch
    # C-order array is passed through as-is — no manual ravel/reverse (the
    # double reversal would corrupt every 2-D+ tensor's declared shape).
    eff_ndim = sum(1 for d in t.shape if d > 1)
    if use_f16 and eff_ndim >= 2 and not name.endswith(".bias") and \
            not name.endswith("norm.weight") and \
            not name.endswith("layer_norm.weight") and \
            name != "a.position_embd.weight":
        # position_embd stays F32: it is consumed by ggml_add, whose Metal
        # kernel requires an F32 src1.
        t = t.astype(np.float16)
    else:
        t = t.astype(np.float32)
    writer.add_tensor(name, t)


# ----------------------------------------------------------------------------
# Main conversion routine
# ----------------------------------------------------------------------------

def convert(hf_dir: Path, output: Path, use_f16: bool):
    print(f"Loading from {hf_dir}")
    with open(hf_dir / "config.json") as f:
        cfg = json.load(f)
    audio_cfg = cfg["audio_config"]
    text_cfg = cfg["text_config"]

    with open(hf_dir / "preprocessor_config.json") as f:
        pp = json.load(f)
    with open(hf_dir / "processor_config.json") as f:
        proc = json.load(f)

    # Encoder hparams
    n_mels       = int(audio_cfg.get("num_mel_bins", 80))
    enc_d_model  = int(audio_cfg.get("d_model", 1024))
    enc_n_head   = int(audio_cfg.get("encoder_attention_heads", 16))
    enc_n_layer  = int(audio_cfg.get("encoder_layers", 24))
    enc_ffn_dim  = int(audio_cfg.get("encoder_ffn_dim", 4096))
    enc_max_pos  = int(audio_cfg.get("max_source_positions", 1500))

    n_fft        = int(pp.get("n_fft", 400))
    hop_length   = int(pp.get("hop_length", 160))
    sample_rate  = int(pp.get("sampling_rate", 16000))
    chunk_frames = int(pp.get("nb_max_frames", 3000))

    merge_size   = int(cfg.get("audio_merge_size", 4))
    adaptor_in   = int(cfg.get("adaptor_input_dim", enc_d_model * merge_size))
    audio_tok_id = int(cfg.get("audio_token_id", 151671))

    # LLM hparams
    n_llm_layer   = int(text_cfg["num_hidden_layers"])
    n_llm_embd    = int(text_cfg["hidden_size"])
    n_llm_head    = int(text_cfg["num_attention_heads"])
    n_llm_head_kv = int(text_cfg["num_key_value_heads"])
    n_llm_vocab   = int(text_cfg["vocab_size"])
    n_llm_ff      = int(text_cfg["intermediate_size"])
    head_dim      = int(text_cfg["head_dim"])
    rope_base     = float(text_cfg.get("rope_theta", 1000000.0))
    rms_eps       = float(text_cfg.get("rms_norm_eps", 1e-6))
    n_ctx_train   = int(text_cfg.get("max_position_embeddings", 131072))

    print(f"Encoder: d={enc_d_model} h={enc_n_head} L={enc_n_layer} "
          f"ffn={enc_ffn_dim} mels={n_mels} chunk_frames={chunk_frames}")
    print(f"Adaptor: merge={merge_size} in={adaptor_in} out={n_llm_embd}")
    print(f"LLM:     L={n_llm_layer} d={n_llm_embd} h={n_llm_head} "
          f"head_dim={head_dim} vocab={n_llm_vocab}")

    writer = GGUFWriter(str(output), "MossTranscribeDiarize")
    writer.add_string("general.architecture", "MossTranscribeDiarize")
    writer.add_string("general.name", "MOSS-Transcribe-Diarize-0.9B")

    # Encoder
    writer.add_int32("mosstd.encoder.n_mels",       n_mels)
    writer.add_int32("mosstd.encoder.n_fft",        n_fft)
    writer.add_int32("mosstd.encoder.hop_length",   hop_length)
    writer.add_int32("mosstd.encoder.sample_rate",  sample_rate)
    writer.add_int32("mosstd.encoder.chunk_frames", chunk_frames)
    writer.add_int32("mosstd.encoder.d_model",      enc_d_model)
    writer.add_int32("mosstd.encoder.n_head",       enc_n_head)
    writer.add_int32("mosstd.encoder.n_layer",      enc_n_layer)
    writer.add_int32("mosstd.encoder.ffn_dim",      enc_ffn_dim)
    writer.add_int32("mosstd.encoder.max_source_positions", enc_max_pos)
    writer.add_float32("mosstd.encoder.layer_norm_eps", 1e-5)

    # Adaptor / audio splicing
    writer.add_int32("mosstd.adaptor.merge_size", merge_size)
    writer.add_int32("mosstd.adaptor.in_dim",     adaptor_in)
    writer.add_float32("mosstd.adaptor.norm_eps", rms_eps)
    writer.add_int32("mosstd.audio.pad_token_id",   audio_tok_id)
    writer.add_int32("mosstd.audio.start_token_id", 151669)
    writer.add_int32("mosstd.audio.end_token_id",   151670)
    writer.add_int32("mosstd.time_marker.enable",
                     1 if proc.get("enable_time_marker", True) else 0)
    writer.add_int32("mosstd.time_marker.every_seconds",
                     int(proc.get("time_marker_every_seconds", 5)))
    writer.add_float32("mosstd.audio.tokens_per_second",
                       float(proc.get("audio_tokens_per_second", 12.5)))

    # LLM (Qwen3) hparams — same keys as convert_qwen3_asr_to_gguf.py
    writer.add_int32("qwen3.block_count",                        n_llm_layer)
    writer.add_int32("qwen3.embedding_length",                   n_llm_embd)
    writer.add_int32("qwen3.attention.head_count",               n_llm_head)
    writer.add_int32("qwen3.attention.head_count_kv",            n_llm_head_kv)
    writer.add_int32("qwen3.attention.key_length",               head_dim)
    writer.add_int32("qwen3.attention.value_length",             head_dim)
    writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", rms_eps)
    writer.add_int32("qwen3.feed_forward_length",                n_llm_ff)
    writer.add_int32("qwen3.context_length",                     n_ctx_train)
    writer.add_float32("qwen3.rope.freq_base",                   rope_base)
    writer.add_int32("qwen3.bos_token_id", 151643)
    writer.add_int32("qwen3.eos_token_id", 151645)

    # ---- Tokenizer (vocab.json + tokenizer.json added_tokens + merges) -----
    with open(hf_dir / "vocab.json", encoding="utf-8") as f:
        base_vocab = json.load(f)
    id_to_token = {int(tid): text for text, tid in base_vocab.items()}
    with open(hf_dir / "tokenizer.json", encoding="utf-8") as f:
        tok_json = json.load(f)
    for added in tok_json.get("added_tokens", []):
        id_to_token[int(added["id"])] = added["content"]

    max_id = max(id_to_token)
    tokens = [id_to_token.get(i, "") for i in range(max_id + 1)]
    writer.add_int32("tokenizer.vocab_size", len(tokens))
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_string("tokenizer.unk_symbol", "<|endoftext|>")

    merges_file = hf_dir / "merges.txt"
    if merges_file.exists():
        with open(merges_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and not ln.startswith("#")]
        writer.add_array("tokenizer.ggml.merges", lines)

    # ---- Tensors -----------------------------------------------------------
    st_files = sorted(hf_dir.glob("model-*.safetensors")) or \
               [hf_dir / "model.safetensors"]
    state = {}
    for fp in st_files:
        print(f"  Loading {fp.name}")
        state.update(load_safetensors(fp))

    converted_enc = converted_mm = converted_llm = skipped = 0
    for name in sorted(state.keys()):
        t = state[name]
        gguf_name = encoder_name_map(name, enc_n_layer)
        if gguf_name is not None:
            _store(writer, gguf_name, t, use_f16)
            converted_enc += 1
            continue
        gguf_name = adaptor_name_map(name)
        if gguf_name is not None:
            _store(writer, gguf_name, t, use_f16)
            converted_mm += 1
            continue
        gguf_name = llm_name_map(name)
        if gguf_name is not None:
            _store(writer, gguf_name, t, use_f16)
            converted_llm += 1
            continue
        print(f"  skipping unmapped tensor: {name}")
        skipped += 1

    print(f"Encoder tensors: {converted_enc}")
    print(f"Adaptor tensors: {converted_mm}")
    print(f"LLM tensors:     {converted_llm}  (lm_head tied to embed_tokens)")
    print(f"Skipped:         {skipped}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWritten {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True,
                        help="Local MOSS-Transcribe-Diarize checkpoint dir")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--f16", action="store_true",
                        help="Store 2-D weights as F16 (default F32)")
    args = parser.parse_args()
    hf_dir = Path(args.hf_dir).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    convert(hf_dir, output, args.f16)
