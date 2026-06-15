#!/usr/bin/env python3
"""
Convert OpenVoice2 (MeloTTS) models to GGUF format.
Reads from a local MeloTTS checkpoint/config pair, or directly from cached
HuggingFace PyTorch checkpoints (.pth).

Usage:
  python3 convert_openvoice2_v2.py --lang ZH --output-dir ./models [--f16]
  python3 convert_openvoice2_v2.py --lang ZH \
      --checkpoint /path/to/G_266000.pth \
      --config /path/to/config.json \
      --output models/openvoice2-custom.gguf [--f16]
"""

import argparse, json, os, struct, sys
import numpy as np

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

class GGUFWriter:
    def __init__(self):
        self.kv_data = []
        self.tensors = []

    def add_string(self, key, value):
        self.kv_data.append(("string", key, value))

    def add_int32(self, key, value):
        self.kv_data.append(("int32", key, value))

    def add_float32(self, key, value):
        self.kv_data.append(("float32", key, value))

    def add_array(self, key, values):
        """Store an array of int32 values."""
        self.kv_data.append(("int32_array", key, list(values)))

    def _write_string(self, f, s):
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)

    def add_tensor(self, name, data, dtype=GGML_TYPE_F32):
        if dtype == GGML_TYPE_F16:
            data = data.astype(np.float16)
        else:
            data = data.astype(np.float32)
        self.tensors.append((name, list(data.shape), dtype, data.tobytes()))

    def write(self, path):
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
                    f.write(struct.pack("<I", 5))  # GGUF_TYPE_INT32 = 5
                    f.write(struct.pack("<i", kv[2]))
                elif kv[0] == "float32":
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 6))
                    f.write(struct.pack("<f", kv[2]))
                elif kv[0] == "int32_array":
                    # GGUF_TYPE_ARRAY = 9, element type GGUF_TYPE_INT32 = 5
                    self._write_string(f, kv[1])
                    f.write(struct.pack("<I", 9))   # array type
                    f.write(struct.pack("<I", 5))   # element type = int32
                    f.write(struct.pack("<Q", len(kv[2])))  # count
                    for v in kv[2]:
                        f.write(struct.pack("<i", v))
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


def get_cached_model_path(lang):
    """Find the cached HuggingFace model path and checkpoint file."""
    if lang == "ZH":
        repo = "models--myshell-ai--MeloTTS-Chinese"
    else:
        repo = "models--myshell-ai--MeloTTS-English"

    base = os.path.expanduser(f"~/.cache/huggingface/hub/{repo}")
    if not os.path.exists(base):
        raise FileNotFoundError(f"Model not found at {base}")

    snapshots = os.path.join(base, "snapshots")
    snapshot = os.listdir(snapshots)[0]
    model_dir = os.path.join(snapshots, snapshot)

    # Find checkpoint file (.pth or .pt)
    for fname in os.listdir(model_dir):
        if fname.endswith(".pth") or fname.endswith(".pt"):
            return model_dir, os.path.join(model_dir, fname)
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def find_checkpoint(model_dir):
    """Find a MeloTTS checkpoint in a local model directory."""
    candidates = []
    for fname in os.listdir(model_dir):
        if fname.endswith(".pth") or fname.endswith(".pt"):
            candidates.append(fname)
    if not candidates:
        raise FileNotFoundError(f"No .pth/.pt checkpoint found in {model_dir}")
    # Prefer MeloTTS generator checkpoints such as G_266000.pth.
    candidates.sort(key=lambda name: (not name.startswith("G_"), name))
    return os.path.join(model_dir, candidates[0])


def resolve_model_paths(args):
    """Resolve checkpoint/config paths for either local or HF-cache conversion."""
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        model_dir = args.model_dir or os.path.dirname(os.path.abspath(checkpoint_path))
    elif args.model_dir:
        model_dir = args.model_dir
        checkpoint_path = find_checkpoint(model_dir)
    else:
        model_dir, checkpoint_path = get_cached_model_path(args.lang)

    config_path = args.config or os.path.join(model_dir, "config.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json not found: {config_path}\n"
            "MeloTTS conversion needs the matching training config.json "
            "for symbols, tones, languages, sample rate, and model dimensions."
        )
    return model_dir, checkpoint_path, config_path


def torch_load(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(checkpoint):
    """Unwrap common PyTorch/Lightning checkpoint containers."""
    state_dict = checkpoint
    for key in ("model", "state_dict", "module"):
        if isinstance(state_dict, dict) and key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
    return state_dict


def reshape_pt_to_ggml(arr):
    """Reverse PyTorch shape to get ggml-compatible layout.

    Keeps raw bytes identical — only changes the declared shape.
    PyTorch C-order: last dim varies fastest. ggml: ne[0] varies fastest.
    So reversing the shape makes them equivalent with no data movement.

    - 2D [out,in] -> [in,out]: arr.ravel().reshape(in, out)
    - 3D [OC,IC,K] -> [K,IC,OC]: arr.ravel().reshape(K, IC, OC)
    - 4D [A,B,C,D] -> [D,C,B,A]
    - 1D: unchanged
    """
    if arr.ndim <= 1:
        return arr
    return arr.ravel().reshape(arr.shape[::-1])


def effective_ndim(arr):
    """Number of dimensions ignoring size-1 dims (for dtype decision)."""
    return sum(1 for d in arr.shape if d > 1)


def should_reshape(name, arr):
    """Check if a weight tensor needs PyTorch->ggml shape reordering."""
    if arr.ndim <= 1:
        return False
    # Skip bias terms
    if name.endswith(".bias"):
        return False
    # Skip weight_norm scale (per-channel scalar), but NOT _v (direction matrix)
    if name.endswith(".weight_g") or name.endswith("_g"):
        return False
    return True


def convert_base_tts(writer, state_dict, dtype):
    """Convert MeloTTS/VITS base model weights with PyTorch->ggml reshape."""
    converted = 0
    for k, v in state_dict.items():
        # Determine dtype: 1D tensors (biases, norms) forced to F32
        eff_dim = effective_ndim(v)
        if eff_dim <= 1:
            t_dtype = GGML_TYPE_F32
        else:
            t_dtype = dtype if dtype == GGML_TYPE_F16 else GGML_TYPE_F32

        if k.startswith("enc_p."):
            name = k.replace("enc_p.", "text_encoder.")
        elif k.startswith("dp."):
            name = k.replace("dp.", "duration_predictor.")
        elif k.startswith("flow."):
            name = k.replace("flow.", "flow_decoder.")
        elif k.startswith("enc_q."):
            name = k.replace("enc_q.", "posterior_encoder.")
        elif k.startswith("dec."):
            name = k.replace("dec.", "vocoder.")
        elif k.startswith("emb_"):
            name = k
            t_dtype = GGML_TYPE_F32  # embeddings always F32
        else:
            name = k

        # Reshape weight layout: PyTorch C-order (last dim fastest)
        # -> ggml (ne[0] fastest). Raw bytes stay the same.
        if should_reshape(name, v):
            v = reshape_pt_to_ggml(v)

        writer.add_tensor(name, v, t_dtype)
        converted += 1
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True, choices=["ZH", "EN"],
                        help="Language: ZH or EN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Local MeloTTS .pth/.pt checkpoint, e.g. G_266000.pth")
    parser.add_argument("--config", type=str, default=None,
                        help="Matching MeloTTS config.json. Defaults to <model-dir>/config.json")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Local directory containing config.json and a .pth/.pt checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output GGUF path. Defaults to <output-dir>/openvoice2-base-<lang>.gguf")
    parser.add_argument("--output-dir", type=str, default="./models")
    parser.add_argument("--f16", action="store_true", help="Store large weights as F16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dtype = GGML_TYPE_F16 if args.f16 else GGML_TYPE_F32

    # Resolve local checkpoint/config, or fall back to the cached HF model.
    try:
        model_dir, checkpoint_path, config_path = resolve_model_paths(args)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    print(f"Loading model from: {checkpoint_path}")

    # Load config
    with open(config_path) as f:
        config = json.load(f)
    data_cfg = config["data"]
    model_cfg = config["model"]

    # Load state dict from PyTorch checkpoint
    try:
        checkpoint = torch_load(checkpoint_path)
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            sys.exit("Python package 'torch' is required: python3 -m pip install torch")
        raise
    # The checkpoint may be a dict with "model" or "state_dict" key, or the state_dict itself.
    state_dict = unwrap_state_dict(checkpoint)

    # Convert torch tensors to numpy
    numpy_state = {}
    for k, v in state_dict.items():
        numpy_state[k] = v.float().numpy()
    print(f"Loaded {len(numpy_state)} tensors")

    # Get symbols directly from config
    symbols = config.get("symbols", [])
    num_tones = config.get("num_tones", 0)
    num_languages = config.get("num_languages", 0)
    print(f"Loaded {len(symbols)} symbols, {num_tones} tones, {num_languages} languages from config.json")

    # Create writer
    writer = GGUFWriter()
    writer.add_string("general.architecture", "openvoice2")
    writer.add_string("general.name", f"OpenVoice2 TTS ({args.lang})")
    writer.add_string("general.language", "zh" if args.lang == "ZH" else "en")

    # Metadata from actual model config
    writer.add_int32("openvoice2.hidden_channels", model_cfg["hidden_channels"])
    writer.add_int32("openvoice2.inter_channels", model_cfg["inter_channels"])
    writer.add_int32("openvoice2.filter_channels", model_cfg["filter_channels"])
    writer.add_int32("openvoice2.n_heads", model_cfg["n_heads"])
    writer.add_int32("openvoice2.n_layers", model_cfg["n_layers"])
    writer.add_int32("openvoice2.n_flow_layers", model_cfg.get("n_layers_trans_flow", 4))
    writer.add_int32("openvoice2.sample_rate", data_cfg["sampling_rate"])
    writer.add_int32("openvoice2.hop_length", data_cfg["hop_length"])
    writer.add_int32("openvoice2.n_fft", data_cfg["filter_length"])
    writer.add_int32("openvoice2.n_mels", model_cfg.get("n_mels", model_cfg["hidden_channels"]))
    writer.add_int32("openvoice2.num_tones", num_tones)
    writer.add_int32("openvoice2.num_languages", num_languages)

    # Store upsample_rates so the C++ runtime can use the correct strides
    upsample_rates = model_cfg.get("upsample_rates", [])
    if upsample_rates:
        writer.add_array("openvoice2.upsample_rates", upsample_rates)

    # Write symbol list
    writer.add_string("openvoice2.symbols", ",".join(symbols))

    n_conv = convert_base_tts(writer, numpy_state, dtype)
    dtype_name = "F16" if args.f16 else "F32"
    print(f"Converted {n_conv} tensors (dtype={dtype_name})")

    out_path = args.output or os.path.join(args.output_dir, f"openvoice2-base-{args.lang.lower()}.gguf")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer.write(out_path)


if __name__ == "__main__":
    main()
