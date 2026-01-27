#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import numpy as np
import torch
import contextlib
from pathlib import Path
from typing import Iterator
from gguf import GGUFWriter, GGMLQuantizationType

# Mapping of SenseVoice internal architecture names to GGUF strings
ARCH_MAP = {
    "SenseVoiceSmall": "SenseVoiceSmall",
    "Funasr-nano": "Funasr-nano",
}
def load_cmvn(mvn_path: str):
    """
    Parses the am.mvn file.
    Usually contains <AddShift> (means) and <Rescale> (vars).
    """
    means = None
    vars = None
    try:
        with open(mvn_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "<AddShift>" in line:
                    # Extract numbers between [ ]
                    data = lines[i+1].split('[')[1].split(']')[0].split()
                    means = np.array([float(x) for x in data], dtype=np.float32)
                if "<Rescale>" in line:
                    data = lines[i+1].split('[')[1].split(']')[0].split()
                    vars = np.array([float(x) for x in data], dtype=np.float32)
        return means, vars
    except Exception as e:
        print(f"Error loading CMVN from {mvn_path}: {e}")
        return None, None

def get_tensors(model_dir: Path):
    """Loads weights from model.pt and optional silero_vad.pt"""
    checkpoints = ["model.pt", "silero_vad.pt"]
    for ckpt in checkpoints:
        path = model_dir / ckpt
        if not path.exists():
            continue

        print(f"gguf: loading weights from '{ckpt}'")
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        for name, data in state_dict.items():
            yield name, data

def write_tensor(writer: GGUFWriter, name: str, data_torch: torch.Tensor, ftype: int):
    """Handles data type conversion and split logic for QKV tensors"""
    # Split merged QKV linear layers as per reference
    if 'linear_q_k_v' in name:
        q_k_v = data_torch.split(data_torch.size(0) // 3)
        write_tensor(writer, name.replace('linear_q_k_v', 'linear_q'), q_k_v[0], ftype)
        write_tensor(writer, name.replace('linear_q_k_v', 'linear_k'), q_k_v[1], ftype)
        write_tensor(writer, name.replace('linear_q_k_v', 'linear_v'), q_k_v[2], ftype)
        return

    # Convert to numpy
    data = data_torch.float().numpy()

    # Specific type handling for FSMN and specific VAD layers as per reference
    if 'fsmn_block.weight' in name or name.startswith('_model.'):
        data = data.astype(np.float16)
    elif ftype == GGMLQuantizationType.F16 and data.ndim == 2 and name.endswith(".weight"):
        data = data.astype(np.float16)
    else:
        data = data.astype(np.float32)

    writer.add_tensor(name, data)

def main():
    parser = argparse.ArgumentParser(description="Convert SenseVoice model to GGUF")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with config.yaml, model.pt, am.mvn")
    parser.add_argument("--output", type=str, required=True, help="Output .gguf file path")
    parser.add_argument("--out-type", type=str, choices=["f32", "f16"], default="f32")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    ftype = GGMLQuantizationType.F32 if args.out_type == "f32" else GGMLQuantizationType.F16

    # 1. Load Config
    config_path = model_dir / "config.yaml"
    with open(config_path, "r") as f:
        hparams = yaml.safe_load(f)

    arch_name = ARCH_MAP.get(hparams.get("model", "SenseVoiceSmall"), "sensevoice")
    writer = GGUFWriter(args.output, arch_name)

    # 2. Set Parameters from YAML (Frontend & Encoder)
    fconf = hparams.get("frontend_conf", {})
    econf = hparams.get("encoder_conf", {})

    writer.add_int32("frontend.sample_rate", fconf.get("fs", 16000))
    writer.add_string("frontend.window", fconf.get("window", "hamming"))
    writer.add_int32("frontend.num_mels", fconf.get("n_mels", 80))
    writer.add_int32("frontend.lfr_m", fconf.get("lfr_m", 7))
    writer.add_int32("frontend.lfr_n", fconf.get("lfr_n", 6))

    writer.add_int32("encoder.output_size", econf.get("output_size", 512))
    writer.add_int32("encoder.attention_heads", econf.get("attention_heads", 8))
    writer.add_int32("encoder.linear_units", econf.get("linear_units", 2048))
    writer.add_int32("encoder.num_blocks", econf.get("num_blocks", 50))
    writer.add_int32("encoder.tp_blocks", econf.get("tp_blocks", 10))

    # 3. Load and Add CMVN from am.mvn
    mvn_path = model_dir / "am.mvn"
    if mvn_path.exists():
        means, vars = load_cmvn(str(mvn_path))
        if means is not None and vars is not None:
            print(f"Writing CMVN metadata from {mvn_path}...")
            # Convert NumPy arrays to lists to satisfy GGUFWriter's requirement for a sequence
            writer.add_array("model.cmvn_means", means.tolist())
            writer.add_array("model.cmvn_vars", vars.tolist())

    # 4. Set Vocabulary
    vocab_model = model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"
    if vocab_model.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(vocab_model))
        tokens = [sp.id_to_piece(i).replace(" ", " ") for i in range(sp.vocab_size())]
        writer.add_int32("tokenizer.vocab_size", sp.vocab_size())
        writer.add_token_list(tokens)
        writer.add_string("tokenizer.unk_symbol", "<unk>")

    # 5. Write Tensors
    print("Writing tensors...")
    for name, data in get_tensors(model_dir):
        if not name.endswith(("Loss", "loss")):
            write_tensor(writer, name, data, ftype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Successfully exported to {args.output}")

if __name__ == "__main__":
    main()