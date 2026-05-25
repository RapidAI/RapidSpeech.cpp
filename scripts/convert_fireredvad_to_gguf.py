#!/usr/bin/env python3
"""
Convert FireRedVAD (DFSMN) to GGUF.

Input:
    --model-dir DIR     directory containing `model.pth.tar` and `cmvn.ark`
    --output FILE       output .gguf path

GGUF tensor names produced:
    cmvn.means                              [D]
    cmvn.inv_std                            [D]
    dfsmn.fc1.weight                        [H, D]
    dfsmn.fc1.bias                          [H]
    dfsmn.fc2.weight                        [P, H]
    dfsmn.fc2.bias                          [P]
    dfsmn.fsmn1.lookback.weight             [P, 1, N1]
    dfsmn.fsmns.{i}.fc1.weight              [H, P]    (i in 0..R-2)
    dfsmn.fsmns.{i}.fc1.bias                [H]
    dfsmn.fsmns.{i}.fc2.weight              [P, H]    (no bias)
    dfsmn.fsmns.{i}.fsmn.lookback.weight    [P, 1, N1]
    dnns.{m}.weight                         [H, P] or [H, H]
    dnns.{m}.bias                           [H]
    out.weight                              [odim, H]
    out.bias                                [odim]

Linear weights stored in PyTorch C-order [out, in]; GGML reads ne=[in, out],
so ggml_mul_mat(W, x) computes W^T @ x = F.linear(x, W).
Depthwise Conv1d weight is PT [P, 1, N1] (groups=P, in/groups=1, kernel=N1);
stored C-order. GGML reads ne=[N1, 1, P], which is exactly what
ggml_conv_1d_dw expects.

CMVN convention:
    means stores +mean (positive); FireRedVadModel computes (x - mean) * inv_std.
    inv_std = 1 / sqrt(max(var, 1e-20)).
"""
import argparse
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter


def read_kaldi_cmvn(cmvn_path):
    """Parse a Kaldi binary cmvn.ark file.

    Two layouts are accepted (kaldiio handles both transparently):
        1. ark archive with utterance key:
            utt_key SP \\0 B FM ... data
        2. raw binary matrix without key:
            \\0 B FM ... data

    We scan for the \\0B binary marker so we don't depend on a leading key.
    Bytes after the marker:
        ascii type tag terminated by SP: "FM " (float) or "DM " (double)
        u8 size_marker (=4) + i32 num_rows
        u8 size_marker (=4) + i32 num_cols
        num_rows * num_cols * elem_size bytes, row-major

    Returns (means: np.ndarray[D], inv_std: np.ndarray[D]).
    """
    with open(cmvn_path, "rb") as f:
        data = f.read()

    # Locate the '\0B' binary header anywhere in the first few hundred bytes.
    marker_idx = data.find(b"\x00B")
    if marker_idx < 0 or marker_idx > 512:
        raise RuntimeError(
            f"CMVN ark: \\0B binary marker not found near start of {cmvn_path}")
    pos = marker_idx + 2

    # Matrix type token, e.g., 'FM ' (float) or 'DM ' (double).
    type_end = data.find(b" ", pos)
    if type_end < 0:
        raise RuntimeError(f"CMVN ark: matrix type tag not found in {cmvn_path}")
    type_tok = data[pos:type_end].decode("ascii", errors="replace")
    pos = type_end + 1
    if type_tok == "FM":
        dtype = np.float32
        elem_size = 4
    elif type_tok == "DM":
        dtype = np.float64
        elem_size = 8
    else:
        raise RuntimeError(f"Unsupported CMVN matrix type: {type_tok!r}")

    # 1 byte size marker (=4), then int32 num_rows, then 1 byte, int32 num_cols.
    if data[pos] != 4:
        raise RuntimeError(f"Unexpected int marker byte: {data[pos]}")
    pos += 1
    (num_rows,) = struct.unpack("<i", data[pos:pos + 4])
    pos += 4
    if data[pos] != 4:
        raise RuntimeError(f"Unexpected int marker byte: {data[pos]}")
    pos += 1
    (num_cols,) = struct.unpack("<i", data[pos:pos + 4])
    pos += 4

    expected_bytes = num_rows * num_cols * elem_size
    matrix = np.frombuffer(data[pos:pos + expected_bytes], dtype=dtype)
    matrix = matrix.reshape(num_rows, num_cols).astype(np.float64)

    if num_rows != 2:
        raise RuntimeError(
            f"CMVN ark expected 2 rows (sums, sum-of-squares); got {num_rows}")

    dim = num_cols - 1
    count = matrix[0, dim]
    if count < 1:
        raise RuntimeError(f"CMVN ark frame count too small: {count}")
    floor = 1e-20

    means = np.empty(dim, dtype=np.float32)
    inv_std = np.empty(dim, dtype=np.float32)
    for d in range(dim):
        mean = matrix[0, d] / count
        var = (matrix[1, d] / count) - mean * mean
        if var < floor:
            var = floor
        means[d] = np.float32(mean)
        inv_std[d] = np.float32(1.0 / math.sqrt(var))
    return means, inv_std


def convert(model_dir: str, output: str):
    model_dir = Path(model_dir)
    model_pth = model_dir / "model.pth.tar"
    cmvn_ark = model_dir / "cmvn.ark"
    if not model_pth.exists():
        sys.exit(f"Missing weights: {model_pth}")
    if not cmvn_ark.exists():
        sys.exit(f"Missing CMVN: {cmvn_ark}")

    print(f"Loading weights from {model_pth}")
    pkg = torch.load(str(model_pth), map_location="cpu", weights_only=False)
    args = pkg["args"]
    sd = pkg["model_state_dict"]

    idim = int(args.idim)
    odim = int(args.odim)
    R = int(args.R)
    M = int(args.M)
    H = int(args.H)
    P = int(args.P)
    N1 = int(args.N1)
    S1 = int(args.S1)
    N2 = int(getattr(args, "N2", 0))
    S2 = int(getattr(args, "S2", 0))
    LP = (N1 - 1) * S1
    print(f"DFSMN config: idim={idim} odim={odim} R={R} M={M} H={H} P={P} "
          f"N1={N1} S1={S1} N2={N2} S2={S2} lookback_padding={LP}")

    print(f"Reading CMVN from {cmvn_ark}")
    means, inv_std = read_kaldi_cmvn(cmvn_ark)
    if means.shape[0] != idim:
        sys.exit(f"CMVN dim {means.shape[0]} mismatches idim={idim}")

    writer = GGUFWriter(output, "firered-vad")
    writer.add_string("general.name", "firered-vad")

    writer.add_int32("vad.sample_rate", 16000)
    writer.add_int32("vad.frame_length_ms", 25)
    writer.add_int32("vad.frame_shift_ms", 10)
    writer.add_float32("vad.speech_threshold", 0.5)
    writer.add_int32("vad.smooth_window_size", 5)
    writer.add_int32("vad.pad_start_frame", 5)
    writer.add_int32("vad.min_speech_frame", 8)
    writer.add_int32("vad.max_speech_frame", 2000)
    writer.add_int32("vad.min_silence_frame", 20)
    writer.add_int32("vad.chunk_max_frame", 30000)

    writer.add_int32("firered-vad.idim", idim)
    writer.add_int32("firered-vad.odim", odim)
    writer.add_int32("firered-vad.R", R)
    writer.add_int32("firered-vad.M", M)
    writer.add_int32("firered-vad.H", H)
    writer.add_int32("firered-vad.P", P)
    writer.add_int32("firered-vad.N1", N1)
    writer.add_int32("firered-vad.S1", S1)
    writer.add_int32("firered-vad.N2", N2)
    writer.add_int32("firered-vad.S2", S2)
    writer.add_int32("firered-vad.lookback_padding", LP)

    def add(name, tensor, expect_shape=None):
        arr = tensor.detach().cpu().float().numpy().astype(np.float32)
        if expect_shape is not None and list(arr.shape) != list(expect_shape):
            sys.exit(f"Shape mismatch for {name}: got {list(arr.shape)} "
                     f"expected {list(expect_shape)}")
        arr = np.ascontiguousarray(arr)
        writer.add_tensor(name, arr)
        print(f"  + {name:50s} {list(arr.shape)}")

    # CMVN
    writer.add_tensor("cmvn.means", np.ascontiguousarray(means))
    writer.add_tensor("cmvn.inv_std", np.ascontiguousarray(inv_std))
    print(f"  + {'cmvn.means':50s} {[idim]}")
    print(f"  + {'cmvn.inv_std':50s} {[idim]}")

    # Initial fc1, fc2 (each is Sequential[Linear, ReLU, Dropout] -> index .0.)
    add("dfsmn.fc1.weight", sd["dfsmn.fc1.0.weight"], [H, idim])
    add("dfsmn.fc1.bias",   sd["dfsmn.fc1.0.bias"],   [H])
    add("dfsmn.fc2.weight", sd["dfsmn.fc2.0.weight"], [P, H])
    add("dfsmn.fc2.bias",   sd["dfsmn.fc2.0.bias"],   [P])

    # Initial FSMN (no skip connection in PT code)
    add("dfsmn.fsmn1.lookback.weight",
        sd["dfsmn.fsmn1.lookback_filter.weight"], [P, 1, N1])
    if N2 > 0:
        add("dfsmn.fsmn1.lookahead.weight",
            sd["dfsmn.fsmn1.lookahead_filter.weight"], [P, 1, N2])

    # R-1 DFSMNBlocks
    for i in range(R - 1):
        prefix = f"dfsmn.fsmns.{i}"
        add(f"{prefix}.fc1.weight", sd[f"{prefix}.fc1.0.weight"], [H, P])
        add(f"{prefix}.fc1.bias",   sd[f"{prefix}.fc1.0.bias"],   [H])
        add(f"{prefix}.fc2.weight", sd[f"{prefix}.fc2.weight"],   [P, H])
        add(f"{prefix}.fsmn.lookback.weight",
            sd[f"{prefix}.fsmn.lookback_filter.weight"], [P, 1, N1])
        if N2 > 0:
            add(f"{prefix}.fsmn.lookahead.weight",
                sd[f"{prefix}.fsmn.lookahead_filter.weight"], [P, 1, N2])

    # M DNN layers (PT Sequential of [Linear, ReLU, Dropout] x M; index 0, 3, 6, ...)
    for m in range(M):
        pt_idx = 3 * m
        out_dim, in_dim = sd[f"dfsmn.dnns.{pt_idx}.weight"].shape
        add(f"dnns.{m}.weight",
            sd[f"dfsmn.dnns.{pt_idx}.weight"], [out_dim, in_dim])
        add(f"dnns.{m}.bias",
            sd[f"dfsmn.dnns.{pt_idx}.bias"], [out_dim])

    # Output linear
    add("out.weight", sd["out.weight"], [odim, H])
    add("out.bias",   sd["out.bias"],   [odim])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {output}")


def main():
    parser = argparse.ArgumentParser(description="FireRedVAD -> GGUF converter")
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing model.pth.tar and cmvn.ark "
                             "(e.g. FireRedVAD/Stream-VAD/)")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    args = parser.parse_args()
    convert(args.model_dir, args.output)


if __name__ == "__main__":
    main()
