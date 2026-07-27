#!/usr/bin/env python3
"""
Convert TEN-VAD (https://github.com/ten-framework/ten-vad) to GGUF.

Input:
    src/onnx_model/ten-vad.onnx from the ten-vad repository.

Usage:
    python3 scripts/convert_tenvad_to_gguf.py <ten-vad.onnx> <output.gguf>

Model graph (per 16 ms frame, verified against the ONNX export):
    input  [1, 3, 41]   3-frame context window x (40 log-mel + 1 pitch) features
    -> SeparableConv2d: depthwise 3x3 (valid) + pointwise 1x1 -> 16 ch, ReLU
    -> MaxPool k=(1,3) s=(1,2)
    -> SeparableConv1d: depthwise k3 s2 pad(1,1) + pointwise, ReLU
    -> SeparableConv1d: depthwise k3 s2 pad(0,1) + pointwise, ReLU
    -> flatten [5,16] position-major -> 80
    -> LSTM1(80->64) -> LSTM2(64->64)      (h/c states carried across frames)
    -> concat(lstm2_h, lstm1_h) = 128 -> Dense(128->32)+ReLU -> Dense(32->1)+Sigmoid

GGUF tensor names / layouts:
    conv1.dw.weight     [1,1,3,3]  ggml ne=[3,3,1,1]   (ggml_conv_2d kernel)
    conv1.pw.weight     [16,1,1]   ggml ne=[1,1,16]    (ggml_conv_1d kernel, K=1)
    conv1.pw.bias       [16]
    conv{2,3}.dw.weight [16,1,3]   ggml ne=[3,1,16]    (ggml_conv_1d_dw kernel)
    conv{2,3}.pw.weight [16,16,1]  ggml ne=[1,16,16]   (ggml_conv_1d kernel, K=1)
    conv{2,3}.pw.bias   [16]
    lstm{1,2}.weight_ih [256,in]   ggml ne=[in,256]    mul_mat(W,x)==F.linear
    lstm{1,2}.weight_hh [256,64]   ggml ne=[64,256]
    lstm{1,2}.bias_ih   [256]      (= ONNX Wb)
    lstm{1,2}.bias_hh   [256]      (= ONNX Rb)
    fc1.weight          [32,128]   ggml ne=[128,32]    (transposed from ONNX MatMul [128,32])
    fc1.bias            [32]
    fc2.weight          [1,32]     ggml ne=[32,1]
    fc2.bias            [1]

LSTM gate order: ONNX stores i,o,f,c; rows are reordered here to i,f,g(=c),o
so the C++ cell can reuse the silero-style gate split (silero_vad.cpp).

Feature normalization means/stds, the Hann-768 window and the mel filterbank
are NOT stored in the GGUF — they live in the vendored third_party/ten-vad
sources (coeff.h) and are compiled into the C++ frontend.
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper
from gguf import GGUFWriter

HIDDEN = 64


def reorder_gates(w: np.ndarray) -> np.ndarray:
    """Reorder LSTM gate blocks from ONNX i,o,f,c to i,f,g(=c),o along dim 0."""
    assert w.shape[0] == 4 * HIDDEN, w.shape
    i, o, f, c = (w[k * HIDDEN:(k + 1) * HIDDEN] for k in range(4))
    return np.concatenate([i, f, c, o], axis=0)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    onnx_path, out_path = sys.argv[1], sys.argv[2]

    m = onnx.load(onnx_path)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    def get(name):
        if name not in inits:
            raise KeyError(f"initializer not found in ONNX: {name}")
        return inits[name].astype(np.float32)

    P = "StatefulPartitionedCall/vad_model/"

    tensors = {}

    # ---- SeparableConv2d (context x feature patch) ----
    tensors["conv1.dw.weight"] = get("const_fold_opt__178")            # [1,1,3,3]
    tensors["conv1.pw.weight"] = get(P + "separable_conv2d/separable_conv2d/ReadVariableOp_1:0").reshape(16, 1, 1)
    tensors["conv1.pw.bias"]   = get(P + "separable_conv2d/BiasAdd/ReadVariableOp:0")

    # ---- SeparableConv1d #1 ----
    tensors["conv2.dw.weight"] = get("const_fold_opt__179").reshape(16, 1, 3)
    tensors["conv2.pw.weight"] = get(P + "separable_conv1d/ExpandDims_2:0").reshape(16, 16, 1)
    tensors["conv2.pw.bias"]   = get(P + "separable_conv1d/BiasAdd/ReadVariableOp:0")

    # ---- SeparableConv1d #2 ----
    tensors["conv3.dw.weight"] = get("const_fold_opt__180").reshape(16, 1, 3)
    tensors["conv3.pw.weight"] = get(P + "separable_conv1d_1/ExpandDims_2:0").reshape(16, 16, 1)
    tensors["conv3.pw.bias"]   = get(P + "separable_conv1d_1/BiasAdd/ReadVariableOp:0")

    # ---- LSTMs (W [1,256,in], R [1,256,64], B [1,512] = Wb||Rb, gates iofc) ----
    for idx, (w, r, b) in enumerate([("W0__70", "R0__71", "B0__72"),
                                     ("W0__99", "R0__100", "B0__101")], start=1):
        W = reorder_gates(get(w)[0])           # [256, in]
        R = reorder_gates(get(r)[0])           # [256, 64]
        B = get(b)[0]                          # [512]
        tensors[f"lstm{idx}.weight_ih"] = W
        tensors[f"lstm{idx}.weight_hh"] = R
        tensors[f"lstm{idx}.bias_ih"]   = reorder_gates(B[:4 * HIDDEN])
        tensors[f"lstm{idx}.bias_hh"]   = reorder_gates(B[4 * HIDDEN:])

    # ---- Dense heads (ONNX MatMul weights are [in,out]; store [out,in]) ----
    tensors["fc1.weight"] = get(P + "dense_3/Tensordot/ReadVariableOp:0").T.copy()  # [32,128]
    tensors["fc1.bias"]   = get(P + "dense_3/BiasAdd/ReadVariableOp:0")
    tensors["fc2.weight"] = get(P + "dense_5/Tensordot/ReadVariableOp:0").T.copy()  # [1,32]
    tensors["fc2.bias"]   = get(P + "dense_5/BiasAdd/ReadVariableOp:0")

    writer = GGUFWriter(out_path, "ten-vad")
    writer.add_string("general.name", "ten-vad")
    writer.add_int32("vad.sample_rate", 16000)
    writer.add_int32("vad.hop_size", 256)
    writer.add_float32("vad.threshold", 0.5)
    writer.add_int32("vad.context_frames", 3)
    writer.add_int32("vad.feature_dim", 41)
    writer.add_int32("vad.hidden_size", 64)
    writer.add_int32("vad.reset_frames", 1875)
    writer.add_int32("vad.min_silence_ms", 200)

    for name, arr in tensors.items():
        writer.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))
        print(f"  {name:20s} {list(arr.shape)}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
