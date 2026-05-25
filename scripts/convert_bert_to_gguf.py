#!/usr/bin/env python3
"""
Convert HuggingFace BERT (or BERT-architecture RoBERTa like
hfl/chinese-roberta-wwm-ext-large) into a single GGUF file containing
weights + WordPiece vocab + KV config.

Usage:
  python scripts/convert_bert_to_gguf.py \
      --hf-model bert-base-multilingual-uncased \
      --output models/bert/mbert-768.gguf [--f16]

The output GGUF can be loaded by the C++ BertModel in rapidspeech/src/arch/bert.cpp.

Tensor naming (HF -> GGUF):
  embeddings.word_embeddings.weight       -> embeddings.word.weight
  embeddings.position_embeddings.weight   -> embeddings.position.weight
  embeddings.token_type_embeddings.weight -> embeddings.token_type.weight
  embeddings.LayerNorm.{weight,bias}      -> embeddings.LayerNorm.{weight,bias}
  encoder.layer.N.attention.self.{query,key,value}.{weight,bias}
  encoder.layer.N.attention.output.dense.{weight,bias}
  encoder.layer.N.attention.output.LayerNorm.{weight,bias}
  encoder.layer.N.intermediate.dense.{weight,bias}
  encoder.layer.N.output.dense.{weight,bias}
  encoder.layer.N.output.LayerNorm.{weight,bias}

Shape convention (same as scripts/convert_openvoice2.py): tensor data is
written as numpy row-major bytes with shape reversed before recording, so
GGUF's ne[0] is the innermost/fastest dim. This matches ggml_mul_mat's
expectation that weight is [K, M] (in, out).
"""

import argparse
import struct
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# GGUF type codes (must match ggml/gguf.h)
GGUF_MAGIC   = 0x46554747
GGUF_VERSION = 3

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1

GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9


class GGUFWriter:
    """Minimal GGUF writer, mirrors scripts/convert_omnivoice_to_gguf.py."""

    def __init__(self):
        self.kv_data = []
        self.tensors = []

    def add_string(self, key, value):
        self.kv_data.append(("string", key, value))

    def add_int32(self, key, value):
        self.kv_data.append(("int32", key, int(value)))

    def add_uint32(self, key, value):
        self.kv_data.append(("uint32", key, int(value)))

    def add_float32(self, key, value):
        self.kv_data.append(("float32", key, float(value)))

    def add_bool(self, key, value):
        self.kv_data.append(("bool", key, bool(value)))

    def add_string_array(self, key, values):
        self.kv_data.append(("string_array", key, list(values)))

    def add_tensor(self, name, data, dtype=GGML_TYPE_F32):
        if dtype == GGML_TYPE_F16:
            data = data.astype(np.float16)
        else:
            data = data.astype(np.float32)
        # numpy row-major bytes; reverse shape so ne[0] is the innermost (fastest) dim.
        ggml_shape = tuple(reversed(data.shape))
        self.tensors.append((name, ggml_shape, dtype, data.tobytes()))

    @staticmethod
    def _write_string(f, s):
        b = s.encode("utf-8")
        f.write(struct.pack("<Q", len(b)))
        f.write(b)

    def write(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.kv_data)))

            for kv in self.kv_data:
                kind = kv[0]
                self._write_string(f, kv[1])
                if kind == "string":
                    f.write(struct.pack("<I", GGUF_TYPE_STRING))
                    self._write_string(f, kv[2])
                elif kind == "int32":
                    f.write(struct.pack("<I", GGUF_TYPE_INT32))
                    f.write(struct.pack("<i", kv[2]))
                elif kind == "uint32":
                    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
                    f.write(struct.pack("<I", kv[2]))
                elif kind == "float32":
                    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
                    f.write(struct.pack("<f", kv[2]))
                elif kind == "bool":
                    f.write(struct.pack("<I", GGUF_TYPE_BOOL))
                    f.write(struct.pack("<B", 1 if kv[2] else 0))
                elif kind == "string_array":
                    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
                    f.write(struct.pack("<I", GGUF_TYPE_STRING))
                    f.write(struct.pack("<Q", len(kv[2])))
                    for s in kv[2]:
                        self._write_string(f, s)
                else:
                    raise RuntimeError(f"Unknown kv kind: {kind}")

            data_offset = 0
            tensor_offsets = []
            for name, shape, dtype, data_bytes in self.tensors:
                self._write_string(f, name)
                f.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", dtype))
                aligned = (data_offset + 31) & ~31
                f.write(struct.pack("<Q", aligned))
                tensor_offsets.append(aligned)
                data_offset = aligned + len(data_bytes)

            cur = f.tell()
            aligned_start = (cur + 31) & ~31
            f.write(b"\x00" * (aligned_start - cur))
            data_base = f.tell()

            for i, (_, _, _, data_bytes) in enumerate(self.tensors):
                target = data_base + tensor_offsets[i]
                cur = f.tell()
                if cur < target:
                    f.write(b"\x00" * (target - cur))
                f.write(data_bytes)

        print(f"Written {path} ({len(self.tensors)} tensors, {len(self.kv_data)} KV)")


# ---------------------------------------------------------------------------
def convert(hf_model, output_path, use_f16):
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading {hf_model} ...")
    tok = AutoTokenizer.from_pretrained(hf_model)
    model = AutoModel.from_pretrained(hf_model)
    model.eval()
    cfg = model.config

    writer = GGUFWriter()

    # ---- Metadata ------------------------------------------------------
    writer.add_string("general.architecture", "bert")
    writer.add_string("bert.model_name", hf_model)
    writer.add_int32("bert.hidden_size", cfg.hidden_size)
    writer.add_int32("bert.num_layers", cfg.num_hidden_layers)
    writer.add_int32("bert.num_heads", cfg.num_attention_heads)
    writer.add_int32("bert.intermediate_size", cfg.intermediate_size)
    writer.add_int32("bert.max_position_embeddings", cfg.max_position_embeddings)
    writer.add_int32("bert.vocab_size", cfg.vocab_size)
    writer.add_int32("bert.type_vocab_size", getattr(cfg, "type_vocab_size", 2))
    writer.add_float32("bert.layer_norm_eps", cfg.layer_norm_eps)
    # MeloTTS uses mean of last 4 hidden states.
    writer.add_int32("bert.feature_layers", 4)

    # ---- Tokenizer (WordPiece) ----------------------------------------
    vocab = tok.get_vocab()  # {token: id}
    id_to_token = [None] * len(vocab)
    for token, idx in vocab.items():
        id_to_token[idx] = token
    # Fill any gaps with placeholders (HF should be dense, but be safe).
    for i, t in enumerate(id_to_token):
        if t is None:
            id_to_token[i] = f"[unused{i}]"
    writer.add_string_array("tokenizer.ggml.tokens", id_to_token)
    writer.add_string("tokenizer.ggml.model", "wordpiece")
    writer.add_int32("tokenizer.ggml.unk_id", tok.unk_token_id)
    writer.add_int32("tokenizer.ggml.cls_id", tok.cls_token_id)
    writer.add_int32("tokenizer.ggml.sep_id", tok.sep_token_id)
    writer.add_int32("tokenizer.ggml.pad_id", tok.pad_token_id)
    mask_id = tok.mask_token_id if tok.mask_token_id is not None else 0
    writer.add_int32("tokenizer.ggml.mask_id", mask_id)
    writer.add_bool("tokenizer.ggml.do_lower_case",
                    bool(getattr(tok, "do_lower_case", False)))

    # ---- Tensors -------------------------------------------------------
    sd = model.state_dict()
    weight_dtype = GGML_TYPE_F16 if use_f16 else GGML_TYPE_F32

    def add_linear(src_prefix, dst_prefix):
        w = sd[f"{src_prefix}.weight"].cpu().numpy()
        writer.add_tensor(f"{dst_prefix}.weight", w, weight_dtype)
        b_key = f"{src_prefix}.bias"
        if b_key in sd:
            writer.add_tensor(f"{dst_prefix}.bias",
                              sd[b_key].cpu().numpy(), GGML_TYPE_F32)

    def add_layernorm(src_prefix, dst_prefix):
        writer.add_tensor(f"{dst_prefix}.weight",
                          sd[f"{src_prefix}.weight"].cpu().numpy(), GGML_TYPE_F32)
        writer.add_tensor(f"{dst_prefix}.bias",
                          sd[f"{src_prefix}.bias"].cpu().numpy(), GGML_TYPE_F32)

    # Embeddings
    writer.add_tensor("embeddings.word.weight",
                      sd["embeddings.word_embeddings.weight"].cpu().numpy(),
                      weight_dtype)
    writer.add_tensor("embeddings.position.weight",
                      sd["embeddings.position_embeddings.weight"].cpu().numpy(),
                      weight_dtype)
    if "embeddings.token_type_embeddings.weight" in sd:
        writer.add_tensor("embeddings.token_type.weight",
                          sd["embeddings.token_type_embeddings.weight"].cpu().numpy(),
                          weight_dtype)
    add_layernorm("embeddings.LayerNorm", "embeddings.LayerNorm")

    # Encoder layers
    for i in range(cfg.num_hidden_layers):
        src = f"encoder.layer.{i}"
        dst = f"encoder.layer.{i}"
        add_linear(f"{src}.attention.self.query",       f"{dst}.attention.self.query")
        add_linear(f"{src}.attention.self.key",         f"{dst}.attention.self.key")
        add_linear(f"{src}.attention.self.value",       f"{dst}.attention.self.value")
        add_linear(f"{src}.attention.output.dense",     f"{dst}.attention.output.dense")
        add_layernorm(f"{src}.attention.output.LayerNorm",
                      f"{dst}.attention.output.LayerNorm")
        add_linear(f"{src}.intermediate.dense",         f"{dst}.intermediate.dense")
        add_linear(f"{src}.output.dense",               f"{dst}.output.dense")
        add_layernorm(f"{src}.output.LayerNorm",        f"{dst}.output.LayerNorm")

    writer.write(output_path)
    print(f"  config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers}"
          f" heads={cfg.num_attention_heads} ff={cfg.intermediate_size}"
          f" vocab={cfg.vocab_size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--f16", action="store_true",
                    help="Store weight tensors as F16 (LayerNorm/biases stay F32)")
    args = ap.parse_args()
    convert(args.hf_model, args.output, args.f16)


if __name__ == "__main__":
    main()
