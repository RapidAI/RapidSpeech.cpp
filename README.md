<div align="center">
<img src="assets/rapid-speech.png" alt="RapidSpeech Logo" />
</div>

English | [简体中文](./README-CN.md)

<a href="https://huggingface.co/RapidAI/RapidSpeech" target="_blank"><img src="https://img.shields.io/badge/🤗-Hugging Face-blue"></a>
<a href="https://www.modelscope.cn/models/RapidAI/RapidSpeech/files?version=main" target="_blank"><img src="https://img.shields.io/badge/ModelScope-blue"></a>
<a href="https://github.com/RapidAI/RapidSpeech.cpp/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidSpeech.cpp?color=ccf"></a>

# RapidSpeech.cpp 🎙️

**RapidSpeech.cpp** is a high-performance, **edge-native speech intelligence framework** built on top of **ggml**.
It aims to provide **pure C++**, **zero-dependency**, and **on-device inference** for large-scale ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

------

## 🌟 Key Differentiators

While the open-source ecosystem already offers powerful cloud-side frameworks such as **vLLM-omni**, as well as mature on-device solutions like **sherpa-onnx**, **RapidSpeech.cpp** introduces a new generation of design choices focused on edge deployment.

### 1. vs. vLLM: Edge-first, not cloud-throughput-first

- **vLLM**
    - Designed for data centers and cloud environments
    - Strongly coupled with Python and CUDA
    - Maximizes GPU throughput via techniques such as PageAttention

- **RapidSpeech.cpp**
    - Designed specifically for **edge and on-device inference**
    - Optimized for **low latency, low memory footprint, and lightweight deployment**
    - Runs on embedded devices, mobile platforms, laptops, and even NPU-only systems
    - **No Python runtime required**

### 2. vs. sherpa-onnx: Deeper control over the inference stack

| Aspect | sherpa-onnx (ONNX Runtime) | **RapidSpeech.cpp (ggml)** |
| --- | --- | --- |
| **Memory Management** | Managed internally by ORT, relatively opaque | **Zero runtime allocation** — memory is fully planned during graph construction to avoid edge-side OOM |
| **Quantization** | Primarily INT8, limited support for ultra-low bit-width | **Full K-Quants family** (Q4_K / Q5_K / Q6_K), significantly reducing bandwidth and memory usage while preserving accuracy |
| **GPU Performance** | Relies on execution providers with operator mapping overhead | **Native backends** (`ggml-cuda`, `ggml-metal`) with speech-specific optimizations, outperforming generic `onnxruntime-gpu` |
| **Deployment** | Requires shared libraries and external config files | **Single binary deployment** — model weights and configs are fully encapsulated in **GGUF** |

------

## 📦 Model Support

**Automatic Speech Recognition (ASR)**
- [x] SenseVoice-small
- [x] FunASR-nano
- [ ] Qwen3-ASR
- [ ] FireRedASR2

**Text-to-Speech (TTS)**
- [ ] CosyVoice3
- [ ] Qwen3-TTS

------

## 🏗️ Architecture Overview

RapidSpeech.cpp is not just an inference wrapper — it is a full-featured speech application framework:

- **Core Engine**
  A `ggml`-based computation backend supporting mixed-precision inference from INT4 to FP32.

- **Architecture Layer**
  A plugin-style model construction and loading system, with support for FunASR-nano, SenseVoice, and planned support for CosyVoice, Qwen3-TTS, and more.

- **Business Logic Layer**
  Built-in ring buffers, VAD (voice activity detection), text frontend processing (e.g., phonemization), and multi-session management.

------

## 🚀 Core Features

- [x] **Extreme Quantization**: Native support for 4-bit, 5-bit, and 6-bit quantization schemes to match diverse hardware constraints.
- [x] **Zero Dependencies**: Implemented entirely in C/C++, producing a single lightweight binary.
- [x] **GPU / NPU Acceleration**: Customized CUDA and Metal backends optimized for speech models.
- [x] **Unified Model Format**: Both ASR and TTS models use an extended **GGUF** format.
- [x] **Python Bindings**: Python API via pybind11, installable with `pip install`.

------

## 🛠️ Quick Start

### Download Models

Models are available on:

- 🤗 Hugging Face: https://huggingface.co/RapidAI/RapidSpeech
- ModelScope: https://www.modelscope.cn/models/RapidAI/RapidSpeech

### Build from Source

```bash
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
git submodule sync && git submodule update --init --recursive
cmake -B build
cmake --build build --config Release
```

Build artifacts are located in the `build/` directory:
- `rs-asr-offline` — Offline ASR command-line tool
- `rs-asr-online` — Online (streaming) ASR command-line tool
- `rs-quantize` — Model quantization tool

### C++ CLI Usage

#### Offline Recognition (rs-asr-offline)

**Basic — single file without VAD:**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

**With VAD segmentation (recommended for long audio):**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

When a VAD model is provided, the tool automatically segments the audio by speech activity and produces timestamped results per segment.

Parameters:

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to GGUF model file (required) | — |
| `-w, --wav` | Path to WAV audio file (16 kHz, required) | — |
| `-v, --vad` | Path to Silero VAD GGUF model (optional, enables VAD segmentation) | — |
| `-t, --threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |
| `--vad-threshold` | VAD speech probability threshold (0–1, lower = more sensitive) | 0.5 |
| `--silence-ms` | Silence duration to split segments (ms) | 600 |
| `--max-segment-s` | Max segment length for ASR input (seconds) | 30.0 |

#### Online / Streaming Recognition (rs-asr-online)

**WAV file (simulate streaming):**

```bash
./build/rs-asr-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

**Microphone (live mode):**

```bash
./build/rs-asr-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  --mic \
  -t 4
```

**Two-pass mode (CTC fast pass + LLM rescoring, FunASR-Nano only):**

```bash
./build/rs-asr-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  --two-pass
```

Parameters:

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to ASR GGUF model file (required) | — |
| `-v, --vad` | Path to Silero VAD model file (required) | — |
| `-w, --wav` | Path to WAV audio file (16 kHz) | — |
| `--mic` | Use microphone input (live mode) | off |
| `--mic-device` | Audio device index for mic input | auto |
| `--mic-chunk-ms` | Mic read chunk size (ms) | 32 |
| `-t, --threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |
| `--vad-threshold` | VAD speech detection threshold (0–1, lower = more sensitive) | 0.5 |
| `--silence-ms` | Silence timeout for segment splitting (ms) | 600 |
| `--two-pass` | Enable 2-pass mode: CTC decode + LLM rescore | off |

#### Model Quantization (rs-quantize)

```bash
./build/rs-quantize /path/to/funasr-nano-fp16.gguf /path/to/output-q4_k.gguf q4_k
```

Supported quantization types: `q4_0`, `q4_k`, `q5_0`, `q5_k`, `q8_0`, `f16`, `f32`

> ⚠️ **Note**: Q2_K quantization causes unacceptable accuracy loss for FunASR Nano, producing garbled output. Not recommended.

### Python Usage

#### Installation

```bash
# Install from PyPI (CPU version)
pip install rapidspeech

# CUDA version
pip install rapidspeech-cuda

# macOS Metal version
pip install rapidspeech-metal
```

#### Build Python Package from Source

```bash
pip install .
# Or specify backend
RS_BACKEND=cuda pip install .
```

#### Python API

```python
import rapidspeech
import numpy as np

# Initialize ASR context
ctx = rapidspeech.asr_offline(
    model_path="funasr-nano-fp16.gguf",
    n_threads=4,
    use_gpu=True
)

# Read WAV audio (16 kHz, float32, mono)
pcm = ...  # np.ndarray, shape=[N], dtype=float32

# Push audio and recognize
ctx.push_audio(pcm)
ctx.process()

# Get recognition result
text = ctx.get_text()
print(f"Result: {text}")
```

See [python-api-examples/asr/asr-offline.py](python-api-examples/asr/asr-offline.py) for a complete example.

------

## 📊 Performance Benchmarks

Test environment: Apple M1 Pro, funasr-nano-fp16.gguf, 15s audio

| Configuration | RTF | Wall Time | Notes |
| --- | --- | --- | --- |
| CPU -t 4 | 0.465 | 12.4s | CPU-only inference |
| GPU -t 4 | 0.170 | 5.2s | Metal acceleration |
| GPU -t 4 Q4_K | 0.756 | — | Quantized model: GPU dequant overhead |
| CPU -t 4 Q4_K | 0.530 | — | Quantized model CPU inference, 596 MB (3.3× compression) |

> RTF (Real-Time Factor) = Processing time / Audio duration. Lower is faster. RTF < 1 means faster than real-time.

------

## 🔧 Model Format Conversion

### ASR Model (HF → GGUF)

A conversion tool from HuggingFace models to GGUF format is provided:

```bash
python scripts/convert_hf_to_gguf.py \
  --model /path/to/hf-model-dir \
  --outfile /path/to/output.gguf \
  --outtype f16
```

### Silero VAD Model (safetensors → GGUF)

To convert the Silero VAD model for use with `rs-asr-online` or offline VAD segmentation:

```bash
python scripts/convert_silero_to_gguf.py \
  --model /path/to/silero_vad_16k.safetensors \
  --output /path/to/silero_vad_v6.gguf
```

The converted VAD model is also available for direct download from [HuggingFace](https://huggingface.co/RapidAI/RapidSpeech) and [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidSpeech).

------

## 🤝 Contributing

If you are interested in the following areas, we welcome your PRs or participation in discussions:

- Adapting more models to the framework.
- Refining and optimizing the project architecture.
- Improving inference performance.

## Acknowledgements

1. [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
2. [llama.cpp](https://github.com/ggml-org/llama.cpp)
3. [ggml](https://github.com/ggml-org/ggml)
