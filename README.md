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
- [x] OpenVoice2 (MeloTTS + voice cloning)
- [x] OmniVoice (single-stage non-autoregressive diffusion TTS, multilingual + voice cloning)
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
- `rs-tts-offline` — Offline TTS command-line tool
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
| `--ctc-precheck` | CTC pre-check before LLM to skip silence (reduces hallucination, slightly increases RTF) | off |

#### Text-to-Speech (rs-tts-offline)

**Basic usage (OpenVoice2):**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  -o output.wav \
  --threads 4
```

**OmniVoice diffusion TTS:**

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

**Voice cloning (OmniVoice):**

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, this is cloned voice." \
  --ref /path/to/reference.wav \
  --ref-text "transcript of the reference audio" \
  -o output.wav
```

Parameters:

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to TTS GGUF model file (required) | — |
| `-t, --text` | Text to synthesize (required) | — |
| `-o, --output` | Output WAV file path | output.wav |
| `--instruct` | Voice description, e.g. `male`, `female`, `young adult` (OmniVoice) | male |
| `--lang` | Target language, e.g. `English`, `zh` (OmniVoice) | English |
| `--seed` | Random seed (OmniVoice) | 42 |
| `--n-steps` | Diffusion steps 1-128, fewer = faster but lower quality (OmniVoice) | 32 |
| `--ref` | Reference audio WAV for voice cloning (OmniVoice) | — |
| `--ref-text` | Transcript of the reference audio (OmniVoice) | — |
| `--threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |

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

**TTS Python API:**

```python
import rapidspeech
import numpy as np

# Initialize TTS synthesizer
tts = rapidspeech.tts_synthesizer(
    model_path="openvoice2-base.gguf",
    n_threads=4,
    use_gpu=True
)

# Synthesize text to audio (returns full PCM as numpy array)
pcm = tts.synthesize("Hello, welcome to RapidSpeech!")

# Streaming synthesis (returns list of numpy array chunks)
chunks = tts.synthesize_streaming("Hello, welcome to RapidSpeech!")
for chunk in chunks:
    print(f"Chunk: {len(chunk)} samples")

# Optional: set reference audio for voice cloning
# reference_pcm = ...  # load reference audio
# tts.set_reference(reference_pcm, sample_rate=16000)
```

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

### TTS Model (OpenVoice2 → GGUF)

Convert MeloTTS (OpenVoice2) base model and optional tone color converter to GGUF:

```bash
# Convert base TTS model
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --output-dir ./models \
  --language EN

# Convert with tone color converter for voice cloning
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --converter-model myshell-ai/OpenVoiceV2 \
  --output-dir ./models
```

Outputs:
- `openvoice2-base.gguf` — Text encoder + duration predictor + flow decoder + HiFi-GAN vocoder
- `openvoice2-converter.gguf` — Tone color converter (optional, for voice cloning)

### TTS Model (OmniVoice → GGUF)

Merge OmniVoice PyTorch model (LLM + audio tokenizer) into a single GGUF:

```bash
python scripts/convert_omnivoice_to_gguf.py \
  --model /path/to/omnivoice-model \
  --tokenizer /path/to/omnivoice-audio-tokenizer \
  --output /path/to/omnivoice-merged.gguf \
  --outtype f16
```

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
