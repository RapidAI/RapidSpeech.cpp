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
  A plugin-style model construction and loading system, with planned support for FunASR-nano, CosyVoice, Qwen3-TTS, and more.

- **Business Logic Layer**
  Built-in ring buffers, VAD (voice activity detection), text frontend processing (e.g., phonemization), and multi-session management.

------

## 🚀 Core Features

- **Extreme Quantization**: Native support for 4-bit, 5-bit, and 6-bit quantization schemes to match diverse hardware constraints.
- **Zero Dependencies**: Implemented entirely in C/C++, producing a single lightweight binary.
- **GPU / NPU Acceleration**: Customized CUDA and Metal backends optimized for speech models.
- **Unified Model Format**: Both ASR and TTS models use an extended **GGUF** format.
- **Python Bindings**: Python interface via pybind11, installable via pip, callable with just one line of code.

------

## 🛠️ Quick Start

### Download Models

Download GGUF model files from:

- 🤗 Hugging Face: https://huggingface.co/RapidAI/RapidSpeech
- ModelScope: https://www.modelscope.cn/models/RapidAI/RapidSpeech

### C++ Build

#### Basic Build (CPU only)

```bash
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
git submodule sync && git submodule update --init --recursive
cmake -B build
cmake --build build --config Release
```

#### Enable GPU Acceleration

<details>
<summary>🍎 macOS Metal (enabled by default)</summary>

Metal acceleration is enabled by default on macOS — no extra configuration needed:

```bash
cmake -B build
cmake --build build --config Release
```

</details>

<details>
<summary>🖥️ NVIDIA CUDA</summary>

```bash
cmake -B build -DRS_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release
```

</details>

<details>
<summary>🌋 Vulkan</summary>

```bash
cmake -B build -DRS_VULKAN=ON
cmake --build build --config Release
```

</details>

<details>
<summary>⚡ Huawei CANN (Ascend NPU)</summary>

```bash
cmake -B build -DRS_CANN=ON
cmake --build build --config Release
```

</details>

### C++ Command-Line Usage

After building, use `rs-asr-offline` for offline speech recognition:

```bash
# Basic usage
./build/rs-asr-offline \
  -m /path/to/model.gguf \
  -w /path/to/audio.wav

# Specify threads and GPU
./build/rs-asr-offline \
  -m /path/to/model.gguf \
  -w /path/to/audio.wav \
  -t 8 \
  --gpu 1
```

**Command-line arguments:**

| Argument | Description | Default |
| --- | --- | --- |
| `-m, --model` | Model file path (required) | - |
| `-w, --wav` | WAV audio file path (optional; uses a test sine wave if not provided) | - |
| `-t, --threads` | Number of CPU threads | 4 |
| `--gpu` | Enable GPU acceleration (`true`/`false`) | true |
| `-h, --help` | Show help message | - |

### Python Usage

#### Installation

**Option 1: pip install (recommended)**

```bash
# CPU version
pip install rapidspeech

# CUDA version
pip install rapidspeech-cuda

# Metal version (macOS)
pip install rapidspeech-metal
```

**Option 2: Build from source**

```bash
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
git submodule sync && git submodule update --init --recursive

# Build with Python bindings
pip install .

# Or specify CUDA backend
RS_BACKEND=cuda pip install .
```

#### Python API Example

```python
import numpy as np
import rapidspeech

# 1. Initialize the offline ASR recognizer
asr = rapidspeech.asr_offline(
    model_path="/path/to/model.gguf",
    n_threads=4,
    use_gpu=True
)

# 2. Load audio data (float32, 16kHz, mono)
# pcm should be a numpy float32 array in the range [-1.0, 1.0]
pcm = load_wav("audio.wav")  # Implement WAV loading yourself, or use soundfile / scipy.io.wavfile

# 3. Push audio data
asr.push_audio(pcm)

# 4. Run inference
asr.process()

# 5. Get the recognition result
text = asr.get_text()
print(f"Result: {text}")
```

A complete offline recognition example script is available at [`python-api-examples/asr/asr-offline.py`](./python-api-examples/asr/asr-offline.py).

Run the example:

```bash
python python-api-examples/asr/asr-offline.py \
  --model /path/to/model.gguf \
  --audio /path/to/audio.wav \
  --threads 4 \
  --gpu 1
```

#### Python API Reference

| Class / Method | Description |
| --- | --- |
| `rapidspeech.asr_offline(model_path, n_threads=4, use_gpu=True)` | Create an offline ASR recognizer |
| `asr.push_audio(pcm)` | Push float32 audio data (1-D numpy array) |
| `asr.process()` | Run inference, returns status code (0=no output, 1=has output, -1=error) |
| `asr.get_text()` | Get the recognized text result |

### C API Reference

RapidSpeech provides a C interface for integration with other languages and projects. Key APIs:

```c
#include "rapidspeech.h"

// Initialization
rs_init_params_t params = rs_default_params();
params.model_path = "/path/to/model.gguf";
params.n_threads   = 4;
params.use_gpu     = true;
rs_context_t* ctx = rs_init_from_file(params);

// Push audio and run inference
rs_push_audio(ctx, pcm_data, num_samples);
int32_t status = rs_process(ctx);

// Get result
const char* text = rs_get_text_output(ctx);

// Release resources
rs_free(ctx);
```

For the complete C API documentation, see [`include/rapidspeech.h`](./include/rapidspeech.h).

------

## 🔧 Development & Build Options

### CMake Options

| Option | Description | Default |
| --- | --- | --- |
| `RS_BUILD_TESTS` | Build test executables | ON |
| `RS_CUDA` | Enable CUDA acceleration | OFF |
| `RS_METAL` | Enable Metal acceleration (macOS only) | Auto-detect |
| `RS_VULKAN` | Enable Vulkan acceleration | OFF |
| `RS_CANN` | Enable Huawei CANN acceleration | OFF |
| `RS_OPENCL` | Enable OpenCL acceleration | OFF |
| `RS_ENABLE_PYTHON` | Enable Python bindings (pybind11) | OFF |

### Model Conversion

Use the provided script to convert Hugging Face models to GGUF format:

```bash
python scripts/convert_hf_to_gguf.py --model /path/to/hf-model --output /path/to/output.gguf
```

------

## 🤝 Contributing

If you are interested in the following areas, we welcome your PRs or participation in discussions:

- Adapting more models (Qwen3-ASR, CosyVoice3, etc.)
- Refining the framework architecture and performance optimization
- Improving documentation and examples

## Acknowledgements

1. [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
2. [llama.cpp](https://github.com/ggml-org/llama.cpp)
3. [ggml](https://github.com/ggml-org/ggml)
