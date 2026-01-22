<div align="center">
<img src="assets/rapid-speech.png" alt="rapid-speech Logo"  />
</div>

[简体中文](./README-CN.md) | English

# RapidSpeech.cpp 🎙️

**RapidSpeech.cpp** is a high-performance, edge-native speech intelligence framework powered by **ggml**. It is designed to provide a pure C++, zero-dependency on-device inference solution for large-scale ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

------

## 🌟 Core Strategic Advantages

In the current open-source landscape, while we have excellent cloud-based frameworks like **vLLM-omni** and mature edge tools like **sherpa-onnx**, **RapidSpeech.cpp** achieves a generational breakthrough in the following dimensions:

### 1. Distinction from vLLM: Edge vs. Cloud Throughput

- **Deployment Environment:** vLLM is engineered for data centers, relying on Python environments and tight CUDA coupling to maximize GPU throughput via PagedAttention.
- **RapidSpeech.cpp** focuses on **Edge Computing**. It prioritizes low latency and minimal footprint, capable of running on embedded devices, mobile phones, laptops, or NPU platforms without any GPU or Python runtime requirements.

### 2. Distinction from sherpa-onnx: Deeper Low-Level Control

| Dimension             | sherpa-onnx (ONNX Runtime)                                   | **RapidSpeech.cpp (GGML)**                                   |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Memory Management** | Relies on internal ORT allocation; memory overhead is relatively "black-box." | **Zero runtime memory allocation**. Memory is locked during the graph-building phase, completely eliminating on-device OOM (Out of Memory) issues. |
| **Quantization**      | Primarily supports INT8; limited support for ultra-low bits (INT4/INT5). | **Cutting-edge Quantization**. Native support for the K-Quants series (Q4_K, Q5_K, Q6_K, etc.), significantly reducing bandwidth pressure while maintaining accuracy. |
| **GPU Performance**   | Depends on ONNX EP mapping, incurring operator conversion overhead. | **Native GPU Optimization**. Directly invokes `ggml-cuda` or `ggml-metal`, offering inference efficiency significantly superior to the generic `onnxruntime-gpu`. |
| **Deployment**        | Depends on dynamic libraries and often requires external config files (.yaml/.txt). | **Single Binary**. The GGUF format encapsulates all configurations and weights—deploy and run instantly. |


------

## 🏗️ Architecture Design

RapidSpeech.cpp is more than just an inference engine; it is a comprehensive speech business framework:

- **Core Engine:** A `ggml`-based computational backend supporting mixed-precision inference from INT4 to FP32.
- **Architecture Layer:** A plugin-based model builder with planned support for **Funasr-nano**, **CosyVoice**, **Qwen3-TTS**, and more.
- **Business Logic:** Built-in circular buffers, VAD (Voice Activity Detection), text frontend (Phonemization), and multi-session management.

------

## 🚀 Key Features

- [ ] **Extreme Quantization:** Native support for 4-bit, 5-bit, 6-bit, and other quantization schemes to adapt to hardware with varying memory bandwidths.
- [ ] **Zero Dependencies:** Pure C/C++ implementation, compiling into a single lightweight binary.
- [ ] **GPU Acceleration:** Tailor-made optimizations for CUDA and Metal backends specifically for large speech model characteristics.
- [ ] **Unified Format:** Both ASR and TTS models utilize a unified, extended **GGUF** format.

------

## 🛠️ Quick Start (Under Development)

Bash

```
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
```

------

## 🤝 Contributing

If you are interested in the following areas, we welcome your PRs or participation in discussions:

- Adapting more models to the framework.
- Refining and optimizing the project architecture.