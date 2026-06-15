<div align="center">
<img src="assets/rapid-speech.png" alt="RapidSpeech Logo" />
</div>

简体中文 | [English](./README.md)

<a href="https://huggingface.co/RapidAI/RapidSpeech" target="_blank"><img src="https://img.shields.io/badge/🤗-Hugging Face-blue"></a>
<a href="https://www.modelscope.cn/models/RapidAI/RapidSpeech/files?version=main" target="_blank"><img src="https://img.shields.io/badge/ModelScope-blue"></a>
<a href="https://colab.research.google.com/drive/16U6k9zhdtfrEwVLP9a6ks99J0bEHNQyS?usp=sharing" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="https://rapidai-rapidspeech-wasm.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face wasm Demo-blue"></a>
<a href="https://rapidai-rapidspeech-wasm.ms.show" target="_blank"><img src="https://img.shields.io/badge/魔搭-wasm Demo-blue"></a>
<a href="https://github.com/RapidAI/RapidSpeech.cpp/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidSpeech.cpp?color=ccf"></a>

# RapidSpeech.cpp 🎙️

在本地用一个 GGUF 驱动的 C++ runtime 跑 ASR 和 TTS。

**RapidSpeech.cpp** 是一个纯 C/C++ 语音推理引擎，面向端侧语音识别、语音合成、VAD、说话人嵌入和声音克隆。它使用 **ggml** 后端和统一的 **GGUF** 模型格式，让部署变成一个原生 runtime 加一个模型文件。

------

## 60 秒搞定

### 文本转语音

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech." \
  --lang English \
  -o output.wav
```

### 语音转文本

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  --gpu true
```

------

## 为什么是 RapidSpeech.cpp

- **一个模型格式**：ASR、TTS、VAD、说话人模型都使用 GGUF。
- **一个原生 runtime**：纯 C/C++，生产环境不需要 Python runtime。
- **一个端侧后端栈**：CPU、Metal、CUDA、Vulkan、CANN、OpenCL、WebGPU。
- **为语音而生**：VAD 分段、流式缓冲、声音克隆、量化，以及 Apple Metal DAC 加速。

------

## 性能快照

测试环境：Apple M1 Pro，funasr-nano-fp16.gguf，15s 音频。

| 配置 | RTF | 耗时 | 备注 |
| --- | --- | --- | --- |
| CPU -t 4 | 0.465 | 12.4s | 纯 CPU 推理 |
| GPU -t 4 | 0.170 | 5.2s | Metal 加速 |
| GPU -t 4 Q4_K | 0.756 | - | 量化模型在 GPU 上反量化开销增大 |
| CPU -t 4 Q4_K | 0.530 | - | 量化模型 CPU 推理，模型体积 596 MB（压缩 3.3x） |

RTF = 处理时间 / 音频时长。越低越快；RTF < 1 表示快于实时。

------

## 当前支持

| 任务 | 模型 | 状态 |
| --- | --- | --- |
| ASR | SenseVoice-small, FunASR-nano | 稳定 |
| VAD | Silero VAD, FireRedVAD | 稳定 |
| TTS | OmniVoice, OpenVoice2, Kokoro | 活跃开发 |
| Speaker | CAMPPlus | 稳定 |

## 开发中

CosyVoice3、Qwen3-ASR、Qwen3-TTS。

------

## 文档

- [技术说明](docs/TECHNICAL-CN.md)：架构、设计取舍、后端、模型转换和绑定接口。
- [Python 示例](python-api-examples/README.md)
- [浏览器 / WASM 示例](wasm-examples/README.md)
- [Node.js 示例](node-api-example/README.md)

------

## 🛠️ 快速开始

### 模型下载

请从以下平台下载对应模型：

- 🤗 Hugging Face：https://huggingface.co/RapidAI/RapidSpeech
- ModelScope：https://www.modelscope.cn/models/RapidAI/RapidSpeech

### 从源码构建

```bash
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
git submodule sync && git submodule update --init --recursive
cmake -B build
cmake --build build --config Release
```

构建产物位于 `build/` 目录：
- `rs-asr-offline` — 离线 ASR 命令行工具
- `rs-asr-vad-online` — VAD 切段的伪流式 ASR 命令行工具
- `rs-tts-offline` — 离线 TTS 命令行工具
- `rs-quantize` — 模型量化工具

### 核心命令

**离线 ASR**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

**VAD 分段 ASR**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

**文本转语音**

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

**模型量化**

```bash
./build/rs-quantize /path/to/input-f16.gguf /path/to/output-q4_k.gguf q4_k
```

### Python

```bash
pip install rapidspeech
```

详细 CLI 参数、Python API、多语言绑定和模型转换说明见 [技术说明](docs/TECHNICAL-CN.md)。

------

## 🤝 参与贡献

如果你对以下领域感兴趣，欢迎提交 PR 或参与讨论：

- 适配更多模型。
- 完善项目框架。
- 优化推理性能。

## 致谢

1. [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
2. [llama.cpp](https://github.com/ggml-org/llama.cpp)
3. [ggml](https://github.com/ggml-org/ggml)
