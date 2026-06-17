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

面向 Python 开发者的本地 ASR / TTS，底层由 C++ GGUF runtime 驱动。

**RapidSpeech.cpp** 给 Python 开发者提供简单的本地语音识别、语音合成、VAD、说话人嵌入和声音克隆 API。底层使用纯 C/C++ 引擎、**ggml** 后端和统一的 **GGUF** 模型格式，所以你写的是 Python，跑的是原生性能，不需要启动语音服务。

------

## Python 60 秒跑起来

### 安装

```bash
pip install rapidspeech
```

GPU wheel：

```bash
pip install rapidspeech-metal   # macOS / Apple Silicon
pip install rapidspeech-cuda    # Linux / NVIDIA
```

### 文本转语音

```bash
python python-api-examples/tts/tts-offline.py \
  --model /path/to/omnivoice-f16.gguf \
  --text "Hello, welcome to RapidSpeech." \
  --output output.wav
```

### 语音转文本

```bash
python python-api-examples/asr/asr-offline.py \
  --model /path/to/funasr-nano-fp16.gguf \
  --audio /path/to/audio.wav
```

### Python API

```python
import rapidspeech

tts = rapidspeech.tts_synthesizer("/path/to/omnivoice-f16.gguf")
tts.set_params(instruct="male, young adult", language="English", seed=42)
pcm = tts.synthesize("Hello from a native speech engine.")
sample_rate = tts.get_sample_rate()
```

```python
import rapidspeech

asr = rapidspeech.asr_offline("/path/to/funasr-nano-fp16.gguf")
sample_rate = asr.get_model_meta()["audio_sample_rate"]
pcm = ...  # 1-D float32 mono PCM，采样率为 sample_rate
asr.push_audio(pcm)
asr.process()
print(asr.get_text())
```

------

## 为什么是 RapidSpeech.cpp

- **Python API，原生核心**：写 Python，底层跑 C++ / ggml 引擎。
- **一个模型格式**：ASR、TTS、VAD、说话人模型都使用 GGUF。
- **NumPy 输入输出**：ASR 接收 float32 PCM；TTS 返回 float32 PCM。
- **默认本地运行**：没有云 API，没有语音服务，没有 Python 模型栈。
- **端侧后端栈**：CPU、Metal、CUDA、Vulkan、CANN、OpenCL、WebGPU。

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

- [Python 示例](python-api-examples/README.md)
- [技术说明](docs/TECHNICAL-CN.md)：架构、设计取舍、后端、模型转换和绑定接口。
- [浏览器 / WASM 示例](wasm-examples/README.md)
- [Node.js 示例](node-api-example/README.md)

------

## 原生 C++ CLI

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

离线 ASR、流式 ASR、离线 TTS、流式 TTS、VAD 和声音克隆见 [Python 示例](python-api-examples/README.md)。

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
4. [cppjieba](https://github.com/yanyiwu/cppjieba) —— 中文分词
5. [WeText](https://github.com/wenet-e2e/wetext) —— 文本归一化（ITN/TN）
6. [miniaudio](https://github.com/mackron/miniaudio) —— 单文件音频 I/O
