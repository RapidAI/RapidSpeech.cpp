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

# RapidSpeech.cpp

端侧语音 AI runtime，支持 ASR、TTS、VAD 和声音克隆。
Python 易用，C++ 原生，GGUF 驱动。

**RapidSpeech.cpp** 在端侧运行语音识别、语音合成、VAD、说话人嵌入和声音克隆。它给 Python 开发者一个简单 API，同时保持底层 runtime 为纯 C/C++，由 **ggml** 后端和统一的 **GGUF** 模型格式驱动。没有云 API，没有语音服务，也没有沉重的 Python 模型栈。

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

- **为端侧而生**：在笔记本、服务器、浏览器和设备级硬件上本地运行语音模型。
- **Python 易用，C++ 原生**：写 Python，底层跑 C++ / ggml 引擎。
- **一个模型格式**：ASR、TTS、VAD、说话人模型都使用 GGUF。
- **NumPy 输入输出**：ASR 接收 float32 PCM；TTS 返回 float32 PCM。
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
| ASR | SenseVoice-small, FunASR-nano, X-ASR（Zipformer2，流式） | 稳定 |
| VAD | Silero VAD, FireRedVAD | 稳定 |
| TTS | OmniVoice, OpenVoice2, Kokoro, IndexTTS-2 | 活跃开发 |
| Speaker | CAMPPlus | 稳定 |

**X-ASR** — 中英文 Zipformer2 transducer（icefall/k2）。一份 GGUF 同时支持**离线**
全上下文解码与**真正的分 chunk 流式**（逐层左上下文缓存、亚秒级出字，
`--chunk-len 16/32/48/96/192` fbank 帧）。带标点和大小写，greedy transducer 解码，
可在 CPU / Metal / CUDA / Vulkan 上运行，可量化到 q4_k_m（99.5 MB）。详见
[docs/x-asr.md](docs/x-asr.md)。

**IndexTTS-2** — 带情感控制的零样本语音克隆 TTS（GPT + S2Mel CFM + BigVGAN-v2
声码器），4 种情感控制模式（参考音频 / 向量 / 文本 / Qwen）。详见
[docs/index2tts.md](docs/index2tts.md)。

## 开发中

CosyVoice3、Qwen3-ASR、Qwen3-TTS。

------

## 文档

- [Python 示例](python-api-examples/README.md)
- [技术说明](docs/TECHNICAL-CN.md)：架构、设计取舍、后端、模型转换和绑定接口。
- 模型使用说明：
  - ASR — [X-ASR](docs/x-asr.md)（Zipformer2，流式）·
    [SenseVoice](docs/sensevoice.md) · [FunASR-Nano](docs/funasr-nano.md)
  - TTS — [IndexTTS-2](docs/index2tts.md)（音色克隆 + 情感）·
    [CosyVoice3](docs/cosyvoice3.md) · [OmniVoice](docs/omnivoice.md) ·
    [OpenVoice2](docs/openvoice2.md) · [Kokoro](docs/kokoro.md)
  - VAD — [Silero / FireRedVAD](docs/vad.md)
  - 说话人 — [CAMPPlus](docs/campplus.md)
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

**独立可执行文件**（无 DLL/.so 运行时依赖）—— 用 `-DRS_STATIC_EXE=ON` 把 core 和
ggml 静态编译进每个 CLI：

```bash
# Windows / MSVC
cmake -B build -G "Visual Studio 17 2022" -A x64 -DRS_STATIC_EXE=ON
cmake --build build --config Release --parallel
# Linux / macOS
cmake -B build -DRS_STATIC_EXE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建产物位于 `build/` 目录：
- `rs-asr-offline` — 离线 ASR 命令行工具
- `rs-asr-vad-online` — VAD 切段的伪流式 ASR 命令行工具
- `rs-asr-online` — 真正的分 chunk 流式 ASR（X-ASR；麦克风或 WAV，实时出字）
- `rs-tts-offline` — 离线 TTS 命令行工具
- `rs-server` — OpenAI 兼容 HTTP API + MCP 服务器（ASR + TTS）
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

**热词偏置（FunASR-Nano）**

```bash
# 纠正专有名词/谐音（如「郭总」→「虢总」）；也可用 --hotword-file 传大词库
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  --hotwords "虢总,阿里巴巴"
```

> 详见 [docs/funasr-nano.md](docs/funasr-nano.md#6-热词rag-上下文偏置)。

**流式 ASR（X-ASR）**

```bash
# WAV：实时速率回放 + 实时出字（加 --fast 则尽快跑完）
./build/rs-asr-online -m /path/to/xasr-q4_k_m.gguf -w /path/to/audio.wav --chunk-len 32
# 麦克风
./build/rs-asr-online -m /path/to/xasr-q4_k_m.gguf --mic --chunk-len 16
```

模型、chunk 大小/延迟权衡与 GGUF 转换详见 [docs/x-asr.md](docs/x-asr.md)。

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

**服务器（OpenAI API + MCP）**

```bash
# 用 OpenAI 兼容 HTTP API 和 MCP 同时提供 ASR + TTS
./build/rs-server --asr-model xasr.gguf --tts-model omnivoice.gguf --port 8080

curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@audio.wav -F model=rapidspeech-asr
curl http://127.0.0.1:8080/v1/audio/speech -H 'content-type: application/json' \
     -d '{"input":"你好","voice":"female","language":"Chinese"}' --output out.wav
```

也可作为 MCP 服务器运行（stdio 给 Claude Desktop，或 `POST /mcp`），并提供
WebSocket 流式接口（带 partial/final 的流式 ASR、VAD 切段 ASR、分段与纯流式
TTS），以及一个浏览器测试台（`--web-dir examples/server` →
`http://host:port/webui.html`）。详见 [examples/server/README.md](examples/server/README.md)。

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
7. 7. [X-ASR](https://github.com/Gilgamesh-J/X-ASR) 重点面向 流式 ASR 和 低延迟部署，同时支持离线识别
