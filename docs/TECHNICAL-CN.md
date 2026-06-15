# RapidSpeech.cpp 技术说明

主 README 负责让新用户快速跑起来。这里放更深的工程背景：为什么这样设计、有哪些后端、CLI 参数怎么用、模型如何转换。

## 设计理念

RapidSpeech.cpp 围绕三个决定构建：

- **GGUF 作为模型容器**：ASR、TTS、VAD、说话人嵌入和量化权重共享同一种部署格式。
- **ggml 作为执行栈**：CPU、Metal、CUDA、Vulkan、CANN、OpenCL、WebGPU 等后端通过同一个原生 runtime 暴露。
- **语音优先的业务逻辑**：VAD 分段、音频前端、文本前端、说话人嵌入、流式缓冲和量化工具都放在 runtime 里，而不是留给 Python 胶水代码。

目标不是包装一个模型，而是让本地语音推理变成一个原生二进制加一个模型文件。

## 差异化

### 对比 vLLM

vLLM 面向数据中心和高吞吐云端服务，强依赖 Python 与 CUDA，并通过 PagedAttention 等系统优化大 batch serving。

RapidSpeech.cpp 面向端侧和本地部署：

- 低延迟、低内存占用。
- 生产环境不需要 Python runtime。
- 通过 ggml 使用原生 CPU/GPU/NPU 后端。
- 提供 C API，便于桌面、移动端、浏览器和服务端嵌入。

### 对比 sherpa-onnx

| 维度 | sherpa-onnx / ONNX Runtime | RapidSpeech.cpp / ggml |
| --- | --- | --- |
| 内存 | 由 ORT 内部管理 | ggml 图规划内存和显式后端 buffer |
| 量化 | 更偏 INT8 | K-quants 和低比特 GGUF 量化 |
| GPU 执行 | Execution Provider 映射 | 原生 ggml 后端和语音专用路径 |
| 部署 | 模型 + runtime 动态库 / 配置 | 原生 runtime + GGUF 模型元数据 |

## 架构

```text
rapidspeech/src/
  arch/       模型实现
  core/       context、模型接口、processor
  frontend/   音频和文本前端
  utils/      日志、WAV IO、miniaudio
  c_api/      外部绑定使用的 C API
```

### 核心引擎

核心 runtime 负责 GGUF 加载、ggml 后端初始化、scheduler、模型注册和持久状态。模型选择由 GGUF 文件中的 `general.architecture` 字段驱动。

### 模型层

模型实现位于 `rapidspeech/src/arch/`。当前实现或活跃开发路径包括：

- ASR：SenseVoice-small、FunASR-nano，Qwen3-ASR 开发中。
- VAD：Silero VAD、FireRedVAD。
- TTS：OmniVoice、OpenVoice2、Kokoro，CosyVoice3 开发中。
- Speaker：CAMPPlus。

### 业务逻辑层

runtime 内置通常会被留在模型外部的语音业务能力：

- 基于 VAD 的分段。
- 伪流式 ASR。
- FunASR-nano 的 CTC 一遍解码 + LLM 重打分。
- 说话人嵌入和 diarization 辅助能力。
- 多语种 TTS 文本前端。
- 模型量化和 importance matrix 收集。

## 后端策略

RapidSpeech.cpp 基于 ggml 后端，并在通用计算图开销过高时加入语音专用路径。

| 后端 | 平台 | 备注 |
| --- | --- | --- |
| CPU | Linux、macOS、Windows | 默认 fallback |
| Metal | macOS / Apple Silicon | Apple 平台默认启用 |
| CUDA | Linux / NVIDIA | 源码构建：`-DRS_CUDA=ON` |
| Vulkan | Linux / Windows | 源码构建：`-DRS_VULKAN=ON` |
| CANN | 华为昇腾 | 源码构建：`-DRS_CANN=ON` |
| OpenCL | Linux / Android | 源码构建：`-DRS_OPENCL=ON` |
| WebGPU | 原生 Dawn / WASM | 源码构建：`-DRS_WEBGPU=ON` |

在 macOS 上，OmniVoice DAC 声码器可以通过 `rapidspeech/src/arch/dac_metal.mm` 使用自定义 Metal 路径，降低声码器图调度开销。

## CLI 参考

### 离线 ASR

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

使用 VAD 分段：

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model` | ASR GGUF 模型文件路径 | 必填 |
| `-w, --wav` | 16 kHz 单声道 WAV 音频路径 | 必填 |
| `-v, --vad` | 可选 VAD GGUF，Silero 或 FireRed | 未设置 |
| `-t, --threads` | CPU 线程数 | 4 |
| `--gpu` | 是否启用 GPU 加速 | true |
| `--vad-threshold` | VAD 语音概率阈值 | 0.5 |
| `--silence-ms` | 静默多久后切段 | 600 |
| `--max-segment-s` | ASR 最大分段长度 | 30.0 |

### VAD 分段在线 ASR

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4
```

麦克风模式：

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  --mic \
  -t 4
```

两遍模式：

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  --two-pass
```

### 文本转语音

OpenVoice2 / MeloTTS：

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-en.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --lang English \
  -o output.wav \
  --threads 4
```

OmniVoice：

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

OmniVoice 声音克隆：

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, this is cloned voice." \
  --ref /path/to/reference.wav \
  --ref-text "transcript of the reference audio" \
  -o output.wav
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model` | TTS GGUF 模型文件路径 | 必填 |
| `-t, --text` | 要合成的文本 | 必填 |
| `-o, --output` | 输出 WAV 路径 | output.wav |
| `--lang` | 目标语种 | English |
| `--ref` | 声音克隆参考 WAV | 未设置 |
| `--ref-text` | 参考音频文本 | 未设置 |
| `--instruct` | OmniVoice 声音描述 | male |
| `--seed` | 随机种子 | 42 |
| `--n-steps` | MaskGIT 扩散步数 | 32 |
| `--threads` | CPU 线程数 | 4 |
| `--gpu` | 是否启用 GPU 加速 | true |

### 量化

```bash
./build/rs-quantize /path/to/input-f16.gguf /path/to/output-q4_k.gguf q4_k
```

常用类型：`q4_0`、`q4_k`、`q5_0`、`q5_k`、`q8_0`、`f16`、`f32`。

Q2_K 不推荐用于 FunASR Nano，因为可能产生乱码输出。

## Python

### 安装

```bash
pip install rapidspeech
pip install rapidspeech-cuda
pip install rapidspeech-metal
```

Python import 名统一为 `rapidspeech`。

### 从源码构建

```bash
pip install .
RS_BACKEND=cuda  pip install .
RS_BACKEND=metal pip install .
RAPIDSPEECH_CMAKE_ARGS="-DRS_VULKAN=ON" pip install .
RAPIDSPEECH_CMAKE_ARGS="-DRS_CANN=ON"   pip install .
RAPIDSPEECH_CMAKE_ARGS="-DRS_OPENCL=ON" pip install .
RAPIDSPEECH_CMAKE_ARGS="-DRS_WEBGPU=ON" pip install .
```

### ASR API

```python
import rapidspeech

ctx = rapidspeech.asr_offline(
    model_path="funasr-nano-fp16.gguf",
    n_threads=4,
    use_gpu=True,
)

ctx.push_audio(pcm)
ctx.process()
print(ctx.get_text())
```

### TTS API

```python
import rapidspeech

tts = rapidspeech.tts_synthesizer(
    model_path="openvoice2-base.gguf",
    n_threads=4,
    use_gpu=True,
)

pcm = tts.synthesize("你好，欢迎使用 RapidSpeech！")
```

## 模型格式转换

### ASR: HuggingFace 到 GGUF

```bash
python scripts/convert_hf_to_gguf.py \
  --model /path/to/hf-model-dir \
  --outfile /path/to/output.gguf \
  --outtype f16
```

### Silero VAD 到 GGUF

```bash
python scripts/convert_silero_to_gguf.py \
  --model /path/to/silero_vad_16k.safetensors \
  --output /path/to/silero_vad_v6.gguf
```

### OpenVoice2 / MeloTTS 到 GGUF

```bash
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --converter-model myshell-ai/OpenVoiceV2 \
  --output-dir ./models \
  --language EN
```

产物：

- `openvoice2-base-<lang>.gguf`：基础 TTS 模型。
- `openvoice2-converter.gguf`：可选音色转换器。

### OmniVoice 到 GGUF

```bash
python scripts/convert_omnivoice_to_gguf.py \
  --model /path/to/omnivoice-model \
  --tokenizer /path/to/omnivoice-audio-tokenizer \
  --output /path/to/omnivoice-merged.gguf \
  --outtype f16
```

## 绑定与示例

| 入口 | 位置 |
| --- | --- |
| C++ CLI | `examples/` |
| Python | `python-api-examples/` |
| 浏览器 / WASM | `wasm-examples/` |
| Node.js | `node-api-example/` |
| Colab | `colab/` |

`include/rapidspeech.h` 中的 C API 是非 C++ 绑定的稳定边界。
