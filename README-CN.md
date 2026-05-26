<div align="center">
<img src="assets/rapid-speech.png" alt="RapidSpeech Logo" />
</div>

简体中文 | [English](./README.md)

<a href="https://huggingface.co/RapidAI/RapidSpeech" target="_blank"><img src="https://img.shields.io/badge/🤗-Hugging Face-blue"></a>
<a href="https://www.modelscope.cn/models/RapidAI/RapidSpeech" target="_blank"><img src="https://img.shields.io/badge/ModelScope-blue"></a>
<a href="https://github.com/RapidAI/RapidSpeech.cpp/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidSpeech.cpp?color=ccf"></a>

# RapidSpeech.cpp 🎙️

**RapidSpeech.cpp** 是一个基于 **ggml** 构建的高性能、边缘原生（Edge-native）语音智能框架，致力于为 ASR（自动语音识别）与 TTS（语音合成）大模型提供 **纯 C++、零依赖、可端侧部署** 的推理解决方案。

------

## 🌟 核心差异化优势

在当前开源生态中，云端侧已有如 **vLLM-omni** 等高吞吐推理框架，端侧也有 **sherpa-onnx** 这样成熟的工具链。而 **RapidSpeech.cpp** 则在以下关键维度实现了代际突破。

### 1. 对比 vLLM：边缘计算优先，而非云端吞吐优先

- **vLLM**
    - 面向数据中心与云端部署
    - 强依赖 Python 运行时与 CUDA
    - 通过 PageAttention 等技术最大化 GPU 吞吐

- **RapidSpeech.cpp**
    - 面向 **边缘计算与端侧推理**
    - 强调 **低延迟、低内存占用与轻量化**
    - 可运行于嵌入式设备、移动端、普通笔记本，甚至无 GPU 的 NPU 平台
    - **无需 Python 运行环境**

### 2. 对比 sherpa-onnx：更深度的底层掌控能力

| 维度 | sherpa-onnx（ONNX Runtime） | **RapidSpeech.cpp（ggml）** |
| --- | --- | --- |
| **内存管理** | 依赖 ORT 内部机制，内存行为相对不可控 | **零运行时内存分配**，在计算图构建阶段完成内存规划，最大限度避免端侧 OOM |
| **量化能力** | 以 INT8 为主，对超低比特支持有限 | **完整 K-Quants 量化体系**（Q4_K / Q5_K / Q6_K 等），在保证精度的同时显著降低带宽与内存压力 |
| **GPU 性能** | 通过 EP 映射，存在通用算子转换开销 | **原生后端优化**，直接使用 `ggml-cuda` / `ggml-metal`，推理效率显著优于 `onnxruntime-gpu` |
| **部署形态** | 通常依赖动态库与外部配置文件 | **单一可执行文件**，模型与配置统一封装于 **GGUF**，部署即运行 |

------

## 📦 模型支持

**语音识别（ASR）**
- [x] SenseVoice-small
- [x] FunASR-nano
- [ ] Qwen3-ASR
- [ ] FireRedASR2

**语音合成（TTS）**
- [x] OpenVoice2（MeloTTS + 声音克隆）
- [x] OmniVoice（单阶段非自回归扩散 TTS，多语种 + 声音克隆）
- [ ] CosyVoice3
- [ ] Qwen3-TTS

------

## 🏗️ 架构设计

RapidSpeech.cpp 并非"单模型推理工具"，而是一套面向真实业务场景设计的完整语音框架：

- **核心引擎（Core Engine）**
  基于 `ggml` 的高性能计算后端，支持从 INT4 到 FP32 的混合精度推理。

- **架构层（Architecture Layer）**
  插件式模型构建与加载机制，支持 FunASR-nano、CosyVoice、Qwen3-TTS 等主流模型体系。

- **业务逻辑层（Business Logic）**
  内置环形缓冲区、VAD（端点检测）、文本前端（如音素化）以及多会话并发管理能力。

------

## 🚀 核心特性

- [x] **极致量化支持**：原生支持 4-bit / 5-bit / 6-bit 量化方案，充分适配不同算力与带宽条件的硬件。
- [x] **零依赖部署**：纯 C/C++ 实现，最终产物为单一、轻量级二进制文件。
- [x] **GPU / NPU 加速**：针对语音模型特点，对 CUDA 与 Metal 后端进行定制化优化。
- [x] **统一模型格式**：ASR 与 TTS 统一采用扩展后的 **GGUF** 模型格式。
- [x] **Python 绑定**：通过 pybind11 提供 Python API，支持 `pip install` 一键安装。

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
- `rs-asr-online` — 在线（流式）ASR 命令行工具
- `rs-tts-offline` — 离线 TTS 命令行工具
- `rs-quantize` — 模型量化工具

### C++ 命令行使用

#### 离线识别（rs-asr-offline）

**基础用法 — 不使用 VAD：**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

**使用 VAD 分段（推荐，适合长音频）：**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

指定 VAD 模型后，工具会自动按语音活动分段，并输出带时间戳的分段识别结果。

参数说明：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model` | GGUF 模型文件路径（必填） | — |
| `-w, --wav` | WAV 音频文件路径（16kHz，必填） | — |
| `-v, --vad` | Silero VAD GGUF 模型路径（可选，启用 VAD 分段） | — |
| `-t, --threads` | CPU 线程数 | 4 |
| `--gpu` | 是否启用 GPU 加速（`true`/`false`） | true |
| `--vad-threshold` | VAD 语音检测阈值（0~1，越低越灵敏） | 0.5 |
| `--silence-ms` | 静默时长超过该值则切分段落（ms） | 600 |
| `--max-segment-s` | ASR 输入的最大分段长度（秒） | 30.0 |

#### 在线/流式识别（rs-asr-online）

**WAV 文件（模拟流式）：**

```bash
./build/rs-asr-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

**麦克风实时识别：**

```bash
./build/rs-asr-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  --mic \
  -t 4
```

**两遍模式（CTC 快速解码 + LLM 重打分，仅 FunASR-Nano 支持）：**

```bash
./build/rs-asr-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  --two-pass
```

参数说明：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model` | ASR GGUF 模型文件路径（必填） | — |
| `-v, --vad` | Silero VAD 模型文件路径（必填） | — |
| `-w, --wav` | WAV 音频文件路径（16kHz） | — |
| `--mic` | 使用麦克风输入（实时模式） | 关闭 |
| `--mic-device` | 麦克风设备索引 | 自动 |
| `--mic-chunk-ms` | 麦克风读取块大小（ms） | 32 |
| `-t, --threads` | CPU 线程数 | 4 |
| `--gpu` | 是否启用 GPU 加速（`true`/`false`） | true |
| `--vad-threshold` | VAD 语音检测阈值（0~1，越低越灵敏） | 0.5 |
| `--silence-ms` | 静默超时切分时长（ms） | 600 |
| `--two-pass` | 启用两遍模式：CTC 解码 + LLM 重打分 | 关闭 |
| `--ctc-precheck` | LLM 解码前 CTC 预检，跳过静音段（减少幻觉，略微增加实时率） | 关闭 |

#### 语音合成（rs-tts-offline）

##### MeloTTS / OpenVoice2

OpenVoice2 基于 [MeloTTS](https://github.com/myshell-ai/MeloTTS) 作为底层声学模型（VITS 风格：文本编码器 + 时长预测器 + Flow 解码器 + HiFi-GAN 声码器）。MeloTTS 每种语言一个权重文件，`--lang` 必须与转换 GGUF 时使用的语言一致。

**英语（MeloTTS-English）：**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-en.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --lang English \
  -o output.wav \
  --threads 4
```

**中文（MeloTTS-Chinese）：**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-zh.gguf \
  -t "你好，欢迎使用 RapidSpeech 语音合成。" \
  --lang Chinese \
  -o output.wav
```

**日语（MeloTTS-Japanese）：**

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-jp.gguf \
  -t "こんにちは、RapidSpeech へようこそ。" \
  --lang Japanese \
  -o output.wav
```

`--lang` 可选值：`English`/`EN`/`en`、`Chinese`/`ZH`/`zh`、`Japanese`/`JA`/`ja`，大小写不敏感，但必须与模型匹配——给英语模型喂中文文本会产生乱码。

**声音克隆（OpenVoice2 = MeloTTS 基础模型 + 音色转换器）：**

OpenVoice2 把说话人音色和韵律解耦。通过 `--ref` 传入参考音频，即可把对应说话人的音色应用到合成语音上。需要音色转换器 GGUF 与基础 GGUF 放在同一目录（加载器会自动发现）。

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-en.gguf \
  -t "Hello, this is cloned voice." \
  --lang English \
  --ref /path/to/reference.wav \
  -o output.wav
```

##### OmniVoice（扩散 TTS，多语种 + 声音克隆）

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

**声音克隆（OmniVoice）：**

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, this is cloned voice." \
  --ref /path/to/reference.wav \
  --ref-text "transcript of the reference audio" \
  -o output.wav
```

参数说明：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-m, --model` | TTS GGUF 模型文件路径（必填） | — |
| `-t, --text` | 要合成的文本（必填） | — |
| `-o, --output` | 输出 WAV 文件路径 | output.wav |
| `--lang` | 目标语种。MeloTTS：`English`/`Chinese`/`Japanese`（必须与 GGUF 一致）；OmniVoice：`English`/`zh`/... | English |
| `--ref` | 参考音频 WAV 文件路径，用于声音克隆（OpenVoice2 / OmniVoice） | — |
| `--ref-text` | 参考音频对应的文本转录（仅 OmniVoice） | — |
| `--bert` | 中文 BERT GGUF（1024 维，OpenVoice2 中文专用，可选） | — |
| `--mbert` | 多语种 BERT GGUF（768 维，可选） | — |
| `--instruct` | 声音描述，如 `male`、`female`、`young adult`（OmniVoice） | male |
| `--seed` | 随机种子（OmniVoice） | 42 |
| `--n-steps` | 扩散步数 1-128，越少越快但音质可能下降（OmniVoice） | 32 |
| `--threads` | CPU 线程数 | 4 |
| `--gpu` | 是否启用 GPU 加速（`true`/`false`） | true |

#### 模型量化（rs-quantize）

```bash
./build/rs-quantize /path/to/funasr-nano-fp16.gguf /path/to/output-q4_k.gguf q4_k
```

支持的量化类型：`q4_0`, `q4_k`, `q4_k_m`, `q5_0`, `q5_k`, `q8_0`, `f16`, `f32`

> ⚠️ **量化注意**：Q2_K 量化对 FunASR Nano 精度损失过大，会导致输出乱码，不推荐使用。推荐使用q4_k_m或q8

### Python 使用

#### 安装

```bash
# 从 PyPI 安装（CPU 版本）
pip install rapidspeech

# CUDA 版本
pip install rapidspeech-cuda

# macOS Metal 版本
pip install rapidspeech-metal
```

#### 从源码构建 Python 包

```bash
pip install .
# 或指定后端
RS_BACKEND=cuda pip install .
```

#### Python API

```python
import rapidspeech
import numpy as np

# 初始化 ASR 上下文
ctx = rapidspeech.asr_offline(
    model_path="funasr-nano-fp16.gguf",
    n_threads=4,
    use_gpu=True
)

# 读取 WAV 音频（16kHz, float32, 单声道）
pcm = ...  # np.ndarray, shape=[N], dtype=float32

# 推送音频并识别
ctx.push_audio(pcm)
ctx.process()

# 获取识别结果
text = ctx.get_text()
print(f"识别结果: {text}")
```

完整示例参见 [python-api-examples/asr/asr-offline.py](python-api-examples/asr/asr-offline.py)。

**TTS Python API：**

```python
import rapidspeech
import numpy as np

# 初始化 TTS 合成器
tts = rapidspeech.tts_synthesizer(
    model_path="openvoice2-base.gguf",
    n_threads=4,
    use_gpu=True
)

# 合成文本为音频（返回完整 PCM numpy 数组）
pcm = tts.synthesize("你好，欢迎使用RapidSpeech！")

# 流式合成（返回 numpy 数组列表）
chunks = tts.synthesize_streaming("你好，欢迎使用RapidSpeech！")
for chunk in chunks:
    print(f"音频块: {len(chunk)} 采样点")

# 可选：设置参考音频用于声音克隆
# reference_pcm = ...  # 加载参考音频
# tts.set_reference(reference_pcm, sample_rate=16000)
```

------

## 🧪 示例与多语言绑定

每种语言绑定都有独立的示例目录与 README，覆盖从安装、命令行参数到底层 API 的端到端用法。

| 目录 | 覆盖内容 | 文档 |
|------|----------|------|
| 🐍 **Python** | `pip install rapidspeech` → 离线 / 流式 ASR（神经 VAD + 2-pass LLM 重打分）、离线 / 流式 TTS、声音克隆 | [`python-api-examples/README.md`](python-api-examples/README.md) |
| 🌐 **浏览器（WebAssembly）** | 三标签页演示：离线 ASR、麦克风在线 ASR、离线 TTS。本地运行，WebGPU + pthreads | [`wasm-examples/README.md`](wasm-examples/README.md) |
| 🟩 **Node.js** | 复用 WASM 模块的 CLI：文件 → ASR（可选 VAD + 2-pass）、文本 → TTS（可选声音克隆） | [`node-api-example/README.md`](node-api-example/README.md) |
| 💻 **C++ CLI** | `rs-asr-offline` / `rs-asr-online` / `rs-tts-offline` / `rs-quantize` | 本 README 上方章节 |

快速预览：

```bash
# Python —— 带 VAD 切分的 2-pass 转写
python python-api-examples/asr/asr-offline.py \
    --model funasr-nano.gguf --audio long.wav \
    --vad silero-vad.gguf --two-pass

# 浏览器 —— 三个标签页一起跑
cd wasm-examples && python3 serve.py 8000  # 然后访问 http://localhost:8000

# Node.js —— 同一份 WASM 模块，文件级 ASR/TTS
node node-api-example/index.js asr -m funasr-nano.gguf -w audio.wav --two-pass
node node-api-example/index.js tts -m omnivoice.gguf -t "你好世界" --lang Chinese -o out.wav
```

------

## 📊 性能基准

测试环境：Apple M1 Pro, funasr-nano-fp16.gguf, 15s 音频

| 配置 | RTF | 耗时 | 备注 |
|------|-----|------|------|
| CPU -t 4 | 0.465 | 12.4s | 纯 CPU 推理 |
| GPU -t 4 | 0.170 | 5.2s | Metal 加速 |
| GPU -t 4 Q4_K | 0.756 | — | 量化模型在 GPU 上反量化开销增大 |
| CPU -t 4 Q4_K | 0.530 | — | 量化模型 CPU 推理，模型体积 596MB（压缩 3.3x） |

> RTF（Real-Time Factor）= 处理时间 / 音频时长，越低越快。RTF < 1 表示快于实时。

------

## 🔧 模型格式转换

### ASR 模型（HF → GGUF）

提供 HF 模型到 GGUF 格式的转换工具：

```bash
python scripts/convert_hf_to_gguf.py \
  --model /path/to/hf-model-dir \
  --outfile /path/to/output.gguf \
  --outtype f16
```

### Silero VAD 模型（safetensors → GGUF）

用于 `rs-asr-online` 或离线 VAD 分段的 Silero VAD 模型转换：

```bash
python scripts/convert_silero_to_gguf.py \
  --model /path/to/silero_vad_16k.safetensors \
  --output /path/to/silero_vad_v6.gguf
```

转换后的 VAD 模型也可直接从 [HuggingFace](https://huggingface.co/RapidAI/RapidSpeech) 或 [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidSpeech) 下载。

### TTS 模型（OpenVoice2 / MeloTTS → GGUF）

将 MeloTTS（OpenVoice2 基础模型）以及可选的音色转换器转换为 GGUF。MeloTTS 每种语言对应一个 HuggingFace 仓库，请选择匹配的 `--base-model` 和 `--language` 标签。

```bash
# 英语
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --output-dir ./models \
  --language EN

# 中文
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-Chinese \
  --output-dir ./models \
  --language ZH

# 日语
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-Japanese \
  --output-dir ./models \
  --language JA

# 同时转换音色转换器（启用 --ref 声音克隆）
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --converter-model myshell-ai/OpenVoiceV2 \
  --output-dir ./models \
  --language EN
```

产物说明：
- `openvoice2-base-<lang>.gguf` — 文本编码器 + 时长预测器 + Flow 解码器 + HiFi-GAN 声码器
- `openvoice2-converter.gguf` — 音色转换器（仅在指定 `--converter-model` 时生成；`--ref` 声音克隆需要它）

### TTS 模型（OmniVoice → GGUF）

将 OmniVoice PyTorch 模型（LLM + audio tokenizer）合并转换为单一 GGUF：

```bash
python scripts/convert_omnivoice_to_gguf.py \
  --model /path/to/omnivoice-model \
  --tokenizer /path/to/omnivoice-audio-tokenizer \
  --output /path/to/omnivoice-merged.gguf \
  --outtype f16
```

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
