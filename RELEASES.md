# Releases

## v1.4.3

### 🐛 Bug Fixes / Bug 修复

- Windows CUDA wheel: upgrade the CI toolkit from CUDA 11.8 to CUDA 12.4, which is required by the MSVC 14.44 STL used on the GitHub Actions runner.
- Windows CUDA wheel：将 CI toolkit 从 CUDA 11.8 升级到 CUDA 12.4，以兼容 GitHub Actions runner 使用的 MSVC 14.44 STL。

## v1.4.2

### 🐛 Bug Fixes / Bug 修复

- Linux CUDA wheel: exclude the NVIDIA driver library `libcuda.so.1` from `auditwheel` repair; it is provided by the host driver and must not be bundled.
- Linux CUDA wheel：在 `auditwheel` 修复阶段排除 NVIDIA 驱动库 `libcuda.so.1`；该库由宿主机驱动提供，不应打包进 wheel。

## v1.4.1

### 🐛 Bug Fixes / Bug 修复

- Linux: link `rs-asr-online` with `libdl` so miniaudio microphone backends resolve `dlclose` correctly.
- Linux：为 `rs-asr-online` 链接 `libdl`，正确解析 miniaudio 麦克风后端使用的 `dlclose`。
- Windows CUDA: allow the CUDA toolkit to build with the newer MSVC toolset used by GitHub Actions runners.
- Windows CUDA：允许 CUDA toolkit 与 GitHub Actions runner 上较新的 MSVC 工具链一起构建。

## v1.4.0

### 本版本解决的问题 / What this release solves

把实时语音、模型下载与 Python 使用方式收拢到同一套 RapidSpeech.cpp 工作流中，减少从模型到可用服务之间的集成工作。

### 用户现在可以 / What users can do now

- 使用原生 `rs-server` 提供 OpenAI 兼容的 ASR/TTS 服务，并通过 WebSocket 实时传输音频。
- 使用 VoxCPM2 进行实时语音合成，使用 MOSS Transcribe 进行转写，并使用 TEN-VAD 做语音活动检测。
- 通过 Python 短命令完成模型注册、下载、转写、合成和服务启动；模型注册表支持运行时热加载。
- 使用连续批处理和双工作线程处理多路 TTS 请求；新增 AOQ 实时示例和 WebUI 观测入口。
- 在 WASM 构建中使用实时协议相关代码，并继续使用现有 C API、CLI 和 GGUF 模型格式。

### 可测量变化 / Verifiable changes

- 实时协议与传输实现：`rapidspeech/src/realtime/`。
- 服务端与 WebUI：`examples/server/rs-server.cpp`、`examples/server/webui.html`。
- Python 命令与模型注册表：`rapidspeech/python/`。
- 可复现的 ASR 基准样本：`benchmarks/datasets/zh-news-15s.wav`；对应基准结果位于 `benchmarks/results/`。

### 兼容性变化 / Compatibility

- GGUF 文件格式无变化。
- 新增实时服务、VoxCPM2、MOSS Transcribe 和 TEN-VAD 能力；原有 ASR、TTS、VAD 和量化 CLI 保持可用。
- 发布包不内置模型文件，模型仍需单独下载。

### 已知问题 / Known issues

- 正式发布前仍需完成发布清单中的 Linux CUDA、Windows、macOS Apple Silicon wheel 和原生静态包验证。
- CUDA 归档需要目标机器提供兼容的 NVIDIA 驱动与 CUDA runtime；不同模型仍需使用其对应的 GGUF 文件。

### 验证命令 / Verification

```bash
python -m pytest tests/bindings/python/test_cli.py
rapidspeech doctor
rapidspeech transcribe benchmarks/datasets/zh-news-15s.wav
rapidspeech speak "你好" -o /tmp/rapidspeech-v1.4.0.wav
```

## v1.3.0

### 🎉 New Features / 新增功能

**X-ASR — Zipformer2 Transducer ASR / X-ASR — Zipformer2 Transducer ASR**
- New streaming-capable ASR engine: Zipformer2 transducer (zh-en) with punctuation and casing, offline + true chunked streaming from a single GGUF
- 全新支持流式的 ASR 引擎：Zipformer2 transducer（中英），带标点与大小写，离线 + 真流式共用同一份 GGUF
- `rs-asr-online` streaming CLI (WAV file + microphone) with per-layer caches and partial results (`--chunk-len 16..192`)
- `rs-asr-online` 流式 CLI（WAV 文件 + 麦克风），逐层缓存与增量结果（`--chunk-len 16..192`）
- Kaldi POVEY fbank frontend (snip_edges=false, mel 20–7600 Hz); matches the official ONNX word-for-word; q4_k_m quantization
- Kaldi POVEY fbank 前端（snip_edges=false，mel 20–7600 Hz），与官方 ONNX 逐字一致；支持 q4_k_m 量化

**FunASR-Nano ASR / FunASR-Nano ASR**
- Lightweight offline ASR: SANM encoder + CTC fast-decode head, with an optional Qwen3 LLM rescore pass (CTC only for speed, or CTC+LLM for accuracy)
- 轻量离线 ASR：SANM 编码器 + CTC 快速解码头，可选 Qwen3 LLM 二次解码（只跑 CTC 更快，CTC+LLM 更准）
- RAG hotword biasing: CTC hypothesis → tone-stripped pinyin retrieval (edit distance, homophone-aware) → LLM prompt injection → deterministic pinyin replacement; corrects Chinese homophone errors
- RAG 热词偏置：CTC 假设 → 去声调拼音检索（编辑距离，支持谐音）→ 注入 LLM prompt → 确定性拼音替换；可纠正中文同音字错误
- `rs_set_hotwords` C API + CLI `--hotwords` / `--hotword-file`; tunable via `RS_HOTWORD_*` env vars
- `rs_set_hotwords` C API + CLI `--hotwords` / `--hotword-file`；可用 `RS_HOTWORD_*` 环境变量调参
- imatrix quantization support (including q4_k_m)
- 支持 imatrix 量化（含 q4_k_m）

**IndexTTS-2 TTS / IndexTTS-2 TTS**
- Full C++ port of IndexTTS-2: GPT AR backbone, S2Mel CFM, BigVGAN-v2 vocoder, voice cloning (CAMPPlus)
- 完整 C++ 移植 IndexTTS-2：GPT 自回归主干、S2Mel CFM、BigVGAN-v2 声码器、音色克隆（CAMPPlus）
- 4-mode emotion control (reference audio / emotion vector / text / Qwen); fused Metal BigVGAN decoder on Apple Silicon
- 四种情感控制模式（参考音频 / 情感向量 / 文本 / Qwen）；Apple Silicon 上使用融合 Metal BigVGAN 解码器

**Stable Audio 3 / Stable Audio 3**
- Core model structure scaffolding for Stable Audio 3
- Stable Audio 3 核心模型结构搭建

**rs-server — OpenAI-compatible API + MCP + WebSocket / rs-server — OpenAI 兼容 API + MCP + WebSocket**
- Single native C++ server exposing ASR/TTS over three protocols; dependencies vendored as single headers (no Python, no external services)
- 单一原生 C++ server，通过三种协议暴露 ASR/TTS；依赖以单头文件内置（无需 Python，无外部服务）
- OpenAI-compatible HTTP API: `POST /v1/audio/transcriptions`, `POST /v1/audio/speech`, `GET /v1/models`, `GET /health`
- OpenAI 兼容 HTTP API：`POST /v1/audio/transcriptions`、`POST /v1/audio/speech`、`GET /v1/models`、`GET /health`
- Model Context Protocol (JSON-RPC 2.0): `transcribe` / `synthesize` tools over stdio and HTTP transports
- Model Context Protocol（JSON-RPC 2.0）：`transcribe` / `synthesize` 工具，支持 stdio 与 HTTP 传输
- WebSocket streaming endpoints: `/asr/stream`, `/asr/vad`, `/tts/segmented`, `/tts/stream`
- WebSocket 流式端点：`/asr/stream`、`/asr/vad`、`/tts/segmented`、`/tts/stream`

**ASR Observer / ASR 观测面板**
- Browser dashboard soak-test harness: streams a live/file source to `rs-server` over WebSocket, shows transcripts, resource usage, and stability alerts
- 浏览器观测面板/长稳测试工具：通过 WebSocket 将实时/文件音频源推给 `rs-server`，展示转写、资源占用与稳定性告警

**Build / 构建**
- Static build support
- 支持静态构建

### 🐛 Bug Fixes / Bug 修复

- Windows: UTF-8 input handling for CLI and TTS
- Windows：CLI 与 TTS 的 UTF-8 输入处理
- MSVC does not support VLA (variable-length arrays)
- MSVC 不支持变长数组（VLA）
- FunASR-Nano q4_k_m error
- FunASR-Nano q4_k_m 错误
- X-ASR: add missing `<stdexcept>` include
- X-ASR：补上缺失的 `<stdexcept>` 头文件

## v1.2.0

### 🎉 New Features / 新增功能

**CosyVoice3 TTS Engine / CosyVoice3 TTS 引擎**
- Full C++ port of CosyVoice3: core structures, Qwen2 LLM backbone, frontend, speaker module, multi-quantization adaptor
- 完整 C++ 移植 CosyVoice3：核心结构、Qwen2 LLM 主干、frontend、speaker 模块、多量化类型适配
- Streaming online mode (`rs-tts-online`) with chunked pipeline
- 流式在线模式（`rs-tts-online`），支持分块流水线
- QKV concat fusion for ~9% LM speedup
- QKV 融合优化，LM 推理提速约 9%
- Voice baking / reuse (`--save-voice` / `--voice`)
- 音色保存与复用（`--save-voice` / `--voice`）

**Kokoro TTS / Kokoro TTS**
- Kokoro TTS engine with frontend (Chinese G2P + ITN) and frontend data tools
- Kokoro TTS 引擎，含 frontend（中文 G2P + ITN）及数据工具

**Other TTS / 其他 TTS**
- MeloTTS checkpoint loading support
- 支持 MeloTTS checkpoint 加载
- WeText text normalization processor + cppjieba submodule integration
- WeText 文本归一化处理器 + cppjieba 子模块集成

**SenseVoice KWS / SenseVoice 关键词检测**
- Keyword spotting on SenseVoice: `rs-kws` CLI + core engine
- 基于 SenseVoice 的关键词检测：`rs-kws` CLI + 核心引擎

**ASR Enhancements / ASR 增强**
- Speaker verification for offline ASR
- 离线 ASR 增加说话人验证
- imatrix quantization for ASR models
- ASR 模型支持 imatrix 量化
- Expanded quantization type coverage across models
- 各模型扩展支持更多量化类型
- Renamed `asr-online` → `rs-asr-vad-online` for consistency
- 重命名 `asr-online` 为 `rs-asr-vad-online` 以统一命名

**CI / Release / CI 与发布**
- GitHub Actions release workflow for prebuilt binaries (Linux/Windows/macOS, CPU/CUDA/Metal)
- GitHub Actions 自动构建发布二进制（Linux/Windows/macOS，CPU/CUDA/Metal）

### 🐛 Bug Fixes / Bug 修复

- **Python 3.12 segfault**: removed `py::call_guard<py::gil_scoped_release>()` from `push_audio` (ASR+VAD) and `synthesize` to avoid crash on Python 3.12
- **Python 3.12 段错误**：移除 `push_audio`（ASR/VAD）与 `synthesize` 上的 `py::call_guard<py::gil_scoped_release>()`，修复 Python 3.12 崩溃
- VAD segfault and test failure
- VAD 段错误与测试失败
- CosyVoice3: apply `output_norm` before `speech_lm_head` projection
- CosyVoice3：`speech_lm_head` 投影前需先应用 `output_norm`
- SenseVoice load error & f16 forward error
- SenseVoice 加载错误与 f16 前向计算错误
- Windows `STATUS_DLL_NOT_FOUND` error
- Windows `STATUS_DLL_NOT_FOUND` 错误
- Compile errors on Windows / Linux CUDA
- Windows 与 Linux CUDA 编译错误
- Missing pthread link on Linux
- Linux 缺失 pthread 链接
- Python package build no longer triggers `RS_BUILD_TESTS`
- Python 包构建不再触发 `RS_BUILD_TESTS`
- AudioProcessor methods properly hidden in bindings
- Python 绑定中正确隐藏 AudioProcessor 内部方法
- Test script + CI binding test fixes (`SKIP_RETURN_CODE 77`)
- 测试脚本与 CI 绑定测试修复（`SKIP_RETURN_CODE 77`）
