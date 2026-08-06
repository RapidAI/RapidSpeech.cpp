<div align="center">
<img src="assets/rapid-speech.png" alt="RapidSpeech Logo" width="640" />

<h1>RapidSpeech.cpp</h1>

<p><strong>Private speech AI that runs where your users are.</strong></p>

<p>
ASR · streaming ASR · TTS · VAD · voice cloning<br />
One native C/C++ runtime. Python-simple. GGUF-powered.
</p>

<p>
<a href="https://rapidai-rapidspeech-wasm.hf.space"><strong>Try in your browser</strong></a>
· <a href="https://colab.research.google.com/github/RapidAI/RapidSpeech.cpp/blob/main/colab/RapidSpeech_Quickstart.ipynb">Run in Colab</a>
· <a href="https://huggingface.co/RapidAI/RapidSpeech">Models</a>
· <a href="docs/cli.md">CLI guide</a>
</p>

<p>
<a href="https://huggingface.co/RapidAI/RapidSpeech"><img src="https://img.shields.io/badge/Hugging_Face-models-yellow" alt="Hugging Face models"></a>
<a href="https://www.modelscope.cn/models/RapidAI/RapidSpeech/files?version=main"><img src="https://img.shields.io/badge/ModelScope-models-blue" alt="ModelScope models"></a>
<a href="https://github.com/RapidAI/RapidSpeech.cpp/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidSpeech.cpp?color=ccf" alt="GitHub stars"></a>
</p>
</div>

English | [简体中文](./README-CN.md)

RapidSpeech.cpp gives applications local speech recognition, text-to-speech,
voice activity detection and voice cloning without sending audio to a cloud
service. The runtime is pure C/C++ on ggml, models use GGUF, and Python users
get installable wheels instead of a heavyweight PyTorch stack.

------

## Install, Then Speak or Transcribe

```bash
# Choose one package for your machine
pip install rapidspeech-metal  # macOS / Apple Silicon
pip install rapidspeech-cuda   # Linux / NVIDIA
pip install rapidspeech        # CPU, Linux / Windows

rapidspeech transcribe your-audio.wav             # speech → text
rapidspeech speak "你好，欢迎使用语音合成。" -o hello.wav   # text → speech
rapidspeech serve                                # OpenAI API + browser console
```

No model paths, no CMake, no config. The first run auto-downloads the default
models (sha256-verified, resumable, cached in `~/.cache/rapidspeech/models/`);
inference uses no network after the files are cached. Set
`RAPIDSPEECH_OFFLINE=1` to enforce offline mode.

Example result from an Apple M1 Pro using Metal:

```text
$ rapidspeech transcribe news-15s.wav
model: FunASRNano  backend: MTL0
15.10s audio in 2.08s (RTF 0.138)
国家发展改革委等部门发布关于开展重点行业节能降碳改造攻坚三年行动的通知。通知提出，重点行业能源消耗和二氧化碳排放的规模大、强度高。
```

Something broken? `rapidspeech doctor` checks your install, backends and model
cache, and prints the exact fix for every problem it finds.
See [docs/cli.md](docs/cli.md) for all commands (`models`, `pull`, offline
mode, mirrors).

Want to see the whole loop at once? `rapidspeech serve` starts the native
server, opens `http://127.0.0.1:8080/webui.html`, and exposes the browser
console, OpenAI-compatible HTTP, and WebSocket streaming ASR/TTS endpoints
from the same process. Models, backend, and per-request latency are shown on
screen; nothing leaves localhost.

> The default TTS model is Mandarin Kokoro v1.1-zh. Curated voice packs are
> available through the model registry, and English words embedded in Chinese
> text are supported. For English and voice-cloning TTS, see the
> [model guides](#documentation).

## Try Before Installing

The browser demo runs inference client-side with WebAssembly and WebGPU:

- [Hugging Face browser demo](https://rapidai-rapidspeech-wasm.hf.space)
- [ModelScope browser demo](https://rapidai-rapidspeech-wasm.ms.show)
- [Google Colab quickstart](https://colab.research.google.com/github/RapidAI/RapidSpeech.cpp/blob/main/colab/RapidSpeech_Quickstart.ipynb)

It includes offline ASR, microphone ASR with VAD, and local TTS. Your model and
audio stay in the browser session.

## Use It From Python

```bash
rapidspeech pull funasr-nano     # prints the local model path
```

```python
import os
import rapidspeech

model = os.path.expanduser("~/.cache/rapidspeech/models/funasr-nano-q4_k_m.gguf")
asr = rapidspeech.asr_offline(model)
asr.push_audio(pcm)              # 1-D float32 mono PCM
asr.process()
print(asr.get_text())
```

TTS, streaming ASR, VAD and voice cloning:
[Python examples](python-api-examples/README.md).

------

## What You Can Build

- **Private transcription:** offline files, VAD-segmented audio, hotword
  biasing for names and domain terms.
- **Live captions and voice input:** microphone streaming with partial and
  final results.
- **Local speech output:** multilingual TTS, expressive synthesis and
  zero-shot voice cloning.
- **A local speech service:** OpenAI-compatible HTTP endpoints, WebSocket
  streaming and MCP from one server.
- **Browser and native applications:** WASM/JavaScript, Python, C API and
  self-contained C++ executables.

------

## Why RapidSpeech.cpp

1. **Local by construction.** Audio is processed by the native runtime on your
   machine. Model downloads are explicit and offline mode is enforceable.
2. **A speech workflow, not only a model runner.** VAD, streaming buffers,
   hotword biasing, Chinese text normalization, speaker embeddings, model
   quantization and serving live in the same project.
3. **One core, several deployment surfaces.** Use Python while prototyping,
   then embed the C API, ship a native executable, run a local server, or move
   the workflow into the browser.

The shared GGUF format targets CPU, Metal, CUDA, Vulkan, CANN, OpenCL and
WebGPU. Backend support and optimization vary by model; the model guides record
the current paths and known limitations.

------

## Measured Performance

Apple M1 Pro, macOS, default models, warm inference, 5 runs each. The table is
generated from raw JSON in [`benchmarks/results/`](benchmarks/results/) — see
[benchmarks/README.md](benchmarks/README.md) for the rules and one-command
reproduction on your own machine:

| Task | Model | Backend | Cold start | RTF (median) | RTF (p95) |
| --- | --- | --- | --- | --- | --- |
| ASR | funasr-nano q4_k_m (692 MB) | Metal | 0.78 s | 0.106 | 0.132 |
| ASR | funasr-nano q4_k_m | CPU ×4 | 0.41 s | 0.172 | 0.210 |
| TTS | kokoro-zh q8_0 (107 MB) | Metal | 0.24 s | 0.362 | 0.363 |
| TTS | kokoro-zh q8_0 | CPU ×4 | 0.13 s | 0.693 | 0.758 |

RTF is processing time divided by audio duration. Lower is faster; RTF below
1 means faster than real time. Results depend on hardware, backend, model,
quantization and input; the CLI prints RTF on every run. CUDA / Windows /
WASM platforms are marked "awaiting community reproduction" in
[benchmarks/README.md](benchmarks/README.md) — we publish measurements, not
estimates. The [OmniVoice guide](docs/omnivoice.md#performance-rtf) includes
a per-model quantization/step matrix.

------

## Supported Today

| Task | Models | Status |
| --- | --- | --- |
| ASR | SenseVoice-small, FunASR-nano, X-ASR (Zipformer2, streaming) | Stable |
| VAD | Silero VAD, FireRedVAD | Stable |
| TTS | Kokoro, CosyVoice3, OmniVoice, OpenVoice2, IndexTTS-2 | Active |
| Speaker | CAMPPlus | Stable |

**X-ASR** — Chinese/English Zipformer2 transducer (icefall/k2). One GGUF serves
both **offline** full-context decoding and **true chunked streaming** (per-layer
left-context caches, sub-second partials, `--chunk-len 16/32/48/96/192` fbank
frames). Punctuation and casing, greedy transducer decode, runs on CPU / Metal /
CUDA / Vulkan and quantizes to q4_k_m (99.5 MB).

**IndexTTS-2** — expressive zero-shot voice-cloning TTS (GPT + S2Mel CFM +
BigVGAN-v2 vocoder) with 4-mode emotion control (reference audio / vector / text
/ Qwen). See [docs/index2tts.md](docs/index2tts.md).

## Use It Your Way

| Surface | Best for | Start here |
| --- | --- | --- |
| `rapidspeech` CLI | First run, scripts and local tools | [CLI guide](docs/cli.md) |
| Python API | Application integration and NumPy audio | [Python examples](python-api-examples/README.md) |
| Native C API | Desktop/mobile bindings and embedded runtimes | [`include/rapidspeech.h`](include/rapidspeech.h) |
| HTTP/WebSocket/MCP | Local services and agent integrations | [Server guide](examples/server/README.md) |
| Browser WASM | Client-side ASR, VAD and TTS | [WASM guide](wasm-examples/README.md) |
| Node.js | JavaScript tooling with the WASM runtime | [Node.js example](node-api-example/README.md) |

## In Progress

Qwen3-ASR, Qwen3-TTS.

------

## Documentation

- [Python examples](python-api-examples/README.md)
- [Technical Notes](docs/TECHNICAL.md): architecture, design tradeoffs, backends,
  model conversion, and binding surfaces.
- Model guides:
  - ASR — [X-ASR](docs/x-asr.md) (Zipformer2, streaming) ·
    [SenseVoice](docs/sensevoice.md) · [FunASR-Nano](docs/funasr-nano.md)
  - TTS — [IndexTTS-2](docs/index2tts.md) (voice clone + emotion) ·
    [CosyVoice3](docs/cosyvoice3.md) · [OmniVoice](docs/omnivoice.md) ·
    [OpenVoice2](docs/openvoice2.md) · [Kokoro](docs/kokoro.md)
  - VAD — [Silero / FireRedVAD](docs/vad.md)
  - Speaker — [CAMPPlus](docs/campplus.md)
- [Browser / WASM examples](wasm-examples/README.md)
- [Node.js example](node-api-example/README.md)

------

## Native C++ CLI

### Download Models

The pip CLI handles this automatically (see the top of this page). For the
C++ CLIs, download GGUF files manually from:

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

**Self-contained executables** (no runtime DLL/.so dependencies) — build the
core and ggml statically into each CLI with `-DRS_STATIC_EXE=ON`:

```bash
# Windows / MSVC
cmake -B build -G "Visual Studio 17 2022" -A x64 -DRS_STATIC_EXE=ON
cmake --build build --config Release --parallel
# Linux / macOS
cmake -B build -DRS_STATIC_EXE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build artifacts are located in the `build/` directory:
- `rs-asr-offline` — Offline ASR command-line tool
- `rs-asr-vad-online` — VAD-segmented quasi-streaming ASR command-line tool
- `rs-asr-online` — True chunked streaming ASR (X-ASR; mic or WAV, live partials)
- `rs-tts-offline` — Offline TTS command-line tool
- `rs-server` — OpenAI-compatible HTTP API + MCP server (ASR + TTS)
- `rs-quantize` — Model quantization tool

### Core Commands

**Offline ASR**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

**VAD-segmented ASR**

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

**Hotword biasing (FunASR-Nano)**

```bash
# Bias proper nouns / fix homophones ; use --hotword-file for a large list
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  --hotwords "阿里巴巴"
```

> See [docs/funasr-nano.md](docs/funasr-nano.md#6-热词rag-上下文偏置) for details.

**Streaming ASR (X-ASR)**

```bash
# WAV, real-time paced with live partials (or --fast to run as fast as possible)
./build/rs-asr-online -m /path/to/xasr-q4_k_m.gguf -w /path/to/audio.wav --chunk-len 32
# Microphone
./build/rs-asr-online -m /path/to/xasr-q4_k_m.gguf --mic --chunk-len 16
```

See [docs/x-asr.md](docs/x-asr.md) for the model, chunk-size / latency tradeoffs,
and GGUF conversion.

**Text to speech**

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

**Quantization**

```bash
./build/rs-quantize /path/to/input-f16.gguf /path/to/output-q4_k.gguf q4_k
```

**Server (OpenAI API + MCP)**

```bash
# From the pip package: auto-download defaults, start API, open browser console
rapidspeech serve

# From a native build: serve ASR + TTS over OpenAI-compatible HTTP API and MCP
./build/rs-server --asr-model xasr.gguf --tts-model omnivoice.gguf --port 8080

curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@audio.wav -F model=rapidspeech-asr
curl http://127.0.0.1:8080/v1/audio/speech -H 'content-type: application/json' \
     -d '{"input":"hello","voice":"female"}' --output out.wav
```

Also runs as an MCP server (stdio for Claude Desktop, or `POST /mcp`) and
exposes WebSocket streaming endpoints (streaming ASR with partial/final,
VAD-segmented ASR, segmented + pure streaming TTS), plus a browser test console
(`--web-dir examples/server` → `http://host:port/webui.html`). See
[examples/server/README.md](examples/server/README.md).

### Python

See [Python examples](python-api-examples/README.md) for offline ASR, streaming
ASR, offline TTS, streaming TTS, VAD, and voice cloning.

------

## 🤝 Contributing

Contributions are most useful when they include a command that reproduces the
change and a result that can be checked. Current high-impact areas:

- Cross-platform installation and wheel testing.
- Reproducible latency, memory and quality benchmarks.
- Streaming, Metal, CUDA, Vulkan and WebGPU performance.
- Documentation and first-run examples.
- Model ports with conversion scripts, parity checks and known limitations.

Open an issue before starting a large port so the implementation can align with
the runtime architecture.

## Acknowledgements

1. [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
2. [llama.cpp](https://github.com/ggml-org/llama.cpp)
3. [ggml](https://github.com/ggml-org/ggml)
4. [cppjieba](https://github.com/yanyiwu/cppjieba) — Chinese word segmentation
5. [WeText](https://github.com/wenet-e2e/wetext) — text normalization (ITN/TN)
6. [miniaudio](https://github.com/mackron/miniaudio) — single-file audio I/O
7. [X-ASR](https://github.com/Gilgamesh-J/X-ASR) Streaming-focused automatic speech recognition models
