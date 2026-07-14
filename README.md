<div align="center">
<img src="assets/rapid-speech.png" alt="RapidSpeech Logo" />
</div>

English | [简体中文](./README-CN.md)

<a href="https://huggingface.co/RapidAI/RapidSpeech" target="_blank"><img src="https://img.shields.io/badge/🤗-Hugging Face-blue"></a>
<a href="https://www.modelscope.cn/models/RapidAI/RapidSpeech/files?version=main" target="_blank"><img src="https://img.shields.io/badge/ModelScope-blue"></a>
<a href="https://colab.research.google.com/drive/16U6k9zhdtfrEwVLP9a6ks99J0bEHNQyS?usp=sharing" target="_blank"><img src="https://raw.githubusercontent.com/RapidAI/RapidOCR/main/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="https://rapidai-rapidspeech-wasm.hf.space" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging Face wasm Demo-blue"></a>
<a href="https://rapidai-rapidspeech-wasm.ms.show" target="_blank"><img src="https://img.shields.io/badge/魔搭-wasm Demo-blue"></a>
<a href="https://github.com/RapidAI/RapidSpeech.cpp/stargazers"><img src="https://img.shields.io/github/stars/RapidAI/RapidSpeech.cpp?color=ccf"></a>


# RapidSpeech.cpp

On-device speech AI runtime for ASR, TTS, VAD, and voice cloning.
Python-simple, C++-native, GGUF-powered.

**RapidSpeech.cpp** runs speech recognition, text-to-speech, VAD, speaker
embedding, and voice cloning on-device. It gives Python developers a simple API
while keeping the runtime pure C/C++, backed by **ggml** and a unified **GGUF**
model format. No cloud API, no speech server, no heavyweight Python model stack.

------

## Python In 60 Seconds

### Install

```bash
pip install rapidspeech
```

GPU wheels:

```bash
pip install rapidspeech-metal   # macOS / Apple Silicon
pip install rapidspeech-cuda    # Linux / NVIDIA
```

### Text to speech

```bash
python python-api-examples/tts/tts-offline.py \
  --model /path/to/omnivoice-f16.gguf \
  --text "Hello, welcome to RapidSpeech." \
  --output output.wav
```

### Speech to text

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
pcm = ...  # 1-D float32 mono PCM at sample_rate
asr.push_audio(pcm)
asr.process()
print(asr.get_text())
```

------

## Why RapidSpeech.cpp

- **Built for the edge**: run speech models locally on laptops, servers,
  browsers, and device-class hardware.
- **Python-simple, C++-native**: write Python, run a C++/ggml engine underneath.
- **One model format**: ASR, TTS, VAD, and speaker models use GGUF.
- **NumPy in, NumPy out**: ASR takes float32 PCM; TTS returns float32 PCM.
- **Edge-first backends**: CPU, Metal, CUDA, Vulkan, CANN, OpenCL, and WebGPU.

------

## Performance Snapshot

Test environment: Apple M1 Pro, funasr-nano-fp16.gguf, 15s audio.

| Configuration | RTF | Wall Time | Notes |
| --- | --- | --- | --- |
| CPU -t 4 | 0.465 | 12.4s | CPU-only inference |
| GPU -t 4 | 0.170 | 5.2s | Metal acceleration |
| GPU -t 4 Q4_K | 0.756 | - | Quantized model: GPU dequant overhead |
| CPU -t 4 Q4_K | 0.530 | - | Quantized model CPU inference, 596 MB (3.3x compression) |

RTF is processing time divided by audio duration. Lower is faster; RTF < 1 is
faster than real time.

------

## Supported Today

| Task | Models | Status |
| --- | --- | --- |
| ASR | SenseVoice-small, FunASR-nano, X-ASR (Zipformer2, streaming) | Stable |
| VAD | Silero VAD, FireRedVAD | Stable |
| TTS | OmniVoice, OpenVoice2, Kokoro, IndexTTS-2 | Active |
| Speaker | CAMPPlus | Stable |

**X-ASR** — Chinese/English Zipformer2 transducer (icefall/k2). One GGUF serves
both **offline** full-context decoding and **true chunked streaming** (per-layer
left-context caches, sub-second partials, `--chunk-len 16/32/48/96/192` fbank
frames). Punctuation and casing, greedy transducer decode, runs on CPU / Metal /
CUDA / Vulkan and quantizes to q4_k_m (99.5 MB).

**IndexTTS-2** — expressive zero-shot voice-cloning TTS (GPT + S2Mel CFM +
BigVGAN-v2 vocoder) with 4-mode emotion control (reference audio / vector / text
/ Qwen). See [docs/index2tts.md](docs/index2tts.md).

## In Progress

CosyVoice3, Qwen3-ASR, Qwen3-TTS.

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
# Bias proper nouns / fix homophones (e.g. 郭总 → 虢总); use --hotword-file for a large list
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  --hotwords "虢总,阿里巴巴"
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
# Serve ASR + TTS over an OpenAI-compatible HTTP API and MCP
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

If you are interested in the following areas, we welcome your PRs or participation in discussions:

- Adapting more models to the framework.
- Refining and optimizing the project architecture.
- Improving inference performance.

## Acknowledgements

1. [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)
2. [llama.cpp](https://github.com/ggml-org/llama.cpp)
3. [ggml](https://github.com/ggml-org/ggml)
4. [cppjieba](https://github.com/yanyiwu/cppjieba) — Chinese word segmentation
5. [WeText](https://github.com/wenet-e2e/wetext) — text normalization (ITN/TN)
6. [miniaudio](https://github.com/mackron/miniaudio) — single-file audio I/O
7. [X-ASR](https://github.com/Gilgamesh-J/X-ASR) Streaming-focused automatic speech recognition models
