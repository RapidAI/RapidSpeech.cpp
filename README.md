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


# RapidSpeech.cpp 🎙️

Python-friendly local ASR and TTS, powered by a native C++ GGUF runtime.

**RapidSpeech.cpp** gives Python developers a simple API for local speech
recognition, text-to-speech, VAD, speaker embedding, and voice cloning. Under
the hood it uses a pure C/C++ engine, **ggml** backends, and a unified **GGUF**
model format, so you get native performance without running a speech server.

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

- **Python API, native core**: write Python, run a C++/ggml engine underneath.
- **One model format**: ASR, TTS, VAD, and speaker models use GGUF.
- **NumPy in, NumPy out**: ASR takes float32 PCM; TTS returns float32 PCM.
- **Local by default**: no cloud API, no speech server, no Python model stack.
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
| ASR | SenseVoice-small, FunASR-nano | Stable |
| VAD | Silero VAD, FireRedVAD | Stable |
| TTS | OmniVoice, OpenVoice2, Kokoro | Active |
| Speaker | CAMPPlus | Stable |

## In Progress

CosyVoice3, Qwen3-ASR, Qwen3-TTS.

------

## Documentation

- [Python examples](python-api-examples/README.md)
- [Technical Notes](docs/TECHNICAL.md): architecture, design tradeoffs, backends,
  model conversion, and binding surfaces.
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

Build artifacts are located in the `build/` directory:
- `rs-asr-offline` — Offline ASR command-line tool
- `rs-asr-vad-online` — VAD-segmented quasi-streaming ASR command-line tool
- `rs-tts-offline` — Offline TTS command-line tool
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
