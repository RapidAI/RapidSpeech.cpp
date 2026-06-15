# RapidSpeech.cpp Technical Notes

This document keeps the deeper engineering context out of the main README.
The README should help a new user run the product. This file explains why the
runtime is built the way it is.

## Design Philosophy

RapidSpeech.cpp is a speech inference runtime built around three decisions:

- **GGUF as the model container**: ASR, TTS, VAD, speaker embedding, and
  quantized weights share one deployment format.
- **ggml as the execution stack**: backends such as CPU, Metal, CUDA, Vulkan,
  CANN, OpenCL, and WebGPU are exposed through one native runtime.
- **Speech-first application logic**: VAD segmentation, audio frontends,
  text frontends, speaker embedding, streaming buffers, and quantization tools
  live beside model execution rather than in Python glue code.

The goal is not to wrap one model. The goal is to make local speech inference
deployable as one native binary plus one model file.

## Key Differentiators

### vs. vLLM

vLLM is designed for data centers and high-throughput cloud serving. It is
strongly tied to Python and CUDA, and optimizes large batch serving with
systems such as PagedAttention.

RapidSpeech.cpp is designed for edge and on-device inference:

- Low latency and low memory footprint.
- No Python runtime required in production.
- Native CPU/GPU/NPU backends through ggml.
- Embeddable C API for desktop, mobile, browser, and server use.

### vs. sherpa-onnx

| Aspect | sherpa-onnx / ONNX Runtime | RapidSpeech.cpp / ggml |
| --- | --- | --- |
| Memory | Managed internally by ORT | Graph-planned ggml memory and explicit backend buffers |
| Quantization | Mostly INT8-oriented | K-quants and low-bit GGUF quantization |
| GPU execution | Execution-provider mapping | Native ggml backends plus speech-specific paths |
| Deployment | Model plus runtime libraries/config | Single native runtime with GGUF model metadata |

## Architecture

```text
rapidspeech/src/
  arch/       model implementations
  core/       context, model interface, processor
  frontend/   audio and text frontends
  utils/      logging, WAV IO, miniaudio
  c_api/      C API for external bindings
```

### Core Engine

The core runtime owns the GGUF loader, ggml backend initialization, scheduler,
model registry, and persistent state. Model selection is driven by the
`general.architecture` key inside the GGUF file.

### Architecture Layer

Model implementations live under `rapidspeech/src/arch/`. Current implemented
or active paths include:

- ASR: SenseVoice-small, FunASR-nano, Qwen3-ASR work in progress.
- VAD: Silero VAD, FireRedVAD.
- TTS: OmniVoice, OpenVoice2, Kokoro, CosyVoice3 work in progress.
- Speaker: CAMPPlus.

### Business Logic Layer

The runtime includes speech application features that are usually left outside
model inference:

- VAD-based segmentation.
- Quasi-streaming ASR.
- CTC first-pass plus LLM rescoring for FunASR-nano.
- Speaker embedding and diarization helpers.
- Text frontend processing for multilingual TTS.
- Quantization and importance-matrix collection.

## Backend Strategy

RapidSpeech.cpp builds on ggml backends and adds speech-specific choices where
the generic graph path is too expensive.

| Backend | Platform | Notes |
| --- | --- | --- |
| CPU | Linux, macOS, Windows | Default fallback |
| Metal | macOS / Apple Silicon | Enabled by default on Apple platforms |
| CUDA | Linux / NVIDIA | Source build flag: `-DRS_CUDA=ON` |
| Vulkan | Linux / Windows | Source build flag: `-DRS_VULKAN=ON` |
| CANN | Huawei Ascend | Source build flag: `-DRS_CANN=ON` |
| OpenCL | Linux / Android | Source build flag: `-DRS_OPENCL=ON` |
| WebGPU | Native Dawn / WASM | Source build flag: `-DRS_WEBGPU=ON` |

On macOS, OmniVoice DAC vocoder can use a custom Metal path through
`rapidspeech/src/arch/dac_metal.mm`, reducing graph dispatch overhead for the
vocoder.

## CLI Reference

### Offline ASR

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --gpu true
```

With VAD segmentation:

```bash
./build/rs-asr-offline \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4 \
  --vad-threshold 0.5 \
  --silence-ms 600
```

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to ASR GGUF model file | required |
| `-w, --wav` | Path to WAV audio file, 16 kHz mono | required |
| `-v, --vad` | Optional VAD GGUF model, Silero or FireRed | unset |
| `-t, --threads` | CPU thread count | 4 |
| `--gpu` | Enable GPU acceleration | true |
| `--vad-threshold` | VAD speech probability threshold | 0.5 |
| `--silence-ms` | Silence duration before segment split | 600 |
| `--max-segment-s` | Max ASR segment length | 30.0 |

### VAD-Segmented Online ASR

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  -t 4
```

Microphone mode:

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  --mic \
  -t 4
```

Two-pass mode:

```bash
./build/rs-asr-vad-online \
  -m /path/to/funasr-nano-fp16.gguf \
  -v /path/to/silero_vad_v6.gguf \
  -w /path/to/audio.wav \
  --two-pass
```

### Text To Speech

OpenVoice2 / MeloTTS:

```bash
./build/rs-tts-offline \
  -m /path/to/openvoice2-base-en.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --lang English \
  -o output.wav \
  --threads 4
```

OmniVoice:

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, welcome to RapidSpeech!" \
  --instruct "male, young adult, moderate pitch" \
  --lang English \
  --n-steps 32 \
  -o output.wav
```

OmniVoice voice cloning:

```bash
./build/rs-tts-offline \
  -m /path/to/omnivoice-f16.gguf \
  -t "Hello, this is cloned voice." \
  --ref /path/to/reference.wav \
  --ref-text "transcript of the reference audio" \
  -o output.wav
```

| Flag | Description | Default |
| --- | --- | --- |
| `-m, --model` | Path to TTS GGUF model file | required |
| `-t, --text` | Text to synthesize | required |
| `-o, --output` | Output WAV path | output.wav |
| `--lang` | Target language | English |
| `--ref` | Reference WAV for voice cloning | unset |
| `--ref-text` | Transcript for reference audio | unset |
| `--instruct` | Voice description for OmniVoice | male |
| `--seed` | Random seed | 42 |
| `--n-steps` | MaskGIT diffusion steps | 32 |
| `--threads` | CPU thread count | 4 |
| `--gpu` | Enable GPU acceleration | true |

### Quantization

```bash
./build/rs-quantize /path/to/input-f16.gguf /path/to/output-q4_k.gguf q4_k
```

Common types: `q4_0`, `q4_k`, `q5_0`, `q5_k`, `q8_0`, `f16`, `f32`.

Q2_K is not recommended for FunASR Nano because it can produce garbled output.

## Python

### Installation

```bash
pip install rapidspeech
pip install rapidspeech-cuda
pip install rapidspeech-metal
```

The Python import name is always `rapidspeech`.

### Build From Source

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

pcm = tts.synthesize("Hello, welcome to RapidSpeech!")
```

## Model Format Conversion

### ASR: HuggingFace to GGUF

```bash
python scripts/convert_hf_to_gguf.py \
  --model /path/to/hf-model-dir \
  --outfile /path/to/output.gguf \
  --outtype f16
```

### Silero VAD to GGUF

```bash
python scripts/convert_silero_to_gguf.py \
  --model /path/to/silero_vad_16k.safetensors \
  --output /path/to/silero_vad_v6.gguf
```

### OpenVoice2 / MeloTTS to GGUF

```bash
python scripts/convert_openvoice2.py \
  --base-model myshell-ai/MeloTTS-English \
  --converter-model myshell-ai/OpenVoiceV2 \
  --output-dir ./models \
  --language EN
```

Outputs:

- `openvoice2-base-<lang>.gguf`: base TTS model.
- `openvoice2-converter.gguf`: optional tone color converter.

### OmniVoice to GGUF

```bash
python scripts/convert_omnivoice_to_gguf.py \
  --model /path/to/omnivoice-model \
  --tokenizer /path/to/omnivoice-audio-tokenizer \
  --output /path/to/omnivoice-merged.gguf \
  --outtype f16
```

## Bindings And Examples

| Surface | Location |
| --- | --- |
| C++ CLI | `examples/` |
| Python | `python-api-examples/` |
| Browser / WASM | `wasm-examples/` |
| Node.js | `node-api-example/` |
| Colab | `colab/` |

The C API in `include/rapidspeech.h` is the stable boundary for non-C++
bindings.
