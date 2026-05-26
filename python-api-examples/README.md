# Python API Examples

End-to-end Python examples for the `rapidspeech` package. Covers both offline
and streaming use of every task type the library exposes (ASR / TTS / VAD).

```
python-api-examples/
├── asr/
│   ├── asr-offline.py    # File-based ASR, optional VAD pre-segmentation, 2-pass
│   └── asr-online.py     # Microphone or WAV replay → neural VAD → ASR, 2-pass
└── tts/
    ├── tts-offline.py    # text → WAV (OmniVoice / OpenVoice2), voice cloning
    └── tts-streaming.py  # text → chunked PCM stream, low-latency consumption
```

## Prerequisites

```bash
# CPU
pip install rapidspeech

# CUDA / Metal
pip install rapidspeech-cuda
pip install rapidspeech-metal

# From source (picks up local backend env, e.g. RS_BACKEND=cuda)
pip install .
```

Download a GGUF model from
[HuggingFace](https://huggingface.co/RapidAI/RapidSpeech) or
[ModelScope](https://www.modelscope.cn/models/RapidAI/RapidSpeech) — the
examples accept any GGUF whose arch is supported.

Live mic capture (only used by `asr-online.py`) additionally needs:

```bash
pip install sounddevice    # PortAudio bindings
```

## ASR

### Offline — `asr/asr-offline.py`

Transcribe a WAV file (any bit-depth, any sample rate — auto-resampled to the
model's native rate, multi-channel mixed to mono).

```bash
# Basic
python asr/asr-offline.py --model funasr-nano.gguf --audio test.wav

# 2-pass: CTC greedy first (fast), then LLM rescoring (accurate)
python asr/asr-offline.py --model funasr-nano.gguf --audio test.wav --two-pass

# VAD-segmented (silero-vad / firered-vad auto-detected from the GGUF)
python asr/asr-offline.py --model funasr-nano.gguf --audio long.wav \
    --vad silero-vad.gguf --vad-threshold 0.5 --vad-min-seg 0.3 --two-pass

# Benchmark — repeat inference N times and print average RTF
python asr/asr-offline.py --model funasr-nano.gguf --audio test.wav --runs 5
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--model <path>` | GGUF ASR model |
| `--audio <path>` | WAV file (mono/stereo, 8/16/24/32-bit) |
| `--threads / --gpu` | CPU threads (4) / use GPU (1) |
| `--two-pass` | CTC → LLM rescore (FunASR-Nano) |
| `--ctc-precheck` | Skip LLM on silence with a quick CTC pre-check |
| `--vad <path>` | Enable VAD-driven pre-segmentation |
| `--vad-threshold <f>` | VAD speech threshold (default 0.5) |
| `--vad-min-seg <s>` | Drop segments shorter than this (default 0.3 s) |
| `--runs <n>` | Benchmark mode |
| `--prompt <text>` | LLM decoder prompt (FunASR-Nano) |

### Online — `asr/asr-online.py`

Continuous streaming from the microphone or from a WAV file played back as if
it were live. Audio is sliced into speech regions by a neural VAD and
transcripts are printed as soon as each segment closes.

```bash
# Live mic (Ctrl-C to stop)
python asr/asr-online.py \
    --model funasr-nano.gguf \
    --vad silero-vad.gguf \
    --vad-threshold 0.5

# 2-pass mode — emit CTC immediately, then LLM-rescored line on the next row
python asr/asr-online.py --model funasr-nano.gguf --vad silero-vad.gguf --two-pass

# Replay a WAV file as a stream (no mic needed)
python asr/asr-online.py --model funasr-nano.gguf --vad silero-vad.gguf \
    --simulate test.wav --simulate-realtime

# List input devices and exit
python asr/asr-online.py --model x --vad y --list-devices
```

Notes:
- VAD always runs at 16 kHz. If the ASR model uses a different sample rate the
  segment is resampled before being pushed to ASR — both VAD and ASR work in
  parallel for this reason.
- `--two-pass` and `--no-llm` are mutually exclusive.
- A rolling 60-second 16 kHz buffer is kept so segments can be sliced out of
  history once the VAD closes them.

## TTS

### Offline — `tts/tts-offline.py`

Synthesize a sentence and write a WAV. Works with OmniVoice and OpenVoice2.

```bash
# Basic (built-in voice description)
python tts/tts-offline.py --model omnivoice.gguf --text "Hello world" --output out.wav

# Tune voice / language / seed / diffusion steps (OmniVoice knobs)
python tts/tts-offline.py --model omnivoice.gguf \
    --text "你好世界" --lang Chinese --instruct "female" \
    --seed 7 --n-steps 16

# Voice cloning — supply a reference clip and its transcript
python tts/tts-offline.py --model omnivoice.gguf \
    --text "This is a cloned voice." \
    --ref reference.wav --ref-text "Transcript of the reference clip." \
    --output cloned.wav
```

### Streaming — `tts/tts-streaming.py`

Consume PCM chunk-by-chunk as the model produces them — useful for pipelined
playback or pushing to another consumer (WebSocket, ffmpeg, etc.). The full
clip is still assembled and written to disk for verification.

```bash
python tts/tts-streaming.py --model omnivoice.gguf --text "Hello, streaming world"
python tts/tts-streaming.py --model omnivoice.gguf --text "你好" --lang Chinese --n-steps 16
```

## Python API surface used here

| Class / function | Description |
|------------------|-------------|
| `rapidspeech.asr_offline(model_path, n_threads, use_gpu)` | Load an offline ASR model |
| `ctx.push_audio(pcm) / process() / get_text()` | Push float32 PCM, run inference, read text |
| `ctx.set_use_llm(bool) / redecode()` | Toggle LLM rescoring + re-run decoder for 2-pass |
| `ctx.set_user_input_prompt(str) / set_ctc_precheck(bool)` | Decoder knobs |
| `rapidspeech.vad(model_path, n_threads, use_gpu)` | Load silero-vad / firered-vad |
| `vad.push_audio / drain_segments / drain_frames / detect_full` | Streaming + one-shot VAD |
| `rapidspeech.tts_synthesizer(model_path, …)` | Load a TTS model |
| `tts.set_params / set_diffusion_steps` | OmniVoice knobs |
| `tts.set_reference_audio / set_reference_text` | Voice cloning |
| `tts.synthesize(text) / synthesize_streaming(text)` | Run synthesis (full / chunked) |

See [`rapidspeech/python/pybind_rapidspeech.cpp`](../rapidspeech/python/pybind_rapidspeech.cpp)
for the full binding source.
