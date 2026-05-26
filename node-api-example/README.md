# RapidSpeech Node.js API Example

Run RapidSpeech locally from Node.js using the WebAssembly build. Both
**ASR** (speech → text, with optional VAD pre-segmentation and 2-pass
LLM rescoring) and **TTS** (text → speech, with optional voice cloning)
are supported. No network calls after the WASM module is loaded —
model and audio never leave your machine.

```
node-api-example/
├── index.js        # ASR + TTS CLI (uses ../wasm-examples/rapidspeech-bridge.js)
├── package.json
└── README.md       # this file
```

## Prerequisites

- Node.js ≥ 18
- A GGUF model
    - ASR: SenseVoice Small, FunASRNano, …
    - TTS: OmniVoice, OpenVoice2, …

## Build the WASM module

```bash
cd rapidspeech/wasm
./build-wasm.sh
```

This writes `rapidspeech-wasm.{js,wasm}` to the build directory; the
example searches for it under `wasm-examples/` and the WASM build dir.

## Run

### ASR

```bash
node index.js asr -m model.gguf -w audio.wav
node index.js asr -m funasr-nano.gguf -w audio.wav --two-pass
node index.js asr -m funasr-nano.gguf -w audio.wav --no-llm        # CTC-only, fastest
node index.js asr -m sense-voice-small.gguf -w audio.wav --threads 4 --runs 10

# VAD-segmented (transcribe each detected speech region)
node index.js asr -m funasr-nano.gguf -w long.wav \
  --vad silero-vad.gguf --vad-threshold 0.5 --vad-min-seg 0.3 --two-pass
```

### TTS

```bash
node index.js tts -m omnivoice.gguf -t "Hello, this is rapidspeech." -o hello.wav
node index.js tts -m omnivoice.gguf -t "你好世界" --lang Chinese --n-steps 16 -o zh.wav

# Voice cloning
node index.js tts -m omnivoice.gguf \
  -t "Whatever you want me to say." \
  --ref reference.wav --ref-text "This is the reference transcript." \
  -o cloned.wav
```

## Usage

```
node index.js asr -m <model.gguf> -w <audio.wav> [options]
node index.js tts -m <model.gguf> -t "text"    [options]

ASR options:
  -w, --wav <path>          Input WAV file (8/16/24/32-bit PCM, mono/stereo)
  --two-pass                CTC greedy → LLM rescore (FunASRNano)
  --no-llm                  Disable LLM rescoring (CTC only — fastest)
  --ctc-precheck            Skip LLM on silence using a quick CTC precheck
  -r, --runs <n>            Inference runs (default: 1)

ASR + VAD pre-segmentation:
  --vad <path>              GGUF VAD model (silero-vad / firered-vad)
  --vad-threshold <f>       Speech threshold (default: 0.5)
  --vad-min-seg <s>         Drop segments shorter than this (default: 0.3)

TTS options:
  -t, --text <text>         Text to synthesize
  -o, --output <path>       Output WAV (default: out.wav)
  --instruct <text>         Voice description    (default: "male")
  --lang <lang>             Target language      (default: English)
  --seed <n>                Random seed          (default: 42)
  --n-steps <n>             Diffusion steps      (default: 32)
  --ref <path>              Reference WAV for voice cloning
  --ref-text <text>         Transcript of the reference audio

Common:
  -m, --model <path>        GGUF model
  --threads <n>             CPU threads (default: 2)
  -h, --help                Show this help
```

## How it works

The example uses the `RapidSpeechWASM` JS bridge in
`../wasm-examples/rapidspeech-bridge.js`, which wraps the Emscripten
module. The bridge handles:

1. **Model loading** — Bytes are written to the Emscripten VFS at
   `/model.gguf`, then `rs_wasm_init_ex(path, task_type, threads)`
   loads it.
2. **ASR pipeline** — `pushAudio()` copies a Float32Array into the WASM
   heap, `await process()` runs encoder + decoder, `get_text()` returns
   the transcript. `setUseLlm(true)` + `await redecode()` performs an
   LLM second pass on the same encoder output. `process()` and
   `redecode()` are async because they may suspend through WebGPU's
   queue ops under ASYNCIFY — always `await` them.
3. **VAD pre-segmentation** — `RapidSpeechVAD` loads a silero-vad or
   firered-vad GGUF, runs at 16 kHz, and emits `{start_s, end_s}`
   segments. Each segment is then transcribed independently.
4. **TTS pipeline** — `setTtsParams()` and `setDiffusionSteps()`
   configure generation, `await synthesize(text)` runs the full
   `push_text → process → get_audio_*` loop and returns a
   Float32Array of PCM at the model's native sample rate.
5. **Voice cloning** — `pushReferenceAudio()` + `pushReferenceText()`
   provide a reference speaker before synthesis (OmniVoice).

## API surface (WASM exports)

| Function | Description |
|----------|-------------|
| `rs_wasm_init_ex(path, task_type, threads)` | Load model with explicit task type |
| `rs_wasm_init(path, threads)` | Legacy ASR-only init |
| `rs_wasm_push_audio(ptr, n)` | Push float32 PCM samples |
| `rs_wasm_push_text(text)` | Push UTF-8 text for TTS |
| `rs_wasm_push_reference_audio/text(...)` | Voice cloning |
| `rs_wasm_process()` | Run one inference step (async — `await` it) |
| `rs_wasm_redecode()` | Re-run decoder only (2-pass ASR, async) |
| `rs_wasm_get_text() / get_audio_ptr() / get_audio_len()` | Read outputs |
| `rs_wasm_set_use_llm / set_ctc_precheck / set_user_input_prompt` | ASR knobs |
| `rs_wasm_set_tts_params / set_tts_diffusion_steps` | TTS knobs |
| `rs_wasm_get_sample_rate / get_arch_name / get_version` | Metadata |
| `rs_wasm_vad_init / push_audio / drain_segments / drain_frames` | VAD streaming API |

## Notes

- WASM is slower than the native build (no SIMD/Metal/CUDA). For
  production use prefer the C API or the Python bindings
  (`pip install rapidspeech`).
- TTS models are typically much larger than ASR models — make sure
  Node has enough memory (`--max-old-space-size=4096`).
- `await` is mandatory on `rs.process()`, `rs.redecode()`, and
  `rs.synthesize()` — they're ASYNCIFY-aware. Forgetting it returns a
  bare `Promise` and the script silently logs `undefined`.

## See also

- [`../wasm-examples/README.md`](../wasm-examples/README.md) — the browser demo using the same WASM module
- [`../python-api-examples/README.md`](../python-api-examples/README.md) — native Python bindings (faster, no WASM overhead)
- [`../README.md`](../README.md) — project overview and C++ CLI usage
