# RapidSpeech Node.js API Example

Run RapidSpeech locally from Node.js using the WebAssembly build. Both
**ASR** (speech → text) and **TTS** (text → speech) are supported. No
network calls after the WASM module is loaded — model and audio never
leave your machine.

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
node index.js asr -m sense-voice-small.gguf -w audio.wav --threads 4 --runs 10
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
  --ctc-precheck            Skip LLM on silence using a quick CTC precheck
  -r, --runs <n>            Inference runs (default: 1)

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
   heap, `process()` runs encoder + decoder, `get_text()` returns the
   transcript. `setUseLlm(true)` + `redecode()` performs an LLM second
   pass on the same encoder output.
3. **TTS pipeline** — `setTtsParams()` and `setDiffusionSteps()`
   configure generation, `synthesize(text)` runs the full
   `push_text → process → get_audio_*` loop and returns a
   Float32Array of PCM at the model's native sample rate.
4. **Voice cloning** — `pushReferenceAudio()` + `pushReferenceText()`
   provide a reference speaker before synthesis (OmniVoice).

## API surface (WASM exports)

| Function | Description |
|----------|-------------|
| `rs_wasm_init_ex(path, task_type, threads)` | Load model with explicit task type |
| `rs_wasm_init(path, threads)` | Legacy ASR-only init |
| `rs_wasm_push_audio(ptr, n)` | Push float32 PCM samples |
| `rs_wasm_push_text(text)` | Push UTF-8 text for TTS |
| `rs_wasm_push_reference_audio/text(...)` | Voice cloning |
| `rs_wasm_process()` | Run one inference step |
| `rs_wasm_redecode()` | Re-run decoder only (2-pass ASR) |
| `rs_wasm_get_text() / get_audio_ptr() / get_audio_len()` | Read outputs |
| `rs_wasm_set_use_llm / set_ctc_precheck / set_user_input_prompt` | ASR knobs |
| `rs_wasm_set_tts_params / set_tts_diffusion_steps` | TTS knobs |
| `rs_wasm_get_sample_rate / get_arch_name / get_version` | Metadata |

## Notes

- WASM is slower than the native build (no SIMD/Metal/CUDA). For
  production use prefer the C API or the Python bindings
  (`pip install rapidspeech`).
- TTS models are typically much larger than ASR models — make sure
  Node has enough memory (`--max-old-space-size=4096`).
