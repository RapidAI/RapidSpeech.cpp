#!/usr/bin/env python3
"""
RapidSpeech Offline TTS Example

Synthesize a sentence with a GGUF TTS model and write the result to a WAV file.
Optional voice cloning is supported by passing a reference WAV + transcript.

Usage:
    # Basic synthesis (built-in voice description)
    python tts-offline.py --model omnivoice.gguf --text "Hello world" --output out.wav

    # Voice cloning
    python tts-offline.py --model omnivoice.gguf \
        --text "Hello world" \
        --ref ref.wav --ref-text "This is the reference utterance" \
        --output cloned.wav

    # Tune voice/lang/seed/diffusion steps
    python tts-offline.py --model omnivoice.gguf \
        --text "你好世界" --lang Chinese --instruct "female" --seed 7 --n-steps 16
"""

import argparse
import struct
import time
import wave

import numpy as np
import rapidspeech


def read_wav_mono_float(path: str) -> tuple[np.ndarray, int]:
    """Minimal mono float32 WAV reader; supports 8/16/24/32-bit PCM."""
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sw == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
        i32 = np.where(i32 & 0x800000, i32 | ~0xFFFFFF, i32)
        pcm = i32.astype(np.float32) / (1 << 23)
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)

    return np.ascontiguousarray(pcm, dtype=np.float32), sr


def write_wav_pcm16(path: str, pcm: np.ndarray, sample_rate: int) -> None:
    """Write float32 PCM in [-1,1] to a 16-bit mono WAV file."""
    clipped = np.clip(pcm, -1.0, 1.0)
    i16 = (clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(i16.tobytes())


def main():
    p = argparse.ArgumentParser(description="RapidSpeech Offline TTS")
    p.add_argument("--model", required=True, help="Path to GGUF TTS model")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--output", default="output.wav", help="Output WAV file (default: output.wav)")
    p.add_argument("--instruct", default="male", help="Voice description (default: male)")
    p.add_argument("--lang", default="English", help="Target language (default: English)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--n-steps", type=int, default=32, help="Diffusion steps 1-128 (default: 32)")
    p.add_argument("--ref", default=None, help="Reference audio WAV for voice cloning")
    p.add_argument("--ref-text", default=None, help="Transcript of the reference audio")
    p.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    p.add_argument("--gpu", type=int, default=1, help="Use GPU: 1=on, 0=off (default: 1)")
    args = p.parse_args()

    print(f"rapidspeech version: {rapidspeech.version()}")
    print(f"Loading model: {args.model}")
    tts = rapidspeech.tts_synthesizer(
        model_path=args.model,
        n_threads=args.threads,
        use_gpu=bool(args.gpu),
    )
    meta = tts.get_model_meta()
    print(f"Model:    {meta['arch_name']} ({meta['audio_sample_rate']} Hz)")
    print(f"Backend:  {tts.get_backend_name()}")

    tts.set_params(instruct=args.instruct, language=args.lang, seed=args.seed)
    tts.set_diffusion_steps(args.n_steps)
    print(f"Params:   instruct=\"{args.instruct}\" lang={args.lang} seed={args.seed} steps={args.n_steps}")

    # ── Voice cloning ──────────────────────────────────────────────
    if args.ref:
        if not args.ref_text:
            raise SystemExit("Error: --ref requires --ref-text (transcript of the reference)")
        ref_pcm, ref_sr = read_wav_mono_float(args.ref)
        tts.set_reference_audio(ref_pcm, sample_rate=ref_sr)
        tts.set_reference_text(args.ref_text)
        print(f"Cloning:  {args.ref}  ({ref_sr} Hz, {len(ref_pcm)/ref_sr:.2f}s)")

    # ── Synthesize ─────────────────────────────────────────────────
    print(f"Text:     {args.text!r}")
    t0 = time.perf_counter()
    pcm = tts.synthesize(args.text)
    elapsed = time.perf_counter() - t0

    sr = tts.get_sample_rate()
    duration = len(pcm) / sr
    rtf = elapsed / max(duration, 1e-9)
    print(f"Output:   {len(pcm)} samples @ {sr} Hz ({duration:.2f}s, "
          f"elapsed={elapsed:.2f}s, RTF={rtf:.3f})")

    write_wav_pcm16(args.output, pcm, sr)
    print(f"Wrote     {args.output}")


if __name__ == "__main__":
    main()
