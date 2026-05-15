#!/usr/bin/env python3
"""
RapidSpeech Offline ASR Example

Transcribe a WAV file and benchmark throughput.

Usage:
    python asr-offline.py --model model.gguf --audio test.wav
    python asr-offline.py --model model.gguf --audio test.wav --threads 8 --gpu 0 --runs 5
"""

import argparse
import struct
import time
import wave

import numpy as np
import rapidspeech


def read_wav(file_path: str) -> tuple[np.ndarray, int]:
    """
    Read a WAV file and return (float32 mono PCM, sample_rate).

    Supports 8/16/24/32-bit PCM. Multi-channel audio is averaged to mono.
    """
    with wave.open(file_path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nf = wf.getnframes()
        raw = wf.readframes(nf)

    dtype_map = {1: "b", 2: "h", 3: "i", 4: "i"}  # 24-bit stored as 3-byte → int32
    dtype = dtype_map.get(sw)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sw}")

    pcm = np.frombuffer(raw, dtype=f"<{dtype}").astype(np.float32)

    if sw == 1:
        pcm = (pcm - 128.0) / 128.0
    elif sw == 3:
        pcm /= (1 << 23)  # 24-bit signed
    else:
        pcm /= (1 << (sw * 8 - 1))

    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)

    return np.ascontiguousarray(pcm, dtype=np.float32), sr


def main():
    parser = argparse.ArgumentParser(description="RapidSpeech Offline ASR")
    parser.add_argument("--model", required=True, help="Path to GGUF ASR model")
    parser.add_argument("--audio", required=True, help="Path to WAV file (any sample rate, mono or stereo)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--gpu", type=int, default=1, help="Use GPU: 1=on, 0=off (default: 1)")
    parser.add_argument("--runs", type=int, default=1, help="Number of inference runs for benchmarking (default: 1)")
    parser.add_argument("--prompt", type=str, default=None, help="LLM decoder prompt (e.g. 'Transcribe:')")
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading model: {args.model}")
    ctx = rapidspeech.asr_offline(
        model_path=args.model,
        n_threads=args.threads,
        use_gpu=bool(args.gpu),
    )

    if args.prompt:
        ctx.set_user_input_prompt(args.prompt)
        print(f"Prompt: {args.prompt}")

    # --- Read audio ---
    pcm, sr = read_wav(args.audio)
    duration = len(pcm) / sr
    print(f"Audio: {args.audio}  ({sr} Hz, {len(pcm)} samples, {duration:.2f}s)")

    # --- Run ASR ---
    print(f"Running ASR ({args.runs} run(s))...")
    elapsed = []

    for i in range(args.runs):
        ctx.push_audio(pcm)
        t0 = time.perf_counter()
        ctx.process()
        t1 = time.perf_counter()
        elapsed.append(t1 - t0)

        text = ctx.get_text()
        if args.runs == 1:
            print(f"  Result: {text or '(no speech)'}")
        else:
            rtf = elapsed[-1] / duration
            print(f"  Run {i+1}: {elapsed[-1]:.3f}s  RTF={rtf:.3f}  {text or '(no speech)'}")

    if args.runs > 1:
        avg = sum(elapsed) / len(elapsed)
        print(f"\n  Average: {avg:.3f}s  RTF={avg/duration:.3f}")


if __name__ == "__main__":
    main()
