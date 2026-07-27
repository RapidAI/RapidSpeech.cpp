#!/usr/bin/env python3
"""
RapidSpeech True Streaming ASR Example (X-ASR)

Drives the model's chunked encoder + continuous transducer directly — audio
goes in as fixed chunks, partial text comes out with sub-second latency, and
the hypothesis is continuous across chunks. This is different from
`asr-online.py`, which is VAD-segmented (each segment decoded offline).

Only X-ASR (Zipformer2 transducer) supports this path; the script checks
`asr.stream_supported()` and exits otherwise.

Live capture requires `sounddevice` (PortAudio). `--simulate` replays a WAV as
if it were streaming and needs no microphone.

Usage:
    # Replay a WAV as a stream (no microphone needed)
    python asr-streaming.py --model xasr-q4_k_m.gguf --simulate test.wav

    # Live microphone — Ctrl-C to stop
    python asr-streaming.py --model xasr-q4_k_m.gguf

    # Lower latency (160 ms chunks) vs higher throughput (960 ms chunks)
    python asr-streaming.py --model xasr-q4_k_m.gguf --simulate test.wav --chunk-len 16
    python asr-streaming.py --model xasr-q4_k_m.gguf --simulate test.wav --chunk-len 96

Notes
-----
* --chunk-len is in fbank frames (10 ms each) and must be a multiple of 16:
  16/32/48/96/192 ≈ 160/320/480/960/1920 ms. Larger = higher latency, lower RTF.
* Audio is 16 kHz mono float32 in [-1, 1]; other formats are resampled.
"""

import argparse
import sys
import time
import wave

import numpy as np

try:
    import rapidspeech as rs
except ImportError:
    sys.stderr.write(
        "rapidspeech module not found. Build with -DRS_ENABLE_PYTHON=ON and "
        "put the built module on PYTHONPATH.\n"
    )
    raise


def load_wav_16k_mono(path):
    """Read a WAV, average to mono, resample to 16 kHz, return float32 [-1,1]."""
    w = wave.open(path, "rb")
    sr = w.getframerate()
    n = w.getnframes()
    sampwidth = w.getsampwidth()
    ch = w.getnchannels()
    raw = w.readframes(n)
    w.close()

    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
    if dt is None:
        raise ValueError(f"unsupported sample width {sampwidth}")
    data = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if sampwidth == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(1 << (8 * sampwidth - 1))
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        # simple linear resample (fine for a demo; use a real resampler for prod)
        idx = np.linspace(0, len(data) - 1, int(len(data) * 16000 / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return np.ascontiguousarray(data)


def run_simulate(asr, wav_path, feed_ms, realtime):
    pcm = load_wav_16k_mono(wav_path)
    step = int(16000 * feed_ms / 1000)
    dur = len(pcm) / 16000.0
    t0 = time.time()
    compute = 0.0
    for i in range(0, len(pcm), step):
        chunk = pcm[i : i + step]
        if realtime:
            target = i / 16000.0
            while time.time() - t0 < target:
                time.sleep(0.002)
        c0 = time.time()
        updated = asr.stream_push(np.ascontiguousarray(chunk))
        compute += time.time() - c0
        if updated:
            sys.stdout.write("\r\033[K" + asr.stream_get_text())
            sys.stdout.flush()
    c0 = time.time()
    asr.stream_finish()
    compute += time.time() - c0
    sys.stdout.write("\r\033[K" + asr.stream_get_text() + "\n")
    sys.stderr.write(
        f"[streaming] audio {dur:.2f}s  compute {compute:.2f}s  RTF {compute/dur:.3f}\n"
    )


def run_mic(asr, chunk_len):
    try:
        import sounddevice as sd
    except ImportError:
        sys.stderr.write("live mic needs `pip install sounddevice`\n")
        sys.exit(1)

    block = 1600  # 100 ms
    sys.stderr.write("[streaming] listening... Ctrl-C to stop\n")
    try:
        with sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", blocksize=block
        ) as stream:
            while True:
                data, _ = stream.read(block)
                if asr.stream_push(np.ascontiguousarray(data[:, 0])):
                    sys.stdout.write("\r\033[K" + asr.stream_get_text())
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    asr.stream_finish()
    sys.stdout.write("\r\033[K" + asr.stream_get_text() + "\n")


def main():
    ap = argparse.ArgumentParser(description="X-ASR true streaming demo")
    ap.add_argument("--model", required=True, help="X-ASR GGUF path")
    ap.add_argument("--simulate", help="WAV to replay as a stream (no mic)")
    ap.add_argument("--chunk-len", type=int, default=32,
                    help="chunk length in fbank frames (16/32/48/96/192)")
    ap.add_argument("--feed-ms", type=int, default=100,
                    help="how much audio to push per step in simulate mode")
    ap.add_argument("--fast", action="store_true",
                    help="simulate as fast as possible (no real-time pacing)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    asr = rs.asr_offline(args.model, args.threads, not args.no_gpu)
    if not asr.stream_supported():
        meta = asr.get_model_meta()
        sys.stderr.write(
            f"model '{meta['arch_name']}' does not support true streaming; "
            "use asr-online.py (VAD-segmented) instead.\n"
        )
        sys.exit(1)
    asr.set_chunk_len(args.chunk_len)

    if args.simulate:
        run_simulate(asr, args.simulate, args.feed_ms, realtime=not args.fast)
    else:
        run_mic(asr, args.chunk_len)


if __name__ == "__main__":
    main()
