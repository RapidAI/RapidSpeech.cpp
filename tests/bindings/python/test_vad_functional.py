"""Layer-2 functional: silero_vad end-to-end roundtrip.

Push 2 s of silence, push 2 s of synthetic noise burst, expect that
`detect_full` returns at least one segment overlapping the burst, and
that `is_speech` flips at least once across the stream. This is a smoke
test, not an accuracy bench — we just want runtime-correct shape and
non-crashing behavior.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def vad(vad_model_path):
    import rapidspeech

    v = rapidspeech.vad(vad_model_path, n_threads=2, use_gpu=False)
    yield v


def _make_clip() -> np.ndarray:
    """1 s silence + 1 s noise burst + 1 s silence, 16 kHz mono."""
    sr = 16000
    silence = np.zeros(sr, dtype=np.float32)
    rng = np.random.default_rng(0)
    burst = rng.standard_normal(sr).astype(np.float32) * 0.5
    return np.concatenate([silence, burst, silence])


def test_arch_is_silero(vad):
    assert vad.get_arch() in ("silero-vad", "silero_vad", "silero")


def test_detect_full_returns_segments(vad):
    clip = _make_clip()
    segments = vad.detect_full(clip)
    assert isinstance(segments, list)
    # At least one segment should overlap with the burst (1.0–2.0 s).
    overlapping = [s for s in segments if s["end_s"] > 1.0 and s["start_s"] < 2.0]
    assert overlapping, f"no segment overlaps burst window: {segments}"


def test_streaming_push_and_drain(vad):
    vad.reset()
    clip = _make_clip()
    # 320-sample (20 ms) chunks
    flips = 0
    last_state = None
    for i in range(0, len(clip), 320):
        chunk = clip[i : i + 320]
        if len(chunk) < 320:
            break
        vad.push_audio(chunk)
        state = vad.is_speech()
        if last_state is not None and state != last_state:
            flips += 1
        last_state = state
    drained = vad.drain_segments()
    assert isinstance(drained, list)
    assert flips >= 1, "is_speech state never changed across burst"
