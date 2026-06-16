"""Layer-3 functional: OmniVoice TTS smoke (skipped unless model present)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def tts(tts_model_path):
    import rapidspeech

    return rapidspeech.tts_synthesizer(tts_model_path, n_threads=2, use_gpu=False)


def test_sample_rate_positive(tts):
    sr = tts.get_sample_rate()
    assert sr > 0


def test_synthesize_short(tts):
    """Short ASCII phrase should produce non-empty mono float32 PCM."""
    tts.set_params(instruct="male", language="English", seed=42)
    tts.set_diffusion_steps(8)  # keep CI fast
    pcm = tts.synthesize("hello world")
    assert isinstance(pcm, np.ndarray)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert pcm.size > 0
    # Should not be all-zero / all-NaN
    assert np.isfinite(pcm).all()
    assert np.abs(pcm).max() > 0.0


def test_synthesize_streaming_chunks(tts):
    chunks = tts.synthesize_streaming("hello world")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, np.ndarray) and c.dtype == np.float32
