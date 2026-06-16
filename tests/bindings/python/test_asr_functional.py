"""Layer-2 functional: SenseVoice ASR runtime smoke.

The point is "load the model, push 3 s of silence, call process(), the
binding survives." We do not check transcription accuracy — silence
plausibly returns empty or '<|nospeech|>' depending on the build.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def asr(asr_model_path):
    import rapidspeech

    return rapidspeech.asr_offline(asr_model_path, n_threads=2, use_gpu=False)


def test_model_meta(asr):
    meta = asr.get_model_meta()
    assert isinstance(meta, dict)
    assert meta.get("audio_sample_rate") in (16000, 22050, 24000)
    arch = meta.get("arch_name", "")
    assert arch, "arch_name empty"


def test_backend_name_is_string(asr):
    name = asr.get_backend_name()
    assert isinstance(name, str)
    assert name  # at least "CPU"


def test_silence_roundtrip(asr):
    asr.reset()
    sr = asr.get_model_meta()["audio_sample_rate"]
    silence = np.zeros(sr * 3, dtype=np.float32)
    asr.push_audio(silence)
    rc = asr.process()
    assert rc in (0, 1)  # >= 0 means no error
    text = asr.get_text()
    assert isinstance(text, str)


def test_reset_clears_state(asr):
    sr = asr.get_model_meta()["audio_sample_rate"]
    asr.push_audio(np.zeros(sr, dtype=np.float32))
    asr.reset()
    # After reset, get_text returns empty or previous-buffered residue;
    # we only assert no crash.
    _ = asr.get_text()
