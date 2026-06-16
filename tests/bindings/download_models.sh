#!/usr/bin/env bash
# Pulls the small models that the binding CI needs.
#
# Layer 1 (always): nothing — surface tests do not load models.
# Layer 2 (every PR): silero_vad (~1.2 MB) + sensevoice-small-q3_k (~150 MB).
# Layer 3 (workflow_dispatch with FULL=1): everything else.
#
# Models are placed under the directory pointed to by $1 (default
# tests/bindings/_models/), so the GitHub Actions cache key can be
# anchored on a single path.

set -euo pipefail

DEST="${1:-$(dirname "$0")/_models}"
mkdir -p "$DEST"

HF_BASE="https://huggingface.co/RapidAI/RapidSpeech/resolve/main"

fetch() {
    local rel="$1"
    local out="$DEST/$(basename "$rel")"
    if [ -s "$out" ]; then
        echo "  cached: $out"
        return 0
    fi
    echo "  fetch:  $HF_BASE/$rel  ->  $out"
    curl -L --fail --silent --show-error --retry 3 --retry-delay 2 \
        -o "$out.part" "$HF_BASE/$rel"
    mv "$out.part" "$out"
}

echo "=== Layer 2 (always) ==="
fetch "VAD/silero_vad_v6.gguf"
fetch "ASR/SenseVoice/sense-voice-small-q3_k.gguf"

if [ "${FULL:-0}" = "1" ]; then
    echo "=== Layer 3 (FULL=1) ==="
    # OmniVoice smallest variant for TTS surface tests
    fetch "TTS/omnivoice-q2_k.gguf"
    # When Kokoro / CosyVoice3 / OpenVoice2 / CAM++ / KWS-context land on
    # the public HF repo, add them here and the matching pytest will pick
    # them up automatically via the env vars below.
fi

echo
echo "Models in $DEST:"
ls -lh "$DEST" 2>/dev/null || true
