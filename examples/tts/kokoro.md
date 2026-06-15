# Kokoro v1.1-zh TTS — Usage Guide

Kokoro v1.1-zh is a **StyleTTS2 + iSTFTNet** Chinese text-to-speech model
([hexgrad/Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)).
Compared to the diffusion-based OmniVoice/CosyVoice3 architectures, Kokoro is a
single deterministic forward pass: tokenized phonemes → 12-layer pLBERT → duration
predictor → AdaIN decoder → iSTFT vocoder → 24 kHz waveform.

- **Model size**: 156 MB (F16, 82 M parameters)
- **Sample rate**: 24 kHz
- **Languages**: Chinese (Mandarin), bopomofo-based phoneme set
- **Voices**: 100 Chinese voices (55 `zf_*` female, 45 `zm_*` male) + 3 English bonus voices
- **Inference**: deterministic per voice + text (no diffusion, no CFG)

## Quick Start

```bash
# 1. Convert PyTorch checkpoint to GGUF (once)
python scripts/convert_kokoro_to_gguf.py \
  --input /path/to/Kokoro-82M-v1.1-zh \
  --output models/kokoro-v1_1-zh-f16.gguf
python scripts/convert_kokoro_voice_to_gguf.py \
  --input-dir /path/to/Kokoro-82M-v1.1-zh/voices \
  --output-dir models/kokoro_voices/

# 2. Dump pypinyin dictionaries used by the C++ G2P (once)
python scripts/dump_pypinyin_data.py
# → rapidspeech/data/kokoro_zh/pinyin_single.bin + pinyin_phrases.bin

# 3. Synthesize from raw Chinese text (G2P runs automatically)
./build/rs-tts-offline \
  -m models/kokoro-v1_1-zh-f16.gguf \
  --voice models/kokoro_voices/kokoro-voice-zf_001.gguf \
  -t "你好，世界，今天天气真好" --lang zh \
  -o output.wav
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `-m <path>` | Main Kokoro GGUF (required) |
| `--voice <path>` | Voice pack GGUF (required for ZH voices) |
| `-t <text>` | Input text (Chinese, or IPA when `--phonemes` set) |
| `--lang zh` | Use the Chinese G2P pipeline (default is English; required for `-t` ZH text) |
| `--phonemes` | Bypass G2P — treat `-t` as already-tokenised bopomofo (debug / parity testing) |
| `--length-scale <f>` | Duration multiplier (default `1.0`; `<1` faster, `>1` slower) |
| `--seed <n>` | Sampler seed (default `42`) |
| `--threads <n>` | CPU threads (default `4`) |
| `--gpu true|false` | GPU on/off (default on; ignored — see [Backend](#backend) below) |
| `-o <path>` | Output WAV path |

### Optional environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `RS_KOKORO_JIEBA_DICT_DIR` | cppjieba dict directory | `third_party/cppjieba/dict` |
| `RS_KOKORO_ZH_DATA_DIR` | pypinyin binary tables | `rapidspeech/data/kokoro_zh` |
| `RS_KOKORO_DUMP_G2P` | If set, prints the C++ G2P bopomofo string to stderr (handy for diffing against `misaki[zh]`) | unset |
| `RS_KOKORO_EN` | `0` disables the misaki[en] English fallback; Latin runs inside ZH text then emit `❓` | `1` |
| `RS_KOKORO_EN_DATA_DIR` | Directory holding `us_gold.bin` | `rapidspeech/data/kokoro_en` |
| `RS_WETEXT` | `0` disables the WeTextProcessing TN pass entirely | `1` |
| `RS_WETEXT_DATA_DIR` | Directory holding `zh_tn_tagger.fst` + `zh_tn_verbalizer.fst` | `rapidspeech/data/wetext` |
| `RS_WETEXT_DUMP` | `1` prints `TN: <raw> -> <normalised>` to stderr per `PushText` | unset |

### Chinese text normalization (WeTextProcessing)

Before G2P the model runs an OpenFST-based TN pass (vendored from
[wenet-e2e/WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)) that
rewrites mixed text to its spoken form, e.g.

```
2.5平方电线可以承受20安培电流。
  ↓ TN
二点五平方电线可以承受二十安培电流.
  ↓ ZHG2P
ʈʂʰɻ̩˧˩˧tjɛn˥˩...
```

The pass is default-on for `--lang zh`. If
`rapidspeech/data/wetext/zh_tn_{tagger,verbalizer}.fst` is missing the
wrapper logs a warning and becomes a pass-through (G2P sees the raw text).
Set `RS_WETEXT=0` to disable, or rebuild the FSTs with
`scripts/download_wetext_fst.sh`.

## Quantization

`rs-quantize` ships Kokoro-friendly mixed-precision strategies for the four
`*_k_m` ftypes. Run after producing the F16 GGUF:

```bash
./build/rs-quantize models/kokoro-v1_1-zh-f16.gguf \
    models/kokoro-v1_1-zh-q4_k_m.gguf q4_k_m
```

| ftype  | size  | quality                          | notes                                  |
|--------|-------|----------------------------------|----------------------------------------|
| f16    | 156 MB | reference                       | default conversion output              |
| q5_k_m |  80 MB | indistinguishable from f16       | recommended for highest fidelity       |
| q4_k_m |  73 MB | indistinguishable from f16       | **recommended default** (-53%)         |
| q3_k_m |  68 MB | minor degradation; intelligible  | LSTM bumped to Q4_K for tone stability |
| q2_k_m |  63 MB | duration predictor unstable      | CPU-only fallback; not recommended     |

Conv1d weights are stored as flattened 2D matrices `[K·Cin, Cout]` (the
convert script does the reshape) so the quantizer's k-quant block size of 256
engages on `ne[0] = K·Cin` instead of demoting tiny `K`-axis tensors to F16.
This was the dominant size win over the previous q4_k_n build (144 MB → 73 MB).
At runtime `kokoro_conv_1d_2d` reconstructs the geometry via `ggml_im2col` +
`ggml_mul_mat` (`rapidspeech/src/arch/kokoro.cpp`).

## Voice Packs

Each voice pack is a ~520 KB GGUF holding the **128-d predictor reference style**
and **128-d decoder reference style** for every phoneme length L ∈ [1, 510]. At
synthesis time the engine slices `voice_pack[L-1]` for the current utterance
length.

```
models/kokoro_voices/
  kokoro-voice-zf_001.gguf  …  kokoro-voice-zf_055.gguf   (55 female voices)
  kokoro-voice-zm_009.gguf  …  kokoro-voice-zm_100.gguf   (45 male voices)
  kokoro-voice-af_maple.gguf, kokoro-voice-af_sol.gguf, kokoro-voice-bf_vale.gguf
```

The `af_*` / `bf_*` voices come from upstream Kokoro multilingual checkpoints and
are kept for completeness; the ZH model emits Mandarin regardless of the voice
pack's nominal language.

## Chinese G2P Pipeline

The C++ G2P is a port of **`misaki[zh]` v1.1** (`ZHG2P(version='1.1')`) — a
[Kokoro Discord](https://github.com/hexgrad/misaki) upstream. Stages:

1. `cn2an.transform(text, 'an2cn')` — Arabic digits → Chinese numerals (`123` → `一百二十三`)
2. `map_punctuation` — full-width punctuation → ASCII (`，` → `, `, `。` → `. `, …)
3. Split into Chinese / non-Chinese segments
4. **Chinese segments** → `cppjieba.PosCut` for `(word, pos)` tokens →
   - `ToneSandhi` (`不`/`一`/三声/儿化/轻声)
   - `pypinyin.lazy_pinyin(style=INITIALS)` + `lazy_pinyin(style=FINALS_TONE3)`
   - `ZH_MAP` lookup → bopomofo string (`ㄋㄧ2`, `ㄓ中1`, etc.)
5. **English segments** → vendored `misaki[en]` gold IPA lookup with
   letter-by-letter fallback for OOD words (see
   `rapidspeech/data/kokoro_en/`). Set `RS_KOKORO_EN=0` to disable and
   restore the old `❓` unk placeholder.

The phoneme stream emitted by step 5 is what the kokoro tokenizer ingests
(`tokenizer.ggml.tokens`, 178 symbols including bopomofo, the `/` word
separator, and ASCII punctuation).

### Verifying parity with the Python reference

```bash
# Python reference
python -c "from misaki.zh import ZHG2P; \
  print(ZHG2P(version='1.1')('你好，世界，今天天气真好')[0])"
# → ㄋㄧ2ㄏㄠ3, ㄕ十4ㄐㄝ4, ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄓㄣ1/ㄏㄠ3

# C++ port
RS_KOKORO_DUMP_G2P=1 ./build/rs-tts-offline \
  -m models/kokoro-v1_1-zh-f16.gguf \
  --voice models/kokoro_voices/kokoro-voice-zf_001.gguf \
  -t "你好，世界，今天天气真好" --lang zh -o /tmp/out.wav 2>&1 | grep "^CXX:"
# → CXX: ㄋㄧ2ㄏㄠ3, ㄕ十4ㄐㄝ4, ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄓㄣ1ㄏㄠ3
```

Across a 10-sentence corpus the C++ port matches the Python reference **100 % on
phoneme tokens**. The only differences are `/` word separators caused by
cppjieba's bundled dictionary cutting some compounds (e.g. `真好`, `我要`) as a
single word where `jieba.posseg` cuts them as two. These do not affect intelligibility.

### Bypassing G2P (debug)

If you want to feed the kokoro tokenizer directly (e.g. to A/B against a different
G2P implementation):

```bash
PY_IPA=$(python -c "from misaki.zh import ZHG2P; print(ZHG2P(version='1.1')('你好')[0])")
./build/rs-tts-offline \
  -m models/kokoro-v1_1-zh-f16.gguf \
  --voice models/kokoro_voices/kokoro-voice-zf_001.gguf \
  --phonemes -t "$PY_IPA" -o /tmp/out.wav
```

## Backend

- **macOS (Apple Silicon)**: the iSTFTNet generator has a stride-10
  `ConvTranspose1d` that hangs Apple's Metal driver. The Kokoro engine forces
  the generator subgraph onto CPU; pLBERT + duration predictor still run on
  Metal. End-to-end RTF is ~1.5 on M1 Pro with 4 CPU threads for `--threads`
  scheduling. This is automatic — no flag needed.
- **Linux/Windows (CPU)**: works as-is. CUDA/Vulkan backends are not yet
  validated for the Kokoro generator.

## Length Control

Kokoro outputs duration per phoneme from a learned regressor. `--length-scale`
multiplies these durations:

```bash
./build/rs-tts-offline … --length-scale 0.9 …   # 10 % faster speech
./build/rs-tts-offline … --length-scale 1.15 …  # ~15 % slower
```

Values outside `[0.7, 1.3]` typically distort prosody.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `KokoroModel: failed to load ZH G2P` | Missing data files | Re-run `scripts/dump_pypinyin_data.py` and confirm `third_party/cppjieba/dict/` exists (git submodule) |
| Silent / near-silent WAV | Wrong `--lang` (default is English; ZH text is dropped) | Pass `--lang zh` |
| `❓` symbols in `RS_KOKORO_DUMP_G2P` output | `RS_KOKORO_EN=0` is set, or `us_gold.bin` is missing/corrupt, or the OOD spell fallback couldn't resolve any letters | Unset `RS_KOKORO_EN`, verify `rapidspeech/data/kokoro_en/us_gold.bin` is present, or split the request |
| Hang on Apple Silicon | Older build without the Metal workaround | Pull latest `rs_context.cpp`; generator must run on CPU |
| `cppjieba` segmentation differs from `jieba.posseg` | Dictionary difference between cppjieba (`jieba.dict.utf8`) and Python jieba | Cosmetic (`/` placement only); does not change phonemes |

## See Also

- Architecture port notes: `rapidspeech/src/arch/kokoro.{h,cpp}`,
  `rapidspeech/src/frontend/kokoro_{g2p_zh,pinyin,zh_tone_sandhi}.{h,cpp}`
- Upstream `misaki[zh]` reference: <https://github.com/hexgrad/misaki>
- Kokoro model card: <https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh>
- cppjieba (MIT, bundled as submodule): <https://github.com/yanyiwu/cppjieba>
