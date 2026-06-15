# WeTextProcessing FST data (Chinese TN)

Pre-compiled OpenFST tagger + verbalizer pair used by
`rs::WeTextNormalizer` ahead of the ZH G2P stage in Kokoro / OpenVoice2.

| File | Size | sha256 |
|---|---:|---|
| `zh_tn_tagger.fst`     | 4 312 998 B | `c18405688a8c7aa41a6ec5c40117e12b5a593da67adcfc575098f82120164ec4` |
| `zh_tn_verbalizer.fst` |   558 366 B | `5686678dec3b6220d1a030ba4e00c729786b91c4863b3d37d0f9ea69df0985c1` |

## Provenance

Built from upstream [wenet-e2e/WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)
Chinese TN grammar via Pynini. Upstream does not ship pre-compiled FSTs as
release artifacts — they are produced on first `ZhNormalizer(overwrite_cache=True)`
call and cached under `~/.cache/WeTextProcessing/`.

The copies in this directory were imported on 2026-06-15. Re-generate with
`scripts/download_wetext_fst.sh` if you need to rebuild against a newer
grammar revision.

## File-naming convention (important)

The first argument to `wetext::Processor(tagger, verbalizer)` must contain
one of `zh_tn_`, `zh_itn_`, `en_tn_`, `ja_tn_` as a substring — the
upstream parser switches on it to choose `ParseType`. **Do not rename**.

## Runtime knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_WETEXT`           | `1` (on) | `0` disables TN entirely; G2P sees raw text |
| `RS_WETEXT_DATA_DIR`  | `rapidspeech/data/wetext` | Override directory holding the two `.fst` files |
| `RS_WETEXT_DUMP`      | `0`      | `1` prints `TN: <raw> -> <normalised>` to stderr per `PushText` |

Missing or unreadable FST files → `WeTextNormalizer::Load` returns false
and `Normalize` becomes a pass-through (a warning is logged once).

## License

Apache-2.0, same as upstream WeTextProcessing. See
`third_party/wetext/LICENSE` for the full text.
