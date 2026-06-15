# misaki[en] gold IPA table (vendored)

Binary dump of misaki's `us_gold.json` consumed by `rs::kokoro_en::EnG2P`
when the Kokoro ZH G2P encounters Latin-script segments.

| File | Size | Records |
|---|---:|---:|
| `us_gold.bin` | 2 239 615 B | 90 201 |

## Provenance

- Upstream: [hexgrad/misaki](https://github.com/hexgrad/misaki) v0.9.4
- Source JSON: `misaki/data/us_gold.json` (US accent, rating-4 gold dictionary)
- POS-tagged entries (~790) contribute only their `DEFAULT` IPA;
  porting the misaki POS tagger is out of scope.
- Re-generate with `scripts/dump_misaki_en_gold.py` (see top of `meta.txt`
  for the exact source path + sha256 of the current binary).

## Binary format

Little-endian, header + flat record list.

```
header:
  magic   uint32   0x4E45474D  ('MGEN' = Misaki Gold EN)
  version uint32   1
  count   uint32

per record (count times):
  key_len   uint8
  key       UTF-8 bytes, key_len long
  ipa_len   uint8
  ipa       UTF-8 bytes, ipa_len long
```

Records are sorted by key for deterministic builds. Entries whose key or
value would exceed 255 UTF-8 bytes are skipped by the dump script (the
longest observed key in misaki is ~30 bytes, longest IPA ~40, so this is
not load-bearing).

## Runtime knobs

| Env var | Default | Effect |
|---|---|---|
| `RS_KOKORO_EN`           | `1` (on) | `0` disables the English fallback; Latin segments emit `❓` |
| `RS_KOKORO_EN_DATA_DIR`  | `rapidspeech/data/kokoro_en` | Override directory holding `us_gold.bin` |

If `us_gold.bin` is missing or corrupt, `EnG2P::Load` returns false and
the KokoroModel logs a warning; `ZHG2P` then falls back to `❓` for Latin
segments — same behaviour as `RS_KOKORO_EN=0`.

## License

Apache-2.0, same as upstream misaki. See `third_party/misaki-LICENSE` or
the `NOTICE.md` next to this README.
