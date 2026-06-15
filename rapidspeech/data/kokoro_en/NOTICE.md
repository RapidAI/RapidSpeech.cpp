# NOTICE

This directory contains data derived from the
[hexgrad/misaki](https://github.com/hexgrad/misaki) project (PyPI package
`misaki` v0.9.4), licensed under the Apache License 2.0.

## Vendored asset

`us_gold.bin` is a 1-for-1 binary re-packaging of `misaki/data/us_gold.json`
(US accent gold IPA dictionary). The dump procedure:

1. Read every `(key, value)` pair.
2. If `value` is a string, keep as-is.
3. If `value` is a POS dict (~790 entries: VBD/NN/JJ/...), keep
   `value["DEFAULT"]`. The misaki POS tagger is intentionally NOT ported —
   coverage loss vs upstream is < 1% and inaudible.
4. Skip records whose key or IPA exceeds 255 UTF-8 bytes (none observed in
   the current snapshot).

The original JSON content is otherwise unmodified — same Unicode IPA
symbols, same set of words. See `scripts/dump_misaki_en_gold.py` for the
generator and `meta.txt` for sha256 + source path of the current binary.

## Apache 2.0 attribution

```
Copyright 2024- hexgrad and the misaki contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

Full upstream LICENSE text:
https://github.com/hexgrad/misaki/blob/main/LICENSE
