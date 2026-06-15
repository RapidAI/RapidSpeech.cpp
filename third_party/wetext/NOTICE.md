# third_party/wetext — Vendored notice

This directory contains a minimal vendored copy of
[wenet-e2e/WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing),
licensed under Apache-2.0 (see `LICENSE`).

## Source

- Upstream: https://github.com/wenet-e2e/WeTextProcessing
- Vendored from branch: `master`
- Fetched: 2026-06-15

## Files copied verbatim

| Vendored path                          | Upstream path                                          |
|----------------------------------------|--------------------------------------------------------|
| `processor/wetext_processor.h`         | `runtime/processor/wetext_processor.h`                 |
| `processor/wetext_processor.cc`        | `runtime/processor/wetext_processor.cc`                |
| `processor/wetext_token_parser.h`      | `runtime/processor/wetext_token_parser.h`              |
| `processor/wetext_token_parser.cc`     | `runtime/processor/wetext_token_parser.cc`             |
| `utils/wetext_string.h`                | `runtime/utils/wetext_string.h`                        |
| `utils/wetext_string.cc`               | `runtime/utils/wetext_string.cc`                       |

## Files replaced (not copied)

- `utils/wetext_log.h` — upstream re-exports `glog/logging.h`. RapidSpeech does
  not depend on glog; the local replacement is a self-contained stderr+abort
  shim that implements the subset of glog macros actually used by the vendored
  sources (`LOG(level) << ...`, `CHECK`, `CHECK_EQ/NE/LT/LE/GT/GE`). No other
  changes to vendored sources.

## Excluded from the vendor

The upstream `runtime/` tree also ships a binary frontend (`bin/`), tests
(`test/`), a C API (`c_api/`), and Python training/data tools — none of which
are needed for the embedded text-normalization use case. Only the
tagger/verbalizer runtime is vendored here.

## Dependencies

The vendored code requires [OpenFST](https://github.com/openfst/openfst) at
runtime; this project pulls
[`csukuangfj/openfst v1.8.5-2026-06-15`](https://github.com/csukuangfj/openfst/releases/tag/v1.8.5-2026-06-15)
via FetchContent (see `cmake/openfst.cmake`).
