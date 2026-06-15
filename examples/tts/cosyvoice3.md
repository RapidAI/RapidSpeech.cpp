# CosyVoice3-0.5B TTS — Usage Guide

CosyVoice3 是一个两阶段 TTS：
**Qwen2-0.5B LLM**（text → 6561-codebook speech tokens）→ **Flow (CFM+DiT)**（speech tokens → 80-mel @ 24 kHz）→ **HiFT** (NSF-HiFiGAN, mel → 24 kHz 波形)。

声音克隆走 **speech_tokenizer_v3**（ref wav → prompt speech tokens）+ **CAM++**（ref wav → 192-d 说话人嵌入）。生产侧已支持把 voice 元组 bake 到 GGUF，运行时不再需要 tokenizer / CAMPPlus。

> **Status (2026-06-11)**：
> - **默认 voice 已 bake 进 unified GGUF**（`cv3.default_voice.*` 四元组），直接 `-t "..."` 即可合成，`--ref` 变成可选。
> - 想换音色：`--ref <wav> --ref-text "<逐字稿>" --speech-tokenizer ... --campplus ...`，或者一次跑通后 `--save-voice cache.gguf` 保存，下次直接 `--voice cache.gguf` 复用（跳过 tokenizer 和 CAMPPlus 的加载与前向）。
> - 全 18 种 ftype 量化已落地，`q4_k_m` 590 MB 是生产推荐；`q3_k` 535 MB 起开始有音质损伤，可配合 **imatrix** 校准缓解（见第 8 节）。

---

## 1. 准备 GGUF

最简单的情况只需 1 个文件（unified + 默认 voice）：

```bash
python scripts/convert_cosyvoice3_to_gguf.py \
  --input  /path/to/Fun-CosyVoice3-0.5B-2512 \
  --output models/cv3-ours.gguf
```

`--input` 可直接是 HuggingFace repo id（`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`），脚本通过 `huggingface_hub.snapshot_download` 拉取。

如果想自带 ref wav bake 一个不同的默认音色：

```bash
python scripts/convert_cosyvoice3_to_gguf.py \
  --input    /path/to/Fun-CosyVoice3-0.5B-2512 \
  --output   models/cv3-ours.gguf \
  --ref-wav  /path/to/my_voice.wav \
  --ref-text "对应这个音频的逐字稿"
```

只有在运行时换音色或调试时，才需要另外两个文件：

```bash
# speech_tokenizer_v3 (ref wav → prompt speech tokens)
python scripts/convert_speech_tokenizer_v3_to_gguf.py \
  --input  /path/to/Fun-CosyVoice3-0.5B-2512/speech_tokenizer_v3.onnx \
  --output models/cv3-speech-tokenizer.gguf

# CAM++ (ref wav → 192-d spk embedding)
python scripts/convert_campplus_to_gguf.py \
  --input  /path/to/Fun-CosyVoice3-0.5B-2512/campplus.onnx \
  --output models/cv3-campplus.gguf
```

转换完成后 unified GGUF 的关键 KV：

| KV | 值 |
|---|---|
| `general.architecture` | `cosyvoice3` |
| `cosyvoice3.llm.{n_layers, d_model, n_heads, n_kv_heads, head_dim, ff_dim}` | 24, 896, 14, 2, 64, 4864 |
| `cosyvoice3.llm.{speech_vocab_size, speech_token_codebook}` | 6761, 6561 |
| `cosyvoice3.flow.{depth, dim, n_heads, mel_dim}` | 22, 1024, 16, 80 |
| `cosyvoice3.hift.{upsample_rates, sample_rate}` | [8, 5, 3], 24000 |
| `cv3.default_voice.{prompt_text_ids, prompt_token, prompt_feat, embedding}` | 四元组（baked） |

T_audio = T_mel · 4 · 120 = T_mel · 480 → 24 kHz。

---

## 2. 编译

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rs-tts-offline -j$(sysctl -n hw.ncpu)
```

`rs-tts-offline` 通过 `general.architecture` 字段自动路由到 CosyVoice3 实现。

---

## 3. 运行

### 3.1 默认 voice（最简）

GGUF 自带 `cv3.default_voice.*`，直接合成：

```bash
./build/bin/rs-tts-offline \
  -m models/cv3-ours.gguf \
  -t "你好，世界。这是 CosyVoice3 的合成测试。" \
  --seed 42 \
  -o /tmp/cv3_out.wav
```

### 3.2 运行时换音色（`--ref`）

```bash
./build/bin/rs-tts-offline \
  -m models/cv3-ours.gguf \
  --speech-tokenizer models/cv3-speech-tokenizer.gguf \
  --campplus         models/cv3-campplus.gguf \
  --ref      assets/my_voice.wav \
  --ref-text "对应这个音频的逐字稿，必须完全一致" \
  -t "你好，世界。" \
  --seed 42 \
  -o /tmp/cv3_out.wav
```

### 3.3 一次 bake、永久复用（推荐生产）

第一次跑通 `--ref` 之后用 `--save-voice` 把 voice 元组保存为单独的 GGUF：

```bash
./build/bin/rs-tts-offline \
  -m models/cv3-ours.gguf \
  --speech-tokenizer models/cv3-speech-tokenizer.gguf \
  --campplus         models/cv3-campplus.gguf \
  --ref       assets/my_voice.wav \
  --ref-text  "..." \
  --save-voice models/my-voice.gguf \
  -t "warm-up" -o /tmp/warmup.wav
```

之后只需 `--voice`，**不再加载 speech_tokenizer + CAMPPlus**：

```bash
./build/bin/rs-tts-offline \
  -m models/cv3-ours.gguf \
  --voice models/my-voice.gguf \
  -t "..." -o out.wav
```

经过验证：相同 seed 下，bake-then-reuse 与原 `--ref` 路径产生 bit-identical 音频。

`--speech-tokenizer` / `--campplus` / `--voice` / `--save-voice` 在底层会写 4 个环境变量：
`RS_CV3_SPEECH_TOKENIZER_PATH` / `RS_CV3_CAMPPLUS_PATH` / `RS_CV3_VOICE_PATH` / `RS_CV3_SAVE_VOICE_PATH`，CI 里直接 export 即可。

### Ref WAV 要求

- 任意采样率（内部重采样到 16 kHz 给 tokenizer/CAM++、24 kHz 给 Flow prompt_feat）；推荐 16 k 或 24 k 单声道。
- 时长 **3–10 秒** 的干净语音最佳，避免背景噪声 / 多人重叠。
- `--ref-text` 必须是 ref wav 的**逐字稿**（不一致会导致 LM prefix 与 prompt_speech_tokens 错位，输出乱码或循环 token）。

---

## 4. 关键参数

| 参数 | 作用 | 默认 |
|---|---|---|
| `--seed <n>` | RAS 采样器随机种子（影响 prosody，复现实验请固定） | 42 |
| `--lang <name>` | 目标语言；**CV3 暂未使用**（系统 prefix 在 LM 中硬编码为 `"You are a helpful assistant."`），仅 OpenVoice2/OmniVoice 走这个 | English |
| `--instruct <text>` | **instruct2 模式**风格描述。非空（且不是 `"male"`/`"female"` 这些 OmniVoice 占位）时进入 instruct2：LM prefix 用 instruct 替换 ref 转写，LM 端不再喂 prompt_speech_token；Flow 端 `prompt_token` / `prompt_feat` / `spk_emb` 仍生效，**音色照常克隆**，**韵律由 instruct 引导**。 | male（OV gender，CV3 视为无 instruct） |
| `--n-steps <n>` | Flow CFM Euler 步数（LLM 是 AR 解码不受此影响） | 10 |
| `--threads <n>` | CPU 线程数 | 4 |
| `--gpu true\|false` | Metal/CUDA/Vulkan 后端开关 | true |
| `--voice <path>` | 加载 baked voice GGUF（跳过 tokenizer/CAMPPlus） | — |
| `--save-voice <path>` | 把当前解析出的 voice 元组写到这个 GGUF | — |
| `--dump-step0-logits <path>` | 调试：dump 首步 6761 维 logits 用于对齐 PyTorch | — |
| `--prompt-tokens-bin <path>` | 调试：用预先 dump 好的 int32 LE prompt-token 文件，绕过 speech_tokenizer | — |

> LLM 的 `max_speech_tokens` 上限是 1500（≈30 秒音频）。

---

## 5. 端到端流程示意

```
text + ref.wav
     │
     ├─► speech_tokenizer_v3.gguf ───► prompt_token        (≈25 tok @ 25 Hz)
     ├─► campplus.gguf           ────► spk_emb [192]
     └─► 80-mel @ 24kHz          ────► prompt_feat [T_pt, 80]
                                                  │
            ┌─── BPE ──── prompt_text_ids ────┐   │
            ▼                                 ▼   ▼
text ──► Qwen2-LLM ────► speech_tokens ──► Flow (CFM/DiT) ──► mel
                            (RAS sample)           │
                                                   ▼
                                            HiFT (NSF-HiFiGAN)
                                                   │
                                                   ▼
                                            24 kHz f32 PCM ──► wav
```

LM prefix 实际拼接：
`[sos=6561] + sys_prefix(lang/instruct) + [eop=151646] + prompt_text_ids + tts_text_ids + [task=6563] + prompt_speech_tokens`

`--voice` 模式：`prompt_text_ids` / `prompt_token` / `prompt_feat` / `spk_emb` 都直接从 baked GGUF 取，不跑 tokenizer/CAMPPlus 前向。

---

## 6. 量化与体积

`cv3-ours.gguf` (f16) = 2273 MB。全部 18 种 ftype 已实现：

| ftype | 体积 | 备注 |
|---|---|---|
| `q2_k` / `q2_k_m` | 494 MB | 最小，音质明显劣化，建议配 imatrix |
| `iq3_xxs` | 527 MB | |
| `q3_k` | 535 MB | 开始有可察觉音质损伤 |
| `q3_k_m` | 555 MB | |
| `iq3_s` | 583 MB | |
| `q4_0` / `q4_k` | 589 MB | 同 bpw 体积一致 |
| `q4_k_m` | 590 MB | **生产推荐**（LM/Flow K-bump 到位） |
| `q4_1` | 646 MB | |
| `q5_0` / `q5_k` | 703 MB | |
| `q5_k_m` | 709 MB | |
| `iq4_xs` | 714 MB | |
| `iq4_nl` | 730 MB | |
| `q5_1` | 761 MB | |
| `q6_k` | 948 MB | |
| `q8_0` | 1046 MB | 几乎无损 |

```bash
./build/bin/rs-quantize models/cv3-ours.gguf models/cv3-q4_k_m.gguf q4_k_m
```

> HiFT vocoder 的 conv 核 `ne[0] ∈ {3..16}` 远小于任何 K-quant block-size，运行时一律以 F16 落地；选择 K/IQ ftype 时影响仅在 HiFT 内极少的 linear 上体现，对体积无显著差异。

---

## 7. 已知问题与排查

| 现象 | 原因 | 修复 |
|---|---|---|
| 内容错乱、token 出现 `564,2410,218,6302,6309` 循环 | `--ref-text` 与 ref wav 不匹配，或 `--ref-text` 漏传 | 校对逐字稿；中英文标点保持与音频一致 |
| 启动报 `CosyVoice3: CAMPPlus::Embed failed` | `--campplus` 指向的 GGUF 不对、或缺少 CAM++ 张量 | 重新跑 `convert_campplus_to_gguf.py` |
| 启动报 `V3 tokenizer failed` | 使用 `--ref` 模式时 `--speech-tokenizer` 没传 / 路径错误 | 设置 `--speech-tokenizer` 或用 `--voice` 走 bake 流程 |
| 内容正确但音质闷糊、有金属声 | Flow CFG / HiFT 数值正常但 ref 太短 | 用 ≥3 秒、干净的 ref wav；24 kHz 单声道 |
| q4_k_m 以下音质明显下降 | 激活分布尾部权重被一刀切 | 用 **imatrix** 重新量化（见第 8 节） |

复现时固定 `--seed`：种子改变只影响 prosody，不应影响吐字内容。若同种子换语言/换 ref 才能复现，说明遇到的是 LM 行为，不是采样器随机性。

---

## 8. Imatrix 激活感知量化

`q4_k_m` 以下、特别是 `q3_k` / `q2_k_m`，没有 imatrix 时高频细节容易失真。`rs-imatrix` 工具收集 LM + Flow DiT 的逐通道激活²统计，给 K-quant / IQ-quant 用于按重要性分配比特。

### 8.1 收集（CV3 模式自动识别）

默认走 GGUF 内置 `cv3.default_voice.*`，不需要 tokenizer / CAMPPlus：

```bash
./build/bin/rs-imatrix \
  -m models/cv3-ours.gguf \
  -o models/cv3.imatrix.dat \
  --gpu \
  --n-seeds 8
```

内置 40 句 zh+en 语料 × 8 seed = 320 次 forward；想换语料用 `--text-list file.tsv`（`lang\ttext` 一行一条）。

需要换音色再收集时加 `--voice my-voice.gguf`。

### 8.2 用 imatrix 重新量化

```bash
./build/bin/rs-quantize \
  models/cv3-ours.gguf \
  models/cv3-q3_k-imat.gguf \
  q3_k \
  --imatrix models/cv3.imatrix.dat
```

### 8.3 CUDA 一键脚本

放 CUDA 机器跑更快，参考 `scripts/cv3_imatrix_cuda.sh`：

```bash
bash scripts/cv3_imatrix_cuda.sh models/cv3-ours.gguf models/cv3-imat
# 环境变量：
#   N_SEEDS=8 THREADS=8
#   QUANT_TYPES="q4_k_m q3_k q2_k_m"
#   VOICE=...  TEXT_LIST=...
#   SKIP_BUILD=1 SKIP_IMATRIX=1
```

脚本会 `-DGGML_CUDA=ON` 配置 `build-cuda`、构建 `rs-imatrix` + `rs-quantize`、收集 imatrix、然后按 `QUANT_TYPES` 列表挨个量化。

---

## 9. 调试技巧

- **首步 logits 对齐**：用 `--dump-step0-logits /tmp/step0.bin` 输出 6761 维 float32，可与 PyTorch 端 reference 做 max-abs-diff 比较，定位 LM 路径 bug。
- **跳过 speech_tokenizer**：用 `--prompt-tokens-bin prompt.bin`（int32 LE）直接喂预先 dump 好的 prompt token，可在调试 Flow/HiFT 时排除 tokenizer 误差。
- **去掉 GPU**：`--gpu false` 强制 CPU，多数数值精度问题用 CPU 复现更稳定。

---

## 10. 后续计划

- [ ] Metal 融合 HiFT kernel（参考 `dac_metal.mm`）。
- [ ] 流式 / chunked 合成。
- [ ] imatrix-aware ftype 的 MOS / RTF benchmark 表。
