# FunASR-Nano 使用说明（离线 ASR，可选 LLM 二次解码）

FunASR-Nano 是轻量离线 ASR：SANM 编码器 + CTC 快速解码头，另带一个 Qwen3 LLM
解码器做二次改写（rescore）。可以只跑 CTC（快），也可以叠加 LLM（更准）。

---

## 1. 编译

```bash
cmake -B build && cmake --build build -j$(nproc)
# 产物：build/rs-asr-offline（离线）、build/rs-asr-vad-online（流式切段 + 二次解码）
```

## 2. 转换 GGUF

```bash
# 完整模型（CTC + LLM）
python scripts/convert_hf_to_gguf.py \
    --model-dir /path/to/FunASRNano \
    --output    funasr-nano-fp16.gguf --out-type f16

# 只要 CTC（更小，跳过 LLM 权重）
python scripts/convert_hf_to_gguf.py \
    --model-dir /path/to/FunASRNano \
    --output    funasr-nano-ctc.gguf --out-type f16 --without_llm
```

转换器会同时写入 CTC 词表（`ctc.tokenizer.ggml.tokens`）和 LLM 词表。

## 3. 离线使用

```bash
# CTC 快速解码（默认）
./build/rs-asr-offline -m funasr-nano-fp16.gguf -w audio.wav -t 4 --gpu true

# 加 VAD 切段 / CAMPPlus 说话人分离，同 SenseVoice
./build/rs-asr-offline -m funasr-nano-fp16.gguf -v silero_vad_v6.gguf \
    -s campplus.gguf -w audio.wav
```

## 4. 二次解码（CTC → LLM rescore）

流式切段 CLI `rs-asr-vad-online` 支持 2-pass：先 CTC 出快结果，静音端点后再用 LLM 改写：

```bash
./build/rs-asr-vad-online -m funasr-nano-fp16.gguf -v silero_vad_v6.gguf \
    -w audio.wav --two-pass --ctc-precheck
# 或麦克风：--mic
```

- `--two-pass`：CTC 快速 partial + LLM 终稿 rescore。
- `--ctc-precheck`：LLM 前先用 CTC 判静音，跳过空段，减少幻觉。

## 5. 量化

```bash
./build/rs-quantize funasr-nano-fp16.gguf funasr-nano-q4_k_m.gguf q4_k_m
```

## 6. 热词（RAG 上下文偏置）

热词让识别更倾向于你指定的专有名词/术语，尤其能纠正**中文同音字/谐音**错误 。

### 6.1 用法

```bash
# 内联列表（逗号分隔）
./build/rs-asr-offline -m funasr-nano-q4_k_m.gguf -w audio.wav \
    --hotwords "阿里巴巴,二氧化碳"

# 文件（每行一个词，# 开头为注释，空行忽略）
./build/rs-asr-offline -m funasr-nano-q4_k_m.gguf -w audio.wav \
    --hotword-file hotwords.txt

# 两者可同时用，自动去重合并
./build/rs-asr-offline -m funasr-nano-q4_k_m.gguf -w audio.wav \
    --hotwords "阿里巴巴" --hotword-file terms.txt
```

热词文件格式：

```
# 人名
阿里巴巴   # 行内注释也支持

# 术语
二氧化碳
```

C API：

```c
const char *words[] = {"阿里巴巴"};
rs_set_hotwords(ctx, words, 2);   // 传 NULL/0 清空
```

> 热词仅在 **LLM 解码开启**时生效（FunASR-Nano 默认开启）。

### 6.2 工作原理（三层）

1. **CTC 假设 → 拼音检索**：先跑一遍 CTC 得到初步假设，把每个热词转成**去声调拼音**，
   与假设做编辑距离检索，只召回读音匹配的热词（支持谐音，如 CTC「章叁」召回热词「张三」）。
2. **注入 LLM prompt**：把召回的热词按 Fun-ASR 官方模板（`热词列表：[...]`）喂给 LLM 做软引导。
3. **确定性拼音替换**：对最终文本按拼音**精确匹配**，把同音字段强制改写成热词原文——
   保证生僻字出现，不依赖 LLM 是否"听话"（这一层默认开启）。

只有前两层时，模型对生僻字的偏置较弱（可能被推向别的同音字）；第三层兜底确保结果稳定。

### 6.3 运行时开关（环境变量，无需重编译）

| 变量 | 作用 | 默认 |
|---|---|---|
| `RS_HOTWORD_FORCE=0` | 关闭确定性拼音替换（只保留 CTC 检索 + prompt 软注入） | 开 |
| `RS_HOTWORD_TAU=0.5` | 检索阈值，归一化编辑距离上限，越大越宽松 | 0.34 |
| `RS_HOTWORD_TOPK=20` | 最多注入的热词数 | 20 |
| `RS_HOTWORD_PROMPT_STYLE=strong` | 更强硬的注入措辞（要求生僻字照写） | 官方模板 |
| `RS_HOTWORD_DEBUG=1` | 打印 CTC 拼音、每个热词的距离/命中、最终注入 prompt、替换动作 | 关 |

### 6.4 依赖与注意事项

- **拼音数据**：谐音纠错依赖 `rapidspeech/data/kokoro_zh/`。从仓库根目录运行会自动找到；
  否则设 `RS_KOKORO_ZH_DATA_DIR=/path/to/kokoro_zh`，或放到
  `<可执行文件>/../share/rapidspeech/kokoro_zh`。缺失时退化为字符级匹配（谐音失效，会打 WARN）。
- **确定性替换会改写所有同音字段**：`--hotwords "xx"` 会把输出里**所有**读音为 `xx`
  的地方改成「xx」。若场景中同时存在同音的其它写法可能误伤，可用 `RS_HOTWORD_FORCE=0` 关闭。

