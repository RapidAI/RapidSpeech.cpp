<div align="center">
<img src="assets/rapid-speech.png" alt="rapid-speech Logo"  />
</div>

简体中文 | [English](./README.md)
# RapidSpeech.cpp 🎙️

**RapidSpeech.cpp** 是一个基于 **ggml** 构建的高性能、边缘原生（Edge-native）语音智能框架。它旨在为 ASR（语音识别）和 TTS（语音合成）大模型提供纯 C++、零依赖的端侧推理解决方案。

------

## 🌟 核心差异化优势

在当前的开源生态中，我们已经拥有了像 **vLLM-omni** 这样优秀的云端推理框架和 **sherpa-onnx** 这样成熟的端侧工具。然而，**RapidSpeech.cpp** 在以下维度实现了代际突破：

### 1. 与 vLLM 的区别：边缘侧 vs 云端吞吐

- **部署环境：** vLLM 专为数据中心设计，依赖 Python 环境和 CUDA 强绑定，旨在通过 PageAttention 压榨 GPU 并发吞吐。
- **RapidSpeech.cpp** 专注于**边缘计算**。主打低延迟和轻量化，可以在嵌入式设备、手机、普通笔记本甚至没有任何 GPU 的 NPU 平台上运行，无需 Python 运行时。

### 2. 与 sherpa-onnx 的区别：更深度的底层控制

| 维度         | sherpa-onnx (ONNX Runtime)                       | **RapidSpeech.cpp (GGML)**                                   |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------ |
| **内存管理** | 依赖 ORT 内部分配，内存开销相对黑盒。            | **零运行时内存分配**。在图构建阶段锁定内存，彻底杜绝端侧 OOM。 |
| **量化支持** | 主要支持 INT8，对极低比特（INT4/INT5）支持受限。 | **极致量化生态**。支持 K-Quants 系列（Q4_K, Q5_K, Q6_K 等），在保持精度的前提下大幅降低带宽压力。 |
| **GPU 性能** | 依靠 ONNX EP 映射，存在算子转换开销。            | **原生 GPU 优化**。直接调用 `ggml-cuda` 或 `ggml-metal`，推理效率显著优于通用型的 `onnxruntime-gpu`。 |
| **部署形态** | 依赖动态库，常需外挂配置文件（.yaml/.txt）。     | **单二进制文件**。GGUF 格式封装所有配置与权重，部署即运行。  |


------

## 🏗️ 架构设计

RapidSpeech.cpp 并非单纯的推理工具，它是一套完整的语音业务框架：

- **核心引擎 (Core Engine):** 基于 `ggml` 的计算后端，支持从 INT4 到 FP32 的混合精度推理。
- **架构层 (Architecture Layer):** 插件式的模型构建器，计划支持Funasr-nano, cosyvoice、 qwen3-tts 等。
- **业务逻辑 (Business Logic):** 内置环形缓冲区、VAD（端点检测）、文本前端（音素化）以及多会话管理。

------

## 🚀 核心特性

- [ ] **极致量化：** 原生支持 4-bit、5-bit、6-bit 等多种量化方案，适配不同内存带宽的硬件。
- [ ] **零依赖：** 纯 C/C++ 实现，编译后仅为一个轻量级二进制文件。
- [ ] **GPU 加速：** 针对语音大模型特征，定制化优化了 CUDA 和 Metal 后端。
- [ ] **统一格式：** ASR 与 TTS 模型均采用统一扩展的 **GGUF** 格式。

------


## 🛠️ 快速开始（开发中）

Bash

```
git clone https://github.com/RapidAI/RapidSpeech.cpp
cd RapidSpeech.cpp
```

------

## 🤝 参与贡献

如果你对以下领域感兴趣，欢迎提交 PR 或参与讨论：

- 适配更多模型。
- 完善项目框架。