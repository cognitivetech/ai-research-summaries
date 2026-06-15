# BitNet b1.58 2B4T Technical Report

**Source**: [arXiv:2504.12285](https://arxiv.org/html/2504.12285)

**Authors:** Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, Furu Wei

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Architecture](#2-architecture)
- [3 Training](#3-training)
- [4 Evaluation](#4-evaluation)
- [5 Inference Implementation](#5-inference-implementation)
- [6 Conclusion](#6-conclusion)
- [7 Future Directions](#7-future-directions)

## Abstract
**[Key point]**:
- Introduction of BitNet b1.58 2B4T, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale.
- Emphasis on native quantization during training rather than post-training quantization.

**Main contribution**:
- Development of a 2B parameter model trained on 4 trillion tokens using 1-bit weights.
- Demonstration that native 1-bit models can match the performance of full-precision counterparts.
- Release of open-source weights and inference implementations for both GPU and CPU.

**Key results:**
- Performance parity with leading open-weight, full-precision LLMs of similar size.
- Significant reduction in memory footprint due to 1-bit weight storage.
- Lower energy consumption during inference compared to full-precision models.
- Reduced decoding latency, particularly on CPU architectures.

**Conclusion:**
- Native 1-bit quantization is a viable path for efficient LLM deployment.
- The model serves as a benchmark for future research in ultra-low precision LLMs.

## 1 Introduction
**Context:**
- Large Language Models (LLMs) have achieved remarkable success but require substantial computational resources.
- Quantization is a common technique to reduce model size and inference cost.
- Most existing quantized models use post-training quantization (PTQ) or quantization-aware training (QAT) with higher bit-widths (e.g., 4-bit, 8-bit).
- Native 1-bit LLMs (using ternary weights: -1, 0, +1) have shown promise in reducing memory bandwidth bottlenecks.
- Previous 1-bit models were either closed-source or lacked comprehensive evaluation at the 2B scale.

**Goal:**
- To present BitNet b1.58 2B4T, a native 1-bit LLM trained from scratch.
- To validate its performance against full-precision and other quantized baselines.
- To provide open-source tools for efficient inference on diverse hardware.

**Contributions:**
- First open-source native 1-bit LLM at the 2B parameter scale.
- Training on a massive corpus of 4 trillion tokens to mitigate information loss from quantization.
- Comprehensive evaluation across language understanding, math, coding, and conversation.
- Detailed analysis of inference efficiency on GPU and CPU.
- Release of weights and inference code to the community.

## 2 Architecture
**Model Structure:**
- Based on the Transformer architecture.
- Uses ternary weights (-1, 0, +1) for all linear layers.
- Maintains full-precision activations to preserve information flow.
- Embedding layers are also quantized to 1-bit.

**BitNet b1.58 Configuration:**
- "b1.58" refers to the average bit-width per weight, accounting for the distribution of ternary values.
- The model retains the standard Transformer block structure: Multi-Head Attention (MHA) and Feed-Forward Network (FFN).
- Both MHA and FFN weights are ternary.
- Layer normalization is preserved in full precision.

**Advantages of Ternary Weights:**
- Replaces multiplications with additions and subtractions.
- Eliminates the need for high-precision weight storage.
- Reduces memory bandwidth requirements significantly.
- Enables higher throughput on hardware optimized for integer operations.

**Comparison with Full Precision:**
- Full-precision models use 16-bit or 32-bit floating-point weights.
- BitNet b1.58 uses 1-bit weights, reducing weight size by ~16x compared to FP16.
- Activation precision remains FP16 to prevent accuracy degradation.

## 3 Training
**3.1 Pre-training**
- **Corpus:**
  - Trained on 4 trillion tokens.
  - High-quality text data curated to maximize information density.
  - Includes web pages, books, articles, and code.
- **Optimization:**
  - Uses AdamW optimizer.
  - Learning rate schedule with warmup and decay.
  - Gradient clipping to stabilize training.
- **Quantization Strategy:**
  - Native ternary quantization applied during forward and backward passes.
  - Straight-Through Estimator (STE) used for gradient approximation.
  - No post-training quantization steps; quantization is integral to training.

**3.2 Supervised Fine-tuning (SFT)**
- **Dataset:**
  - Instruction-following datasets.
  - High-quality human-written instructions and responses.
  - Diverse domains: STEM, humanities, coding, etc.
- **Process:**
  - Fine-tunes the pre-trained 1-bit model on instruction data.
  - Maintains ternary weights throughout SFT.
  - Ensures the model retains its ability to follow complex instructions.

**3.3 Direct Preference Optimization (DPO)**
- **Objective:**
  - Aligns the model with human preferences.
  - Improves helpfulness and safety.
- **Dataset:**
  - Preference pairs (chosen vs. rejected responses).
  - Curated from diverse sources to avoid bias.
- **Process:**
  - Applies DPO loss to the SFT model.
  - Optimizes for preference without requiring a separate reward model.
  - Preserves the 1-bit weight constraint.

## 4 Evaluation
**4.1 Main Results**
- **Benchmarks:**
  - Language Understanding: MMLU, HellaSwag, ARC.
  - Mathematical Reasoning: GSM8K, MATH.
  - Coding Proficiency: HumanEval, MBPP.
  - Conversational Ability: AlpacaEval, Chatbot Arena.
- **Performance:**
  - Matches or exceeds full-precision LLMs of similar size (e.g., Llama-2-7B, Qwen-2-7B equivalents in relative scale).
  - Outperforms previous 1-bit models in accuracy.
  - Demonstrates robustness across diverse tasks.

**4.2 Comparison with Post-training Quantized Models**
- **Baselines:**
  - PTQ models (e.g., GPTQ, AWQ) applied to full-precision models.
  - QAT models with 4-bit and 8-bit precision.
- **Findings:**
  - Native 1-bit models achieve better accuracy than PTQ 4-bit models.
  - Native 1-bit models have lower memory footprint than PTQ models.
  - PTQ models suffer from accuracy degradation due to quantization error accumulation.
  - Native training mitigates this by adapting weights to the quantization noise.

**4.3 Comparison with Open-weight 1-bit Models**
- **Baselines:**
  - Previous open-source 1-bit models (e.g., BitNet b1.58 at smaller scales).
  - Other ternary LLMs.
- **Findings:**
  - BitNet b1.58 2B4T scales effectively to 2B parameters.
  - Larger scale compensates for information loss from 1-bit quantization.
  - Superior performance compared to smaller 1-bit models.

## 5 Inference Implementation
**5.1 GPU Inference**
- **Optimization:**
  - Custom CUDA kernels for ternary weight operations.
  - Exploits sparsity in ternary weights (0 values).
  - Reduces memory bandwidth usage.
- **Performance:**
  - Faster decoding latency compared to FP16 models.
  - Lower memory usage allows for larger batch sizes.
  - Compatible with existing GPU infrastructure.

**5.2 CPU Inference**
- **Optimization:**
  - Vectorized instructions (AVX2/AVX-512) for ternary operations.
  - Bit-packing of weights for efficient storage.
  - Minimal dependency on specialized hardware.
- **Performance:**
  - Significant speedup on CPU compared to FP16 models.
  - Enables deployment on edge devices and servers without GPUs.
  - Energy-efficient inference for mobile and IoT applications.

**Open-Source Tools:**
- Release of inference code for both GPU and CPU.
- Integration with Hugging Face Transformers.
- Documentation for custom hardware deployment.

## 6 Conclusion
**Summary:**
- BitNet b1.58 2B4T demonstrates the viability of native 1-bit LLMs.
- Achieves performance parity with full-precision models at a fraction of the cost.
- Provides a scalable solution for efficient LLM deployment.

**Implications:**
- Reduces barriers to entry for LLM research and application.
- Promotes sustainable AI by lowering energy consumption.
- Encourages further exploration of ultra-low precision models.

## 7 Future Directions
**Research Areas:**
- Scaling to larger parameter counts (e.g., 7B, 13B).
- Exploring mixed-precision quantization strategies.
- Optimizing for specific hardware architectures (e.g., TPUs, NPUs).
- Investigating the impact of 1-bit quantization on model robustness and safety.

**Applications:**
- Deployment on resource-constrained devices.
- Real-time inference for interactive applications.
- Democratizing access to powerful LLMs globally.