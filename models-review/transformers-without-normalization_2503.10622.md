# Transformers without Normalization

**Source**: [arXiv:2503.10622](https://arxiv.org/html/2503.10622)

**Authors:** Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background: Normalization Layers](#2-background-normalization-layers)
- [3 What Do Normalization Layers Do?](#3-what-do-normalization-layers-do)
- [4 Dynamic Tanh (DyT)](#4-dynamic-tanh-dyt)
- [5 Experiments](#5-experiments)
- [6 Analysis](#6-analysis)
- [7 Initialization of α](#7-initialization-of-alpha)
- [8 Related Work](#8-related-work)
- [9 Limitations](#9-limitations)
- [10 Conclusion](#10-conclusion)

## Abstract
**[Key point]**: Normalization layers, traditionally deemed essential for training deep networks, can be entirely removed from Transformers without performance degradation.
**Main contribution**: Introduction of Dynamic Tanh (DyT), an element-wise non-linearity that replaces normalization layers, enabling "normalization-free" Transformers.
**Key results:**
- DyT achieves performance matching or exceeding normalized Transformers across vision and language tasks.
- The method requires minimal hyperparameter tuning, primarily involving the scaling parameter α.
- Validated across diverse settings: supervised/self-supervised learning, recognition/generation, and varying model scales.
- Challenges the dogma that normalization is indispensable for stabilizing deep network training.
**Conclusion:**
- Normalization layers are not strictly necessary if appropriate non-linearities are used.
- DyT provides a simpler, more efficient alternative to LayerNorm and other normalization techniques.
- Offers new theoretical insights into the role of normalization in deep learning.

## 1 Introduction
**Context:**
- Normalization layers (e.g., BatchNorm, LayerNorm) are ubiquitous in modern deep learning architectures.
- They are widely believed to be critical for mitigating internal covariate shift and stabilizing gradient flow.
- Transformers, the backbone of modern AI, heavily rely on LayerNorm for stability and performance.
- Removing normalization is often thought to lead to training instability or poor convergence.
**Goal:**
- To demonstrate that Transformers can be trained effectively without any normalization layers.
- To identify a simple, effective replacement for normalization that maintains or improves performance.
- To challenge the conventional wisdom regarding the necessity of normalization in deep networks.
**Contributions:**
- Proposes Dynamic Tanh (DyT) as a drop-in replacement for normalization layers.
- Shows that DyT-based Transformers match or beat normalized counterparts on standard benchmarks.
- Provides extensive empirical validation across computer vision and natural language processing domains.
- Analyzes the theoretical and practical implications of removing normalization.
- Offers guidelines for initializing the key parameter α in DyT.

## 2 Background: Normalization Layers
**Context:**
- Brief review of common normalization techniques: BatchNorm, LayerNorm, InstanceNorm, GroupNorm.
- Explanation of how LayerNorm is specifically used in Transformers (pre-LN vs. post-LN architectures).
- Discussion of the theoretical motivations for normalization: stabilizing activations, smoothing loss landscapes.
- Acknowledgment of the computational overhead and memory footprint associated with normalization layers.
- Note on the complexity normalization adds to model implementation and inference.

## 3 What Do Normalization Layers Do?
**Context:**
- Investigation into the actual functional role of normalization layers in Transformers.
- Observation that LayerNorm often produces input-output mappings that resemble S-shaped curves.
- Analysis of the distribution of activations within normalized layers.
- Hypothesis that the primary benefit of normalization is not just scaling, but shaping the activation distribution.
- Comparison of the effect of normalization on the gradient dynamics during backpropagation.
- Identification of the "tanh-like" behavior as a key characteristic of effective normalization in this context.

## 4 Dynamic Tanh (DyT)
**Context:**
- Definition of Dynamic Tanh (DyT) as the core methodological contribution.
- Mathematical formulation: $DyT(x) = \tanh(\alpha x)$.
- Explanation of α as a learnable or fixed scaling parameter that controls the slope of the tanh function.
- Description of DyT as an element-wise operation, making it computationally lightweight.
- Comparison of DyT to standard tanh: DyT allows for adaptive scaling via α.
- Integration of DyT into Transformer blocks: replacing LayerNorm layers with DyT.
- Discussion of where DyT is placed (e.g., before attention, before MLP) in the architecture.
- Explanation of why DyT mimics the stabilizing effect of normalization without the computational cost.

## 5 Experiments
**Context:**
- Overview of experimental setup and datasets used for validation.
- Description of baseline models (standard Transformers with LayerNorm).
- Description of DyT-based models (Transformers with DyT replacing LayerNorm).
- **Computer Vision:**
  - Tasks: Image classification, object detection, semantic segmentation.
  - Datasets: ImageNet, COCO, ADE20K.
  - Models: ViT (Vision Transformers), Swin Transformer variants.
  - Results: DyT models achieve comparable or superior accuracy to normalized baselines.
- **Natural Language Processing:**
  - Tasks: Language modeling, text generation, sequence classification.
  - Datasets: WikiText, OpenWebText, GLUE benchmark.
  - Models: GPT-style decoder-only transformers, BERT-style encoder-only transformers.
  - Results: DyT models match perplexity and accuracy metrics of normalized models.
- **Self-Supervised Learning:**
  - Tasks: Contrastive learning (e.g., MAE, DINO).
  - Results: DyT enables stable training in self-supervised regimes without normalization.
- **Training Stability:**
  - Analysis of loss curves and gradient norms during training.
  - Demonstration that DyT prevents exploding/vanishing gradients similarly to LayerNorm.

## 6 Analysis
**Context:**
- Deep dive into the properties of DyT and the parameter α.
- **6.1 Ablations of tanh and α:**
  - Comparison of DyT with other non-linearities (ReLU, Sigmoid, Swish).
  - Justification for choosing tanh over other S-shaped functions.
  - Ablation study on the impact of α on performance.
  - Finding that tanh's bounded nature is crucial for stability.
- **6.2 Values of α:**
  - Exploration of different fixed values for α.
  - Observation that α significantly affects the effective learning rate and convergence speed.
  - Identification of optimal ranges for α across different model depths and widths.
  - Discussion of the trade-off between α magnitude and gradient magnitude.
- **6.3 Comparison with Other Methods:**
  - Comparison of DyT against other normalization-free approaches (e.g., RMSNorm, Weight Standardization).
  - Demonstration that DyT is simpler and often more effective than these alternatives.
  - Analysis of computational efficiency: DyT is faster and uses less memory than LayerNorm.

## 7 Initialization of α
**Context:**
- Critical discussion on how to set the initial value of α for successful training.
- **7.1 Initialization of α for Non-LLM Models:**
  - Guidelines for initializing α in Vision Transformers and other non-language models.
  - Recommendation of specific initial values based on layer width and depth.
  - Strategy for setting α to ensure initial gradients are well-scaled.
- **7.2 Initialization of α for LLMs:**
  - Guidelines for initializing α in large language models.
  - Discussion of the challenges in scaling α for very deep networks.
  - Proposed initialization schemes that maintain stability in LLM pre-training.
  - Note on the potential for α to be learned dynamically during training vs. fixed.

## 8 Related Work
**Context:**
- Review of literature on normalization layers and their alternatives.
- Discussion of previous attempts to remove or replace normalization (e.g., RMSNorm, NoRM).
- Comparison with works on activation functions and their role in deep networks.
- Reference to theoretical analyses of normalization in deep learning.
- Positioning DyT within the broader context of efficient and simplified neural network design.

## 9 Limitations
**Context:**
- Acknowledgment of scenarios where DyT might not be optimal.
- Potential sensitivity of α to specific dataset distributions or task types.
- Limitations in extremely deep networks where gradient flow might still benefit from normalization.
- Lack of extensive testing on multimodal or specialized architectures beyond standard Transformers.
- Note on the need for careful hyperparameter selection for α in some edge cases.

## 10 Conclusion
**Context:**
- Summary of the main findings: Normalization is not strictly necessary for Transformers.
- Reiteration of DyT's effectiveness as a simple, efficient replacement.
- Implications for future neural network design: simplification and efficiency gains.
- Call for re-evaluation of the necessity of normalization in other architectures.
- Final statement on the potential of DyT to become a standard component in efficient AI models.