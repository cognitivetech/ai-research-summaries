# Energy-Based Transformers are Scalable Learners and Thinkers

**Source**: [arXiv:2507.02092](https://arxiv.org/html/2507.02092)

**Authors:** Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Energy-Based Transformers (EBT) Intuition](#2-energy-based-transformers-ebt-intuition)
- [3 Energy-Based Transformers (EBT) Approach](#3-energy-based-transformers-ebt-approach)
- [4 Experimentation and Results](#4-experimentation-and-results)
- [5 Discussion](#5-discussion)
- [6 Related Work](#6-related-work)
- [7 Limitations and Conclusion](#7-limitations-and-conclusion)
- [Appendix A Future Works and Broader Impact](#appendix-a-future-works-and-broader-impact)
- [Appendix B Additional Experimentation](#appendix-b-additional-experimentation)
- [Appendix C Additional EBT Details](#appendix-c-additional-ebt-details)

## Abstract
**[Key point]**: The paper introduces Energy-Based Transformers (EBTs), a novel architecture that enables models to "think" via inference-time optimization without additional supervised training.
**Main contribution**: A new class of Energy-Based Models (EBMs) that learns to verify input-prediction compatibility, allowing for gradient-descent-based prediction during inference.
**Key results:**
- EBTs scale faster than Transformer++ during training, achieving up to 35% higher scaling rates across data, batch size, parameters, FLOPs, and depth.
- EBTs improve performance with System 2 Thinking by 29% more than Transformer++ on language tasks.
- EBTs outperform Diffusion Transformers on image denoising using fewer forward passes.
- EBTs generalize better than existing approaches, achieving superior downstream results even with worse pretraining performance.
**Conclusion:**
- EBTs represent a promising new paradigm for scaling both learning and thinking capabilities in AI models.
- It is possible to generalize System 2 Thinking approaches to be modality-agnostic and unsupervised.

## 1 Introduction
**Context:**
- Inference-time computation techniques, analogous to human System 2 Thinking, are gaining popularity for enhancing model performance.
- Existing approaches suffer from significant limitations:
    - Modality-specificity (e.g., working only for text).
    - Problem-specificity (e.g., limited to verifiable domains like math or coding).
    - Requirement for additional supervision or training (e.g., verifiers or verifiable rewards) on top of unsupervised pretraining.
**Goal:**
- To determine if System 2 Thinking approaches can be generalized to develop models that learn to think solely from unsupervised learning.
- To create a unified framework that handles both discrete (text) and continuous (visual) modalities without extra supervision.
**Contributions:**
- Introduction of Energy-Based Transformers (EBTs), a new class of Energy-Based Models.
- Demonstration that EBTs can explicitly verify compatibility between inputs and candidate predictions.
- Reframing prediction problems as optimization tasks via energy minimization.
- Empirical validation of superior scaling laws and inference-time performance across multiple modalities.

## 2 Energy-Based Transformers (EBT) Intuition
### 2.1 Learning to Verify
- The core intuition is to learn a verifier that assesses the compatibility of an input with a candidate prediction.
- This verifier assigns an "energy" value to every input-prediction pair.
- Lower energy indicates higher compatibility or likelihood.
- The model learns this energy landscape through unsupervised pretraining.
- Unlike traditional models that predict tokens directly, EBTs evaluate the quality of potential outputs.

### 2.2 Learning to Understand
- EBTs aim to understand the underlying structure of the data distribution.
- By minimizing energy, the model navigates towards high-probability regions in the data space.
- This process mimics human deliberation, where one considers multiple options before settling on the best one.
- The "thinking" process is essentially an optimization trajectory towards a low-energy state.
- This approach allows for dynamic computation during inference, adjusting the depth of thought based on the complexity of the input.

## 3 Energy-Based Transformers (EBT) Approach
### 3.1 Energy-Based Models (EBM) Background
- EBMs define a scalar energy function $E(x, y)$ for inputs $x$ and outputs $y$.
- The probability distribution is defined via the Boltzmann distribution: $P(y|x) \propto \exp(-E(x, y))$.
- Traditional EBMs face challenges in training due to the intractable partition function.
- EBTs leverage recent advances in scalable EBM training to overcome these limitations.
- The energy function is parameterized by a neural network, specifically a Transformer architecture.

### 3.2 Scalable EBM Learning
- Training involves minimizing the energy of correct data pairs while maximizing the energy of incorrect pairs.
- Contrastive divergence or similar sampling-based methods are used to approximate the gradient.
- The paper introduces scalable techniques to make this training feasible for large models.
- Loss functions are designed to encourage the model to assign low energy to valid predictions.
- The training process is entirely unsupervised, relying only on the structure of the input data.

### 3.3 Scalable EBM Thinking
- During inference, predictions are generated by minimizing the energy function with respect to the output.
- Gradient descent is applied to the output variables (or their embeddings) to find the minimum energy state.
- This process is iterative, allowing the model to refine its predictions over multiple steps.
- The number of inference steps can be adjusted, providing a trade-off between computation and accuracy.
- This "thinking" phase is distinct from the forward pass of traditional autoregressive models.

### 3.4 Energy-Based Transformers (EBTs) Architecture
- EBTs combine the representational power of Transformers with the optimization capabilities of EBMs.
- The architecture uses standard Transformer blocks to compute the energy score.
- Input and candidate predictions are concatenated or processed jointly to compute the energy.
- The model is trained end-to-end to produce a smooth energy landscape.
- The architecture supports both discrete and continuous outputs by adjusting the optimization domain.

## 4 Experimentation and Results
### 4.1 Autoregressive Language Modeling Experiments
- Evaluated on standard language modeling benchmarks.
- Compared against Transformer++ and other baseline models.
- EBTs showed faster scaling with respect to data size and model parameters.
- Inference-time optimization provided significant performance boosts.
- The 29% improvement in System 2 Thinking tasks highlights the benefit of deliberation.
- Results were consistent across different model sizes and dataset scales.

### 4.2 Autoregressive Video Experiments
- Extended the EBT framework to continuous video data.
- Demonstrated the modality-agnostic nature of the approach.
- EBTs effectively captured temporal dependencies in video sequences.
- Scaling laws held for continuous modalities as well.
- The model could generate coherent video frames by minimizing energy over time.

### 4.3 Bidirectional Image Experiments
- Tested on image denoising tasks using Diffusion Transformers as a baseline.
- EBTs outperformed Diffusion Transformers in terms of quality metrics.
- Crucially, EBTs achieved this with fewer forward passes.
- This efficiency gain is attributed to the direct optimization of the energy landscape.
- The results suggest EBTs are more sample-efficient for generative tasks.

## 5 Discussion
- EBTs offer a unified framework for both learning and thinking.
- The ability to scale faster than Transformer++ suggests fundamental advantages in representation learning.
- The generalization capabilities are superior, as evidenced by downstream task performance.
- The approach avoids the need for task-specific verifiers or reward models.
- EBTs bridge the gap between discrete and continuous generation tasks.
- The "thinking" process is interpretable as an optimization trajectory.
- Potential applications include complex reasoning, creative generation, and robust perception.

## 6 Related Work
### 6.1 Traditional Transformers
- Traditional Transformers rely on autoregressive or masked prediction.
- They lack explicit mechanisms for inference-time verification.
- EBTs extend this by adding an energy-based evaluation layer.

### 6.2 RNNs
- Recurrent Neural Networks process data sequentially.
- EBTs use parallelizable Transformer architectures for better scalability.
- However, EBTs incorporate iterative refinement similar to recurrent thinking.

### 6.3 Dynamic Computation with LLMs
- Previous work on dynamic computation in LLMs often requires specialized training.
- EBTs achieve dynamic computation through unsupervised energy minimization.
- This reduces the dependency on additional supervision signals.

### 6.4 Dynamic Computation with Diffusion
- Diffusion models use iterative denoising, which is a form of dynamic computation.
- EBTs differ by optimizing a learned energy function rather than a fixed noise schedule.
- This allows for more flexible and task-adaptive inference.

### 6.5 Energy-Based Models (EBMs)
- EBMs have a long history in statistical physics and machine learning.
- Recent work has focused on making EBMs trainable at scale.
- EBTs build on these advances by integrating them with Transformer architectures.

## 7 Limitations and Conclusion
**Limitations:**
- Inference-time computation increases latency compared to single-pass models.
- The optimization process may get stuck in local minima.
- Training stability can be challenging due to the nature of EBMs.
- Computational cost during inference is higher per step, though fewer steps may be needed.

**Conclusion:**
- EBTs successfully demonstrate that models can learn to think from unsupervised data.
- They offer superior scaling properties and generalization capabilities.
- The framework is applicable to both discrete and continuous modalities.
- EBTs are a promising direction for future AI research in scalable and thoughtful models.

## Appendix A Future Works and Broader Impact
### A.1 Reversal Curse
- Investigating how EBTs handle the reversal curse (where models fail to reverse learned sequences).
- Energy-based formulation may offer insights into this phenomenon.

### A.2 Improved Stability
- Developing techniques to stabilize the training of large-scale EBTs.
- Exploring normalization and initialization strategies specific to energy landscapes.

### A.3 World Models
- Potential application of EBTs in learning world models.
- The energy function could represent the consistency of a predicted world state.

### A.4 EBTs as Complementary Models
- Using EBTs alongside traditional models for verification.
- Hybrid architectures could leverage the strengths of both paradigms.

### A.5 Recurrent Energy-Based Models
- Combining recurrent structures with energy-based objectives.
- This could lead to more efficient processing of sequential data.

### A.6 Improved Thinking Algorithms
- Exploring advanced optimization algorithms for the inference phase.
- Methods like Langevin dynamics or variational inference could be integrated.

### A.7 Multimodal Energy-Based Models
- Extending EBTs to handle complex multimodal inputs (e.g., text, image, audio).
- Unified energy functions could capture cross-modal relationships.

### A.8 Thinking Scalability
- Analyzing the scaling laws of the "thinking" process itself.
- How does the number of inference steps scale with problem complexity?

### A.9 Learning Multimodal Distributions
- Using EBTs to model complex, high-dimensional multimodal distributions.
- This could improve generative capabilities across diverse data types.

### A.10 Understanding Predictions
- Using the energy landscape to interpret model predictions.
- Low-energy regions correspond to high-confidence predictions.

## Appendix B Additional Experimentation
### B.1 Additional Natural Language Processing Experiments
- Detailed results on specific NLP benchmarks (e.g., GSM8K, MMLU).
- Comparison with other reasoning-enhanced models.
- Analysis of error cases and improvements.

### B.2 EBT Failure Cases
- Analysis of scenarios where EBTs fail to converge or produce poor results.
- Identification of patterns in difficult inputs.
- Suggestions for mitigating these failures.

## Appendix C Additional EBT Details
### C.1 Formalizing Thinking
- Mathematical formalization of the "thinking" process as optimization.
- Definitions of convergence criteria and energy barriers.

### C.2 Energy-Based Transformer (EBT) Thinking Types
- Classification of different types of thinking (e.g., fast vs. slow).
- How the architecture supports different inference strategies.