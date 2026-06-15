# Unextractable Protocol Models: Collaborative Training and Inference without Weight Materialization

**Source**: [arXiv:2605.23464](https://arxiv.org/html/2605.23464)

**Authors:** Alexander Long, Chamin Hewa Koneputugodage, Thalaiyasingam Ajanthan, Yan Zuo, Gil Avraham, Violetta Shevchenko, Hadi Mohaghegh Dolatabadi, Sameera Ramasinghe

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background and Motivation](#2-background-and-motivation)
- [3 Unextractable Protocol Models (UPM)](#3-unextractable-protocol-models-upm)
- [4 Training Framework](#4-training-framework)
- [5 Inference and Latency Analysis](#5-inference-and-latency-analysis)
- [6 Security and Attack Analysis](#6-security-and-attack-analysis)
- [7 Experimental Results](#7-experimental-results)
- [8 Discussion](#8-discussion)
- [9 Conclusion](#9-conclusion)

## Abstract
**[Key point]**: The paper introduces a decentralized framework for training and serving large neural networks where no single participant ever possesses the complete set of model weights.
**Main contribution**: The proposal of Unextractable Protocol Models (UPMs), a system that uses time-varying, random, invertible transforms at participant boundaries to ensure model shards are incompatible across different time steps.
**Key results:**
- On Qwen-2.5-0.5B and Llama-3.2-1B, applying 10,000 transforms results in negligible performance degradation (FP32 perplexity change < 0.01).
- Jensen-Shannon drift remains extremely low (< 4 × 10^-5), indicating functional equivalence.
- Inference overhead is minimal: 3% latency increase, 0.1% bandwidth increase, and 10% GPU memory overhead when applying transforms every 30 seconds.
- Training overhead is even lower: 1.6% time increase and < 1% memory increase.
- Gradient-based fine-tuning of stitched partitions requires ≥ 60% of the tokens needed to train a model from scratch, effectively neutralizing extraction attacks.
**Conclusion:**
- UPMs enable practical programmatic incentive mechanisms in community-driven decentralized training by making weight extraction computationally infeasible.
- The approach balances privacy, security, and performance, allowing collaborative model development without centralization risks.

## 1 Introduction
**Context:**
- Large language models (LLMs) are typically trained and served by centralized entities, creating single points of failure and control.
- Decentralized training has emerged as a solution to distribute computational load and data privacy concerns among multiple participants.
- However, existing decentralized approaches often require participants to hold or reconstruct full model weights at some point, leading to potential weight extraction and intellectual property theft.
- Current methods for protecting weights in decentralized settings are either computationally prohibitive or compromise model utility.

**Goal:**
- To design a collaborative training and inference framework where the full weight set is never materialized for any single participant.
- To ensure that while the network function is preserved globally, the local shards held by participants are incompatible at different time steps.
- To demonstrate that this "unextractable" property can be achieved with negligible overhead on standard LLM architectures.

**Contributions:**
- Introduction of Unextractable Protocol Models (UPMs), a novel protocol for decentralized model management.
- Development of a mechanism using time-varying, random, invertible transforms injected at participant boundaries.
- Comprehensive evaluation on Qwen-2.5-0.5B and Llama-3.2-1B models demonstrating functional equivalence and low overhead.
- Security analysis proving that direct extraction and gradient-based fine-tuning attacks are impractical or inefficient.
- Demonstration of how UPMs facilitate programmatic incentives in decentralized communities.

## 2 Background and Motivation
**Context:**
- Decentralized machine learning often relies on sharding, where model layers or parameters are distributed across nodes.
- Traditional sharding requires synchronization of weights for training updates, which can leak information about the global model structure.
- Homomorphic encryption and secure multi-party computation (SMPC) offer privacy but introduce significant computational and communication overhead.
- The concept of "weight materialization" refers to the state where a participant can reconstruct or access the full dense weight matrix.

**Motivation:**
- The need for privacy-preserving collaborative training in open communities where participants cannot fully trust each other.
- The desire to prevent "model stealing" attacks where malicious participants reconstruct the full model from shards.
- The limitation of current defenses that rely on static sharding, which allows attackers to stitch shards together over time if they can observe the same shard across multiple steps.

**Key Challenges:**
- Maintaining model accuracy while constantly changing the representation of weights.
- Ensuring invertibility of transforms to allow for correct gradient computation during training.
- Minimizing the latency and memory overhead introduced by dynamic transformations.

## 3 Unextractable Protocol Models (UPM)
**Context:**
- UPMs operate on the principle that the global function is invariant under specific local transformations.
- The system divides the model into shards, each held by a different participant.
- Instead of static weights, each shard is associated with a dynamic transformation key.

**Core Mechanism:**
- **Sharded Model Setup:** The neural network is partitioned into subsets (shards) distributed among participants.
- **Time-Varying Transforms:** At regular intervals, random, invertible linear transforms are applied to the boundaries between shards.
- **Invertibility:** The transforms must be invertible to allow gradients to flow correctly during backpropagation.
- **Incoherence:** The transforms render the shards incompatible across different time steps, preventing the assembly of a coherent global model from snapshots taken at different times.

**Properties:**
- **Functional Equivalence:** The overall input-output behavior of the network remains unchanged despite the internal transformations.
- **Unextractability:** No participant can reconstruct the original weights because they only see transformed versions that change over time.
- **Collaborative Compatibility:** Participants can still collaborate on training because the transforms are known and applied consistently within each time step.

## 4 Training Framework
**Context:**
- Training in UPMs requires careful handling of gradients to account for the dynamic transforms.
- The forward pass applies the transforms to maintain functional equivalence.
- The backward pass must apply the inverse transforms to ensure correct gradient updates.

**Training Process:**
- **Initialization:** Model shards are initialized and distributed among participants.
- **Transform Injection:** At each training step or epoch, new random invertible transforms are generated for each boundary.
- **Forward Pass:** Inputs pass through the shards, with transforms applied at boundaries. The output is mathematically equivalent to the untransformed model.
- **Loss Computation:** Loss is computed on the final output as usual.
- **Backward Pass:** Gradients are computed from the loss and propagated backward. Inverse transforms are applied to gradients to align them with the original parameter space for updates.
- **Weight Updates:** Parameters are updated using standard optimizers (e.g., Adam) on the local shards.

**Overhead Management:**
- **Memory:** Transforms are stored temporarily and discarded after the backward pass, minimizing persistent memory overhead.
- **Computation:** Matrix multiplications for transforms are optimized to reduce computational cost.
- **Synchronization:** Participants synchronize transform keys at the start of each step to ensure consistency.

## 5 Inference and Latency Analysis
**Context:**
- Inference in UPMs requires the application of the current set of transforms to generate outputs.
- The latency and resource costs of these transforms are critical for practical deployment.

**Inference Process:**
- **State Maintenance:** Each participant maintains the current set of invertible transforms for their shard boundaries.
- **Transform Application:** During inference, inputs are transformed at boundaries before passing to the next shard.
- **Output Generation:** The final output is produced after the last shard, with no additional transformation needed if the last shard is the output layer.

**Latency Metrics:**
- **Transform Frequency:** Transforms are applied every 30 seconds in the reported experiments.
- **Latency Increase:** Applying transforms every 30s adds approximately 3% latency to inference.
- **Bandwidth:** The communication overhead for sharing transform keys is minimal, adding only 0.1% bandwidth usage.
- **GPU Memory:** The overhead for storing and computing transforms adds 10% GPU memory usage.

**Precision Considerations:**
- **FP32:** Negligible impact on perplexity and drift.
- **Lower Precision:** The paper discusses methods to control growth of numerical errors in lower precision datatypes (e.g., FP16, BF16) to maintain stability.
- **Control Mechanisms:** Scaling factors and normalization techniques are used to prevent error accumulation in low-precision settings.

## 6 Security and Attack Analysis
**Context:**
- The primary security goal is to prevent the reconstruction of the original model weights.
- Various attack vectors are analyzed to assess the robustness of UPMs.

**Attack Vectors:**
- **Direct Attack:** Attempting to reconstruct weights by observing shards over time.
  - **Feasibility:** Impractical because shards are incompatible across time steps due to time-varying transforms.
  - **Defense:** The random nature of transforms makes it impossible to align shards from different times.
- **Gradient-Based Fine-Tuning:** Attempting to fine-tune a stitched model using gradients from the UPM.
  - **Feasibility:** Requires ≥ 60% of the tokens needed to train a model from scratch.
  - **Implication:** This is computationally prohibitive and inefficient compared to training a new model.
- **Stitching Attack:** Attempting to combine shards from different participants at the same time step.
  - **Feasibility:** Difficult because participants do not share raw weights, only transformed activations.
  - **Defense:** The transforms obscure the original weight values, making stitching useless for extracting the original model.

**Security Guarantees:**
- **Unmaterializable Weights:** Full weight sets are never available to any single participant.
- **Cross-Time Incoherence:** Assemblies of shards from different times are incoherent and non-functional.
- **Computational Infeasibility:** Attacks require resources that exceed practical limits.

## 7 Experimental Results
**Context:**
- Experiments were conducted on Qwen-2.5-0.5B and Llama-3.2-1B models.
- The goal was to verify functional equivalence and measure overhead.

**Model Performance:**
- **Perplexity:** FP32 perplexity remained unchanged (ΔPPL < 0.01) after applying 10,000 transforms.
- **Distributional Drift:** Jensen-Shannon drift between original and transformed outputs was < 4 × 10^-5.
- **Accuracy:** Task-specific accuracy metrics showed no significant degradation.

**Overhead Measurements:**
- **Training Time:** Overhead fell to 1.6% time increase.
- **Training Memory:** Overhead was < 1% memory increase.
- **Inference Latency:** 3% increase with transforms every 30s.
- **Inference Bandwidth:** 0.1% increase.
- **Inference Memory:** 10% GPU memory overhead.

**Scalability:**
- **Number of Transforms:** Tested with up to 10,000 transforms without performance loss.
- **Model Size:** Results were consistent across different model sizes (0.5B and 1B parameters).
- **Participant Count:** The framework scales with the number of participants as long as the sharding strategy is maintained.

## 8 Discussion
**Context:**
- The results demonstrate that UPMs are a viable solution for decentralized training.
- The trade-offs between security, performance, and complexity are analyzed.

**Advantages:**
- **Privacy:** Prevents weight extraction and protects intellectual property.
- **Decentralization:** Enables true collaborative training without central authority.
- **Efficiency:** Low overhead makes it practical for real-world deployment.
- **Incentive Compatibility:** Allows for programmatic incentives in community-driven projects.

**Limitations:**
- **Complexity:** The system is more complex than centralized training due to transform management.
- **Synchronization:** Requires precise synchronization of transform keys among participants.
- **Trust Model:** Assumes participants follow the protocol; malicious participants could still disrupt training.

**Comparison with Alternatives:**
- **vs. Homomorphic Encryption:** UPMs have significantly lower computational overhead.
- **vs. SMPC:** UPMs have lower communication overhead and are easier to scale.
- **vs. Static Sharding:** UPMs provide stronger security guarantees against extraction.

## 9 Conclusion
**Context:**
- The paper concludes by summarizing the contributions and future directions.

**Summary:**
- UPMs successfully enable collaborative training and inference without weight materialization.
- The use of time-varying, invertible transforms ensures unextractability while maintaining model utility.
- The overhead is negligible, making the approach practical for large-scale models.

**Future Work:**
- **Larger Models:** Testing on models larger than 1B parameters.
- **Heterogeneous Hardware:** Adapting the framework for heterogeneous participant hardware.
- **Advanced Attacks:** Further analysis of sophisticated extraction attacks.
- **Incentive Mechanisms:** Developing robust economic models for decentralized communities using UPMs.

**Final Statement:**
- UPMs provide a foundational protocol for secure, decentralized AI development, enabling community-driven innovation without the risks of centralization or weight theft.