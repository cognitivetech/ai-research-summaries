# Inference-Time Scaling for Generalist Reward Modeling

**Source**: [arXiv:2504.02495](https://arxiv.org/html/2504.02495)

**Authors:** Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Preliminaries](#2-preliminaries)
- [3 Self-Principled Critique Tuning (SPCT)](#3-self-principled-critique-tuning-spct)
- [4 Inference-Time Scaling with SPCT](#4-inference-time-scaling-with-spct)
- [5 Results on Reward Modeling Benchmarks](#5-results-on-reward-modeling-benchmarks)
- [6 Related Work](#6-related-work)
- [7 Conclusion and Future Work](#7-conclusion-and-future-work)
- [Ethics Statement](#ethics-statement)

## Abstract
**[Key point]**: The paper investigates the scalability of reward modeling (RM) during inference time for generalist queries, moving beyond verifiable or rule-based domains.
**Main contribution**: Introduction of Self-Principled Critique Tuning (SPCT) to enable generalist reward models to generate adaptive principles and accurate critiques via online RL.
**Key results:**
- SPCT significantly improves the quality and scalability of Pointwise Generative Reward Models (GRMs).
- DeepSeek-GRM models outperform existing methods on various RM benchmarks without severe biases.
- Inference-time scaling via parallel sampling and meta-RM voting achieves better performance than training-time scaling in certain contexts.
- The approach allows for flexibility across different input types and potential for inference-time compute expansion.
**Conclusion:**
- While challenges remain in some tasks, the proposed method demonstrates that proper learning methods can enable effective inference-time scalability for generalist RMs.
- Models are released on Hugging Face and ModelScope.

## 1 Introduction
**Context:**
- Reinforcement Learning (RL) is widely adopted in post-training for Large Language Models (LLMs) at scale.
- Recent incentives for reasoning capabilities in LLMs suggest that proper learning methods can enable effective inference-time scalability.
- A major challenge in RL is obtaining accurate reward signals for LLMs in domains beyond verifiable questions or artificial rules.
- Traditional Reward Models often struggle with general queries due to lack of flexibility and scalability during inference.
- Existing methods often rely on training-time scaling, which is computationally expensive and less flexible than inference-time adjustments.
**Goal:**
- To improve reward modeling with more inference compute for general queries, specifically focusing on the inference-time scalability of generalist RM.
- To enhance the effectiveness of performance-compute scaling through proper learning methods.
- To develop a generalist reward model that can adaptively generate principles and critiques for diverse inputs.
**Contributions:**
- Adoption of Pointwise Generative Reward Modeling (GRM) to enable flexibility for different input types and potential for inference-time scaling.
- Proposal of Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL.
- Implementation of parallel sampling to expand compute usage for inference-time scaling.
- Introduction of a meta RM to guide the voting process for better scaling performance.
- Empirical validation showing SPCT improves quality and scalability, outperforming existing methods and models.
- Release of DeepSeek-GRM models on Hugging Face and ModelScope.

## 2 Preliminaries
**2.1 Comparisons of Different RM approaches**
- **Pointwise vs. Pairwise**:
    - Pointwise RM assigns a score to a single response.
    - Pairwise RM compares two responses to determine preference.
    - Pointwise approaches offer greater flexibility for general queries and diverse input types.
- **Discriminative vs. Generative**:
    - Discriminative RMs output a scalar score directly.
    - Generative RMs (GRMs) generate text-based critiques and scores, allowing for more nuanced feedback.
    - GRMs are better suited for inference-time scaling as they can leverage chain-of-thought reasoning.
- **Limitations of Traditional RMs**:
    - Often biased towards specific domains or formats.
    - Lack adaptability to unseen or complex general queries.
    - Difficulty in scaling compute during inference to improve accuracy.
**2.2 Boosting Reward Quality with Principles**
- **Role of Principles**:
    - Principles serve as high-level guidelines for evaluating responses.
    - They help standardize the critique process across diverse topics.
- **Adaptive Principle Generation**:
    - Static principles may not cover all edge cases in generalist settings.
    - Adaptive generation allows the model to tailor principles to the specific context of the query.
- **Critique Accuracy**:
    - Accurate critiques are essential for reliable reward signals.
    - Principles guide the LLM to focus on relevant aspects of the response.
    - Reduces hallucination and irrelevant feedback in critiques.

## 3 Self-Principled Critique Tuning (SPCT)
**3.1 Unpinning Principles from Understanding to Generation**
- **Concept of Unpinning**:
    - Traditional methods pin principles to specific understanding tasks.
    - SPCT unlinks principles from rigid understanding, allowing them to be generated dynamically.
- **Dynamic Principle Generation**:
    - The model generates principles relevant to the specific input query.
    - This ensures that the evaluation criteria are context-aware.
    - Enhances the generalizability of the reward model.
- **Integration with GRM**:
    - Principles are integrated into the generative process of the GRM.
    - The model outputs principles, critiques, and scores in a structured manner.
    - Facilitates a more coherent and logical evaluation process.
**3.2 Rule-Based Reinforcement Learning**
- **RL Framework**:
    - SPCT utilizes online Reinforcement Learning to optimize the GRM.
    - The goal is to maximize the quality of generated principles and critiques.
- **Rule-Based Rewards**:
    - Rewards are derived from rule-based evaluations of the generated outputs.
    - Rules check for consistency, relevance, and logical coherence of principles and critiques.
    - Avoids reliance on external human labels during the scaling phase.
- **Optimization Process**:
    - The model is trained to generate principles that lead to higher reward scores.
    - Online RL allows for continuous adaptation to new data distributions.
    - Improves the robustness of the reward model over time.
- **Benefits of Rule-Based RL**:
    - Scalable to large datasets without manual annotation.
    - Provides clear and consistent feedback signals.
    - Enables the model to learn complex evaluation strategies autonomously.

## 4 Inference-Time Scaling with SPCT
**4.1 Parallel Sampling Strategy**
- **Compute Expansion**:
    - Inference-time scaling involves using more compute during the evaluation phase.
    - Parallel sampling generates multiple critiques and scores for a single response.
- **Diversity of Critiques**:
    - Multiple samples capture diverse perspectives and potential errors.
    - Reduces the variance in reward estimation.
    - Enhances the robustness of the final reward signal.
**4.2 Meta RM for Voting**
- **Meta Reward Model**:
    - A meta RM is introduced to aggregate results from parallel samples.
    - It guides the voting process to determine the final reward.
- **Voting Mechanism**:
    - Votes are cast based on the consistency and quality of individual critiques.
    - The meta RM weighs votes to mitigate outliers or low-quality critiques.
    - Improves the accuracy of the aggregated reward signal.
- **Scaling Performance**:
    - The combination of parallel sampling and meta RM enables effective scaling.
    - Performance improves with increased compute, demonstrating scalability.
    - Outperforms methods that rely solely on training-time scaling.
**4.3 Efficiency Considerations**
- **Trade-offs**:
    - Balancing compute cost with performance gains.
    - Parallel sampling increases latency but improves accuracy.
- **Optimization**:
    - Strategies to optimize the number of samples for optimal cost-performance ratio.
    - Adaptive sampling based on query complexity.

## 5 Results on Reward Modeling Benchmarks
**5.1 Experiment Settings**
- **Benchmarks Used**:
    - Various standard Reward Modeling benchmarks are employed.
    - Benchmarks cover diverse domains and query types.
    - Includes both verifiable and subjective evaluation tasks.
- **Baselines**:
    - Comparison with existing state-of-the-art RM models.
    - Baselines include traditional discriminative RMs and other generative RMs.
    - Includes models trained with different scaling strategies.
- **Implementation Details**:
    - Hyper-parameters for SPCT and parallel sampling.
    - Training data sources and preprocessing steps.
    - Evaluation metrics used for comparison.
**5.2 Results and Analysis**
- **Performance Comparison**:
    - DeepSeek-GRM outperforms existing methods on most benchmarks.
    - Significant improvements in accuracy and consistency.
    - Superior performance on generalist queries compared to domain-specific models.
- **Scalability Analysis**:
    - Demonstration of performance gains with increased inference compute.
    - Linear or super-linear scaling in certain regimes.
    - Comparison with training-time scaling methods.
- **Bias Analysis**:
    - Evaluation of biases in the generated rewards.
    - SPCT shows reduced severe biases compared to baselines.
    - More balanced performance across different demographic or topic groups.
- **Ablation Studies**:
    - Impact of SPCT components on overall performance.
    - Effectiveness of adaptive principle generation.
    - Contribution of parallel sampling and meta RM.
- **Failure Modes**:
    - Identification of tasks where DeepSeek-GRM struggles.
    - Analysis of reasons for failure (e.g., ambiguity, lack of data).
    - Suggestions for future improvements.

## 6 Related Work
- **Reward Modeling**:
    - Overview of traditional and modern RM approaches.
    - Discussion of discriminative vs. generative models.
    - Limitations of current generalist RMs.
- **Inference-Time Scaling**:
    - Review of methods for scaling LLMs during inference.
    - Comparison with training-time scaling strategies.
    - Applications in reasoning and evaluation tasks.
- **Reinforcement Learning for LLMs**:
    - Use of RL in post-training for LLMs.
    - Rule-based vs. reward-model-based RL.
    - Challenges in applying RL to generalist tasks.
- **Principle-Based Evaluation**:
    - Existing work on using principles for evaluation.
    - Static vs. dynamic principle generation.
    - Integration of principles into LLM pipelines.

## 7 Conclusion and Future Work
**Conclusion:**
- SPCT successfully enables inference-time scalability for generalist reward models.
- DeepSeek-GRM models demonstrate superior performance and reduced biases.
- Inference-time scaling via parallel sampling and meta RM is effective.
- The approach provides a flexible framework for diverse evaluation tasks.
**Future Work:**
- Addressing remaining challenges in specific tasks.
- Improving generalization beyond training data.
- Enhancing the efficiency of inference-time scaling.
- Exploring more complex principle generation mechanisms.
- Expanding the scope of generalist reward systems.

## Ethics Statement
- **Bias Mitigation**:
    - Efforts made to reduce severe biases in reward generation.
    - Continuous monitoring for potential biases in future iterations.
- **Transparency**:
    - Release of models and code for reproducibility.
    - Detailed documentation of training and evaluation processes.
- **Responsible Use**:
    - Guidelines for the responsible use of reward models.
    - Warning against misuse in sensitive applications.
- **Data Privacy**:
    - Use of publicly available and anonymized data.
    - Compliance with data protection regulations.