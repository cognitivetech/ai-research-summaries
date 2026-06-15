# Demystifying Long Chain-of-Thought Reasoning in LLMs

**Source**: [arXiv:2502.03373](https://arxiv.org/html/2502.03373)

**Authors:** Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, Xiang Yue

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Problem Formulation](#2-problem-formulation)
- [3 Impact of SFT on Long CoT](#3-impact-of-sft-on-long-cot)
- [4 Impact of Reward Design on Long CoT](#4-impact-of-reward-design-on-long-cot)
- [5 Scaling up Verifiable Reward](#5-scaling-up-verifiable-reward)
- [6 Exploration on RL from the Base Model](#6-exploration-on-rl-from-the-base-model)
- [7 Discussions and Future Work](#7-discussions-and-future-work)

## Abstract
**[Key point]**: The paper systematically investigates the mechanics of long Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs), focusing on the conditions under which these capabilities emerge during Reinforcement Learning (RL) training.
**Main contribution**: Identification of four key factors enabling long CoT trajectories through extensive Supervised Fine-Tuning (SFT) and RL experiments, providing practical guidance for training strategies.
**Key results:**
- SFT is not strictly necessary for long CoT emergence but significantly simplifies training and improves efficiency.
- Reasoning capabilities emerge with increased training compute, but their development is not guaranteed without careful reward shaping to stabilize CoT length growth.
- Scaling verifiable reward signals is critical; using noisy, web-extracted solutions with filtering mechanisms shows strong potential, especially for Out-of-Distribution (OOD) tasks like STEM reasoning.
- Core abilities like error correction are inherently present in base models, but incentivizing them for complex tasks via RL demands significant compute and nuanced measurement.
**Conclusion:**
- The findings provide actionable insights for optimizing training strategies to enhance long CoT reasoning.
- The study highlights the importance of reward design and data quality in scaling reasoning capabilities.

## 1 Introduction
**Context:**
- Scaling inference compute enhances reasoning in LLMs.
- Long chains-of-thought enable advanced strategies such as backtracking and error correction.
- Reinforcement Learning (RL) has become a crucial method for developing these reasoning capabilities.
- Despite RL's success, the specific conditions under which long CoTs emerge remain unclear.
- RL training requires careful design choices regarding reward functions and data sources.
**Goal:**
- To systematically investigate the mechanics of long CoT reasoning.
- To identify the key factors that enable models to generate long CoT trajectories.
- To provide empirical evidence on the roles of SFT, reward design, and verifiable data.
**Contributions:**
- Systematic analysis of SFT's impact on long CoT generation.
- Detailed examination of reward design impacts, including length stability and hacking.
- Investigation of scaling verifiable rewards using noisy, web-extracted data.
- Exploration of RL from base models versus RL from SFT-initialized models.
- Practical guidelines for optimizing training strategies for long CoT reasoning.

## 2 Problem Formulation
### 2.1 Notation
- Defines standard mathematical notation for LLMs and RL processes.
- Specifies variables for states, actions, rewards, and policy parameters.
- Clarifies the distinction between pre-training, SFT, and RL phases.

### 2.2 Supervised Fine-Tuning (SFT)
- Describes the process of fine-tuning models on high-quality reasoning data.
- Highlights the role of SFT in initializing the model for complex reasoning tasks.
- Discusses the quality and source of SFT data as critical variables.

### 2.3 Reinforcement Learning (RL)
- Outlines the RL framework used, typically PPO or REINFORCE.
- Defines the reward structure, including correctness rewards and length penalties.
- Explains the objective function for optimizing long CoT generation.

### 2.4 Training Setup
- Details the computational resources and infrastructure used.
- Specifies hyperparameters for SFT and RL phases.
- Describes the dataset splits for training, validation, and testing.

### 2.5 Evaluation Setup
- Defines the metrics used to evaluate reasoning performance (e.g., pass@k).
- Describes the test sets, including in-distribution and out-of-distribution tasks.
- Explains how long CoT length is measured and analyzed.

## 3 Impact of SFT on Long CoT
### 3.1 SFT Scaling
- Investigates the effect of varying SFT data volume on long CoT emergence.
- Finds that while SFT helps, it is not strictly necessary for long CoT to emerge.
- Notes that SFT significantly improves training efficiency and stability.

### 3.2 SFT Initialization for RL
- Analyzes the impact of initializing RL with SFT-trained models.
- Shows that SFT initialization leads to faster convergence in RL.
- Demonstrates that SFT helps prevent early collapse of CoT length.

### 3.3 Sources of Long CoT SFT Data
- Compares different sources of SFT data (e.g., synthetic vs. human-written).
- Highlights the importance of data quality over quantity for SFT.
- Discusses the challenges of sourcing high-quality long CoT data.

## 4 Impact of Reward Design on Long CoT
### 4.1 CoT Length Stability
- Examines the stability of CoT length during RL training.
- Identifies that without proper reward shaping, CoT length can fluctuate wildly.
- Shows that reward shaping is crucial for stabilizing long CoT growth.

### 4.2 Active Scaling of CoT Length
- Explores methods for actively encouraging longer CoTs.
- Discusses the use of length rewards to scale CoT length.
- Notes the trade-off between length and correctness.

### 4.3 Cosine Reward Hyperparameters
- Details the use of cosine reward functions to balance length and correctness.
- Analyzes the sensitivity of results to cosine reward hyperparameters.
- Provides recommendations for setting these hyperparameters.

### 4.4 Context Window Size
- Investigates the impact of context window size on long CoT reasoning.
- Finds that larger context windows allow for more complex reasoning steps.
- Notes computational constraints associated with larger context windows.

### 4.5 Length Reward Hacking
- Identifies the phenomenon of "length reward hacking" where models generate unnecessarily long CoTs.
- Discusses strategies to mitigate this, such as using verifiable rewards.
- Shows that verifiable rewards reduce hacking compared to simple length rewards.

### 4.6 Optimal Discount Factors
- Analyzes the impact of discount factors in RL on long CoT emergence.
- Finds that optimal discount factors vary depending on the task complexity.
- Provides guidelines for selecting discount factors.

## 5 Scaling up Verifiable Reward
### 5.1 SFT with Noisy Verifiable Data
- Explores the use of noisy, web-extracted solutions for SFT.
- Demonstrates that filtering mechanisms can effectively clean noisy data.
- Shows that SFT with noisy data can still improve reasoning capabilities.

### 5.2 Scaling up RL with Noisy Verifiable Data
- Investigates the scaling of RL with noisy verifiable rewards.
- Finds that noisy rewards can be effective for OOD tasks like STEM reasoning.
- Highlights the potential of leveraging large-scale web data for RL.
- Notes the importance of filtering mechanisms to maintain reward signal quality.

## 6 Exploration on RL from the Base Model
### 6.1 Nuances in Analysis Based on Emergent Behaviors
- Analyzes emergent behaviors in models trained with RL from base models.
- Identifies specific behaviors that emerge only after significant compute.
- Discusses the difficulty in measuring the emergence of these behaviors.

### 6.2 Nuances in Analysis Based on Length Scaling
- Examines how CoT length scales with compute in base model RL.
- Finds that length scaling is not linear and depends on reward design.
- Notes that some models may plateau in CoT length despite increased compute.

### 6.3 Potential Reasons Why Emergent Behavior is Not Observed with Qwen2.5-Math-7B
- Investigates why certain models (e.g., Qwen2.5-Math-7B) do not show emergent behaviors.
- Hypothesizes that model architecture and pre-training data play a role.
- Suggests that smaller models may require more specialized training for long CoT.

### 6.4 Comparison between RL from the Base Model and RL from Long CoT SFT
- Compares the performance of RL from base models vs. RL from SFT-initialized models.
- Finds that RL from SFT generally outperforms RL from base models in terms of stability.
- Notes that RL from base models can still achieve comparable performance with sufficient compute.

### 6.5 Long CoT Patterns in Pre-training Data
- Analyzes the presence of long CoT patterns in pre-training data.
- Finds that pre-training data contains latent reasoning patterns.
- Suggests that these patterns contribute to the base model's inherent reasoning abilities.

## 7 Discussions and Future Work
### 7.1 Scaling up Model Size
- Discusses the potential benefits of scaling model size for long CoT reasoning.
- Notes that larger models may require less RL to achieve similar performance.
- Highlights the computational costs associated with scaling model size.

### 7.2 RL Infrastructure Is Still in Its Infancy
- Critiques the current state of RL infrastructure for LLMs.
- Notes the lack of standardized tools and best practices.
- Calls for improved infrastructure to support large-scale RL training.

### 7.3 REINFORCE Is More Tricky to Tune than PPO
- Compares REINFORCE and PPO algorithms for long CoT reasoning.
- Finds that REINFORCE is more sensitive to hyperparameter tuning.
- Recommends PPO for its stability, despite REINFORCE's simplicity.

### 7.4 Scaling up Verification
- Discusses the challenges of scaling verification mechanisms.
- Highlights the need for robust and scalable verification tools.
- Suggests that verification is key to preventing reward hacking.

### 7.5 Latent Capabilities in Base Models
- Emphasizes the importance of latent capabilities in base models.
- Notes that these capabilities can be unlocked with the right RL strategy.
- Suggests that future work should focus on better leveraging these latent capabilities.