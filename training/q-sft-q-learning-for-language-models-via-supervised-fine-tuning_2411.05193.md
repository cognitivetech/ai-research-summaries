# Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning

**Source**: [arXiv:2411.05193](https://arxiv.org/html/2411.05193)

**Authors:** Joey Hong, Anca Dragan, Sergey Levine

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Preliminaries](#3-preliminaries)
- [4 Q-Learning via Supervised Fine-Tuning](#4-q-learning-via-supervised-fine-tuning)
- [5 Experiments](#5-experiments)
- [6 Discussion](#6-discussion)
- [References](#references)
- [Appendix A Theoretical Proofs](#appendix-a-theoretical-proofs)
- [Appendix B Implementation Details](#appendix-b-implementation-details)

## Abstract
**[Key point]**: The paper introduces Q-SFT, a novel offline reinforcement learning algorithm that reformulates Q-learning as a supervised fine-tuning problem for large language models.
**Main contribution**: A method that maps token probabilities directly to Q-values, allowing LLMs to leverage pretraining weights without reinitialization or new value heads.
**Key results:**
- Demonstrates strong performance on multi-turn dialogue tasks using pretrained LLMs.
- Successfully applies to vision-language models (VLMs) for robotic manipulation and navigation.
- Achieves theoretical performance bounds comparable to state-of-the-art Q-learning methods.
- Eliminates the need for separate advantage or value prediction heads during fine-tuning.
**Conclusion:**
- Q-SFT bridges the gap between policy gradient methods and value-based RL for LLMs.
- Offers a scalable, efficient alternative to traditional offline RL for large-scale architectures.

## 1 Introduction
**Context:**
- Value-based reinforcement learning (RL) is theoretically capable of solving complex multi-turn problems.
- Applications include games, dialogue systems, and robotic control.
- Offline RL allows learning from static datasets without environment interaction.
- Policy gradient methods are standard for training LLMs on single-turn tasks like QA.
- Value-based methods for multi-turn RL in LLMs have struggled to scale.
- Challenges include leveraging pretraining, scaling to billions of parameters, and handling large datasets.
- Current value-based RL methods often require reinitializing weights or adding new heads.
**Goal:**
- To propose a scalable offline RL algorithm for LLMs that overcomes these scaling challenges.
- To cast Q-learning as a modified supervised fine-tuning (SFT) problem.
- To enable smooth transition from pretraining likelihood maximization to Q-function learning.
**Contributions:**
- Proposes Q-SFT, which interprets token probabilities as Q-values.
- Provides theoretical analysis showing performance bounds similar to standard Q-learning.
- Demonstrates empirical success on both LLMs and VLMs across diverse tasks.
- Removes the need for reinitializing weights or initializing new value/advantage heads.
- Enables the use of full pretraining benefits during RL fine-tuning.

## 2 Related Work
**Context:**
- Review of existing methods for training LLMs with RL.
- Distinction between policy gradient methods (e.g., PPO, GRPO) and value-based methods.
- Discussion of offline RL techniques for tabular and small-scale models.
- Analysis of challenges in applying value-based RL to large-scale neural networks.
**Key Comparisons:**
- Policy gradient methods dominate LLM alignment (e.g., RLHF).
- Value-based methods are less common due to instability and scaling issues.
- Previous attempts to combine value-based RL with LLMs often required significant architectural changes.
- Q-SFT differs by maintaining the standard autoregressive architecture.
- Comparison with methods that use separate value heads or advantage networks.
- Q-SFT integrates value estimation directly into the token prediction objective.

## 3 Preliminaries
**Context:**
- Formal definition of the Markov Decision Process (MDP) framework.
- Definition of offline RL setting and dataset constraints.
- Review of standard Q-learning and its theoretical underpinnings.
- Explanation of supervised fine-tuning (SFT) for LLMs.
**Key Concepts:**
- MDP tuple: $(S, A, R, P, \gamma)$.
- Offline dataset $\mathcal{D}$ collected from a behavior policy.
- Objective of maximizing expected return in the offline setting.
- Standard SFT objective: maximizing likelihood of next tokens.
- Limitations of standard SFT: does not account for long-term rewards.
- Need for a bridge between likelihood maximization and value maximization.

## 4 Q-Learning via Supervised Fine-Tuning
**Context:**
- Detailed derivation of the Q-SFT algorithm.
- Explanation of how Q-values are embedded into token probabilities.
**4.1 Learning Values as Probabilities**
- Core insight: Q-values can be represented as log-probabilities of actions.
- Modification of the standard cross-entropy loss.
- Introduction of a temperature or scaling factor to map logits to Q-values.
- Derivation of the Q-SFT objective function.
- Relationship between the modified loss and Bellman optimality.
- Explanation of how the algorithm encourages high-Q actions.
**4.2 Theoretical Analysis**
- Proof of performance bounds for Q-SFT.
- Comparison of Q-SFT bounds to standard Q-learning bounds.
- Analysis of convergence properties in the offline setting.
- Discussion of the impact of dataset coverage on performance.
- Theoretical justification for using SFT-like objectives for RL.
**4.3 Practical Implementation**
- Algorithm steps for training Q-SFT.
- Handling of multi-turn dialogue and robotic tasks.
- Integration with existing LLM training pipelines.
- No need for separate value networks or advantage estimators.
- Computational efficiency compared to traditional RL methods.
- Stability improvements due to the SFT-like objective.

## 5 Experiments
**Context:**
- Empirical evaluation of Q-SFT on various benchmarks.
- Comparison against baseline methods (SFT, PPO, other offline RL).
**5.1 Task Descriptions**
- Natural language dialogue tasks (e.g., AlpacaEval, Arena-Hard).
- Robotic manipulation tasks using VLMs.
- Navigation tasks from images.
- Description of the offline datasets used.
- Details on the pretrained LLMs and VLMs used as backbones.
**5.2 Results**
- Performance metrics: win rates, success rates, reward scores.
- Q-SFT outperforms standard SFT on reward-maximization tasks.
- Q-SFT performs competitively with policy gradient methods (e.g., PPO).
- Q-SFT shows robustness across different model sizes.
- VLM experiments show successful transfer to robotic control.
- Ablation studies on the scaling factor and dataset composition.
- Comparison of training stability and convergence speed.
- Analysis of the impact of pretraining quality on Q-SFT performance.

## 6 Discussion
**Context:**
- Interpretation of results and implications for the field.
- Limitations of the Q-SFT approach.
**Key Insights:**
- Q-SFT effectively leverages pretraining for RL without architectural overhead.
- The method is simpler to implement than traditional offline RL for LLMs.
- Strong theoretical grounding provides confidence in its applicability.
**Limitations:**
- Performance may degrade if the offline dataset lacks coverage of high-Q actions.
- Sensitivity to the scaling factor in the probability mapping.
- Current evaluation focuses on specific task types; generalization needs more study.
**Future Work:**
- Extension to online RL settings.
- Application to more complex multi-modal tasks.
- Investigation of dynamic scaling factors.
- Exploration of theoretical bounds in non-stationary environments.

## References
- List of cited works including foundational RL papers, LLM training papers, and related offline RL studies.
- Key references on Q-learning, policy gradients, and supervised fine-tuning.
- Citations for the datasets and benchmarks used in experiments.

## Appendix A Theoretical Proofs
**Context:**
- Detailed mathematical proofs for the claims made in Section 4.2.
**Contents:**
- Proof of the performance bound for Q-SFT.
- Derivation of the relationship between the Q-SFT objective and the Bellman error.
- Lemmas supporting the convergence analysis.
- Conditions for the validity of the theoretical bounds.

## Appendix B Implementation Details
**Context:**
- Technical specifications for the experiments.
**Contents:**
- Hyperparameters used for Q-SFT training (learning rate, batch size, etc.).
- Details on the scaling factor and its selection.
- Hardware and software environment setup.
- Data preprocessing steps for the offline datasets.
- Code availability and reproducibility notes.