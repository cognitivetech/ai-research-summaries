# A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models

**Source**: [arXiv:2506.18485](https://arxiv.org/html/2506.18485)

**Authors:** Junjie Zhang, Guozheng Ma, Shunyu Liu, Haoyu Wang, Jiaxing Huang, Ting-En Lin, Fei Huang, Yongbin Li, Dacheng Tao

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 Method](#3-method)
- [4 Experiment](#4-experiment)
- [5 Analysis on the mechanism behind MeRF](#5-analysis-on-the-mechanism-behind-merf)
- [6 Conclusion](#6-conclusion)
- [Ethics Statements](#ethics-statements)
- [References](#references)
- [Appendix A Appendix](#appendix-a-appendix)

## Abstract
**[Key point]**:
- The paper addresses the inefficiency of current Reinforcement Learning with Verifiable Rewards (RLVR) paradigms in training large reasoning models.
- RLVR typically relies on trial-and-error exploration, which is data-inefficient and blind to global reward patterns.
- The authors propose leveraging the in-context learning capabilities of Large Language Models (LLMs) to provide "motivation" during training.
- This motivation is defined as an explicit awareness of the reward function's rules.

**Main contribution**:
- Introduction of Motivation-enhanced Reinforcement Finetuning (MeRF).
- A method that injects the reward specification directly into the prompt as in-context motivation.
- Demonstration that aligning generation with optimization objectives improves performance.
- Empirical validation of MeRF's effectiveness across various reasoning tasks.

**Key results:**
- MeRF achieves substantial performance gains over standard RLVR baselines.
- The method leverages the model's ability to understand natural language descriptions of rewards.
- Consistency between in-context motivation and external reward signals is critical for optimal performance.
- Models exhibit robustness, capable of adapting to misleading motivations through continued reinforcement finetuning.

**Conclusion:**
- Explicitly telling LLMs the "rules of the game" enhances their reinforcement learning process.
- This simple modification bridges the gap between the model's internal generation process and the external optimization objective.
- The approach is intuitive, effective, and computationally feasible.

## 1 Introduction
**Context:**
- Large Reasoning Models (LRMs) have shown significant potential in solving complex tasks requiring multi-step logical deduction.
- Reinforcement Learning with Verifiable Rewards (RLVR) has become a dominant paradigm for fine-tuning these models.
- RLVR allows models to learn from feedback where the correctness of the output can be objectively verified (e.g., math problems, code execution).
- Despite its success, RLVR suffers from low sample efficiency.
- The trial-and-error nature of RL requires the model to explore the reward space extensively.
- Models often receive fragmented reward signals without understanding the overarching structure of the reward function.
- This lack of global awareness hinders the model's ability to generalize and optimize efficiently.

**Goal:**
- To improve the efficiency and effectiveness of reinforcement finetuning for large reasoning models.
- To investigate whether providing explicit "motivation" (awareness of the reward function) can help models learn better.
- To draw inspiration from human learning, where understanding the rules of a task aids in mastering it.
- To explore the synergy between in-context learning (ICL) and reinforcement learning (RL).

**Contributions:**
- Proposes MeRF, a novel framework that injects reward specifications into the prompt.
- Demonstrates that in-context motivation serves as a powerful prior for the model's generation process.
- Shows that MeRF aligns the model's internal generation distribution with the external optimization objective.
- Provides extensive empirical evaluations showing superior performance compared to baseline RLVR.
- Conducts ablation studies to analyze the impact of motivation consistency and adaptability.
- Offers mechanistic insights into how LLMs utilize in-context reward information during RL.

## 2 Related Work
**Reinforcement Learning with Verifiable Rewards (RLVR):**
- RLVR has been widely adopted for training reasoning models like those in the Qwen2.5-Math and DeepSeek-R1 families.
- Standard RLVR methods (e.g., PPO, GRPO) optimize policies based on scalar rewards derived from verifiable outputs.
- These methods treat the reward function as a black box, providing only scalar feedback.
- Prior work has focused on improving reward shaping, reward models, and exploration strategies.
- However, few works have explicitly integrated the *description* of the reward function into the training loop.

**In-Context Learning (ICL) in LLMs:**
- LLMs demonstrate strong capabilities to learn patterns from examples provided in the context window.
- ICL allows models to adapt to new tasks without weight updates.
- Recent studies show that LLMs can understand and follow complex instructions and rules provided in-context.
- The intersection of ICL and RL is an emerging area of interest.
- Some works explore using ICL to improve RL exploration or reward modeling.
- MeRF differs by using ICL specifically to inform the *policy* about the optimization goal during finetuning.

**Motivation and Awareness in AI:**
- The concept of "motivation" in AI often relates to intrinsic rewards or curiosity-driven exploration.
- This work redefines motivation as explicit awareness of the reward function's structure.
- It draws parallels to human cognitive processes where understanding rules enhances learning speed.
- Related to the field of "learning to learn" or meta-learning, but applied specifically to RLVR contexts.

## 3 Method
### 3.1 Motivation-enhanced Reinforcement Finetuning
**Core Concept:**
- MeRF introduces a "motivation" component to the standard RLVR process.
- This motivation is not a separate reward signal but a contextual cue.
- It explicitly describes the reward function in natural language.
- The goal is to make the model "aware" of what it is being optimized for.

**Mechanism:**
- During the prompt construction for each training step, the reward specification is appended.
- This specification includes the rules, constraints, and criteria for receiving a positive reward.
- The model reads this specification before generating the response.
- This leverages the model's in-context learning ability to align its generation strategy with the reward criteria.
- The process is "intuitive" because it mimics how humans are taught: first by explaining the rules, then by practicing.

**Integration with RL:**
- The standard RLVR loop remains intact (rollout, reward calculation, policy update).
- The key modification is in the input prompt structure.
- The reward specification is treated as part of the system prompt or user prompt context.
- This ensures that the motivation is present during every generation step.
- The model uses this information to guide its internal reasoning process.

**Benefits:**
- Reduces the exploration burden by providing a strong prior.
- Aligns the model's implicit understanding of the task with the explicit optimization objective.
- Enhances sample efficiency by reducing the number of trials needed to discover effective strategies.

### 3.2 Motivation with Verifiable Reward
**Design of Motivation:**
- The motivation is crafted as a natural language description of the verifiable reward function.
- It includes:
    - The task definition.
    - The criteria for correctness.
    - Any specific constraints or formatting requirements.
    - Examples of correct vs. incorrect outputs (optional, depending on design).
- The description is concise yet comprehensive enough to capture the reward logic.

**Consistency Requirement:**
- The in-context motivation must be consistent with the actual reward function used for scoring.
- If the motivation says "reward is 1 if answer is integer" but the reward function checks for "exact string match," confusion arises.
- MeRF assumes a high degree of alignment between the text description and the code-based reward.

**Adaptability:**
- The model is not rigidly bound to the motivation.
- Through reinforcement updates, the model can learn to override or adapt to the motivation if it proves suboptimal.
- This is particularly relevant when the motivation is misleading or incomplete.
- The RL process acts as a correction mechanism for potential misunderstandings of the in-context text.

## 4 Experiment
### 4.1 Experimental Setup
**Datasets:**
- Evaluated on standard reasoning benchmarks.
- Common datasets include:
    - GSM8K (Grade School Math).
    - MATH (High School Math).
    - Code generation benchmarks (e.g., HumanEval, MBPP).
    - Logic reasoning datasets.
- Focus on tasks with verifiable rewards.

**Models:**
- Base large reasoning models (e.g., Qwen2.5-Math, DeepSeek-R1 variants).
- Comparison across different model sizes to assess scalability.

**Baselines:**
- Standard RLVR (e.g., GRPO, PPO) without in-context motivation.
- Supervised Fine-Tuning (SFT) as a lower bound.
- Other advanced RL methods if applicable.

**Implementation Details:**
- Reward functions are implemented as code-based verifiers.
- In-context motivation is injected via prompt templates.
- Training hyperparameters are kept consistent across baselines and MeRF.
- Number of training steps and rollout lengths are controlled.

**Metrics:**
- Pass@1 accuracy on test sets.
- Training efficiency (samples per second).
- Convergence speed (accuracy vs. training steps).

### 4.2 Main Results
**Performance Gains:**
- MeRF consistently outperforms the RLVR baseline across all tested datasets.
- Significant improvements in early training stages, indicating faster convergence.
- Final accuracy gains are substantial, often exceeding 5-10% absolute improvement.
- Gains are observed in both math and code reasoning tasks.

**Efficiency Analysis:**
- MeRF achieves higher accuracy with fewer training samples.
- Reduced variance in reward signals during training.
- More stable learning curves compared to standard RLVR.

**Ablation Studies:**
- **Motivation Consistency:**
    - Performance drops when motivation is inconsistent with the reward function.
    - High consistency leads to the best results.
- **Motivation Length:**
    - Too short: insufficient information.
    - Too long: noise and distraction.
    - Optimal length balances detail and clarity.
- **Motivation Type:**
    - Explicit rule description vs. implicit examples.
    - Explicit rules generally perform better for complex tasks.

**Comparison with SFT:**
- MeRF surpasses SFT, demonstrating the value of RL.
- However, MeRF narrows the gap between RL and SFT in terms of sample efficiency.

## 5 Analysis on the mechanism behind MeRF
**How Models Use Motivation:**
- Analysis of attention patterns shows models attend to the motivation section.
- The motivation influences the initial tokens of the generated response.
- It acts as a constraint filter during the generation process.

**Adaptation to Misleading Motivation:**
- Experiments with intentionally misleading motivations.
- Models initially follow the misleading motivation.
- Over time, RL updates correct the behavior based on actual rewards.
- Demonstrates robustness and the dominance of external reward signals in the long run.

**Internal Representation Changes:**
- Probing hidden states reveals changes in representation alignment.
- MeRF-trained models have representations more closely aligned with the reward function's logic.
- This suggests that motivation helps shape the model's internal understanding of the task.

**Generalization:**
- MeRF improves generalization to unseen problem types within the same domain.
- The explicit rules help the model abstract the underlying principles of the reward.

## 6 Conclusion
**Summary:**
- MeRF is a simple yet effective method for enhancing RLVR.
- It leverages in-context learning to provide explicit motivation.
- This approach aligns generation with optimization, improving efficiency and performance.

**Implications:**
- Highlights the importance of explicit instruction in RL training for LLMs.
- Suggests that "telling the model the rules" is a powerful prior.
- Opens avenues for integrating more complex in-context information into RL loops.

**Future Work:**
- Exploring dynamic motivation generation (e.g., adapting motivation based on model state).
- Applying MeRF to multi-agent RL scenarios.
- Investigating the limits of motivation consistency and adaptability.

## Ethics Statements
**Bias and Fairness:**
- The method does not introduce new biases but relies on the existing reward functions.
- Care must be taken to ensure reward functions are fair and unbiased.

**Transparency:**
- The use of in-context motivation makes the optimization objective more transparent to the user.
- This aligns with goals of interpretable AI.

**Resource Usage:**
- MeRF does not significantly increase computational overhead compared to standard RLVR.
- Prompt length increases slightly, but this is negligible for large models.

## References
- Key citations include:
    - Original RLVR papers (e.g., DeepSeek-R1, Qwen2.5-Math).
    - Foundational RL papers (PPO, GRPO).
    - In-Context Learning studies.
    - Works on reward shaping and verifiable rewards.
    - Recent advances in large reasoning models.

## Appendix A Appendix
### A.1 Use of LLMs
- Details on how LLMs were used in the research process (e.g., generating prompts, analyzing results).
- Disclosure of any automated tools used.

### A.2 Preliminary
- Mathematical formulation of RLVR.
- Definition of GRPO and PPO algorithms.
- Explanation of verifiable rewards.

### A.3 Prompt Design of In-Context Motivation
- Detailed templates for motivation prompts.
- Variations in motivation style (e.g., rule-based, example-based).
- Guidelines for crafting effective motivations.

### A.4 Case Study
- Detailed examples of model behavior with and without MeRF.
- Step-by-step analysis of reasoning paths.
- Comparison of generated responses.

### A.5 Additional results
- Extended tables and figures.
- Results on additional datasets.
- Sensitivity analysis of hyperparameters.

### Instructions for reporting errors
- Contact information for authors.
- Guidelines for submitting bug reports or feedback.