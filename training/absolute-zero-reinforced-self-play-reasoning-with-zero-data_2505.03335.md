# Absolute Zero: Reinforced Self-play Reasoning with Zero Data

**Source**: [arXiv:2505.03335](https://arxiv.org/html/2505.03335)

**Authors:** Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 The Absolute Zero Paradigm](#2-the-absolute-zero-paradigm)
- [3 Absolute Zero Reasoner](#3-absolute-zero-reasoner)
- [4 Experiments](#4-experiments)
- [5 Related Work](#5-related-work)
- [6 Conclusion and Discussion](#6-conclusion-and-discussion)

## Abstract
**[Key point]**: The paper introduces "Absolute Zero," a novel Reinforcement Learning with Verifiable Rewards (RLVR) paradigm that enables large language models to learn reasoning skills entirely without external human-curated data.
**Main contribution**: The proposal of the Absolute Zero Reasoner (AZR), a system that self-evolves its training curriculum by generating its own tasks and solving them, using a code executor as a unified source of verifiable reward.
**Key results:**
- AZR achieves state-of-the-art (SOTA) performance on coding and mathematical reasoning benchmarks.
- It outperforms existing zero-setting models that rely on tens of thousands of in-domain human-curated examples.
- The approach is scalable across different model sizes and compatible with various model classes.
- The system demonstrates the ability to self-evolve its training curriculum and reasoning ability without external supervision.
**Conclusion:**
- Reliance on human supervision for RLVR is a scalability bottleneck.
- Self-play reasoning with zero external data is a viable and superior path for long-term AI learning.
- Code execution provides a robust, grounded, and open-ended reward signal for self-improvement.

## 1 Introduction
**Context:**
- Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful method for enhancing the reasoning capabilities of Large Language Models (LLMs).
- RLVR allows models to learn directly from outcome-based rewards, bypassing the need for extensive supervised fine-tuning on reasoning traces.
- Recent advancements in RLVR have focused on the "zero-setting," which avoids supervision in labeling the reasoning process itself.
- However, current zero-setting approaches still depend heavily on manually curated collections of questions and answers for training.
- This dependency creates a significant scalability bottleneck due to the scarcity of high-quality, human-produced examples.
- The challenge of relying on human supervision is already evident in the domain of language model pretraining.
- In a hypothetical future where AI surpasses human intelligence, tasks provided by humans may offer limited learning potential for a superintelligent system.
- Human-curated data may become a ceiling for AI capabilities, preventing further autonomous growth.
**Goal:**
- To address the scalability and dependency issues of current RLVR methods.
- To propose a new RLVR paradigm called "Absolute Zero."
- To enable a single model to learn to propose tasks that maximize its own learning progress.
- To allow the model to improve its reasoning by solving these self-proposed tasks.
- To achieve this without relying on any external data or human supervision.
**Contributions:**
- Introduction of the Absolute Zero paradigm, where learning is driven by internal curiosity and self-generated challenges.
- Development of the Absolute Zero Reasoner (AZR), a system that uses a code executor for both validation and verification.
- Demonstration that AZR can self-evolve its training curriculum and reasoning ability.
- Empirical evidence that AZR achieves SOTA performance on coding and mathematical reasoning tasks.
- Proof that AZR outperforms models using tens of thousands of human-curated examples.
- Validation that AZR is effective across different model scales and compatible with various model classes.

## 2 The Absolute Zero Paradigm
### 2.1 Preliminaries
**Context:**
- The paper builds upon the foundation of Reinforcement Learning with Verifiable Rewards (RLVR).
- RLVR typically involves an agent generating a solution and receiving a reward based on the correctness of the final answer.
- Traditional RLVR methods often use datasets like GSM8K or MATH for training.
- These datasets are static and limited by human creativity and availability.
- The concept of "zero-setting" in RLVR refers to avoiding supervision in the reasoning process labels.
- However, "zero-setting" in prior works still implies the use of external problem statements.
- The paper distinguishes between "zero-supervision" (no labels) and "zero-data" (no external problems).
- Absolute Zero aims for true zero-data learning.
**Key Concepts:**
- Verifiable rewards: Rewards derived from objective correctness checks (e.g., code execution, mathematical equality).
- Self-play: A learning paradigm where agents interact with themselves or their own generated environments.
- Curriculum learning: The process of gradually increasing the difficulty of training tasks.
- Code executor: A tool that runs generated code and provides immediate feedback on execution results.

### 2.2 Absolute Zero
**Core Mechanism:**
- Absolute Zero eliminates the need for any external dataset of questions.
- The model acts as both the problem proposer and the problem solver.
- The system generates a task, attempts to solve it, and receives a reward based on the solution's correctness.
- The reward signal is derived from the code executor, which validates the code and verifies the answer.
- This creates a closed-loop learning system.
**Self-Evolution:**
- The model learns to propose tasks that maximize its own learning progress.
- Tasks are not random; they are curated by the model to be challenging yet solvable.
- The curriculum evolves dynamically based on the model's current capabilities.
- If a task is too easy, the model learns nothing; if too hard, it fails to learn.
- The system balances exploration and exploitation to find the optimal learning zone.
**Grounded Learning:**
- Learning is "grounded" in the sense that the reward is based on concrete, executable outcomes.
- The code executor serves as a unified source of truth.
- This avoids the ambiguity of natural language rewards or human judgments.
- The learning is open-ended because the model can generate infinite variations of tasks.

## 3 Absolute Zero Reasoner
### 3.1 Two Roles in One: Proposer and Solver
**Dual Role Architecture:**
- AZR integrates two distinct roles within a single model framework: Proposer and Solver.
- The Proposer role generates new coding or mathematical tasks.
- The Solver role attempts to solve the tasks generated by the Proposer.
- This integration allows for seamless feedback between task generation and solution quality.
**Proposer Role:**
- The Proposer generates a problem statement and the corresponding code solution.
- It uses the code executor to validate that the code runs without errors.
- It ensures the problem is well-formed and has a verifiable answer.
- The Proposer aims to generate tasks that are informative for learning.
- It avoids generating trivial or already-solved tasks.
**Solver Role:**
- The Solver receives the task from the Proposer.
- It generates a reasoning trace and a final answer.
- The answer is verified against the ground truth provided by the Proposer's code execution.
- The Solver's performance determines the reward signal for the Proposer.
- The Solver learns to tackle increasingly complex problems.
**Interaction:**
- The Proposer and Solver are trained jointly.
- The reward for the Proposer depends on the Solver's success and the difficulty of the task.
- The reward for the Solver depends on the correctness of its answer.
- This creates a cooperative dynamic where the Proposer helps the Solver improve, and the Solver's improvement motivates the Proposer to create harder tasks.

### 3.2 Learning Different Modes of Reasoning: Deduction, Induction, and Abduction
**Reasoning Modes:**
- AZR supports three primary modes of reasoning: Deduction, Induction, and Abduction.
- These modes correspond to different types of logical inference and problem-solving strategies.
**Deduction:**
- Deductive reasoning involves deriving specific conclusions from general rules.
- In AZR, this corresponds to generating code that strictly follows logical constraints.
- The model learns to apply known algorithms and mathematical theorems.
- Tasks often involve implementing standard algorithms with specific inputs.
**Induction:**
- Inductive reasoning involves generalizing from specific examples.
- In AZR, this corresponds to discovering patterns in code or data.
- The model learns to infer rules from observed behaviors.
- Tasks may involve predicting outputs based on input patterns.
**Abduction:**
- Abductive reasoning involves inferring the most likely explanation for an observation.
- In AZR, this corresponds to debugging or reverse-engineering code.
- The model learns to identify missing or incorrect components in a system.
- Tasks may involve fixing buggy code or completing partial implementations.
**Integration:**
- AZR dynamically switches between these modes based on the task.
- The Proposer can generate tasks that require specific reasoning modes.
- The Solver adapts its strategy to match the required mode.
- This flexibility allows AZR to handle a wide variety of reasoning challenges.

### 3.3 Absolute Zero Reasoner Learning Algorithm
**Algorithm Overview:**
- The learning algorithm is based on Reinforcement Learning with Verifiable Rewards.
- It uses a policy gradient method to update the model parameters.
- The reward function is designed to encourage both correctness and complexity.
**Reward Function:**
- The reward consists of two components: correctness reward and complexity reward.
- Correctness reward: Binary reward for solving the task correctly.
- Complexity reward: Encourages the Proposer to generate harder tasks.
- The complexity reward is based on the difficulty of the task relative to the model's current ability.
- Tasks that are too easy or too hard receive lower rewards.
**Training Loop:**
1. The Proposer generates a task (code and problem statement).
2. The code is executed to verify correctness and obtain the ground truth.
3. The Solver attempts to solve the task.
4. The Solver's answer is compared to the ground truth.
5. Rewards are calculated based on correctness and complexity.
6. The model parameters are updated using policy gradients.
7. The process repeats, with the curriculum evolving over time.
**Optimization:**
- The algorithm uses advantage estimation to reduce variance.
- It employs a baseline to stabilize training.
- The learning rate is adjusted dynamically to ensure convergence.
- The model is trained for a fixed number of steps or until performance plateaus.

## 4 Experiments
### 4.1 Experiment Setup
**Benchmarks:**
- AZR is evaluated on standard coding and mathematical reasoning benchmarks.
- Coding benchmarks include HumanEval and MBPP.
- Mathematical reasoning benchmarks include GSM8K and MATH.
- These benchmarks are widely used for evaluating LLM reasoning capabilities.
**Baselines:**
- AZR is compared against existing zero-setting RLVR models.
- Baselines include models trained on tens of thousands of human-curated examples.
- Examples include models trained on GSM8K, MATH, and HumanEval datasets.
- The comparison highlights the efficiency of AZR in terms of data usage.
**Model Configurations:**
- AZR is tested on various model scales (e.g., 7B, 13B, 70B parameters).
- Different base model architectures are used to test compatibility.
- The code executor is implemented using a standard Python interpreter.
- The reward function parameters are tuned for optimal performance.
**Training Details:**
- AZR is trained from scratch without any external data.
- The training process involves self-play for a fixed number of iterations.
- The curriculum evolves automatically during training.
- The system is evaluated periodically during training to monitor progress.

### 4.2 Results
**Performance Comparison:**
- AZR achieves SOTA performance on all tested benchmarks.
- It outperforms models trained on tens of thousands of human-curated examples.
- The performance gap is significant, especially on complex reasoning tasks.
- AZR demonstrates superior generalization to out-of-distribution tasks.
**Scalability:**
- AZR's performance scales with model size.
- Larger models benefit more from the self-play mechanism.
- The approach is compatible with various model classes (e.g., Transformer, Mamba).
- The system remains stable across different initializations.
**Curriculum Evolution:**
- The complexity of proposed tasks increases over time.
- The diversity of tasks also increases, covering a wide range of reasoning modes.
- The model learns to balance difficulty and solvability.
- The curriculum adapts to the model's improving capabilities.
**Ablation Studies:**
- Removing the complexity reward reduces performance.
- Removing the Proposer role degrades the system's ability to generate challenging tasks.
- Using a fixed curriculum leads to suboptimal learning.
- The code executor is crucial for providing reliable rewards.

## 5 Related Work
**Reinforcement Learning with Verifiable Rewards:**
- Prior work has explored RLVR for reasoning tasks.
- Most methods rely on external datasets for problem generation.
- AZR extends this by eliminating the need for external data.
**Self-Play in AI:**
- Self-play has been successful in games (e.g., AlphaGo, AlphaZero).
- Recent work applies self-play to language models.
- AZR adapts self-play for reasoning tasks using code execution.
**Curriculum Learning:**
- Curriculum learning has been used to improve training efficiency.
- Traditional methods use static or manually designed curricula.
- AZR uses a dynamic, self-generated curriculum.
**Code Generation and Reasoning:**
- Code generation is a key area for evaluating reasoning.
- Previous methods use human-written code for training.
- AZR generates its own code tasks, enabling autonomous learning.
**Zero-Shot and Few-Shot Learning:**
- AZR relates to zero-shot learning by avoiding training data.
- It differs by actively generating its own training signal.
- The approach bridges the gap between zero-shot and supervised learning.

## 6 Conclusion and Discussion
**Summary:**
- The paper introduces Absolute Zero, a novel RLVR paradigm.
- AZR enables LLMs to learn reasoning without external data.
- The system uses self-play and code execution for grounded learning.
**Key Findings:**
- AZR achieves SOTA performance on coding and math benchmarks.
- It outperforms models using large human-curated datasets.
- The approach is scalable and compatible with various models.
**Implications:**
- Absolute Zero challenges the reliance on human-curated data.
- It suggests a path towards autonomous AI learning.
- The method could be extended to other domains with verifiable rewards.
**Limitations:**
- The current implementation relies on code execution.
- It may not be directly applicable to domains without verifiable outputs.
- The computational cost of self-play can be high.
**Future Work:**
- Extending AZR to other reasoning domains (e.g., science, law).
- Improving the efficiency of the self-play process.
- Exploring hybrid approaches combining self-play and limited human data.
**Final Thought:**
- Absolute Zero represents a significant step towards self-improving AI systems.
- It demonstrates the potential of autonomous learning in reasoning tasks.
- The success of AZR highlights the importance of verifiable rewards and self-generated curricula.