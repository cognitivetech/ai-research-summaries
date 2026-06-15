# Mutual Theory of Mind in Human-AI Collaboration: An Empirical Study with LLM-driven AI Agents in a Real-time Shared Workspace Task

**Source**: [arXiv:2409.08811](https://arxiv.org/html/2409.08811)

**Authors:** Shao Zhang, Xihuai Wang, Wenhao Zhang, Yongshan Chen, Landi Gao, Dakuo Wang, Weinan Zhang, Xinbing Wang, Ying Wen

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Works](#2-related-works)
- [3 Cooperative Task Setup and Formalization in Shared Workspace](#3-cooperative-task-setup-and-formalization-in-shared-workspace)
- [4 LLM-driven Agent with Theory of Mind and Communication Module](#4-llm-driven-agent-with-theory-of-mind-and-communication-module)
- [5 Methods](#5-methods)
- [6 Results](#6-results)
- [7 Discussion](#7-discussion)
- [8 Conclusion](#8-conclusion)

## Abstract
**[Key point]**: The study investigates Mutual Theory of Mind (MToM) in Human-AI Teams (HATs) where AI agents possess ToM capabilities.
**Main contribution**: Empirical evaluation of an LLM-driven AI agent with ToM and communication modules in a real-time shared workspace task.
**Key results:**
- Agent's ToM capability does not significantly improve team performance metrics.
- ToM capability significantly enhances human understanding of the agent.
- ToM capability increases the human feeling of being understood by the agent.
- Participants perceive verbal communication as increasing human cognitive burden.
- Bidirectional communication leads to lower overall HAT performance compared to other conditions.
**Conclusion:**
- Designing AI agents for real-time collaboration requires balancing ToM capabilities with communication efficiency.
- Verbal communication may hinder rather than help in certain real-time shared workspace contexts.

## 1 Introduction
**Context:**
- Theory of Mind (ToM) is a critical capability for understanding others' intentions, beliefs, and knowledge in human collaboration.
- As AI agents become more integrated into human workflows, the concept of Mutual Theory of Mind (MToM) emerges in Human-AI Teams (HATs).
- MToM involves interactive communication and strategy adjustment based on mutual understanding.
- Current research often focuses on AI's ToM in isolation or in static tasks, lacking real-time empirical studies in shared workspaces.
- There is a gap in understanding how LLM-driven ToM agents actually perform in dynamic, real-time collaborative settings.
**Goal:**
- To explore the MToM process in HATs through a mixed-design experiment.
- To evaluate the impact of an LLM-driven AI agent's ToM capability on team performance and human perceptions.
- To analyze the role of communication (verbal vs. non-verbal) in the collaboration process.
**Contributions:**
- Introduction of a real-time shared workspace task framework for evaluating HATs.
- Development of an LLM-driven AI agent equipped with both ToM and communication modules.
- Empirical findings on the trade-offs between ToM capabilities, communication burden, and team performance.
- Insights for designing future AI agents that collaborate effectively with humans in real-time.

## 2 Related Works
### 2.1. Theory of Mind in Human-AI Teams
- ToM allows agents to infer human mental states to predict behavior and intentions.
- Previous studies show ToM improves AI's ability to assist humans in complex tasks.
- Most existing work relies on simulated environments or offline datasets.
- Limited empirical evidence exists on how humans perceive AI's ToM in real-time interactions.
- The concept of MToM implies a bidirectional understanding, which is rarely studied in HATs.

### 2.2. Communication in Human-AI Teams
- Communication is essential for coordinating actions and sharing information in HATs.
- Verbal communication allows for nuanced expression but can be cognitively demanding.
- Non-verbal cues (e.g., gaze, gestures) are efficient but may lack explicit clarity.
- Prior research suggests that excessive communication can lead to information overload.
- The impact of communication modality on team performance in real-time tasks remains under-explored.

### 2.3. LLM-driven Theory of Mind AI Agents
- Large Language Models (LLMs) have shown promise in simulating ToM capabilities.
- LLMs can generate human-like inferences about beliefs and intentions.
- Integrating LLMs with reinforcement learning or policy networks enables dynamic decision-making.
- Challenges include latency, hallucination, and maintaining consistent mental state tracking.
- Few studies have validated LLM-driven ToM agents in real-world collaborative scenarios.

## 3 Cooperative Task Setup and Formalization in Shared Workspace
### 3.1. Layout Design
- The experimental setup involves a real-time shared digital workspace.
- The layout is designed to mimic a collaborative design or planning environment.
- Both human and AI agents interact with the same visual interface simultaneously.
- Spatial relationships between objects are critical for task execution.
- The workspace supports real-time updates and synchronized actions.

### 3.2. Task Design
- Participants engage in a cooperative task requiring joint decision-making.
- The task involves arranging objects or elements within the shared workspace.
- Success depends on coordinating actions to achieve a specific configuration.
- The task is designed to be complex enough to require communication and strategy adjustment.
- Time constraints are imposed to simulate real-time pressure.

### 3.3. Communication System Design
- The AI agent is equipped with a communication module capable of generating text.
- Humans can choose to communicate verbally with the agent or work silently.
- The communication interface is integrated into the shared workspace.
- Messages are displayed in real-time to both parties.
- The system supports bidirectional communication, allowing humans to send instructions and receive feedback.

### 3.4. Formulation
- The task is formalized as a partially observable Markov decision process (POMDP).
- The state space includes the positions and states of workspace elements.
- The action space includes movement, selection, and communication actions.
- The reward function is based on task completion time and accuracy.
- The AI agent's policy is driven by an LLM with ToM capabilities.

### 3.5. Objective Metrics
- Team performance is measured by task completion time.
- Accuracy of the final workspace configuration is recorded.
- Communication frequency and length are tracked.
- Error rates in action execution are monitored.
- Subjective metrics include user satisfaction and perceived workload.

## 4 LLM-driven Agent with Theory of Mind and Communication Module
### 4.1. Theory of Mind Module
- The ToM module infers the human's current mental state and intentions.
- It maintains a dynamic model of the human's knowledge and beliefs.
- The module updates its beliefs based on observed human actions and communications.
- It predicts the human's next likely actions to anticipate needs.
- The ToM module uses LLM prompts to generate mental state representations.

### 4.2. Policy Module
- The policy module determines the AI agent's actions based on the ToM model.
- It integrates task goals with the inferred human state.
- The policy balances efficiency with alignment to human preferences.
- It uses reinforcement learning principles to optimize actions.
- The module adapts its strategy based on real-time feedback.

### 4.3. Communication Module
- The communication module generates natural language responses.
- It decides when to communicate based on uncertainty or task criticality.
- Messages are tailored to the human's inferred knowledge level.
- The module supports both proactive and reactive communication.
- It ensures messages are concise and relevant to the current context.

### 4.4. Validation of The Agent Framework
- The agent framework is validated through preliminary simulations.
- Tests confirm the agent's ability to infer human intentions accurately.
- The communication module is evaluated for clarity and relevance.
- Latency issues are addressed to ensure real-time responsiveness.
- The framework is robust to variations in human behavior.

## 5 Methods
### 5.1. Procedure
- Participants are recruited and assigned to different experimental conditions.
- Conditions vary by the presence of ToM capability and communication mode.
- Participants complete a training phase to familiarize themselves with the task.
- The main experiment involves multiple rounds of the cooperative task.
- Post-experiment questionnaires are administered to gather subjective feedback.

### 5.2. Participants
- The study includes a diverse group of human participants.
- Participants have varying levels of experience with collaborative tools.
- Sample size is determined to ensure statistical power.
- Demographic data is collected for analysis.
- Participants are compensated for their time.

### 5.3. Data Analysis
- Quantitative data is analyzed using statistical tests (e.g., ANOVA).
- Qualitative data from questionnaires is coded and analyzed thematically.
- Communication logs are analyzed for patterns and frequency.
- Performance metrics are compared across conditions.
- Correlations between subjective perceptions and objective performance are examined.

## 6 Results
### 6.1. Team Performance
- No significant difference in task completion time between ToM and non-ToM conditions.
- Accuracy of workspace configuration remains consistent across conditions.
- Bidirectional communication condition shows lower performance than silent or unidirectional modes.
- Variability in performance is higher in conditions with verbal communication.
- The AI agent's ToM capability does not directly translate to faster task completion.

### 6.2. Team Collaboration Process
- Communication frequency is higher in the bidirectional condition.
- Messages in the bidirectional condition are often redundant or unclear.
- Humans spend more time reading and processing AI messages.
- The AI agent's actions are more aligned with human intentions in the ToM condition.
- Misunderstandings occur less frequently in the ToM condition despite no performance gain.

### 6.3. Human Preference and Perceptions of AI Agents
- Participants report higher trust in the AI agent with ToM capabilities.
- Humans feel more understood by the ToM agent compared to the non-ToM agent.
- Verbal communication is perceived as a significant cognitive burden.
- Participants prefer concise, non-verbal cues over lengthy text messages.
- Overall satisfaction is higher in conditions with lower communication load.

## 7 Discussion
### 7.1. Team Performance in Real-time Shared Workspace Task
- The lack of performance improvement suggests ToM alone is insufficient for efficiency gains.
- Real-time constraints may limit the effectiveness of complex ToM inferences.
- The shared workspace dynamics introduce complexities not captured in static tasks.
- Performance is more influenced by communication efficiency than ToM capability.
- Future work should explore hybrid communication strategies.

### 7.2. How Do Humans Understand The Agent in MToM Process?
- Humans rely on the agent's actions and brief messages to infer its state.
- ToM capabilities help humans predict the agent's behavior more accurately.
- Consistency in the agent's behavior builds trust and understanding.
- Ambiguity in messages can lead to confusion despite ToM capabilities.
- Understanding is enhanced when the agent's actions align with its stated intentions.

### 7.3. How Do Humans Perceive The Agent’s ToM Capability?
- Participants explicitly recognize the agent's ability to understand their needs.
- The feeling of being understood is a key factor in positive perception.
- ToM capabilities are valued for their transparency and predictability.
- However, the benefits are offset by the burden of processing verbal communication.
- Perception of ToM is distinct from the actual impact on task performance.

### 7.4. Challenges of LLM-driven AI Agent in HATs
- Latency in LLM inference can disrupt real-time collaboration.
- Hallucinations in ToM inferences can lead to incorrect actions.
- Verbal communication can overwhelm users in high-pressure environments.
- Balancing proactive assistance with user autonomy is difficult.
- Ensuring consistent mental state tracking over long interactions is challenging.

### 7.5. Limitations and Future Works
- The study is limited to a specific task type and workspace layout.
- Sample size and participant diversity may affect generalizability.
- Long-term effects of ToM capabilities are not explored.
- Future work should investigate multimodal communication (e.g., gaze, gestures).
- Studies with more complex, open-ended tasks are needed.
- Integration of ToM with other AI capabilities (e.g., planning) should be explored.

## 8 Conclusion
- The study provides empirical insights into MToM in real-time HATs.
- ToM capabilities enhance human understanding and trust but not necessarily performance.
- Verbal communication can be detrimental to efficiency in shared workspace tasks.
- Designing AI agents requires careful consideration of communication modalities.
- Future AI agents should prioritize concise, non-verbal interactions.
- The findings guide the development of more effective human-AI collaboration systems.