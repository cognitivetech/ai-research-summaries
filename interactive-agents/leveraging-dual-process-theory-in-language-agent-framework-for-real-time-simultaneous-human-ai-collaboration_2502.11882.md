# Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration

**Source**: [arXiv:2502.11882](https://arxiv.org/html/2502.11882)

**Authors:** Shao Zhang, Xihuai Wang, Wenhao Zhang, Chaoran Li, Junru Song, Tingyu Li, Lin Qiu, Xuezhi Cao, Xunliang Cai, Wen Yao, Weinan Zhang, Xinbing Wang, Ying Wen

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Works](#2-related-works)
- [3 Why We Need Dual Process Theory?](#3-why-we-need-dual-process-theory)
- [4 DPT-Agent Framework](#4-dpt-agent-framework)
- [5 Experimental Design](#5-experimental-design)
- [6 Results](#6-results)
- [7 Discussion and Future Works](#7-discussion-and-future-works)
- [8 Conclusion](#8-conclusion)
- [Limitations](#limitations)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Abstract
**[Key point]**:
- Current LLM-based agents excel in turn-by-turn collaboration but fail in real-time simultaneous tasks due to latency and strategy inference challenges.
- The paper validates the necessity of Dual Process Theory (DPT) for handling variable human strategies in real-time settings.
- Introduces DPT-Agent, a framework integrating fast System 1 and slow System 2 processes.
- System 1 uses Finite-state Machines (FSM) and code-as-policy for speed and control.
- System 2 uses Theory of Mind (ToM) and asynchronous reflection for reasoning and intention inference.

**Main contribution**:
- Proposal of DPT-Agent, the first language agent framework to achieve successful real-time simultaneous human-AI collaboration autonomously.
- Demonstration that DPT helps LLMs convert slow reasoning into executable actions effectively.
- Significant performance improvements over mainstream LLM-based frameworks in both rule-based and human-collaborator settings.

**Key results:**
- DPT-Agent outperforms independent System 1 and System 2 methods in real-time tasks.
- Effective handling of variable human strategies through ToM integration.
- Successful conversion of correct slow thinking into executable actions.

**Conclusion:**
- DPT is essential for real-time simultaneous collaboration.
- The proposed framework bridges the gap between intuitive fast responses and deliberate slow reasoning.
- Code is publicly available for reproducibility.

## 1 Introduction
**Context:**
- Large Language Models (LLMs) have shown remarkable capabilities in various AI tasks.
- Traditional human-AI collaboration is often turn-by-turn, allowing time for LLM processing.
- Real-time simultaneous collaboration requires immediate, continuous interaction without explicit instructions.
- Latency issues in LLM inference hinder real-time responsiveness.
- Inferring variable human strategies in dynamic environments is a significant challenge for autonomous agents.

**Goal:**
- To develop a language agent framework capable of real-time simultaneous human-AI collaboration.
- To address the limitations of current agents in handling latency and strategy inference.
- To leverage Dual Process Theory to balance speed and reasoning depth.

**Contributions:**
- Identification of the gap in current LLM agents regarding real-time simultaneous tasks.
- Validation of Dual Process Theory's necessity through experiments with independent System 1 and System 2 methods.
- Development of DPT-Agent framework.
- Integration of System 1 (FSM, code-as-policy) and System 2 (ToM, asynchronous reflection).
- Comprehensive evaluation against rule-based agents and human collaborators.
- Publication of code and detailed implementation guidelines.

## 2 Related Works
**Context:**
- Review of existing literature on human-AI collaboration.
- Analysis of turn-by-turn vs. simultaneous collaboration paradigms.
- Examination of latency mitigation strategies in LLM applications.

**Key Areas:**
- **Turn-by-turn Collaboration:**
    - Existing frameworks rely on explicit instructions and pauses.
    - Suitable for complex reasoning but unsuitable for real-time dynamics.
- **Simultaneous Collaboration:**
    - Requires low-latency decision-making.
    - Current methods struggle with adaptability to human behavior.
- **Dual Process Theory in AI:**
    - System 1: Fast, intuitive, heuristic-based.
    - System 2: Slow, deliberate, analytical.
    - Previous attempts to combine them often lack integration or real-time applicability.
- **Theory of Mind (ToM) in Agents:**
    - Used for inferring intentions and beliefs of others.
    - Critical for predicting human moves in collaborative tasks.
- **Finite-state Machines (FSM) in RL:**
    - Used for structured policy representation.
    - Provides interpretability and controllability.

**Gap Identification:**
- Lack of frameworks that effectively combine fast intuitive responses with slow deliberate reasoning in real-time simultaneous settings.
- Insufficient handling of variable human strategies without explicit feedback.

## 3 Why We Need Dual Process Theory?
**Context:**
- Experimental validation of why single-process approaches fail in real-time tasks.
- Comparison of independent System 1 and System 2 implementations.

**Findings:**
- **Independent System 1 Limitations:**
    - Fast but lacks depth in strategy inference.
    - Prone to errors in complex, dynamic scenarios.
    - Cannot adapt to novel human strategies effectively.
- **Independent System 2 Limitations:**
    - High latency due to complex reasoning.
    - Unsuitable for real-time interaction where immediate action is required.
    - May miss critical timing windows for collaboration.
- **Necessity of Integration:**
    - Real-time tasks require both speed (System 1) and adaptability (System 2).
    - System 1 handles routine, low-latency decisions.
    - System 2 handles high-stakes, strategic reasoning when time permits or asynchronously.
- **Validation:**
    - Experiments confirm that neither pure System 1 nor pure System 2 suffices.
    - Hybrid approach is necessary for robust performance.

## 4 DPT-Agent Framework
**Context:**
- Detailed description of the proposed DPT-Agent architecture.
- Integration of System 1 and System 2 components.

**4.1 System 2: Deliberate and Analytical Reasoning**
**Purpose:**
- To infer human intentions and perform high-level strategic planning.
- To handle complex reasoning tasks that require time.

**Components:**
- **Theory of Mind (ToM):**
    - Models the human collaborator's beliefs, goals, and likely actions.
    - Updates beliefs based on observed human behavior.
    - Predicts future human moves to anticipate needs.
- **Asynchronous Reflection:**
    - Runs in parallel or after action execution.
    - Analyzes past interactions to refine future strategies.
    - Does not block real-time action selection.
- **Reasoning Process:**
    - Generates hypotheses about human intent.
    - Evaluates potential outcomes of different actions.
    - Outputs high-level goals or policy adjustments for System 1.

**4.2 System 1: Fast and Intuitive Decision Making**
**Purpose:**
- To execute immediate actions based on current state and System 2 guidance.
- To ensure low-latency responses required for real-time interaction.

**Components:**
- **Finite-state Machine (FSM):**
    - Defines discrete states and transitions for agent behavior.
    - Ensures structured and interpretable decision-making.
    - Maps System 2's high-level goals to specific actions.
- **Code-as-Policy:**
    - Represents policies as executable code.
    - Allows for precise control and modification.
    - Facilitates rapid execution of predefined behaviors.
- **Integration Mechanism:**
    - System 2 updates FSM parameters or triggers state transitions.
    - System 1 executes actions based on current FSM state.
    - Continuous feedback loop between System 1 and System 2.

**Key Features:**
- **Controllability:**
    - FSM provides clear boundaries for agent behavior.
    - Code-as-policy allows for easy debugging and adjustment.
- **Efficiency:**
    - System 1 handles routine tasks instantly.
    - System 2 handles complex tasks asynchronously.
- **Adaptability:**
    - ToM enables adaptation to variable human strategies.
    - Asynchronous reflection allows learning from past interactions.

## 5 Experimental Design
**Context:**
- Description of the environment and setup used for evaluation.
- Justification for choosing the Overcooked Challenge.

**5.1 Overcooked Challenge for Real-time Simultaneous Human-AI Collaboration**
**Environment Details:**
- **Overcooked:**
    - A cooperative cooking game requiring coordination.
    - Agents and humans must work together to prepare meals.
    - Requires real-time interaction and strategy adaptation.
- **Challenges:**
    - Latency sensitivity: Actions must be taken quickly.
    - Complexity: Multiple ingredients, steps, and potential obstacles.
    - Human variability: Human players have different styles and speeds.

**5.2 Experimental Setup**
**Participants:**
- **Rule-based Agents:**
    - Baseline agents with predefined behaviors.
    - Used to establish performance lower bounds.
- **LLM-based Frameworks:**
    - Act, ReAct, and Reflexion frameworks.
    - Represent mainstream approaches to LLM agent reasoning.
- **Human Collaborators:**
    - Real humans interacting with DPT-Agent.
    - Tests real-world applicability and robustness.

**Metrics:**
- **Performance Metrics:**
    - Score: Number of meals prepared.
    - Efficiency: Time taken to complete tasks.
    - Coordination: Measures of joint action success.
- **Latency Metrics:**
    - Response time of the agent.
    - Impact of latency on task success.

**Implementation Details:**
- **Models Used:**
    - Various LLMs for System 2 reasoning.
    - Specific configurations for FSM and code-as-policy.
- **Hyperparameters:**
    - Settings for ToM inference.
    - Parameters for asynchronous reflection.

## 6 Results
**Context:**
- Presentation of experimental outcomes.
- Comparison of DPT-Agent with baselines.

**6.1 Capability in Real-time Task**
**Findings:**
- **Latency Handling:**
    - DPT-Agent maintains low latency due to System 1.
    - Outperforms pure System 2 methods in speed.
- **Task Success:**
    - Higher success rate in real-time tasks compared to baselines.
    - Effective handling of dynamic changes in the environment.

**6.2 Capability in Simultaneous Collaboration**
**Findings:**
- **Coordination:**
    - Improved coordination with rule-based agents.
    - Better alignment with human strategies than independent systems.
- **Adaptability:**
    - Successfully adapts to different human playstyles.
    - ToM component effectively predicts human moves.

**6.3 Experiments with Real Humans**
**Findings:**
- **Human-Agent Interaction:**
    - Positive feedback from human collaborators.
    - DPT-Agent is perceived as a helpful and responsive partner.
- **Performance:**
    - Significant improvement in meal preparation scores.
    - Reduced frustration due to agent responsiveness.
- **Robustness:**
    - Consistent performance across different human participants.
    - Ability to handle unexpected human actions.

**Comparison with Baselines:**
- **vs. Act/ReAct/Reflexion:**
    - DPT-Agent shows superior real-time performance.
    - Better handling of latency-sensitive scenarios.
- **vs. Rule-based Agents:**
    - Higher adaptability and strategic depth.
    - Outperforms in complex, variable environments.

## 7 Discussion and Future Works
**Context:**
- Interpretation of results and implications.
- Identification of limitations and future directions.

**Discussion:**
- **Effectiveness of DPT:**
    - Confirms the hypothesis that dual process theory is essential for real-time collaboration.
    - Highlights the complementary nature of System 1 and System 2.
- **Role of ToM:**
    - ToM is critical for inferring human intentions.
    - Enables proactive rather than reactive collaboration.
- **Role of FSM:**
    - FSM provides necessary structure for System 1.
    - Ensures reliability and interpretability.

**Implications:**
- **For AI Research:**
    - Suggests new architectures for real-time AI agents.
    - Emphasizes the need for hybrid reasoning systems.
- **For Human-AI Interaction:**
    - Improves user experience in collaborative tasks.
    - Enables more natural and fluid interactions.

**Future Works:**
- **Scaling:**
    - Applying DPT-Agent to larger, more complex environments.
    - Testing with larger groups of humans.
- **Generalization:**
    - Extending to other domains beyond cooking.
    - Investigating transferability of learned strategies.
- **Refinement:**
    - Improving ToM accuracy.
    - Optimizing latency further.

## 8 Conclusion
**Context:**
- Summary of the paper's main achievements.
- Final thoughts on the significance of the work.

**Summary:**
- DPT-Agent is a novel framework for real-time simultaneous human-AI collaboration.
- Integrates System 1 (FSM, code-as-policy) and System 2 (ToM, reflection).
- Addresses latency and strategy inference challenges.

**Key Achievements:**
- First framework to achieve successful real-time simultaneous collaboration autonomously.
- Demonstrates significant improvements over existing LLM-based frameworks.
- Validates the necessity of Dual Process Theory in this context.

**Final Thoughts:**
- DPT-Agent paves the way for more capable and responsive AI collaborators.
- Highlights the importance of balancing speed and reasoning in AI design.
- Provides a foundation for future research in real-time human-AI interaction.

## Limitations
**Context:**
- Acknowledgment of the study's constraints.

**Limitations:**
- **Environment Specificity:**
    - Evaluated primarily in the Overcooked environment.
    - Generalizability to other domains needs further testing.
- **Human Variability:**
    - Limited number of human participants.
    - Results may vary with different human populations.
- **Computational Cost:**
    - System 2 reasoning may be computationally expensive.
    - Latency could increase with more complex reasoning tasks.
- **ToM Accuracy:**
    - ToM inference is not perfect.
    - Errors in intention prediction can lead to suboptimal actions.

## Acknowledgments
**Context:**
- Recognition of contributors and funding sources.

**Acknowledgments:**
- Thanks to contributors for their assistance.
- Recognition of funding agencies supporting the research.
- Appreciation for the open-source community and datasets used.

## References
**Context:**
- List of cited works.

**References:**
- Key papers on Dual Process Theory.
- Literature on LLM-based agents.
- Studies on human-AI collaboration.
- Works on Theory of Mind in AI.
- Technical reports on Overcooked and related environments.