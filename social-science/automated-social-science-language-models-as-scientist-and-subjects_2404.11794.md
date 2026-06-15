# Automated Social Science: Language Models as Scientist and Subjects

**Source**: [arXiv:2404.11794](https://arxiv.org/html/2404.11794)

**Authors:** Benjamin S. Manning, Kehang Zhu, John J. Horton

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Overview of the system](#2-overview-of-the-system)
- [3 Results of experiments](#3-results-of-experiments)
- [3.1 Bargaining over a mug](#31-bargaining-over-a-mug)
- [3.2 A bail hearing](#32-a-bail-hearing)
- [3.3 Interviewing for a job as a lawyer](#33-interviewing-for-a-job-as-a-lawyer)
- [3.4 An auction for a piece of art](#34-an-auction-for-a-piece-of-art)
- [4 LLM predictions for paths and points](#4-llm-predictions-for-paths-and-points)
- [4.1 Predicting y_i](#41-predicting-y_i)
- [4.2 Predicting β̂](#42-predicting-β̂)
- [5 Identifying causal structure ex-ante](#5-identifying-causal-structure-ex-ante)
- [5.1 Assuming causal structure from data](#51-assuming-causal-structure-from-data)
- [5.2 Searching for causal structure in data](#52-searching-for-causal-structure-in-data)
- [6 Conclusion](#6-conclusion)
- [6.1 Controlled experimentation at scale](#61-controlled-experimentation-at-scale)
- [6.2 Interactivity](#62-interactivity)
- [6.3 Replicability](#63-replicability)
- [6.4 Future research](#64-future-research)

## Abstract
**[Key point]**: The paper introduces a framework for automating the generation and testing of social scientific hypotheses using Large Language Models (LLMs) as both the researchers and the experimental subjects.
**Main contribution**: The core innovation is the use of Structural Causal Models (SCMs) to bridge the gap between abstract hypotheses and executable LLM-based agent simulations.
**Key results:**
- The system successfully generated and tested hypotheses in four distinct social scenarios: negotiation, bail hearing, job interview, and auction.
- Causal relationships were both proposed by the system and empirically tested within the simulation environment.
- Evidence was found for some hypothesized causal links, while others were refuted, demonstrating the system's ability to distinguish valid from invalid theories.
- The insights derived from these in silico simulations are not directly accessible to the LLM through simple prompt elicitation alone.
- When provided with the fitted SCM, the LLM can accurately predict the direction (signs) of causal effects but fails to predict their magnitudes accurately.
- In the auction scenario, simulation results aligned closely with established auction theory, whereas direct LLM predictions of clearing prices were inaccurate.
- Conditioning the LLM on the fitted SCM dramatically improved its predictive accuracy, suggesting the model possesses latent knowledge it cannot explicitly articulate without structural scaffolding.
**Conclusion:**
- LLMs contain more causal knowledge than they can immediately express.
- SCMs serve as a necessary interface to unlock and utilize this latent causal understanding for rigorous scientific inquiry.

## 1 Introduction
**Context:**
- Traditional social science relies on human subjects, field data, and manual experimental design, which are resource-intensive and difficult to scale.
- Recent advances in LLMs have enabled the creation of synthetic agents capable of simulating human-like behavior.
- However, using LLMs as subjects requires a rigorous methodological framework to ensure hypotheses are testable and results are interpretable.
- Existing methods often lack a formal language for specifying causal structures, leading to ambiguous experimental designs.
**Goal:**
- To present an automated approach for generating and testing social scientific hypotheses in silico.
- To demonstrate that LLMs can function as both the scientist (designing experiments) and the subject (participating in them).
- To validate this approach against established social science theories and empirical benchmarks.
**Contributions:**
- Introduction of a pipeline that uses SCMs to define hypotheses, construct agents, design experiments, and analyze data.
- Demonstration of the system across four diverse social interaction scenarios.
- Empirical evidence that LLMs hold implicit causal knowledge that can be extracted and validated through structural modeling.
- Analysis of the gap between what LLMs know (implicit) and what they can tell (explicit).

## 2 Overview of the system
**Context:**
- The system integrates LLMs with causal inference methodologies to create a closed-loop scientific process.
- It moves beyond simple role-playing by enforcing structural constraints on agent behavior and interaction.
**Core Components:**
- **Structural Causal Models (SCMs):** Used as the formal language for hypotheses. SCMs define variables, causal paths, and functional relationships.
- **Hypothesis Generation:** The LLM proposes potential causal structures based on social science theories or intuitive reasoning.
- **Agent Construction:** The SCM serves as a blueprint for building LLM-based agents. Each agent is programmed to adhere to the causal rules defined in the SCM.
- **Experimental Design:** The SCM dictates the experimental conditions, interventions, and data collection protocols.
- **Data Analysis:** The system runs simulations, collects data, and fits the SCM to the observed outcomes to test the validity of the proposed hypotheses.
**Workflow:**
- Step 1: Define a social scenario (e.g., negotiation).
- Step 2: LLM proposes an SCM representing a hypothesis about the scenario.
- Step 3: Agents are instantiated based on the SCM.
- Step 4: Simulation runs, generating data on agent interactions.
- Step 5: The SCM is fitted to the simulation data.
- Step 6: Statistical tests determine if the data supports the proposed causal structure.
- Step 7: If supported, the SCM becomes an object for prediction or further experimental planning.
**Key Feature:**
- The SCM acts as a "blueprint" that ensures consistency between the theoretical hypothesis and the simulated reality.
- It allows for counterfactual reasoning and intervention planning within the simulation.

## 3 Results of experiments
**Context:**
- The system was tested on four specific social science scenarios to validate its generalizability and accuracy.
- Each scenario involved distinct social dynamics, power structures, and economic incentives.
**3.1 Bargaining over a mug**
- **Scenario:** Two agents negotiate the price of a mug, one being the seller and the other the buyer.
- **Hypothesis:** The system tested hypotheses regarding the influence of anchor prices and agent endowments on the final settlement price.
- **Results:**
  - The simulation successfully replicated the anchoring effect, where initial offers significantly influenced the final price.
  - Causal paths from anchor prices to final settlement were statistically significant.
  - The system identified that the seller's reservation price had a stronger causal impact on the outcome than the buyer's valuation in certain conditions.
  - Evidence supported standard bargaining theory predictions within the simulated environment.
**3.2 A bail hearing**
- **Scenario:** A judge decides on bail amount based on defendant characteristics (e.g., risk level, community ties).
- **Hypothesis:** The system tested for biases in bail decisions based on demographic proxies and risk factors.
- **Results:**
  - The system detected causal links between specific defendant attributes and bail amounts.
  - It identified potential biases where certain proxies correlated with higher bail amounts, independent of actual risk.
  - The SCM allowed for the isolation of confounding variables, such as prior criminal history versus socioeconomic status.
  - The simulation revealed that without controlling for specific variables, the model might incorrectly attribute causality to demographic factors.
**3.3 Interviewing for a job as a lawyer**
- **Scenario:** A candidate interviews for a legal position, with the interviewer evaluating competence and fit.
- **Hypothesis:** The system explored how interview questions and candidate responses influence hiring decisions.
- **Results:**
  - The system modeled the causal chain from question type to candidate response quality to hiring probability.
  - It found that structured interview questions led to more predictable and fairer outcomes than unstructured ones.
  - The SCM highlighted the mediating role of "perceived competence" in the causal path between candidate responses and hiring.
  - The simulation allowed for testing the impact of interviewer bias by manipulating the interviewer's prior beliefs.
**3.4 An auction for a piece of art**
- **Scenario:** Multiple bidders participate in an auction for a piece of art, with varying valuations.
- **Hypothesis:** The system tested predictions from auction theory, specifically regarding clearing prices and bidder strategies.
- **Results:**
  - The in silico simulation results closely matched the theoretical predictions of auction theory (e.g., revenue equivalence).
  - The system accurately predicted the clearing price based on the distribution of bidder valuations.
  - Direct elicitation of the clearing price from the LLM without the SCM was inaccurate.
  - The SCM provided the necessary structure for the LLM to reason about the strategic interactions of bidders.
  - This scenario demonstrated the system's ability to validate complex economic theories in a synthetic environment.

## 4 LLM predictions for paths and points
**Context:**
- This section analyzes the LLM's ability to predict outcomes both with and without the fitted SCM.
- It distinguishes between predicting individual outcomes ($y_i$) and causal effects ($\hat{\beta}$).
**4.1 Predicting y_i**
- **Goal:** To assess if the LLM can predict the specific outcome for a given agent in a specific scenario.
- **Findings:**
  - The LLM is generally good at predicting the sign (direction) of the effect.
  - For example, it can predict that higher income leads to higher willingness to pay.
  - However, it struggles with the magnitude of the prediction.
  - The predicted values often deviate significantly from the actual simulated outcomes.
  - This suggests that while the LLM understands qualitative relationships, it lacks precise quantitative calibration.
**4.2 Predicting β̂**
- **Goal:** To assess if the LLM can predict the estimated causal coefficient ($\hat{\beta}$) from the fitted SCM.
- **Findings:**
  - When provided with the fitted SCM, the LLM can accurately predict the signs of the estimated effects.
  - It correctly identifies which variables have positive or negative causal impacts.
  - It fails to reliably predict the magnitudes of these coefficients.
  - The LLM's predictions are dramatically improved when it can condition on the fitted SCM.
  - This indicates that the SCM acts as a "reasoning scaffold" that helps the LLM access its latent causal knowledge.
  - Without the SCM, the LLM's predictions are noisy and often incorrect.
  - With the SCM, the LLM's predictions align closely with the simulation results.
**Implication:**
- The LLM "knows more than it can (immediately) tell."
- The SCM is essential for translating the LLM's implicit knowledge into explicit, accurate predictions.

## 5 Identifying causal structure ex-ante
**Context:**
- This section explores how the system can identify causal structures before running full simulations.
- It compares two methods: assuming structure from data and searching for structure in data.
**5.1 Assuming causal structure from data**
- **Method:** The LLM uses prior knowledge or existing literature to propose an SCM.
- **Process:**
  - The LLM generates a hypothesis based on social science theories.
  - The SCM is defined based on this hypothesis.
  - The system then tests this SCM against simulation data.
- **Outcome:**
  - This method relies on the LLM's ability to retrieve and apply relevant theoretical knowledge.
  - It is effective when strong theoretical priors exist.
  - It allows for rapid hypothesis generation and testing.
**5.2 Searching for causal structure in data**
- **Method:** The system uses causal discovery algorithms to infer the SCM from the simulation data.
- **Process:**
  - The LLM generates agents with minimal structural constraints.
  - The system runs simulations and collects data.
  - Causal discovery algorithms analyze the data to find the most likely causal structure.
- **Outcome:**
  - This method is useful for exploring novel scenarios where theory is lacking.
  - It can uncover unexpected causal relationships.
  - However, it is computationally more intensive and may be less reliable with small sample sizes.
  - The LLM can then be used to interpret the discovered structure.
**Comparison:**
- Assuming structure is faster and more theory-driven.
- Searching for structure is more exploratory and data-driven.
- Both methods complement each other in the automated social science pipeline.

## 6 Conclusion
**Context:**
- The paper concludes by summarizing the implications of the automated social science approach.
- It discusses the broader impact on social science research and future directions.
**6.1 Controlled experimentation at scale**
- The system enables large-scale experimentation that is impossible with human subjects.
- Researchers can test thousands of hypotheses in a fraction of the time and cost.
- It allows for precise control over variables and conditions, reducing noise and confounding.
**6.2 Interactivity**
- The agents in the simulation are interactive, allowing for dynamic social processes.
- This captures emergent phenomena that static models might miss.
- The LLMs can adapt their strategies based on the actions of other agents.
**6.3 Replicability**
- The in silico experiments are fully replicable.
- The same SCM and agent configurations will produce the same results.
- This addresses the replication crisis in social science by providing a transparent and reproducible experimental platform.
**6.4 Future research**
- **Scaling:** Applying the approach to more complex and larger-scale social systems.
- **Validation:** Comparing simulation results with real-world empirical data to validate the models.
- **Ethics:** Addressing the ethical implications of using LLMs to simulate human behavior.
- **Integration:** Integrating the system with other AI tools for automated theory generation and analysis.
- **Human-in-the-loop:** Exploring hybrid models where humans and LLMs collaborate in the scientific process.
**Final Thought:**
- The approach demonstrates that LLMs can be powerful tools for scientific discovery.
- By combining LLMs with structural causal models, we can unlock new avenues for understanding social phenomena.
- The key insight is that LLMs possess latent causal knowledge that can be accessed and utilized through proper structural scaffolding.