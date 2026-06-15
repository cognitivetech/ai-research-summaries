# Identifying Non-Replicable Social Science Studies with Language Models

**Source**: [arXiv:2503.10671](https://arxiv.org/html/2503.10671)

**Authors:** Denitsa Saynova, Kajsa Hansson, Bastiaan Bruinsma, Annika Fredén, Moa Johansson

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Methods](#2-methods)
- [2.1 Data](#21-data)
- [2.2 Sampling](#22-sampling)
- [2.3 Temperature](#23-temperature)
- [3 Results](#3-results)
- [3.1 Temperature](#31-temperature)
- [4 Related Work](#4-related-work)
- [5 Discussion and Conclusions](#5-discussion-and-conclusions)
- [Limitations](#limitations)
- [6 Ethical Considerations](#6-ethical-considerations)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [Appendix A Replicability of results](#appendix-a-replicability-of-results)
- [Appendix B Summary of studies](#appendix-b-summary-of-studies)
- [Appendix C Statistical tests used in Many Labs](#appendix-c-statistical-tests-used-in-many-labs)
- [Appendix D Histograms of answer responses](#appendix-d-histograms-of-answer-responses)
- [Appendix E Effect size with temperature](#appendix-e-effect-size-with-temperature)

## Abstract
**[Key point]**: The study investigates the capability of Large Language Models (LLMs) to predict the replicability of behavioral social science studies by generating synthetic participant responses.
**Main contribution**: A novel framework for using instruction-tuned LLMs to simulate study outcomes and estimate effect sizes, comparing these predictions against actual human replication data.
**Key results:**
- Evaluated on a dataset of 14 previously replicated studies (9 successful, 5 unsuccessful).
- Tested both open-source models (Llama 3 8B, Qwen 2 7B, Mistral 7B) and proprietary models (GPT-4o).
- Achieved F1 scores of up to 77% with Mistral 7B for discriminating replicable vs. non-replicable findings.
- GPT-4o and Llama 3 8B achieved F1 scores of 67%.
- Qwen 2 7B achieved an F1 score of 55%.
- Found that sampling temperature significantly impacts effect size calculations, with low variance leading to biased estimates.
**Conclusion:**
- LLMs show significant potential as tools for indicating study replicability.
- The method offers a scalable alternative to traditional human replication efforts.
- Careful calibration of generation parameters is required to avoid statistical bias.

## 1 Introduction
**Context:**
- Social science research faces a replication crisis, with many published findings failing to reproduce.
- Traditional replication efforts are resource-intensive, time-consuming, and expensive.
- There is a growing need for efficient methods to identify potentially non-replicable studies before they are widely accepted or acted upon.
- Large Language Models have demonstrated proficiency in simulating human behavior and generating realistic text data.
- Previous work has explored LLMs for various scientific tasks, but their application to predicting replication outcomes remains under-explored.
**Goal:**
- To determine if instruction-tuned LLMs can accurately discriminate between replicable and non-replicable social science studies.
- To evaluate the performance of various open-source and proprietary LLMs on this specific task.
- To analyze how hyperparameters, specifically sampling temperature, affect the statistical validity of the generated synthetic data.
**Contributions:**
- Introduction of a pipeline for generating synthetic participant responses using LLMs.
- Empirical evaluation of multiple LLM architectures on a curated dataset of replication studies.
- Analysis of the relationship between generation temperature and effect size estimation bias.
- Demonstration that LLMs can serve as a preliminary filter for replication feasibility.

## 2 Methods
**Context:**
- The methodology involves simulating participant responses to replicate the statistical power and outcome of original studies.
- The approach relies on instruction-tuning to guide the LLM to act as a participant in specific experimental contexts.
**Goal:**
- To establish a rigorous protocol for generating synthetic data that mirrors human responses.
- To define the metrics for comparing synthetic results against ground-truth replication outcomes.
**Contributions:**
- Detailed description of the data selection process.
- Explanation of the sampling strategy for generating responses.
- Definition of the temperature parameter's role in controlling response variance.

### 2.1 Data
**Context:**
- The dataset consists of studies from the "Many Labs" project, a large-scale collaboration aimed at replicating social psychology findings.
- The selection criteria focused on studies with available original data and clear replication outcomes.
**Goal:**
- To provide a balanced and representative dataset for evaluating LLM performance.
**Contributions:**
- Selection of 14 studies with known replication statuses.
- Classification of studies into "successful" (replicated) and "unsuccessful" (non-replicated) categories.
- Breakdown of the dataset: 9 successful replications and 5 unsuccessful replications.
- Inclusion of diverse behavioral tasks to test generalizability.

### 2.2 Sampling
**Context:**
- LLMs are prompted to generate responses as if they were participants in the original studies.
- The prompts include detailed instructions about the study context, tasks, and response formats.
**Goal:**
- To generate a sufficient number of synthetic responses to estimate statistical effects.
- To ensure the synthetic responses are contextually appropriate and varied.
**Contributions:**
- Use of instruction-tuned models to enhance adherence to study protocols.
- Generation of synthetic samples for each study condition.
- Aggregation of synthetic responses to calculate effect sizes.
- Comparison of synthetic effect sizes with original study effect sizes and replication results.

### 2.3 Temperature
**Context:**
- Sampling temperature controls the randomness of the LLM's output.
- Lower temperatures produce more deterministic and consistent responses.
- Higher temperatures introduce more variability and creativity.
**Goal:**
- To investigate how temperature settings influence the statistical properties of the generated data.
- To identify optimal temperature settings for accurate effect size estimation.
**Contributions:**
- Systematic variation of temperature during the sampling process.
- Analysis of the impact of temperature on the variance of synthetic responses.
- Preliminary findings suggesting that temperature is a critical hyperparameter for this task.

## 3 Results
**Context:**
- The results present the performance of different LLMs in predicting replication outcomes.
- The analysis includes both classification metrics (F1 score) and statistical effect size comparisons.
**Goal:**
- To quantify the accuracy of LLMs in distinguishing replicable from non-replicable studies.
- To assess the reliability of synthetic effect sizes compared to human replication data.
**Contributions:**
- Presentation of F1 scores for each model.
- Comparison of model performances across different study types.
- Analysis of the correlation between synthetic and actual replication outcomes.

### 3.1 Temperature
**Context:**
- This section details the specific findings regarding the effect of sampling temperature.
- It highlights the trade-off between response consistency and statistical validity.
**Goal:**
- To demonstrate the bias introduced by low-variance sampling.
- To provide recommendations for temperature selection in future work.
**Contributions:**
- Observation that low temperature leads to biased effect estimates due to reduced variance.
- Identification of an optimal temperature range that balances consistency and variance.
- Evidence that ignoring temperature effects can lead to incorrect conclusions about replicability.
- Suggestion that temperature calibration is essential for valid statistical inference using LLMs.

## 4 Related Work
**Context:**
- The paper situates its contribution within the broader literature on replication crises and AI in science.
- It reviews existing methods for assessing replicability and prior uses of LLMs in social science.
**Goal:**
- To highlight the novelty of using LLMs for synthetic replication.
- To distinguish the current work from previous attempts at automating replication assessment.
**Contributions:**
- Review of traditional replication efforts and their limitations.
- Discussion of prior work on LLMs simulating human behavior.
- Identification of gaps in current literature regarding the statistical validity of synthetic data.
- Comparison with other computational methods for predicting replication outcomes.

## 5 Discussion and Conclusions
**Context:**
- The discussion interprets the results in the context of the replication crisis.
- It addresses the implications of using LLMs as a tool for scientific validation.
**Goal:**
- To summarize the key findings and their significance.
- To discuss the practical applications of the proposed method.
**Contributions:**
- Confirmation that LLMs can effectively indicate replicability with high accuracy.
- Recognition of the potential for LLMs to reduce the cost and time of replication efforts.
- Emphasis on the importance of careful parameter tuning (temperature) to avoid bias.
- Proposal for LLMs to be used as a preliminary screening tool in research pipelines.

## Limitations
**Context:**
- The study acknowledges constraints in its design and scope.
- It identifies areas where the current approach may fall short.
**Goal:**
- To provide a balanced view of the method's capabilities and weaknesses.
- To guide future research directions.
**Contributions:**
- Small sample size of studies (14) limits the generalizability of the results.
- Dependence on the quality and detail of the original study descriptions.
- Potential bias in LLM training data affecting the realism of synthetic responses.
- Inability to capture complex, nuanced human psychological states fully.
- Focus on binary replication outcomes (success/failure) rather than effect size magnitude precision.
- Lack of evaluation on studies with complex experimental designs or longitudinal data.

## 6 Ethical Considerations
**Context:**
- The paper addresses the ethical implications of using AI to simulate human participants.
- It discusses the potential for misuse of such technology.
**Goal:**
- To ensure responsible use of the proposed method.
- To highlight ethical risks associated with synthetic data generation.
**Contributions:**
- Discussion of the risk of LLMs reinforcing biases present in training data.
- Concerns about the authenticity of synthetic data in scientific reporting.
- Need for transparency when using LLM-generated data in research.
- Potential for misuse to fabricate evidence for non-replicable findings.
- Recommendation for clear labeling of synthetic data in future publications.

## Acknowledgements
**Context:**
- The authors thank contributors and funding bodies.
**Goal:**
- To recognize support received during the research.
**Contributions:**
- Thanks to the Many Labs collaboration for providing data.
- Acknowledgment of funding sources.
- Gratitude to reviewers and colleagues for feedback.

## References
**Context:**
- The paper cites relevant literature on replication, LLMs, and social science methodology.
**Goal:**
- To provide a comprehensive bibliography for further reading.
**Contributions:**
- List of academic papers, books, and online resources.
- Citations for the Many Labs studies.
- References for the LLM architectures used (Llama, Qwen, Mistral, GPT-4o).
- Key works on the replication crisis in social psychology.

## Appendix A Replicability of results
**Context:**
- This appendix provides additional details on the replicability of the synthetic data generation process.
**Goal:**
- To demonstrate the stability of the LLM outputs across multiple runs.
**Contributions:**
- Tables showing consistency of results across different seeds.
- Analysis of variance in synthetic responses over repeated generations.
- Confirmation that the method produces stable estimates under controlled conditions.

## Appendix B Summary of studies
**Context:**
- This appendix lists the 14 studies included in the dataset.
**Goal:**
- To provide transparency regarding the data used.
**Contributions:**
- Detailed description of each study (title, authors, original effect size, replication status).
- Information on the sample size and methodology of the original studies.
- Classification of each study as successful or unsuccessful replication.

## Appendix C Statistical tests used in Many Labs
**Context:**
- This appendix details the statistical methods employed in the original Many Labs replications.
**Goal:**
- To ensure comparability between the original statistical analyses and the LLM-based estimates.
**Contributions:**
- List of statistical tests used (e.g., t-tests, ANOVA).
- Description of how effect sizes were calculated in the original studies.
- Justification for the choice of statistical metrics.

## Appendix D Histograms of answer responses
**Context:**
- This appendix includes visualizations of the synthetic response distributions.
**Goal:**
- To illustrate the distribution of LLM-generated answers.
**Contributions:**
- Histograms comparing synthetic responses to original human responses.
- Visual inspection of the similarity in distribution shapes.
- Identification of any notable deviations or biases in the synthetic data.

## Appendix E Effect size with temperature
**Context:**
- This appendix provides detailed data on how effect sizes vary with temperature.
**Goal:**
- To offer granular evidence for the temperature analysis in the main text.
**Contributions:**
- Tables of effect sizes calculated at different temperature settings.
- Graphs plotting effect size bias against temperature.
- Statistical analysis of the relationship between temperature and estimation accuracy.