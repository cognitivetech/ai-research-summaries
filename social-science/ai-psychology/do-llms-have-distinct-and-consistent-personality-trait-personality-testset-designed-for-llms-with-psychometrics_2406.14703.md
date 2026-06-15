# Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics

**Source**: [arXiv:2406.14703](https://arxiv.org/html/2406.14703)

**Authors:** Seungbeen Lee, Seungwon Lim, Seungju Han, Giyeong Oh, Hyungjoo Chae, Jiwan Chung, Minju Kim, Beong-woo Kwak, Yeonsoo Lee, Dongha Lee, Jinyoung Yeo, Youngjae Yu

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Measuring Personality of LLM](#2-measuring-personality-of-llm)
- [3 TRAIT: Reliable and Valid LLM Personality Tests](#3-trait-reliable-and-valid-llm-personality-tests)
- [4 Assessing LLMs’ Personality with TRAIT](#4-assessing-llms-personality-with-trait)
- [5 Related Works](#5-related-works)
- [6 Conclusions](#6-conclusions)
- [7 Limitations](#7-limitations)
- [8 Ethical Considerations](#8-ethical-considerations)

## Abstract
**[Key point]**: The paper investigates whether Large Language Models (LLMs) possess distinct and consistent personalities, analogous to human personality traits, by introducing a new benchmark called TRAIT.
**Main contribution**: Introduction of TRAIT, a benchmark consisting of 8,000 multiple-choice questions designed to assess LLM personality using psychometrically validated frameworks (Big Five and Dark Triad) enhanced with real-world scenarios via the ATOMIC-10X knowledge graph.
**Key results:**
- TRAIT outperforms existing personality tests for LLMs in terms of reliability and validity metrics, including Content Validity, Internal Validity, Refusal Rate, and Reliability.
- LLMs exhibit distinct and consistent personality traits that are highly influenced by their training data, particularly the data used for alignment tuning.
- Current prompting techniques have limited effectiveness in eliciting extreme personality traits, such as high psychopathy or low conscientiousness, indicating a gap in current evaluation methods.
**Conclusion:**
- Personality assessment is a viable and necessary method for understanding LLM behavior.
- Future research must focus on improving prompting techniques to fully elicit the range of personality traits present in LLMs.

## 1 Introduction
**Context:**
- Recent advancements in LLMs have led to their widespread adaptation as conversational agents in various domains.
- Understanding the behavioral tendencies of these agents is crucial for safety, alignment, and user experience.
- Human personality tests are standard tools for analyzing human behavior, but their applicability to AI agents is under-explored.
- Existing methods for assessing LLM personality often lack psychometric rigor, reliability, and validity.
**Goal:**
- To determine if LLMs have distinct and consistent personalities similar to humans.
- To develop a robust, psychometrically validated benchmark for measuring LLM personality.
- To analyze how training processes, specifically alignment tuning, influence these personality traits.
**Contributions:**
- Proposal of TRAIT, a new benchmark for LLM personality assessment.
- Construction of TRAIT using Big Five Inventory (BFI) and Short Dark Triad (SD-3) frameworks.
- Enhancement of these frameworks with ATOMIC-10X knowledge graph to create diverse, real-world scenarios.
- Comprehensive auditing of TRAIT to ensure high reliability and validity.
- Empirical analysis of personality traits across various LLMs, revealing insights into the impact of alignment data and prompting limitations.

## 2 Measuring Personality of LLM
### 2.1 Big Five and Dark Triad
- The Big Five model (OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) is a widely accepted framework for human personality.
- The Dark Triad (Machiavellianism, Narcissism, Psychopathy) represents socially aversive personality traits.
- These frameworks provide the theoretical basis for TRAIT, ensuring alignment with established psychological science.
- Applying these to LLMs allows for a standardized comparison of behavioral tendencies.

### 2.2 Existing Self-assessment Personality Tests
- Previous works often rely on self-assessment questionnaires where LLMs answer questions about themselves.
- These methods suffer from several issues:
  - Lack of real-world context, leading to abstract and unreliable answers.
  - Susceptibility to social desirability bias, where LLMs answer in ways they perceive as "correct" or "safe."
  - Low reliability due to the static nature of the questions.
- Self-assessment fails to capture how LLMs behave in specific, nuanced situations.

### 2.3 Assessing the Quality of Personality Tests
- Validity and reliability are critical metrics for any psychological test.
- Content Validity: Does the test cover the full range of the personality trait?
- Internal Validity: Do the items within the test correlate with each other as expected?
- Reliability: Does the test produce consistent results across different runs or prompts?
- Refusal Rate: The percentage of questions the LLM refuses to answer, which indicates safety filters or lack of understanding.
- Existing benchmarks often score poorly on these metrics, particularly regarding real-world applicability and consistency.

## 3 TRAIT: Reliable and Valid LLM Personality Tests
### 3.1 Dataset Construction Pipeline
- TRAIT is constructed by combining two psychometrically validated human questionnaires: BFI and SD-3.
- The ATOMIC-10X knowledge graph is used to generate diverse, real-world scenarios for each question.
- This process transforms abstract personality questions into situational judgment tests.
- Each question is a multiple-choice format, asking the LLM to choose the most appropriate response in a given scenario.
- The pipeline ensures that the scenarios are varied and cover a wide spectrum of social interactions.

### 3.2 Auditing TRAIT
- TRAIT undergoes rigorous auditing to ensure psychometric quality.
- Auditing involves checking for:
  - Ambiguity in questions.
  - Bias in scenarios.
  - Consistency in scoring.
- The audit process confirms that TRAIT meets high standards for reliability and validity.
- Comparison with existing tests shows TRAIT’s superior performance in key metrics.

### 3.3 Diverse and Detailed Scenarios are Needed when Measuring LLM Personality
- Abstract questions lead to inconsistent and unreliable results.
- Detailed scenarios provide context that helps LLMs generate more natural and consistent responses.
- The use of ATOMIC-10X ensures that scenarios are grounded in common sense and real-world logic.
- This approach reduces the impact of superficial pattern matching by the LLM.

### 3.4 Multi-Choice Evaluation
- TRAIT uses a multiple-choice format to facilitate automated and consistent evaluation.
- The options are designed to represent different levels of personality traits.
- This format allows for the calculation of standardized scores for each trait.
- It also enables the analysis of refusal rates, as LLMs may refuse to answer certain scenarios.

## 4 Assessing LLMs’ Personality with TRAIT
### 4.1 Do LLMs have Distinct Personality?
- The study evaluates a wide range of LLMs using TRAIT.
- Results show that different LLMs exhibit distinct personality profiles.
- These profiles are consistent across multiple evaluations, indicating stability.
- The distinctness of personalities suggests that LLMs are not just random text generators but have learned consistent behavioral patterns.

### 4.2 Influence of alignment tuning for LLM personality.
- Alignment tuning (e.g., RLHF) significantly shapes the personality of LLMs.
- Models tuned for helpfulness and harmlessness tend to show higher Agreeableness and Conscientiousness.
- The specific data used for alignment tuning directly influences the resulting personality traits.
- This finding highlights the role of human feedback in shaping AI behavior.

### 4.3 Eliciting LLM’s Personality with Simple Prompting
- Simple prompting techniques are often insufficient to elicit extreme personality traits.
- LLMs tend to default to "safe" or "neutral" responses unless prompted carefully.
- Traits like high psychopathy or low conscientiousness are rarely elicited through standard prompts.
- This suggests that current evaluation methods may underestimate the range of LLM behaviors.

### 4.4 Intercorrelation in Traits
- The study analyzes the correlations between different personality traits in LLMs.
- Some traits show expected correlations (e.g., Agreeableness and Conscientiousness).
- Other correlations differ from human psychology, indicating unique AI-specific behavioral patterns.
- Understanding these intercorrelations is important for predicting LLM behavior in complex scenarios.

## 5 Related Works
- Review of existing literature on LLM personality assessment.
- Discussion of previous benchmarks and their limitations.
- Comparison of TRAIT with other psychometric tools applied to AI.
- Emphasis on the need for psychometrically validated tools in AI research.

## 6 Conclusions
- TRAIT provides a reliable and valid method for assessing LLM personality.
- LLMs do have distinct and consistent personalities, influenced by their training data.
- Alignment tuning plays a crucial role in shaping these personalities.
- Current prompting techniques need improvement to fully capture the range of LLM traits.
- Future work should focus on developing better elicitation methods and exploring the implications of AI personality.

## 7 Limitations
- TRAIT is limited to the traits covered by BFI and SD-3.
- The scenarios, while diverse, are generated from a knowledge graph and may not cover all possible real-world situations.
- The study focuses on multiple-choice responses, which may not capture the full nuance of LLM behavior.
- Results may vary across different versions of the same LLM.

## 8 Ethical Considerations
- The use of personality tests for AI raises ethical questions about anthropomorphism.
- There is a risk of misinterpreting LLM behaviors as human-like intentions.
- The benchmark is designed to improve safety and alignment, not to manipulate users.
- Care must be taken in applying these results to real-world AI systems to avoid unintended consequences.