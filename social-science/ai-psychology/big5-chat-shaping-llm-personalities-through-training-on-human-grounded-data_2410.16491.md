# BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data

**Source**: [arXiv:2410.16491](https://arxiv.org/html/2410.16491)

**Authors:** Wenkai Li, Jiarui Liu, Andy Liu, Xuhui Zhou, Mona Diab, Maarten Sap

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Background](#2-background)
- [3 Methodology](#3-methodology)
- [4 Big5-Chat Dataset](#4-big5-chat-dataset)
- [5 Experiments](#5-experiments)
- [6 Conclusion](#6-conclusion)
- [7 Limitations](#7-limitations)
- [8 Ethical Concerns](#8-ethical-concerns)

## Abstract
**[Key point]**: The paper addresses the challenge of embedding realistic human personality traits into Large Language Models (LLMs) by moving beyond prompt-based methods to training-based approaches grounded in real human data.
**Main contribution**: Introduction of BIG5-CHAT, a large-scale dataset of 100,000 dialogues designed to ground models in how humans express personality, and the application of Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).
**Key results:**
- Models trained on BIG5-CHAT outperform prompt-based methods on personality assessments like BFI and IPIP-NEO.
- Trait correlations in trained models more closely match human data distributions.
- Specific personality traits (high conscientiousness, high agreeableness, low extraversion, low neuroticism) correlate with improved reasoning performance.
- First comprehensive study demonstrating training-based personality shaping via learning from real human behaviors.
**Conclusion:**
- Training-based methods offer a more valid and realistic way to shape LLM personalities compared to prompting.
- Personality traits have measurable impacts on cognitive performance in LLMs, mirroring psychological findings in humans.

## 1 Introduction
**Context:**
- LLMs are increasingly used in social and interactive applications where personality consistency is crucial.
- Current methods for inducing personality rely heavily on prompt engineering (e.g., system prompts describing traits).
- Prompt-based methods suffer from "realism" issues (lack of naturalness) and "validity" issues (inconsistent or superficial trait expression).
- Psychological research indicates that personality traits (Big Five) significantly influence human behavior and cognitive performance.
- There is a gap in understanding how to effectively embed these traits into LLMs using data-driven approaches rather than textual descriptions.

**Goal:**
- To develop a method for embedding realistic, stable, and valid personality traits into LLMs.
- To create a dataset that captures how humans naturally express personality in dialogue.
- To evaluate whether training-based alignment (SFT/DPO) is superior to prompting for personality induction.
- To investigate the relationship between induced personality traits and downstream reasoning capabilities.

**Contributions:**
- **BIG5-CHAT Dataset**: Creation of a 100,000-turn dialogue dataset grounded in human personality expressions.
- **Training Framework**: Application of SFT and DPO to align LLMs with human personality patterns.
- **Empirical Validation**: Comprehensive evaluation using standard psychological scales (BFI, IPIP-NEO) and reasoning benchmarks.
- **Insight on Cognition**: Discovery that specific personality configurations enhance reasoning performance in LLMs.

## 2 Background
**Context:**
- **The Big Five Personality Traits**: The standard psychological model includes Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN).
- **Personality in AI**: Previous work has focused on character generation and role-playing, often using static profiles.
- **Prompting vs. Training**: Prompting is flexible but inconsistent; training is stable but requires data.
- **Psycholinguistics**: Language use is a strong indicator of personality; lexical choices correlate with Big Five traits.
- **Limitations of Prior Work**:
    - Lack of large-scale, human-grounded dialogue data for personality.
    - Over-reliance on synthetic data or short text snippets rather than conversational context.
    - Insufficient evaluation of whether induced traits affect cognitive tasks.

**Key Concepts:**
- **Supervised Fine-Tuning (SFT)**: Training a model to mimic specific output distributions.
- **Direct Preference Optimization (DPO)**: Aligning models based on preference pairs without a separate reward model.
- **Validity**: The extent to which the model's output actually reflects the intended psychological construct.
- **Realism**: The naturalness and coherence of the personality expression in dialogue.

## 3 Methodology
**Context:**
- The core methodology involves generating high-quality personality-grounded data and using it to train models.
- Two main components: The Expert Generator (for data creation) and the Alignment Framework (for training).

### 3.1 DExperts Framework
**Context:**
- Uses the DExperts framework to control the personality of the generated dialogue.
- DExperts allows for dynamic steering of model outputs during generation.
- **Mechanism**:
    - Combines a base language model with a classifier that predicts personality traits.
    - Adjusts logits during generation to bias the output toward desired personality traits.
    - Ensures the generated text remains coherent while adhering to personality constraints.

### 3.2 Expert Generator Model Based on Social Media Posts
**Context:**
- To create the BIG5-CHAT dataset, an "Expert Generator" is needed to produce diverse, personality-specific dialogues.
- **Data Source**: Social media posts are used as the primary source for training the Expert Generator.
    - Social media provides natural, unscripted expressions of personality.
    - Posts are annotated with Big Five traits using existing classifiers.
- **Training Process**:
    - The Expert Generator is trained to mimic the linguistic style of users with specific Big Five profiles.
    - It generates synthetic dialogues that reflect these profiles.
    - The generator is evaluated to ensure it can steer personality accurately.

## 4 Big5-Chat Dataset
**Context:**
- BIG5-CHAT is the central resource of this work, designed to ground LLMs in human personality expression.
- It contains 100,000 dialogue turns, ensuring sufficient scale for effective training.

### 4.1 Dataset Construction
**Context:**
- **Step 1: Data Collection**: Aggregated social media posts with known personality labels.
- **Step 2: Dialogue Generation**: Used the Expert Generator to convert static posts into conversational turns.
    - Dialogues are structured to maintain personality consistency across turns.
    - Contextual diversity is ensured by varying topics and scenarios.
- **Step 3: Filtering and Cleaning**:
    - Removed low-quality or inconsistent dialogues.
    - Ensured ethical compliance and removed sensitive content.
- **Step 4: Annotation**: Each dialogue is labeled with the Big Five trait scores of the "speaker."

### 4.2 Dataset Statistics
**Context:**
- **Size**: 100,000 dialogue turns.
- **Distribution**: Balanced across the spectrum of Big Five traits.
- **Diversity**: Covers a wide range of topics, intents, and conversational styles.
- **Comparison**: Larger and more personality-diverse than previous datasets used for LLM personality training.
- **Linguistic Richness**: Analyzed for lexical diversity and syntactic complexity to ensure high quality.

### 4.3 Evaluating Personality-Steering of the Data Generator
**Context:**
- Before using the data for training, the Expert Generator's ability to produce personality-specific text was evaluated.
- **Method**:
    - Generated text with specific trait targets.
    - Used independent personality classifiers to measure the accuracy of the induced traits.
- **Results**:
    - The generator successfully steered personality traits with high accuracy.
    - Correlations between target and generated traits were strong.
    - Demonstrated that the data is valid for training purposes.

## 5 Experiments
**Context:**
- Experiments compare SFT and DPO methods against prompt-based baselines.
- Evaluations cover personality induction accuracy and downstream reasoning performance.

### 5.1 Experiment Setup
**Context:**
- **Models**: Used LLaMA-2 and LLaMA-3 as base models.
- **Baselines**:
    - **Prompting**: System prompts describing personality traits (e.g., "You are highly conscientious").
    - **Zero-shot/Few-shot**: Standard LLM outputs without personality induction.
- **Training Methods**:
    - **SFT**: Fine-tuned on BIG5-CHAT data.
    - **DPO**: Aligned using preference pairs derived from BIG5-CHAT.
- **Evaluation Metrics**:
    - **Personality Assessment**: BFI (Big Five Inventory) and IPIP-NEO scales.
    - **Reasoning**: Standard benchmarks (e.g., GSM8K, MMLU) to test cognitive impact.

### 5.2 Personality Trait Assessment Results
**Context:**
- **Primary Finding**: Training-based methods significantly outperform prompting.
- **Metrics**:
    - Correlation between model outputs and target Big Five traits.
    - Agreement with human-annotated personality labels.
- **Results**:
    - **SFT and DPO**: Showed high correlation with target traits (r > 0.7).
    - **Prompting**: Showed weak and inconsistent correlations (r < 0.4).
    - **Realism**: Human evaluators rated trained models as more natural and consistent.
    - **Validity**: Trained models' trait distributions matched human data distributions more closely.

### 5.3 Reasoning Evaluation Results
**Context:**
- Investigated whether personality induction affects reasoning capabilities.
- **Hypothesis**: Based on psychological literature, certain traits may enhance or hinder cognitive performance.
- **Key Findings**:
    - **High Conscientiousness**: Correlated with better performance on logical and mathematical reasoning tasks.
    - **High Agreeableness**: Correlated with improved performance on collaborative or nuanced reasoning tasks.
    - **Low Extraversion**: Correlated with better focus on complex, solitary reasoning tasks.
    - **Low Neuroticism**: Correlated with more stable performance under varying conditions.
- **Implication**: Personality shaping is not just about social interaction; it impacts cognitive processing in LLMs.
- **Comparison**: Prompted models did not show these cognitive benefits, suggesting the effect is due to deep learning of behavioral patterns, not just surface-level text generation.

## 6 Conclusion
**Context:**
- Summarizes the main achievements and implications of the work.
- **Summary**:
    - BIG5-CHAT provides a robust foundation for personality-aware LLMs.
    - Training-based methods (SFT/DPO) are superior to prompting for realistic personality induction.
    - Personality traits have measurable effects on reasoning, aligning with human psychology.
- **Implications**:
    - Enables creation of more natural and consistent AI companions.
    - Highlights the importance of data quality in shaping model behavior.
    - Suggests that personality engineering can be used to optimize LLMs for specific cognitive tasks.

## 7 Limitations
**Context:**
- Acknowledges the constraints and potential weaknesses of the study.
- **Data Limitations**:
    - Social media data may not represent all demographics equally.
    - Personality expression on social media may differ from face-to-face interaction.
- **Model Limitations**:
    - Experiments focused on specific base models (LLaMA); results may vary with other architectures.
    - Long-term stability of personality traits in extended conversations was not fully tested.
- **Evaluation Limitations**:
    - Psychological scales (BFI/IPIP-NEO) are self-reported and may not capture nuanced behavioral traits.
    - Reasoning benchmarks may not fully capture the complexity of human cognitive performance.

## 8 Ethical Concerns
**Context:**
- Discusses the potential risks and ethical considerations of personality-inducing LLMs.
- **Manipulation Risks**:
    - Personalized personalities could be used to manipulate user behavior or emotions.
    - Deception: Users may attribute human-like intent to the AI.
- **Bias and Fairness**:
    - Social media data may contain biases that are amplified in the trained models.
    - Personality traits are culturally dependent; the dataset may not generalize across cultures.
- **Mitigation Strategies**:
    - Transparency: Clearly disclosing that the AI has a programmed personality.
    - User Control: Allowing users to adjust or disable personality features.
    - Diverse Data: Ensuring the training data represents diverse populations.
- **Recommendation**:
    - Future work should focus on ethical guidelines and safety mechanisms for personality-aware AI.