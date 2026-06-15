# CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of Large Language Models in Mental Health Question Answering

**Source**: [arXiv:2506.08584](https://arxiv.org/html/2506.08584)

**Authors:** Yahan Li, Jifan Yao, John Bosco S. Bunyi, Adam C. Frank, Angel Hsing-Chi Hwang, Ruishan Liu

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 CounselBench-EVAL: Expert Evaluation Benchmark](#3-counselbench-eval-expert-evaluation-benchmark)
- [4 CounselBench-Adv: Adversarial Benchmarking](#4-counselbench-adv-adversarial-benchmarking)
- [5 Experimental Setup](#5-experimental-setup)
- [6 Results and Analysis](#6-results-and-analysis)
- [7 Discussion](#7-discussion)
- [8 Conclusion](#8-conclusion)

## Abstract
**[Key point]**: Existing medical QA benchmarks fail to address the complexity of open-ended mental health queries, which require balancing clinical safety with emotional sensitivity.
**Main contribution**: The introduction of CounselBench, a dual-component benchmark comprising expert evaluations (CounselBench-EVAL) and adversarial stress tests (CounselBench-Adv) for LLMs in mental health contexts.
**Key results:**
- LLMs (GPT-4, LLaMA 3, Gemini) achieve high scores on general dimensions but suffer from unconstructive feedback, overgeneralization, and lack of personalization.
- Significant safety risks are present, particularly the provision of unauthorized medical advice.
- LLM-based judges systematically overrate model responses and miss safety violations that human experts identify.
- Adversarial testing reveals consistent, model-specific failure patterns when probing for specific clinical issues.
**Conclusion:**
- CounselBench provides a clinically grounded framework for rigorous benchmarking.
- Current LLMs are not yet safe or effective enough for direct mental health support without significant improvements in safety and personalization.

## 1 Introduction
**Context:**
- Medical QA benchmarks traditionally focus on multiple-choice questions or factual retrieval tasks.
- These formats do not reflect the reality of patient interactions in mental health settings.
- Mental health queries are often open-ended, mixing symptoms, treatment concerns, and emotional needs.
- Effective responses require a delicate balance of clinical caution, empathy, and contextual sensitivity.
- Current benchmarks lack the depth to evaluate these nuanced requirements.
**Goal:**
- To create a comprehensive benchmark that evaluates LLMs on realistic, open-ended mental health question answering.
- To stress-test models against specific failure modes and safety risks.
- To provide a framework for future development of safer and more effective mental health AI.
**Contributions:**
- Development of CounselBench-EVAL: A dataset of 2,000 expert evaluations of LLM and human therapist responses.
- Development of CounselBench-Adv: An adversarial dataset of 120 expert-authored questions designed to trigger specific model failures.
- Comprehensive evaluation of 9 LLMs across both benchmarks.
- Analysis of the discrepancy between LLM judges and human expert judges.
- Identification of recurring failure modes in current LLMs.

## 2 Related Work
**Context:**
- Prior work in medical QA has largely relied on standardized datasets like MedQA or PubMedQA.
- These datasets focus on diagnostic accuracy and factual knowledge.
- Mental health AI research has explored chatbot interventions but lacks standardized, rigorous evaluation frameworks.
- Existing benchmarks for mental health are often small-scale or lack expert validation.
**Gap:**
- Lack of large-scale, expert-validated benchmarks for open-ended mental health QA.
- Insufficient focus on safety and ethical considerations in existing evaluations.
- Over-reliance on automated metrics or non-expert human raters.
**Comparison:**
- CounselBench differs by focusing on open-ended responses and clinical nuance.
- It includes both evaluation and adversarial components.
- It leverages a large panel of 100 mental health professionals for ground truth.

## 3 CounselBench-EVAL: Expert Evaluation Benchmark
**Context:**
- The first component of CounselBench is designed to evaluate the quality of responses to real patient questions.
- It aims to capture the multidimensional nature of mental health counseling.
**Data Collection:**
- Source data: Patient questions from the public forum CounselChat.
- Expert panel: 100 mental health professionals with diverse specializations.
- Annotation process: Experts rated responses across six clinically grounded dimensions.
**Evaluation Dimensions:**
- Empathy: Ability to acknowledge and validate patient emotions.
- Clinical Accuracy: Correctness of information and advice.
- Safety: Adherence to safety guidelines and risk management.
- Personalization: Tailoring responses to the specific patient context.
- Clarity: Readability and structure of the response.
- Helpfulness: Actionable and constructive nature of the advice.
**Annotation Details:**
- Span-level annotations to identify specific strengths and weaknesses.
- Written rationales provided by experts to justify scores.
- Comparison of LLM responses (GPT-4, LLaMA 3, Gemini) against online human therapists.
**Key Findings from EVAL:**
- LLMs score highly on surface-level metrics like clarity and empathy.
- LLMs struggle with clinical accuracy and safety in complex cases.
- Human therapists outperform LLMs in personalization and contextual understanding.
- Recurring issues in LLMs include unconstructive feedback and overgeneralization.

## 4 CounselBench-Adv: Adversarial Benchmarking
**Context:**
- Standard evaluation may not reveal subtle or critical failure modes.
- Adversarial testing is necessary to probe specific vulnerabilities.
**Dataset Construction:**
- 120 expert-authored mental health questions.
- Questions designed to trigger specific model issues (e.g., self-harm, diagnosis, medication advice).
- Focus on edge cases and high-risk scenarios.
**Adversarial Strategy:**
- Questions are crafted to bypass standard safety filters.
- Designed to test the model's ability to handle ambiguity and urgency.
- Covers a wide range of mental health conditions and crisis situations.
**Evaluation Method:**
- Responses from 9 LLMs collected for each adversarial question.
- Expert evaluation focused on safety violations and failure modes.
- Analysis of model-specific patterns in responding to adversarial prompts.
**Key Findings from Adv:**
- LLMs frequently fail to recognize high-risk situations.
- Models often provide inappropriate or dangerous advice in crisis scenarios.
- Failure patterns are consistent across different model architectures.
- Safety filters are easily bypassed by sophisticated adversarial prompts.

## 5 Experimental Setup
**Context:**
- Detailed description of the models and evaluation methodology.
**Models Evaluated:**
- GPT-4: Proprietary model from OpenAI.
- LLaMA 3: Open-source model from Meta.
- Gemini: Proprietary model from Google.
- Additional models: Total of 9 LLMs evaluated in the study.
- Human Therapists: Baseline comparison from online counseling platforms.
**Evaluation Protocol:**
- CounselBench-EVAL: 2,000 instances evaluated by 100 experts.
- CounselBench-Adv: 120 instances evaluated by experts.
- Inter-annotator agreement measured to ensure quality of expert ratings.
- Statistical significance testing applied to compare model performances.
**Metrics:**
- Scores across the six dimensions (Empathy, Clinical Accuracy, Safety, etc.).
- Safety violation rates.
- Correlation between LLM judges and human experts.
**LLM Judge Analysis:**
- Comparison of automated LLM-based evaluation vs. human expert evaluation.
- Identification of biases in LLM judges.

## 6 Results and Analysis
**Context:**
- Presentation of quantitative and qualitative results from both benchmarks.
**CounselBench-EVAL Results:**
- LLMs achieve high scores on Empathy and Clarity.
- Significant gaps in Clinical Accuracy and Safety compared to human therapists.
- Overgeneralization is a common issue, leading to less personalized advice.
- Unconstructive feedback is frequent, especially in complex cases.
**Safety Analysis:**
- High frequency of unauthorized medical advice in LLM responses.
- LLMs often fail to provide crisis resources when needed.
- Safety risks are more prevalent in LLaMA 3 and Gemini compared to GPT-4.
**CounselBench-Adv Results:**
- Adversarial questions successfully trigger specific failure modes.
- Models show consistent patterns of failure in high-risk scenarios.
- GPT-4 performs better than open-source models in safety but still has significant flaws.
- All models struggle with nuanced ethical dilemmas.
**LLM Judge vs. Human Expert:**
- LLM judges systematically overrate model responses.
- LLM judges overlook safety concerns identified by human experts.
- Correlation between LLM judge scores and human expert scores is low for safety dimensions.
- Bias in LLM judges towards surface-level fluency rather than clinical validity.
**Model-Specific Patterns:**
- GPT-4: Better safety but still prone to overgeneralization.
- LLaMA 3: Higher risk of providing unauthorized medical advice.
- Gemini: Struggles with empathy and personalization.
- Open-source models: Generally lower performance across all dimensions.

## 7 Discussion
**Context:**
- Interpretation of results and implications for the field.
**Implications for Mental Health AI:**
- Current LLMs are not ready for direct deployment in mental health support.
- Significant improvements in safety and clinical accuracy are needed.
- Personalization is critical for effective mental health interventions.
**Limitations of Current Benchmarks:**
- Existing benchmarks do not capture the complexity of mental health QA.
- Over-reliance on automated evaluation is misleading.
- Need for expert-in-the-loop evaluation frameworks.
**Strengths of CounselBench:**
- Large-scale and expert-validated.
- Covers both evaluation and adversarial testing.
- Provides detailed, multidimensional analysis.
**Future Directions:**
- Development of more robust safety filters.
- Training LLMs on more diverse and high-quality mental health data.
- Integration of CounselBench into standard evaluation pipelines.
- Exploration of hybrid human-AI models for mental health support.
**Ethical Considerations:**
- Importance of transparency in AI mental health tools.
- Need for clear disclaimers and crisis resources.
- Responsibility of developers to ensure safety and efficacy.

## 8 Conclusion
**Context:**
- Summary of the paper's main contributions and final thoughts.
**Summary:**
- CounselBench addresses the gap in mental health QA benchmarking.
- It provides a rigorous framework for evaluating LLMs in realistic scenarios.
- Expert evaluation reveals significant safety and quality issues in current LLMs.
**Key Takeaways:**
- LLMs overestimate their own performance in mental health contexts.
- Safety is the most critical area for improvement.
- Adversarial testing is essential for identifying hidden vulnerabilities.
**Final Statement:**
- CounselBench establishes a new standard for benchmarking AI in mental health.
- It calls for greater caution and rigorous evaluation in the development of mental health AI.
- Future work must prioritize safety, accuracy, and personalization.