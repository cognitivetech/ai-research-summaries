# InfiniPot: Efficiently Handling Long Input Contexts in Large Language Models (LLMs) on Memory-Constrained Devices

**Source**: [arXiv:2410.01518](https://arxiv.org/html/2410.01518)
Minsoo Kim1, Kyuhong Shim, Jungwook Choi, Simyung Chang

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related Work](#2-related-work)
- [3 InfiniPot: Infinite Context Processing on Memory-Constrained LLMs](#3-infinipot-infinite-context-processing-on-memory-constrained-llms)
  - [3.2 Continual Context Distillation (CCD)](#32-continual-context-distillation-ccd)
  - [3.3 Importance Measure from Past and Future](#33-importance-measure-from-past-and-future)
  - [3.4 Analysis of Representative and Novelty Scores](#34-analysis-of-representative-and-novelty-scores)
  - [3.5 Context-Reset Rotary Positional Embedding](#35-context-reset-rotary-positional-embedding)
- [4 Experiments](#4-experiments)
  - [4.2 Performance on LongBench](#42-performance-on-longbench)
  - [4.3 Performance on Needle In a Haystack](#43-performance-on-needle-in-a-haystack)
  - [4.4 InfiniPot Analysis](#44-infinipot-analysis)
  - [4.5 Efficiency Analysis: Memory and Speed Measurement](#45-efficiency-analysis-memory-and-speed-measurement)
- [5 Conclusion](#5-conclusion)
- [6 Limitations](#6-limitations)

## Abstract
**Background**
- Significant challenge for LLMs to manage long input contexts, especially on resource-limited devices
- Proposed solution: InfiniPot, a new KV cache control framework for efficient handling of extensive sequences within fixed memory constraints

**InfiniPot Framework**
- Introduced to enable pre-trained LLMs to manage long input contexts without requiring additional training
- Leverages Continual Context Distillation (CCD), an iterative process compressing and retaining essential information through novel importance metrics

**Benefits of InfiniPot**
- Significantly outperforms models trained for long contexts in various Natural Language Processing (NLP) tasks
- Establishes efficacy and versatility

**Key Components**
- Continual Context Distillation (CCD): an iterative process that compresses and retains essential information through novel importance metrics

**Evaluation Results**
- Demonstrates significant improvements over models trained for long contexts in various NLP tasks

**Significance of the Work**
- Represents a substantial advancement towards making LLMs applicable to broader real-world scenarios on memory-constrained devices.

## 1 Introduction

**Large Language Models (LLMs) Limitations:**
- Struggle to handle long input contexts due to predefined maximum length during training
- Fail to maintain coherence over extended sequences, especially in resource-constrained environments
- Traditional approaches: increasing memory capacity or using streaming inputs have drawbacks

**Proposed Solution: InfiniPot**
- Novel KV-cache control framework for handling infinitely long contexts within fixed memory constraints
- Processes incoming token sequences and compresses necessary parts when approaching limit
- Introduces Continual Context Distillation (CCD) process with importance metrics
  - CaP: Catalyst Prompt provides guidance in generating attention scores
  - NuC: Novelty under Compression prioritizes new information for the pre-trained model and previous contexts
- Enables efficient handling of long contexts without future context, making LLMs applicable to a broader range of NLP tasks.

**Benefits:**
- Allows pre-trained LLMs to handle very long contexts within fixed memory requirements without additional training
- Effectively manages long sequences through CCD cycles and importance metrics
- Demonstrates comparable or superior performance on various long-context NLP tasks compared to models explicitly trained for long contexts.

## 2 Related Work

**Long Context Window in LLMs: Background**
- Recognition of need for extended information retention leads to diverse approaches (Jiang et al., [2023](https://arxiv.org/html/2410.01518v1#bib.bib22); Anthropic, [2024](https://arxiv.org/html/2410.01518v1#bib.bib3); Reid et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib40))
- LongLoRA: fine-tunes LLMs for long contexts (Chen et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib10))
- TransformerFAM: feedback attention mechanism as working memory (Hwang et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib21))
- LongLM: sparsely modifying positional encoding for OOD prevention (Jin et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib23))

**Long Context Window in LLMs: Computational Memory Overhead**
- **Importance of efficient KV cache management**: limited computational memory environments (Beltagy et al., [2020](https://arxiv.org/html/2410.01518v1#bib.bib6); Hutchins et al., [2022](https://arxiv.org/html/2410.01518v1#bib.bib20))
- Sliding Window Attention (SWA) mechanism: focus on recent tokens, reducing computational overhead (Xiao et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib45))
- StreamingLLM: initial tokens as 'attention sinks' (Hwang et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib21))
- H2O and TOVA: retain highest attention score tokens (Zhang et al., [2023](https://arxiv.org/html/2410.01518v1#bib.bib48]; Oren et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib36))
- SirLLM: selects tokens based on cross-entropy (Yao et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib46))
- SnapKV: retains critical information at instruction prompt end but requires full context processing before compression (Li et al., [2024](https://arxiv.org/html/2410.01518v1#bib.bib28))

**Proposed Continual Context Distillation (CCD) Method**: distills essence of context into a fixed-size cache, enabling comprehensive caching within limited memory resources. Introduces Catalytic Prompt (CaP) and Novelty under Compression (NuC) scores to ensure core elements are efficiently preserved.

## 3 InfiniPot: Infinite Context Processing on Memory-Constrained LLMs

**Problem Definition and Objective**

**Primary Objective**:
- Manage long contexts within strict memory constraints in on-device environments

**Focus on KV-cache**:
- Keeping a limited number of entries (denoted by |M|) as proxy for actual memory usage
- Recent studies focus on reducing KV-cache memory pressure due to memory bottleneck

**Challenges**:
1. **Parallel Processing**: Cannot parallel-process long input over length |M|
2. **Long Context in Token Generation**: Cannot consider long context over length |M| during token generation process

**Proposed Solution: InfiniPot Framework**
- Enables memory-constrained LLMs to handle extremely long contexts using a novel context compression method, Continual Context Distillation (CCD)

**Components of CCD**:
1. **Catalyst Prompt (CaP)**: Importance measure from future
2. **Novelty under Compression (NuC)**: Importance measure from past

### 3.2 Continual Context Distillation (CCD)

**Continual Context Distillation (CCD)**

**Prior Methods Overlooked Memory Constraints**:
- All input context KV embeddings participate in generation
- SnapKV retains past tokens with high attention scores from context-end prompts, assuming entire set of KV embeddings are accessible
- Creating unconstrained memory usage due to need to load entire context input into model

**CCD Methodology**:
- Proposes **Continual Context Distillation (CCD)**, a novel method that compresses long context continuously in divide-and-conquer manner
- Initially fills cache with KV embeddings it can hold (Context-0 in Figure 1)
- Performs **Context Distillation** process based on proposed metric to evaluate token importance and retain only crucial tokens, reducing memory size to |C| (|M|≫|C|)
- Continues forward pass computation, filling remaining cache space sequentially (Context-1 in Figure 1)
- Once cache becomes full, repeats distillation process with previously distilled tokens and newly taken ones
- Manages KV-cache in memory-constrained environment

### 3.3 Importance Measure from Past and Future

**Infinite Context Processing on Memory-Constrained LLMs (Transformers)**
* **Importance Measure from Past and Future**
  * Overview: CaP and NuC for considering both past (x0:t−1) and future (xt+1:∞) contexts in finite cache size
* **Importance from Future: representative Score from Catalyst Prompt (CaP)**
  * Approximation of future importance u_t due to limited future context of memory pot
  * Uses a prompt (auxiliary context) for guidance in attention score generation
    * Token length of CaP |P|, approximated future importance: u_t = ∑i=t+1∞Attn(xi→xt) becomes infeasible
  * CaP always appended at last before pot overflows
  * Calculated per head, remaining KV-cache entries may not be synchronized in token axis after CCD.
* **Importance from Past: representative Score from Nearest Neighbor (NuC)**
  * To be discussed in the following sections.

#### 3.3.2 Importance from Past: Novelty Score from Compressed Context

**Novelty Score from Compressed Context (NuC)**
- Proposed metric for evaluating importance of previous context within memory pot
- Emphasizes information distinct from existing context
- Synergistic with CCD's representative capability indicator

**Calculation of NuC score**
1. **Cross-entropy**: Quantify novelty of t⁢-th token using cross-entropy (nt)
   - nt = −log⁡Pθ⁢(xt\|x0:t−1) for t > \|C\|
   - nt = −log⁡Pθ⁢(ct\|c0:t−1), otherwise
2. **Continuous update**: Reflect continuous update with approximation equation (4)
3. **Per-token based**: NuC score operates on different axis compared to CaP's representative score

**Approximate calculation of nt**
1. If t > \|C\|: nt = −log⁡Pθ⁢(xt\|c0:\|C\|−1;x\|C\|:t−1)
2. Otherwise: nt = −log⁡Pθ⁢(ct\|c0:t−1)
3. **Compressed region**: cj is the j⁢-th element in compressed region.
4. \|C\| represents the length of the compressed region.

#### 3.3.3 Combine Representative and Novelty Scores

**Token Selection Process:**
- **Representative Score (u~t)** and **Novelty Score (n~t)** are complementary
- Select tokens with highest novelty or representativeness among current tokens in pot (|C|=|CCaP|+|CNuC|)
- Prioritize novelty in token selection by allocating T slots within total cache slots |C| for tokens measured by their novelty scores
- Remaining |C|−T slots filled with Top|C|−T tokens based on representative scores generated by CaP
- Two-step process (n~t→u~t) prioritizes coarse-grained novelty score before applying per-head-based representativeness score
- Distinct granularity scales and different impacts on token significance

**Pseudo Code:**
- Token selection process provided in Appendix B of the paper

**Performance:**
- Hit rate comparison between Continual Context Distillation (CCD) with Catalyst Prompt general (CaP-G) and question (CaP-Q), Figure 2
- Comparison of hit rates between CaP and CaP w/ NuC across the CCD cycle, Figure 3
- Summation of selected token’s entropy comparison for various CCD configurations, including global token entropy comparison, Figure 3.

### 3.4 Analysis of Representative and Novelty Scores

**Analysis of Representative and Novelty Scores**

**Effectiveness of CaP**:
- Experiments removing CaP or altering prompt type show improvement when using CaP in Context Distillation phase (CCD)
- Hit Rate analysis (Figure 2, top) shows that without CaP, CCD's hit rate is similar to Sliding Window Attention (SWA), indicating it struggles to retain crucial tokens.
- With CaP, Hit Rate significantly improves and is closer to global scoring (Figure 2, bottom)

**Impact of integrating NuC with CaP**:
- Token overlap rate analysis in final memory pot shows that using both CaP and NuC results in fewer tokens retained compared to just CaP alone
- Novelty scores of tokens retained by CCD increase resemblance to global novelty scores, indicating NuC shifts attention head configuration across contexts (Figure 3)

**Comparison of Aggregated Novelty Scores**:
- Novelty scores of tokens in each attention head's memory pot approach those from global scoring as more context is processed
- This alignment improves performance on NarrativeQA task, approaching the level of when full past token sets are available

### 3.5 Context-Reset Rotary Positional Embedding

**Context-Reset Rotary Positional Embedding (CR-RoPE)**

**Handling Long Contexts**:
- Requires careful management of positional embeddings to avoid out-of-distribution (OOD) issues

**CR-RoPE Policy**:
- Re-organizes positional information across retained tokens using RoPE method

**Key Differences from Memory-Unconstrained Methods**:
- After each distillation phase, CR-RoPE applies RoPE based on positional indices to newly selected entries, effectively preventing OOD compared to SnapKV

**Advantages of CR-RoPE**:
- Safteguards any model with a limited context window against OOD issues due to re-organization of positional encoding within the memory-pot size
- More efficient in processing long contexts compared to StreamingLLM, which requires recalculating RoPE for each token generated

**Performance Benefits and Latency Improvements**:
- Addressed in Sections 4.4.3 and 4.5 of the study

## 4 Experiments

**Experimental Setup**
- Run experiments across multiple long context benchmarks
- Use input context length as a proxy for actual memory usage

**Benchmarks**
- LongBench Bai et al. (2023)
  * Multi-task benchmark designed for long context understanding
  * Consists of 6 task categories and 21 diverse tasks
  * Context lengths range from 3K to 36K
- Needle In A Haystack Kamradt (2023)
  * Evaluates ability to retrieve critical information from extensive contexts
  * Varying length from 4K to 1M

**Baselines**
- Long context window method and KV-cache compression baselines
- For memory-unconstrained scenarios: Self-Extend Jin et al. (2024) (SE), Snap-KV Li et al. (2024)
- For memory-constrained scenarios: StreamingLLM Xiao et al. (2024) (Streaming), H2O Zhang et al. (2023)
- Recently proposed methods: SirLLM Yao et al. (2024) and TOVA Oren et al. (2024)
  * Tested under identical conditions using their criteria for retaining critical KV-cache within the proposed CCD pipeline
- Truncated method (TR) as a baseline: official LongBench approach which truncates middle part of entire context to fit predefined context memory.

### 4.2 Performance on LongBench

**Performance on LongBench+ Memory-Unconstrained:**
- **InfiniPot**: Outperforms memory-constrained LLMs like GPT-3.5-16K (43.74) and M3-PT-32K (47.84), achieving a competitive score of 44.22
- Compared to recent memory-unconstrained techniques:
  - L3-InfiniPot-4K scores 41.50, closely following L3-SnapKV-4K's 41.94
  - Memory-constrained approaches like StreamingLLM and H2O yield suboptimal performance (29.87 and 31.01 respectively) compared to InfiniPot
- Employing token importance metrics from SirLLM (based on token entropy) and TOVA (attention score from the last token) in CCD pipeline of L3 results in lower performance (35.43 and 36.26) than TR's 37.20
- InfiniPot significantly surpasses all baselines across LLaMA and Mistral models, especially with recent LLM models like LLaMA-3.1/3.2, Gemma-2, Phi-3
- **InfiniPot** shows more remarkable performance improvements with recent LLMs (e.g., LLaMA-3.1/3.2, Gemma-2, and Phi-3) as detailed in Appendix C.3

**Memory-Constrained:**
- In conditions where memory is strictly limited, InfiniPot consistently outperforms other methods in handling long contexts
- When applying StreamingLLM and H2O which involve dropping tokens within a restricted window during the long context processing stage, these approaches yield suboptimal performance (29.87 and 31.01 respectively) compared to TR (31.24)
- InfiniPot significantly surpasses all of these baselines across LLaMA and Mistral models, especially with recent LLM models like LLaMA-3.1/3.2, Gemma-2, Phi-3.

### 4.3 Performance on Needle In a Haystack

**Performance on Needle In a Haystack (Figure 4)**

**InfiniPot vs Long Context Models:**
- **InfiniPot with LLaMA-3-4K and Mistral-7B-v0.3-4K (Ours)** maintain high accuracy even at long context lengths of 512K tokens
- Models like **LongChat-v1.5-32K** and **LongAlpaca-16K** experience steep performance declines beyond 32K tokens, highlighting their limitations in handling very long contexts
- Integrated InfiniPot with Mistral-v0.1-4K displays stable performance up to 128K context length, extending the usable context range while preserving accuracy
- Models like **SE-Mistral-v0.1-24K** showed sharp drops in accuracy beyond 32K, indicating they struggle with extremely long contexts without InfiniPot
- InfiniPot can handle inputs as long as 1M tokens, demonstrating unparalleled performance not seen in traditionally constrained or pretrained context window extended models
- Exceptional scalability underscores the impact of InfiniPot in enabling models to process extraordinarily long context windows efficiently.

**LongBench Performance Comparison:**
- Table 2: Comparison of LLaMA-3-8B instruct model with 4K memory on LongBench tasks (bottom)
- Critical points summarized from the provided text:
  - Figure 4 illustrates InfiniPot's superior scalability and accuracy at extended context lengths in NIH benchmark
  - Models like LongChat and LongAlpaca struggle with handling very long contexts beyond 32K tokens
  - Integrated InfiniPot with Mistral displays stable performance up to 128K context length, extending the usable context range while preserving accuracy
  - SE-Mistral model shows sharp drops in accuracy beyond 32K without InfiniPot.

### 4.4 InfiniPot Analysis

**InfiniPot Analysis: CaP and NuC Impact on Performance**

**Impact of Individual Components**:
- CaP: various prompt designs explored, CaP-G achieves highest LongBench scores (Document QA, Summarization, Few-Shot Learning, Others)
- NuC: Table 2 shows that NuC significantly impacts performance (α ratio affects LongBench score)

**Robustness of CaP to Prompt Design**:
- CaP-G1 and G2 achieve similar performances to CaP-G, confirming robustness to prompt design variation.

**Effectiveness of Query-Aware Context Compression**:
- In predefined query situations, incorporating the question into CaP leads to performance improvements in QA tasks (from 30.54 to 35.71)
- In long context scenarios without pre-accessible queries, CaP-G effectively achieves query-agnostic context compression

**Components Analysis: Table 3**
| NuC Ratio (α)| QA | Summ | FSL | Others | Avg. |
|---|---|---|---|---|---|
| w/o CaP, NuC | 28.89 | 24.47 | 67.35 | 32.84 | 36.26 |
| 0% (CaP only) | 33.35 | 25.04 | 68.11 | 39.68 | 39.89 |
| 25% | 33.71 | 25.19 | 68.65 | 41.64 | 40.65 |
| 50% | 35.71 | 25.14 | 68.89 | 41.91 | 41.50 |
| 75% | 35.02 | 25.13 | 68.12 | 41.86 | 41.08 |
| 100% (NuC only) | 26.46 | 24.16 | 68.31 | 32.66 | 35.43 |

#### 4.4.2 NuC and CaP Ablation Study

**NuC and CaP Ablation Study Results:**
* Table 3 shows performance comparisons of CaP and NuC components in CCD pipeline
* Incorporating CaP enhances performance (36.26 → 39.89)
* Using only NuC leads to a decline (36.26 → 35.43) due to uniformity in novelty scores
* Harmonizing CaP and NuC scores yields significant gain over CaP alone (39.89 → 41.50)
* Proportion of NuC tokens (α) treated as hyperparameter, used at a 50% ratio in experiments
* Combining these metrics effective within the CCD framework

**Impact on Retrieval Accuracy:**
* Figure 5: Retrieval accuracy for Needle in a Haystack task across varying context lengths
* Task involved hiding passkey at different depths and measuring retrieval accuracy with increasing context length

**Memory Usage Comparison:**
* Figure 6: Average LongBench score across memory budgets
* Maximum memory usage for handling all key-values: 32K (Mistral-v0.3) vs 8K (LLaMA-3).

#### 4.4.3 NIH Ablation Study

**CDC Components Impact on NIH Task Performance**

**NIH Ablation Study**:
- Explores how each CDC component enhances performance in NIH task
- Evaluates impact of components over context lengths from 4K to 1M

**Initial Results (Figure 5a)**:
- No CR-RoPE, CaP, or NuC: Performance deteriorates for contexts shorter than Mistral's 32K window
- Reorganizing RoPE within memory pot recovers performance, underscoring CR-RoPE's importance

**Addition of CaP (Figure 5c)**:
- Enables handling context lengths up to four times longer
- Enhances NIH task performance

**Addition of NuP (Figure 5d)**:
- Allows memory-constrained model to maintain high NIH accuracy at 1M context length
- Demonstrates how CDC components synergistically extend conventional 32K context window by over 30 times

**Detailed Results (Table 4)**:
- Provided in the study for each ablation experiment

#### 4.4.4 Performance Across Memory Budget

**Performance Across Memory Budget**

**Comparison of Truncated (TR) Method and InfiniPot**:
- Figure 6 shows comparison of performance for both methods across increasing memory budgets, from 1K to 8K on LongBench scores
- Both methods exhibit an upward trend in performance as memory budget sizes increase
- However, InfiniPot consistently outperforms the baseline across all memory sizes

**Rationale for InfiniPot's Superior Performance**:
- Demonstrates InfiniPot's superior ability to efficiently distill and retain essential context information within given memory constraints
- Effectively utilizes the memory budget to achieve enhanced performance in long-context scenarios, even under strict memory constraints.

### 4.5 Efficiency Analysis: Memory and Speed Measurement

**InfiniPot Efficiency Analysis: Memory and Speed Measurement**

**Assessing InfiniPot's Performance**:
- Evaluate latency and memory usage during text generation with long context inputs under unconstrained and constrained scenarios

**Performance Metrics**:
- **Memory consumption (GB)**: Snap-KV loads entire context for cache compression, resulting in increased memory demands and reduced throughput. AllKV caches all KV embeddings shows sharp rise in memory use and performance degradation. StreamingLLM exhibits minimal memory use but high latency due to single token processing and RoPE recomputation during generation
- **Latency of Time-To-First-Token (TTFT) in prefill stage (sec)**: InfiniPot matches or surpasses Snap-KV's speed, particularly for contexts over 50K
- **Generation throughput (Tokens/sec)** and **wall-clock time (sec)**: InfiniPot maintains high throughput and shortest wall-clock time during generation

**Conclusion**:
- InfiniPot offers consistent performance, optimizes memory use, and maintains high throughput. It matches or surpasses Snap-KV's speed in the prefill stage and displays best token throughput and shortest wall-clock time during generation

## 5 Conclusion

**Conclusion**
- Addressing challenge of enabling Large Language Models (LLMs) to handle long input contexts efficiently in memory-constrained environments
- Proposed InfiniPot, a novel KV-cache control framework using Continual Context Distillation (CCD)
  * Iteratively compresses and distills essential information without future context access
- Evaluations showed InfiniPot-equipped LLMs manage extended sequences effectively, achieving performance comparable to or surpassing models trained for long-context tasks.
- Significantly extends capabilities of pre-trained LLMs, making them more versatile and applicable to a broader range of NLP tasks without additional training.

## 6 Limitations

**Limitations of InfiniPot**

**Predefined Compression Ratio**:
- Current implementation of **Continuous Context Delivery (CCD)** relies on a predefined compression ratio
- This may not be optimal for all types of input data
- Future work could explore **adaptive compression techniques** that dynamically adjust based on context importance

**Preserving Long-Term Dependencies**:
- InfiniPot effectively manages context length within memory limits
- Its ability to preserve very long-term dependencies across compressed contexts has not been exhaustively tested
- Future studies should investigate how well the retained information captures essential long-term dependencies in diverse perspectives

**Real-World Applicability**:
- The method is designed with on-device constraints, but has not yet been evaluated in actual on-device environments
- Future work should include comprehensive testing on various mobile and edge devices to verify its practical applicability and efficiency under real-world conditions

