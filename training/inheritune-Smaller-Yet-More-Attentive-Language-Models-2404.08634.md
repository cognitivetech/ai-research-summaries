# Inheritune: Training Smaller Yet More Attentive Language Models

by Sunny Sanyal 
**Source**: [arXiv:2404.08634v2](https://arxiv.org/html/2404.08634v2)

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Attention Degeneration in Standard Decoder-Style LLMs](#2-attention-degeneration-in-standard-decoder-style-llms)
  - [2.1 Attention Pattern Visualization](#21-attention-pattern-visualization)
- [3 Inheritune: Our Proposed Training Recipe](#3-inheritune-our-proposed-training-recipe)
- [4 Experiments](#4-experiments)
  - [4.1 Results and Discussions](#41-results-and-discussions)
  - [4.2 Ablations](#42-ablations)
- [5 Training without Data Repetition](#5-training-without-data-repetition)
- [6 Related Works](#6-related-works)
- [7 Conclusion](#7-conclusion)
- [Appendix A Supplementary Experiments](#appendix-a-supplementary-experiments)
- [Appendix B Developing a 1.5B Small Base LM in a Low Data Regime with Inheritune](#appendix-b-developing-a-15b-small-base-lm-in-a-low-data-regime-with-inheritune)
  - [B.1 Main Results in Low Data Regime](#b1-main-results-in-low-data-regime)
  - [B.2 Additional analysis with larger reference LMs and 50B data](#b2-additional-analysis-with-larger-reference-lms-and-50b-data)
  - [B.3 Implications of Low Data Regime](#b3-implications-of-low-data-regime)
- [Appendix C Implementation Details](#appendix-c-implementation-details)
  - [C.1 Training details of GPT-2 models](#c1-training-details-of-gpt-2-models)
  - [C.2 Training details of 1.5B OpenLLaMA model](#c2-training-details-of-15b-openllama-model)

## Abstract

**Large Language Models (LLMs)**
- Achieved remarkable performance across natural language processing tasks
- Due to transformer architecture and self-attention mechanism

**Observation**:
- In standard decoder-style LLMs, attention matrices degenerate to single-column for deeper layers
- These **lazy layers** are unable to learn meaningful information and mostly redundant

**Goal**:
- Train smaller models by eliminating this structural inefficiency without compromising performance

**Proposed Solution: Inheritune**
- A simple yet effective training recipe for developing smaller, high-performing language models

**Approach**:
- Smaller models inherit early transformer layers from a larger pre-trained model
- Retrain and progressively expand until they match or exceed the performance of the larger model

**Results**:
- Demonstrated to enable the training of various sizes of **GPT-2 models** on OpenWebText-9B and FineWeb_Edu datasets
- Smaller models trained with Inheritune achieve comparable or even better performance than their larger counterparts
  - 16-layer GPT-2 medium variant achieves comparable performance to the standard 24-layer GPT-2 medium model

**Code**:
- https://github.com/sanyalsunny111/LLM-Inheritune

## 1 Introduction

**Language Models (LLMs)**
- Large Language Models (LLMs) built using decoder-style transformer blocks: Vaswani et al., [2017](https://arxiv.org/html/2404.08634v2#bib.bib44)
- Designed for depth and size, improving performance with increased model capacity: Kaplan et al., [2020](https://arxiv.org/html/2404.08634v2#bib.bib25); Hoffmann et al., [2022](https://arxiv.org/html/2404.08634v2#bib.bib23)

**Attention Mechanism in LLMs**
- Crucial component for capturing long-range dependencies and contextual relationships within text data
- As models grow deeper, encounter attention degeneration: Noci et al., [2022](https://arxiv.org/html/2404.08634v2#bib.bib32); Dong et al., [2021](https://arxiv.org/html/2404.08634v2#bib.bib13); He et al., [2023](https://arxiv.org/html/2404.08634v2#bib.bib20)
- Not studied in standard LLMs: Section [2](https://arxiv.org/html/2404.08634v2#S2 "2 Attention Degeneration in Standard Decoder-Style LLMs ‣ Inheritune: Training Smaller Yet More Attentive Language Models")

**Attention Degeneration and Lazy Layers**
- Observed in GPT-2 medium (24 layers) and large (36 layers) models: Radford et al., [2019](https://arxiv.org/html/2404.08634v2#bib.bib36)
- Characterized by rank-1 attention matrices with single-column structures: Figure 1(a) and (d)
- Termed "lazy layers": Figure 1(c) and (f)

**Inheritune Training Approach**
- Proposed method to develop performant small base language models using weights from larger base LMs
- Initializes smaller model with a few early blocks from the larger pre-trained model
- Continuously grows and trains the smaller model, incrementally adding more blocks until it surpasses the reference model's performance: Figure 2(a) and (b)
- Effectively trains high-performing, smaller models while preserving effective attention patterns: Figure 2(c)

**Key Contributions**
1. Analysis of Attention Degeneration and Lazy Layers in standard LLMs
2. Introduction of Inheritune training approach for smaller yet more attentive language models
3. Validation of Inheritune's effectiveness through comprehensive experiments on various baselines: Tables 1 and 2; Figure 3
4. Improved performance in settings with non-repeated training tokens: Figure 4

## 2 Attention Degeneration in Standard Decoder-Style LLMs

**Preliminaries:**
* A vanilla transformer-model consists of L transformer blocks (layers)
* Model operates on input sequence X∈ℝT×d
	+ T: sequence length, number of tokens
	+ d: embedding dimension or hidden size
* Output of each layer l is denoted as X(l)∈ℝT×d
* Each transformer block consists of self-attention block and position-wise FFN
* Self-attention mechanism enables relevance weighing of tokens in sequence
	+ Queries Q, keys K, values V are linear transformations of input X
	+ Attention matrix A(X) captures pairwise attention scores between tokens
	+ Softmax applied row-wise to compute weighted sum of value vectors
* Previous research shows self-attention networks (SANs) can exhibit rank collapse of attention matrices
	+ Results in loss of expressive power as model attends to all tokens uniformly
	+ This phenomenon affects both SANs and feed-forward networks (FFNs)
* Findings do not directly apply to standard LLMs as they include residual connections, layer norms, and FFNs

**Approximate Rank Computation of Attention Matrices:**
* Analyze structure of attention matrices in standard transformer-based LLMs using GPT-2 models
* Compute approximate rank (k*) of A(X) for all attention heads within each layer
	+ Using Singular Value Decomposition (SVD) and explained variance method
* Determine minimal number of columns (m*) required to capture 90% of total mass
* Analyze degeneration of attention matrices in deeper layers
	+ Many deeper layers exhibit rank-1 attention matrices, indicating reduced performance and less effective token mixing.

**Lazy Layers:**
* Some deeper layers may have completely degenerate attention matrices across all attention heads
* Lazy layers hold less transferable knowledge compared to early counterparts
* Initializing new models with lazy layers extracted from pre-trained models performs similarly to random initialization.

### 2.1 Attention Pattern Visualization

**Visualization of Attention Patterns in Language Models**
- **Vanilla 24-layer GPT-2 model trained from scratch**: Shows a progression in attention patterns as the model deepens
  - Early layers (L4, L7) exhibit structured patterns with local and global attention
  - Deeper layers (L20, L22) display more uniform patterns, indicating loss of focus and lazy layers
- **16-layer model trained using proposed method**: Demonstrates more focused and effective attention patterns even in later layers (L11, L15)

**Inheritune: Training Smaller Yet More Attentive Language Models**
- Algorithm 1: Inheritune training recipe for small language models
  1. **Reference model**: `ℳref` with `k` layers, datasets `𝒟train` and `𝒟val`, steps `𝖳`
  2. Initialize target model `ℳtgt` with first `n=k/2` layers from `ℳref`
  3. Train `ℳtgt` on `𝒟train` for `𝖳` steps
  4. While `ℳtgt` performance < `ℳref` performance on `𝒟val`:
     - Grow `ℳtgt` by inheriting additional layers
     - Train `ℳtgt` for `𝖳` steps
  5. Return optimized model `ℳtgt`
- This method addresses attention degeneration and makes models more attentive, potentially leading to more efficient models in a compact size.

## 3 Inheritune: Our Proposed Training Recipe

**Method Overview:**
* Goal: Develop smaller base language models (LMs) with equivalent or better validation loss compared to larger models
* Principles: Zero-shot initialization and progressive growth
* Three main steps: Inherit, Train, Grow

**Setup:**
* Split dataset into training set 𝒟train and validation subset 𝒟val
* Pre-trained reference model ℳref with k layers (deep LLMs)
* Initialize smaller model ℳtgt with first n=k/2 layers of ℳref
* Train ℳtgt on 𝒟train for T steps, evaluate on 𝒟val
* If needed, increase size and repeat steps 1-2

**Method Description:**
* Inherit: Initialize smaller model with first half of larger model's layers
* Train: Train smaller model for a specified number of steps on training set
* Grow: Increase size and repeat training process until desired performance is achieved.

## 4 Experiments

**Inheritune: Training Smaller Yet More Attentive Language Models**

**Experimental Setup**:
- Evaluated using GPT-2 xlarge (1.5B), GPT-2 large (770M), and GPT-2 medium (355M) models
- Pre-trained on the 9B tokens OpenWebText dataset

**Evaluation Metrics**:
- Model validation loss (log-perplexity) on 𝒟val
- Comparison with baseline models:
  - Baselines trained from scratch with random initialization
  - Baselines trained using various zero-shot initialization techniques
  - Baselines trained with knowledge distillation

**Ablation Study**:
- Detailed specifications for all models used in experiments (Table 10)
- Focuses on the 16-layer GPT-2 medium variants

**Training Procedure**:
1. Reference Model:
   - Train full 36-layer GPT-2 Large model on 𝒟train for 100K steps
   - Evaluate validation loss (log-perplexity) on 𝒟val
2. Model Initialization:
   - Initialize an 18-layer (n=k/2) model using the trained 36-layer GPT-2 Large model as a reference
3. Training and Evaluation:
   - Train the 18-layer model on 𝒟train for T steps
   - Evaluate its validation loss
4. Iterative Refinement:
   - If the smaller model's performance is inferior, incrementally increase its size by two layers and repeat steps 2-3 until parity with the reference model's validation loss is achieved

**Baselines**:
1. **Baselines trained from scratch (rand init.)**:
   - Compare Inheritune-derived models against larger GPT-2 reference models trained from scratch for the same number of steps
   - Compare Inheritune-derived models against smaller GPT-2 models trained from scratch for double the number of training steps
2. **Baselines trained with various model initialization and efficient training techniques**:
   - Stacking (Gong et al., 2019; J. Reddi et al., 2023)
   - Hybrid stacking (large pre-trained reference model for initialization)
   - Half width (initializing the baseline GPT-2 large variant across the width dimension and preserving the entire depth)
3. **Baselines trained with Knowledge Distillation**:
   - Logit-based knowledge distillation (Hinton et al., 2015)
   - DistillBERT-style approach (Sanh et al., 2019)

**Results**:
- Table 2 shows that Inheritune-derived models consistently achieve lower validation loss compared to models initialized with stacking, hybrid stacking, and half-width techniques.

### 4.1 Results and Discussions

**Study Findings on Models Trained with Inheritune**

**Key Results**:
- Smaller models derived using Inheritune achieve comparable or lower validation losses than their full-sized counterparts when trained for the same number of steps.
- GPT-2 variants trained with Inheritune outperform their same-sized counterparts trained from scratch, both when trained for the same number of steps and when trained for double the steps (200K).
- Models trained with Inheritune consistently outperform various baselines, including stacking, hybrid stacking, and vanilla or DistillBERT-style knowledge distillation.

**Table 1**:
- GPT-2 xlarge, GPT-2 large, and GPT-2 medium variants trained with Inheritune achieve comparable or lower validation losses than their full-sized counterparts when trained for the same number of steps (100K).
- GPT-2 models derived using Inheritune match their much larger reference models' downstream performance on two next-word prediction tasks in a zero-shot setting.

**Convergence Perspective**:
- Some prior works have made connections between over-parameterization and faster convergence.
- Small LMs derived with Inheritune still converge as fast as their large-size reference models, despite being smaller compared to them.

**Supplementary Figure 5**:
- Shows that small LMs derived with Inheritune converge as fast as their large-size reference models.

**Table 2**:
- Compares GPT-2 xlarge, GPT-2 large, and GPT-2 medium variants trained with Inheritune against same-sized variants trained with stacking, hybrid, and half-width initialization baselines.
- Half-width baseline performs poorly, revealing the limitations of naive width reduction.
- Stacking and hybrid stacking demonstrate reasonable performance but still fall short compared to 𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾.

**Figure 3**:
- A 16-Layer GPT-2 medium variant derived using Inheritune converges faster and generalizes better than a same-sized model trained with Logit-based distillation baselines (vanilla KD and DistillBERT-style KD).

### 4.2 Ablations

**Experiments on Sub-Module Initializations within a Transformer Block**

**Key Findings**:
- **𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾 initialization and attention + MLP initialization** result in similar performance improvements.
- **Layernorm initialization** shows minimal impact.

**Ablations**:
- Comparison of validation loss for a 16-layer GPT-2 medium variant with different sets of sub-modules initialized using weights from a 24-layer GPT-2 medium reference model.
- All models trained on OpenWebText-9B dataset.

**Sub-Module Initializations**:
- **1) Attention (key, query, value, and projection, as well as layernorm2 weights)** and mlp weights without the layer norm (attn+mlp w/o layernorm).
- **2) Attention and MLP weights without the layer norm**.
- **3) MLP weights with the layer norm**.

**Initialization Method**:
- **𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾** performs initialization by inheriting attention and MLP weights with the layer norm (attn+mlp w/ layernorm).

**Results**:
- Table 3 shows validation loss for each of the three initialization methods.
- Detailed validation loss vs training steps plot in supplementary Figure 7.
- Initializing both **attention and MLP weights** provides a clear advantage.
- Initializing either the **attention or mlp weights** results in similar improvements in convergence speed and final validation loss.

## 5 Training without Data Repetition

**Model Performance Comparison:**

**Background:**
- Figure 4 shows that GPT2 models trained using Inheritune without data repetition perform as well as their full-sized counterparts and outperform same-sized models in terms of training loss.

**Investigation:**
- To investigate if the gains are due to overfitting from data repetition, additional training experiments were conducted following standard LLM pre-training practices.
- A high-quality dataset (Fineweb_edu) was used without data repetition for training 32-layer GPT2 large and 24-layer medium reference models, as well as their 16-layer counterparts derived from them using Inheritune recipe and baseline variants.
- All models were trained for 100K steps.

**Findings:**
- Figure 4 indicates that the GPT2 variants trained with Inheritune consistently perform on par with their full-size counterparts and outperform same-sized models in terms of training loss.
- Zero-shot downstream evaluations on ARCE, LAMBADA, SciQ, Hellaswag, PIQA showed that models derived from Inheritune demonstrate superior performance than baseline models trained from scratch (Table 4).
- The average zero-shot downstream performance of the Inheritune-derived models is better than their larger and same-size counterparts.

**Conclusion:**
- The gains observed with Inheritune are not merely a consequence of overfitting due to data repetition.

## 6 Related Works

**Attention Degeneration:**
* Studied through attention rank collapse Dong et al. (2021) and attention entropy collapse Zhai et al. (2023) leading to representation collapse and training instability
* Theoretical setup for transformer models: Noci et al. (2022) and Barbero et al. (2024)
* Rank collapse in self-attention networks without residual connections or layer norms: He et al. (2023) using two model initialization techniques
	+ Enables faithful signal propagation: ΣL of A(XL) does not collapse in deeper layers but slows down training

**Addressing Attention Degeneration:**
* Smaller models eliminating structural inefficiencies to match performance of larger, inefficient counterparts

**LLM Training Recipes and Model Initialization:**
* Stacking method: Gong et al. (2019), J. Reddi et al. (2023), Du et al. (2024)
	+ Stage-wise training strategy with weights from initial layers initialized in later layers
	+ Effective for LLM training empirically and theoretically Agarwal et al. (2024)
* Knowledge distillation: Hinton et al. (2015), Turc et al. (2020), Sanh et al. (2019)
	+ Smaller student model initialized with teacher layers, though often without clear explanation or intuition
* Recent works in model initialization: Trockman & Kolter (2023) and Xu et al. (2024)
	+ Synthetic attention patterns for initialization in vision settings have limited success in language models
	+ Proposed recipe focuses on creating a smaller model by eliminating specific structural inefficiency in lazy layers.

## 7 Conclusion

**Study Findings:**
* Identified structural flaw in deep decoder-style transformers used as LLMs (largest language models)
* Attention matrices in deeper layers lose rank and converge into single-column matrices, resulting in "lazy layers"
* This phenomenon causes the model to lose attention and leads to inefficiencies

**Proposed Solution:**
* Simple training recipe: inherit early blocks from a larger reference model and continue training/expanding
* Evaluated using GPT-2 xlarge, GPT-2 large, and GPT-2 medium models on OpenWebText-9B dataset
* Successfully developed smaller models (24, 18, 16 layers) without performance loss compared to larger counterparts

**Additional Findings:**
* Similar results observed with GPT-2 large and GPT-2 medium models trained on FineWeb_edu without data repetition
* Offers novel perspective on developing efficient small base language models from existing large pre-trained models
* Potentially democratizes LLM pre-training for a wider community.

## Appendix A Supplementary Experiments

**Inheritune: Training Smaller Yet More Attentive Language Models**

**Additional Experiments:**
- In Figures [5](https://arxiv.org/html/2404.08634v2#A1.F5) and [6](https://arxiv.org/html/2404.08634v2#A1.F6), the authors provide additional training plots for their main results.
- In Figure [5], they compare GPT-2 variants with baseline models trained from scratch.
- In Figure [6], they compare GPT-2 variants with baseline models trained using knowledge distillation techniques.

**Knowledge Distillation:**
- The authors previously discussed knowledge distillation as a baseline in Section [4.1].
- They perform an additional experiment using knowledge distillation as a baseline: training 12-layer GPT-2 medium variants with distilled weights from a 24-layer GPT-2 medium teacher.
- They find that models trained using the **𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾 recipe** outperform both distilled and vanilla GPT-2 medium models.

**Attention Degeneration:**
- The authors previously discussed attention degeneration in Section [2].
- In Figure [9], the authors demonstrate that models trained using the **𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾 recipe** have fewer "lazy layers" compared to their larger counterparts.
- The authors attribute attention degeneration to vanishing gradients in keys and queries, which are caused by small norms of the gradients.
- Smaller models trained using **𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾** have higher gradient norms, which allows them to converge better and generalize better than larger models.

**Training Curves:**
- The authors present training curves for 16-layer GPT-2 variants trained using different initialization methods in Figure [8].
- They find that the **𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾** approach outperforms other initialization methods, especially when initializing specific sub-modules within transformer blocks.

## Appendix B Developing a 1.5B Small Base LM in a Low Data Regime with Inheritune

**Study: Evaluating a Small Base Language Model Using Inheritune Method**
* **Objective**: Investigate effectiveness of Inheritune method in data-constrained setting by training a small base language model with limited resources.
* **Methods**:
  * Adapted Inheritune method for training a small base LM using first 13 layers from reference model.
  * Trained on randomly sampled 1B tokens from RedPajama dataset for eight epochs with data repetition and batch size of 131K per batch.
  * Evaluated using few-shot accuracy on ten different downstream tasks: commonsense reasoning, natural language understanding, factuality, and natural language inference.
* **Results**:
  * Our 1.5B base LM achieved comparable performance to reference model and other similarly sized models despite being trained with fewer tokens (Table 5).
  * Highlighted scores in bold where our model outperforms at least two baseline similar-size LMs or achieves 90% of the score compared to reference LM.
* **Discussion**:
  * Training a small base language model using Inheritune method and limited resources.
  * Comparison of our target model, reference model, and other baseline models in terms of performance (Table 5).
  * All tasks evaluated using 0-shot except MMLU which is 5-shot.

**Referenced Models**:
* **OpenLLaMA-3B**: Pretrained with 1T tokens from RedPajama V1 dataset, contains data from various domains.
* **OPT-1.3B**: Pretrained on similar dataset as OpenLLaMA-3B but with fewer parameters (1.3B).
* **Pythia-1.4B**: Pretrained on similar dataset as OpenLLaMA-3B but with fewer layers and parameters (1.4B).
* **MPT-1.3B**: Pretrained on 200B tokens from RedPajama dataset, contains data from various domains.
* **Sheared LLaMA-1.3B**: Pretrained using a pruning method with 50B tokens from LLaMA2-7B model.
* **Ours-1.5B**: Trained using Inheritune method with first 13 layers from OpenLLaMA-3B and 1B tokens from RedPajama dataset.

**Evaluation Metrics**:
* Few-shot accuracy on ten different downstream tasks: commonsense reasoning, natural language understanding, factuality, and natural language inference.

**Training Recipe**:
* Used the lit-gpt framework for training all small base LMs discussed in this paper.
* Training hyperparameters not further discussed.

### B.1 Main Results in Low Data Regime

**Performance Evaluation of 1.5B Model**
* Table [5](https://arxiv.org/html/2404.08634v2#A2.T5) presents results on various tasks
* **1.5B model**: excels in 7 out of 10 individual tasks
* Achieves scores higher than reference LM twice its size and trained with more data
* Outperforms at least two base LMs of similar size with less data
* Matches accuracy with MPT-1.3B4 model on all nine downstream tasks and MMLU (5-shot) score
* Beats OPT-1.3B and Pythia-1.3B in MMLU (5-shot) score, performs comparably on other datasets
* Inheriting a large reference LM allows for more sample-efficient training than from scratch

**Ablation of Inheritune Across Different Model Sizes with 1B Tokens**
* Previous section considered n=k/2 (half layers) as the smaller model size
* Investigating different choices of n using OpenLLAMA-3B as large pre-trained reference model
* Developed eight submodels with n={4,6,8,10,13,16,18,20}
* Figure [11](https://arxiv.org/html/2404.08634v2#A2.F11) shows MMLU (5-shot) score as a function of n
* Positive trend line: more layers, better performance
* 20 layers decreases performance potentially due to data overfitting
* Training details consistent with target 1.5B small base LM, detailed in appendix
* Future work for comprehensive investigation on choice of n and broader set of tasks.

### B.2 Additional analysis with larger reference LMs and 50B data

**Additional Analysis with Larger Reference LMs and 50B Data**

**Observations:**
- Clear improvement in overall MMLU (5-shot) score with more data
- 1.5B models developed using larger reference models show greater improvements on 50B subset of non-repetitive tokens

**Comparison of Models:**
| Model | Training Data (# tokens) |
|--------|-----------------------|
| OpenLLaMA-3B (ref) | RedPajama(1T) |
| Our-1.5B* | RedPajama (1B) |
| Shear-LLaMA-1.3B* | RedPajama(50B) |
| MPT-1.3B | RedPajama(200B) |
| Pythia-1.4B | The Pile(300B) |
| OPT-1.3B | Custom data(300B) |

**Computational Efficiency:**
- Comparison of pre-training compute requirements for small base LMs and our 𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾-derived model
- Significant reduction in computational resources using Inheritune

**Performance Comparison:**
| Model (# tokens), ref | MMLU(5) score |
|---------------------|---------------|
| Our-1.6B (1B), LLaMA2-7B | 24.27 |
| Our-1.5B (1B), OpenLLaMA-3B | 25.67 |
| Our-1.5B (50B), OpenLLaMA-3B | 25.71 |
| Our-1.6B (50B), LLaMA2-7B | 26.07 |
| Our-1.6B (50B), OpenLLaMA-7B | 26.72 |

**Data Repetition:**
- Ablations on the number of epochs to observe repetition's impact on MMLU performance
- Peaks at 5 epochs and then deteriorates for most datasets except MMLU
- Safe to reuse tokens up to 10-20 epochs without significant degradation in performance (Table 9)

### B.3 Implications of Low Data Regime

**Key Implications of Work in Low Data Regime**

**Cheap and Easy Development of Small Base Language Models (LM)**
- Pre-training a small base LM from scratch is expensive: mpt-1.3B, Pythia-1.4B, TinyLLaMA-1.1B
- 1.5B (1B data variant) LM shows competitive performance with minimal resources
  - Trained using 1 A6000 GPU for less than 12 hours
  - Comparative computational details in Table [7]

**Developing a Small Base LM before Deployment**
- Typically, small base LMs are finetuned after pre-training
- Presenting an easy and cheap way to develop a small base LM for later use: 𝖨𝗇𝗁𝖾𝗋𝗂𝗍𝗎𝗇𝖾

**Naive Baseline for Pre-Training Scaled Down Variants of Large Base LMs**
- Small variants of large base LMs are usually pre-trained on the same data Peiyuan Zhang & Lu ([2023](https://arxiv.org/html/2404.08634v2#bib.bib34)); Groeneveld et al. ([2024](https://arxiv.org/html/2404.08634v2#bib.bib19))
- Our method introduces a new perspective on identifying sufficient depth while maintaining generalization on held out validation set
- Pre-training a small base LM with a small fraction of pre-train data and initial layers from large base LM as naive baseline for developing smaller variants

## Appendix C Implementation Details

### C.1 Training details of GPT-2 models

**GPT-2 Model Configurations and Training Details**

**Focused Models**:
- GPT-2 xlarge: 1.5B parameters
- GPT-2 large: 770M parameters
- GPT-2 medium: 355M parameters

**Model Variants**:
- Adjusted number of layers and hidden size
- Trained using OpenWebText dataset with data repetition
- All models trained on a single node with 3 A100 GPUs
- Scaled attention logits inversely to the layer index

**Hyperparameter Details**:
- **GELU activations**, disabled bias terms, and removed dropout
- **AdamW optimizer** with β1=0.90 and β2=0.95
- **Batch sizes**, learning rates, warmup steps, and scheduler types
- **Weight decay**, gradient clipping values, and total training steps

**Knowledge Distillation Training**:
- Used a distillation based training loss
- Softmax temperature: 1
- α: 0.6
- Batch size: 50K tokens
- Learning rate, warmup steps, scheduler type, weight decay, and gradient clipping value
- Total training steps: 50K

**Reference Models and Baselines**:
- GPT2-xlarge(1.5B), GPT2-large(770M), GPT2-large(680M), GPT2-medium, GPT2-large, and GPT2-medium variants

### C.2 Training details of 1.5B OpenLLaMA model

**Our-1.5B Model Training with OpenLLaMA Version 1 and 1B Tokens:**
* Our main results presented using Our-1.5B model trained on:
  + Existing OpenLLaMA version 1 (Geng & Liu, [2023](https://arxiv.org/html/2404.08634v2#bib.bib16))
  + 1B tokens randomly sampled from 1T redpajama version1 data
* Hyperparameters:
  + Training tokens: 1B
  + Training epochs: 8
  + Training steps: 64K
  + Learning rate: 3×10^(-4)
  + Scheduler: Cosine
  + Weight decay: 0.1
  + Optimizer: AdamW
  + Warm up steps: 1000
  + Batch size: 131K tokens
  + GPU hours: ~8 hours
* Consistent training details for submodels (Figure [11](https://arxiv.org/html/2404.08634v2#A2.F11)) with increasing layers leading to longer training times

**Hyperparameter Details of Our 1.5B Base LM:**
* Derived using OpenLLaMA-3B as reference LM
* Training tokens: 1B
* Training epochs: 8
* Training steps: 64K
* Learning rate: 3×10^(-4)
* Scheduler: Cosine
* Weight decay: 0.1
* Optimizer: AdamW
* Warm up steps: 1000
* Batch size: 131K tokens
* GPU hours: ~8 hours/epoch (~54 minutes)
* GPU count: 1
* GPU type: A6000

**Training Details of Small Base LMs with 50B Data:**
* Our-1.5B model trained on larger subsets of data (Figure [12](https://arxiv.org/html/2404.08634v2#A2.F12))
* All intermediate tokens until 50B are from a single training run
* Key hyperparameters:
  + Training tokens: 50B
  + Training epochs: ~1
  + Training steps: 191K
  + Learning rate: 3×10^(-4)
  + Scheduler: Cosine
  + Weight decay: 0.1
  + Optimizer: AdamW
  + Warm-up steps: 1000
  + Batch size: 131K tokens
* Training details consistent across all models trained with 50B subset of pre-train data

**Training Hyperparameters for Target 1.5B and 1.6B Small Base LMs:**
* Training tokens: 50B
* Training epochs: ~1
* Training steps: 191K
* Learning rate: 3×10^(-4)
* Scheduler: Cosine
* Weight decay: 0.1
* Optimizer: AdamW
* Warm-up steps: 1000
* Batch size: 131K tokens
* GPU hours: ~18 hours
* GPU count: 1
* GPU type: A100

