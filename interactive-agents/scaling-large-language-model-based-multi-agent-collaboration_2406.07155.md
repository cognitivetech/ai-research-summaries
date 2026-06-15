# Scaling Large Language Model-based Multi-Agent Collaboration

**Source**: [arXiv:2406.07155](https://arxiv.org/html/2406.07155)

**Authors:** Chen Qian, Zihao Xie, YiFei Wang, Wei Liu, Kunlun Zhu, Hanchen Xia, Yufan Dang, Zhuoyun Du, Weize Chen, Cheng Yang, Zhiyuan Liu, Maosong Sun

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Multi-Agent Collaboration Network](#2-multi-agent-collaboration-network)
- [2.1 Network Construction](#21-network-construction)
- [2.2 Interactive Reasoning](#22-interactive-reasoning)
- [2.3 Memory Control](#23-memory-control)
- [3 Evaluation](#3-evaluation)
- [3.1 Does Our Method Lead to Improved Performance?](#31-does-our-method-lead-to-improved-performance)
- [3.2 How Do Different Topologies Perform Against Each Other?](#32-how-do-different-topologies-perform-against-each-other)
- [3.3 Could A Collaborative Scaling Law Be Observed?](#33-could-a-collaborative-scaling-law-be-observed)
- [3.4 What Factors Might Contribute to Collaborative Emergence?](#34-what-factors-might-contribute-to-collaborative-emergence)
- [4 Related Work](#4-related-work)
- [5 Conclusion](#5-conclusion)

## Abstract
**[Key point]**:
- Multi-agent collaboration using Large Language Models (LLMs) often outperforms individual agents through collective reasoning.
- The study investigates whether adding more collaborative agents yields performance gains similar to neural scaling laws in single models.

**Main contribution**:
- Introduction of MacNet (Multi-Agent Collaboration Network), a framework organizing agents via Directed Acyclic Graphs (DAGs).
- Demonstration of effective collaboration among over 1,000 agents.
- Identification of a "collaborative scaling law" where performance follows a logistic growth pattern.

**Key results:**
- Irregular topologies in MacNet outperform regular topologies.
- Performance scales with the number of agents, exhibiting a logistic growth curve.
- Collaborative emergence occurs earlier than traditional neural emergence in single models.
- Scaling agents enhances multidimensional considerations during interactive reflection.

**Conclusion:**
- Continuous addition of collaborative agents catalyzes comprehensive artifact production.
- The code for MacNet is available on GitHub under the ChatDev repository.

## 1 Introduction
**Context:**
- Recent breakthroughs in autonomous agents driven by LLMs have highlighted the potential of multi-agent systems.
- Collective reasoning in multi-agent setups frequently surpasses the capabilities of individual agents.
- Neural scaling laws demonstrate that increasing neurons in a single model enhances performance.
- It remains unclear if similar scaling benefits apply to the number of collaborating agents.

**Goal:**
- To explore whether continuously adding collaborative agents yields similar performance benefits to neural scaling.
- To design a scalable architecture for organizing multi-agent interactions.
- To empirically evaluate the impact of agent count and network topology on task performance.

**Contributions:**
- Proposal of MacNet, a framework using DAGs to orchestrate interactive reasoning for autonomous task solving.
- Extensive evaluations supporting collaboration among over a thousand agents.
- Discovery that irregular topologies are superior to regular ones in this context.
- Identification of a collaborative scaling law characterized by logistic growth.
- Analysis of factors contributing to collaborative emergence, specifically multidimensional considerations.

## 2 Multi-Agent Collaboration Network
**Context:**
- Existing multi-agent systems often lack scalable structures for large numbers of agents.
- Unstructured communication leads to inefficiencies and information loss.
- A structured approach is needed to manage complex interactions and data flow.

**Goal:**
- To define a topological structure that facilitates efficient and scalable multi-agent collaboration.
- To ensure that information flows logically and effectively between agents.

**Contributions:**
- Definition of MacNet as a Directed Acyclic Graph (DAG) based network.
- Specification of rules for agent placement and connection within the DAG.
- Prevention of cycles to ensure deterministic and finite reasoning paths.

### 2.1 Network Construction
**Context:**
- The structure of the network dictates how agents interact and pass information.
- Nodes represent individual agents, while edges represent communication channels.

**Goal:**
- To construct a DAG that supports the specific requirements of the task.
- To allow for flexibility in topology design (regular vs. irregular).

**Contributions:**
- Agents are assigned roles and placed as nodes in the DAG.
- Edges are established based on dependency and information flow requirements.
- The construction process allows for the integration of over 1,000 agents.
- Topology can be customized to optimize for specific task characteristics.

### 2.2 Interactive Reasoning
**Context:**
- Agents must not only process information but also refine it through interaction.
- Simple pass-through of information is insufficient for complex tasks.

**Goal:**
- To enable agents to engage in reflective and iterative reasoning.
- To ensure that outputs from previous agents inform the inputs of subsequent agents.

**Contributions:**
- Agents perform local reasoning based on received information.
- Outputs are passed along the DAG edges to downstream agents.
- Downstream agents refine, critique, or expand upon the received information.
- The process continues until the final node produces the final artifact.

### 2.3 Memory Control
**Context:**
- As the number of agents increases, the volume of information grows exponentially.
- Uncontrolled memory usage leads to context window overflow and degradation.

**Goal:**
- To manage the memory footprint of the collaboration network.
- To ensure that only relevant information is retained and passed on.

**Contributions:**
- Implementation of memory compression techniques.
- Selection of key information points for propagation.
- Discarding of redundant or low-value intermediate states.
- Efficient handling of long-context dependencies across the DAG.

## 3 Evaluation
**Context:**
- Empirical validation is required to assess the efficacy of MacNet.
- Comparisons must be made against baseline methods and varying parameters.

**Goal:**
- To evaluate performance improvements with increasing agent counts.
- To compare different network topologies.
- To identify and characterize the collaborative scaling law.
- To analyze the factors driving collaborative emergence.

**Contributions:**
- Comprehensive benchmarking on multiple tasks.
- Quantitative analysis of performance metrics.
- Qualitative analysis of agent interactions and reasoning paths.

### 3.1 Does Our Method Lead to Improved Performance?
**Context:**
- The primary question is whether MacNet improves upon single-agent or smaller multi-agent baselines.

**Goal:**
- To measure the absolute performance gain provided by MacNet.

**Contributions:**
- MacNet significantly outperforms single-agent baselines.
- Performance gains are consistent across various task types.
- The improvement is attributed to the collective reasoning capability.
- Results demonstrate the viability of large-scale agent collaboration.

### 3.2 How Do Different Topologies Perform Against Each Other?
**Context:**
- The structure of the DAG may influence the efficiency and effectiveness of collaboration.

**Goal:**
- To compare regular topologies (e.g., linear, tree) against irregular topologies.

**Contributions:**
- Irregular topologies consistently outperform regular ones.
- Regular topologies may suffer from bottlenecks or information silos.
- Irregular structures allow for more diverse and robust information flow.
- The flexibility of irregular DAGs is crucial for complex tasks.

### 3.3 Could A Collaborative Scaling Law Be Observed?
**Context:**
- If performance scales with agent count, a law similar to neural scaling might exist.

**Goal:**
- To determine the relationship between the number of agents and performance.

**Contributions:**
- Performance follows a logistic growth pattern as agents scale.
- Initial increases in agents yield rapid performance gains.
- Gains plateau as the number of agents becomes very large.
- This pattern is defined as the "collaborative scaling law."

### 3.4 What Factors Might Contribute to Collaborative Emergence?
**Context:**
- Understanding the mechanism behind the scaling law is essential for optimization.

**Goal:**
- To identify why and how collaborative emergence occurs.

**Contributions:**
- Scaling agents catalyzes multidimensional considerations.
- Agents engage in deeper interactive reflection and refinement.
- The diversity of perspectives leads to more comprehensive artifacts.
- Collaborative emergence occurs earlier than in single-model neural scaling.

## 4 Related Work
**Context:**
- Previous research has explored multi-agent systems and LLM scaling.

**Goal:**
- To position MacNet within the existing literature.
- To highlight the novelty of the collaborative scaling approach.

**Contributions:**
- Review of early multi-agent frameworks.
- Discussion of limitations in existing scalable multi-agent architectures.
- Comparison with single-model neural scaling laws.
- Identification of gaps in understanding large-scale agent collaboration.

## 5 Conclusion
**Context:**
- The study concludes that multi-agent collaboration is a viable path for scaling AI capabilities.

**Goal:**
- To summarize the key findings and implications.

**Contributions:**
- MacNet successfully supports collaboration among over 1,000 agents.
- Irregular topologies are preferred for complex tasks.
- A collaborative scaling law is identified, following logistic growth.
- Future work should focus on optimizing topology and memory control.
- The approach offers a new paradigm for enhancing LLM capabilities through collaboration.