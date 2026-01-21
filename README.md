# PolicyChat — An AI Copilot for Policymakers 🤖📄⚖️

PolicyChat is an AI-powered research assistant that helps policymakers, analysts, and researchers discover, evaluate, and synthesize high-quality academic evidence for public policy decisions.

The system blends ideas from:

- Retrieval-Augmented Generation (RAG)  
- Tool-using LLM agents  
- Evidence-aware ranking and causal inference heuristics  
- Multi-document question answering and synthesis 

Unlike generic chatbots, PolicyChat does not just retrieve documents or summarize them.  
It builds a structured, evidence-aware pipeline that:

- Retrieves papers from multiple academic sources  
- Grades evidence quality using a transparent rubric  
- Generates comparative questions across studies  
- Synthesizes findings into structured policy memos  

The goal is to make **evidence-based policymaking faster, more rigorous, and more transparent**.

---

## ✨ What Makes PolicyChat Different

Compared to tools like ChatGPT or simple RAG systems, PolicyChat adds several layers that are critical for policy work:

### 1. Evidence-Quality–Aware Retrieval
Papers are not only ranked by semantic similarity, but also scored with an **Evidence Quality Rating (EQR)** based on:

- Methodological rigor (RCT, quasi-experimental, observational, etc.)
- Data quality and measurement
- Context match to the user’s geography and population
- Recency and policy relevance
- Venue credibility and scholarly impact
- Transparency and reproducibility

This allows the system to prefer **high-credibility causal evidence** over weak or irrelevant sources.

---

### 2. Comparative Question Generation
Instead of summarizing papers independently, PolicyChat:

- Generates structured comparison questions (mechanisms, outcomes, heterogeneity, limitations)
- Asks each paper the same set of questions
- Filters out papers that cannot answer the comparison schema

This enables **cross-paper reasoning** rather than isolated summaries.

---

### 3. Multi-Source Academic Retrieval
PolicyChat integrates multiple academic metadata endpoints:

- Semantic Scholar  
- OpenAlex  
- World Bank / policy working papers (where available)  

Results are deduplicated, ranked, and filtered before being passed to the LLM.

---

### 4. Policy-Focused Synthesis
The final output is a **policy memo**, not a chat answer:

- Comparison tables across studies  
- Structured findings  
- Evidence-weighted recommendations  
- Explicit sourcing and traceability  

This is designed for analysts, policy teams, and research workflows.

---

## 🧠 Architecture (High-Level)

The system follows an agentic research pipeline:



Each layer acts as an **autonomous reasoning module** with explicit inputs, outputs, and failure modes.

---

## 🧠 Layer 1 — Situation & Context Builder (World Model Agent – Planned)

> **Role:** Build a lightweight internal representation of the policy environment before retrieval.

This agent performs **intent parsing + context grounding**:

- Named entity extraction:
  - Jurisdiction (city / country / region)  
  - Population group  
  - Policy instrument / intervention  
- Policy domain classification (transport, education, social protection, etc.)

Planned extensions:

- Retrieval of **external situational priors**:
  - Baseline indicators (poverty rate, congestion, enrollment, etc.)
  - Known flagship programs (e.g., TransMilenio, Bolsa Família, Progresa)  
  - Institutional constraints  

This corresponds to the **“World Model / Situation Layer”** used in advanced agent systems (research copilots, analyst agents).

Without this layer, the system behaves as a **context-blind RAG agent**.

---

## 🔎 Layer 2 — Evidence Retrieval & Ranking  
### Multi-Agent Retrieval-Augmented Generation (RAG)

This layer implements a **tool-using retrieval agent** that maximizes recall while preserving evidence quality.

### 🔹 Query Planner Agent
- Uses LLM-based query expansion  
- Decomposes the user question into:
  - Core mechanism queries  
  - Policy instrument variants  
  - Regional / jurisdiction hints  

This is a form of **semantic query rewriting** and **multi-hop retrieval planning**.

---

### 🔹 Multi-Source Retrieval Agents

Parallel retrievers query heterogeneous scholarly APIs:

- Semantic Scholar  
- OpenAlex  
- (Optional) Local corpora  

Outputs:
- Deduplicated paper graph  
- Metadata + abstracts  
- Coverage diagnostics (abstract rate, policy reports, API usage)

This resembles a **federated retrieval layer** with provenance tracking.

---

### 🔹 Evidence Quality Rater Agent (Core Innovation)

Instead of ranking by relevance alone, PolicyChat uses an **LLM-based Evidence Scoring Agent**.

This agent performs:

- **Method classification** (RCT, quasi-experimental, observational, qualitative, etc.)  
- **Rubric-based causal credibility scoring** across:
  - Identification strategy  
  - Data quality  
  - External validity  
  - Venue reputation  
  - Transparency  
  - Scholarly impact  

Output:

- EQR score (0–100)  
- Tier (A/B/C/D)  
- Method class  
- Context scope  
- Structured rubric breakdown  

This introduces a **causal-aware ranking layer**, closer to how human economists and policy analysts filter literature.

> This replaces naive “semantic similarity RAG” with an **evidence-aware retrieval system**.

Only top-tier papers survive downstream.

---

## 🧠 Layer 3 — Comparative Reasoner  
### Multi-Document QA + Schema Induction Agent

This layer performs **structured multi-document reasoning** rather than free-form summarization.

---

### 🔹 Question Generator Agents

Two cooperating agents induce a **shared comparison schema**:

- **Naive Question Agent**  
  - Identifies baseline mechanisms, outcomes, interventions  

- **Nuanced Question Agent**  
  - Probes heterogeneity, assumptions, identification threats, external validity  

This implements a form of:

- **Schema induction**  
- **Cross-document alignment**  
- **Hypothesis-driven reading**

---

### 🔹 Per-Paper Answering Agent

For each selected paper:

- Loads abstract or partial full text  
- Executes constrained question answering  
- Produces structured evidence slots  

Papers that fail to answer the schema are automatically filtered.

This stage corresponds to:

- **Multi-document QA**  
- **Evidence extraction with abstention**  
- **Grounded reasoning under partial observability**

---

## 📝 Layer 4 — Policy Synthesizer  
### Grounded Long-Form Generation Agent

The final agent performs **retrieval-grounded synthesis**:

Inputs:
- Ranked evidence set  
- Cross-paper comparison matrix  
- Methodology metadata  
- Warnings and gaps  

Outputs:
- Evidence Quality Summary  
- Ranked Top Evidence Table  
- Comparative Findings Section  
- Policy Recommendations grounded in mechanisms  

This is a form of:

- **Evidence-constrained generation**  
- **Comparative summarization**  
- **Chain-of-thought over documents**  

Unlike generic chat models, this agent is **not allowed to hallucinate outside the retrieved evidence**.

---

## ✨ Why This Architecture Matters (vs. ChatGPT)

PolicyChat introduces several agentic/NLP capabilities that generic chat systems lack:

### 🔹 1. Tool-Using Retrieval Agents
- Multi-API academic search  
- Query planning + expansion  
- Provenance tracking  

### 🔹 2. Causal-Aware Ranking
- Explicit modeling of:
  - Identification strategies  
  - Internal vs. external validity  
- Tiered evidence gating  

### 🔹 3. Schema-Driven Multi-Doc Reasoning
- Automatic generation of shared analytical questions  
- Enables **true cross-study comparison**, not just summarization  

### 🔹 4. Grounded Synthesis
- Recommendations tied to extracted mechanisms  
- Explicit evidence gaps and warnings  

### 🔹 5. Transparent Agent Logs
- Full observability of:
  - Retrieval  
  - Ranking  
  - Filtering  
  - Rate-limits  
  - Paper drop-outs  

This makes the system **inspectable and auditable**, critical for policy workflows

---

## ⚠️ Current Limitation (Next Step)

PolicyChat currently lacks a **World / Situation Model layer**.

The system understands:
- The user question  
- The academic literature  

But it does not yet model:
- The current policy environment  
- Existing programs and institutions  
- Political and fiscal constraints  
- Recent reforms and local context  

Future work will add a **Situation & Context Builder** before retrieval, enabling:

- Country and sector briefings  
- Program mapping (what already exists)  
- Constraint-aware recommendations  

---

## 🚀 Running PolicyChat Locally

### 0. Create environment & install dependencies

```bash
python3 -m venv myenv
source myenv/bin/activate
pipreqs .
pip install -r requirements.txt

##Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
docker start qdrant

##Run Streamlit App
streamlit run ux.py
