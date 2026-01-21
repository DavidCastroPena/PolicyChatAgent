# PolicyChat — An AI Copilot for Evidence-Based Policy Research 🤖📄⚖️

PolicyChat is an AI-powered research assistant that helps policymakers, analysts, and researchers discover, evaluate, and synthesize high-quality academic evidence for public policy decisions.

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
