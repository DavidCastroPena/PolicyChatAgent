PolicyChatAgent — Improvements Summary

Date: 2026-01-20

This document consolidates the recent production-grade improvements made to PolicyChatAgent (summary, files changed, rationale, and how the pieces integrate). Use this as a single reference to understand what changed and where to look.

**High-level summary**
- **Unified query expansion**: query expansion logic was moved into `retriever/multi_source_retriever.py` and the expansion prompt was enriched with explicit guardrails for research-style academic search variants (jurisdiction variants, methodology terms, synonyms, angles, hard caps on number of outputs, and deterministic style).
- **Evidence quality rating (EQR) improvements**: `retriever/evidence_quality.py` now performs a 6-step rubric: classify paper type, apply type-based caps, compute scores prioritizing policy mechanism relevance, apply hard caps for mismatch or methods books, compute deterministic confidence, and assign tier. The EQR result now includes `paper_type` and `policy_relevance` fields.
- **Coordinated question & memo timestamps**: question generation (naive + nuanced) and memo generation now share a single timestamp ID so `comparison_questions_{TS}.txt`, `question_results_{TS}.jsonl`, and `memo_{TS}.md` are aligned for easier traceability.

**Files changed / of interest**
- `retriever/multi_source_retriever.py` — moved and improved the query expansion prompt and unified the expansion logic.
- `retriever/evidence_quality.py` — major rubric rewrite; added `paper_type` classification and mechanism relevance scoring, plus deterministic confidence and hard caps.
- `retriever/QuestionsAndAnswers/naiveQuestions.py` — now stores `self.current_timestamp` when saving questions.
- `retriever/QuestionsAndAnswers/nuancedQuestions.py` — accepts `shared_timestamp` to align file naming.
- `retriever/QuestionsAndAnswers/answer_questions.py` — propagates `shared_timestamp` from `naiveQuestions` to `nuancedQuestions`.
- `retriever/QuestionsAndAnswers/generateMemo.py` — updated to retrieve `shared_timestamp` from the `QuestionAnswerer` instance and name memo file accordingly.
- `ux.py` — Streamlit UI reads `memo_path` returned by the coordinator and attempts to render it if present.

**Integration details / Flow**
1. `NaiveQuestions.generate_comparison_questions()` creates `comparison_questions_{TS}.txt` and sets `self.current_timestamp = TS`.
2. `AnswerQuestions.retrieve_naive()` runs `NaiveQuestions`, stores `self.shared_timestamp = naive_questions.current_timestamp`.
3. `NuancedQuestions` is instantiated with `shared_timestamp=...` and writes `question_results_{TS}.jsonl`.
4. `GenerateMemo.run()` executes the `QuestionAnswerer`, then reads `answer.shared_timestamp` (if present) and writes `memo_{TS}.md`.
5. `Coordinator.run_pipeline()` returns `memo_path` to `ux.py` which reads and displays the memo.

**Why this matters**
- Single timestamp per run simplifies correlating generated questions, answers, and memo files for auditing and reproducibility.
- The stronger EQR rubric reduces false high scores for textbooks/method books and emphasizes mechanism relevance for policy decision-making.
- Enriched query expansion yields more robust, search-engine-friendly academic queries (suitable for Semantic Scholar, OpenAlex, CrossRef).

**How to verify / run tests**
1. Activate your venv.
2. Run the test file(s):

```powershell
.venv\\Scripts\\python -m pytest test_multi_source.py -q
```

If tests fail, inspect `retriever/multi_source_retriever.py` and `retriever/evidence_quality.py` for runtime errors or missing imports.

**Where to look for outputs after a pipeline run**
- Questions: `reports/questions/comparison_questions_{TS}.txt`
- Question results: `reports/questions/question_results_{TS}.jsonl`
- Memos: `reports/memos/memo_{TS}.md`

**Notes & next steps**
- If `ux.py` shows "No memo file found", check that `memo_path` in coordinator contains the absolute path to the memo file (the generator returns absolute path if file exists).
- Consider adding a `reports/index.md` or machine-readable `runs.csv` for future runs to list TS, query, memo_path, and top_papers for faster navigation.

If you want, I can:
- Run the test suite now and fix any generated failures.
- Add a small `reports/runs.md` that lists recent run timestamps and links to files.

— PolicyChatAgent team, 2026-01-20

---

## Detailed changes: `retriever/evidence_quality.py`

- Implemented a 6-step LLM rubric executed deterministically by the rater:
	- STEP 1 — Paper type classification into explicit buckets (OUTCOME_STUDY, PROTOCOL, THEORY_ETHICS, METHODS_HANDBOOK, REVIEW_META, DESCRIPTIVE, UNKNOWN).
	- STEP 2 — Apply type-based caps for scores (e.g., methods handbooks and protocols cannot receive full practice-grade scores).
	- STEP 3 — Scoring prioritizes **policy mechanism relevance** (0–20 points) as the single most important criterion, then methodology (0–30), then reproducibility/transparency, population/setting relevance, venue (reduced weight), and impact metrics (binned).
	- STEP 4 — Hard caps applied automatically (e.g., EQR ≤ 75 if mechanism mismatch; EQR ≤ 60 for methods books) to prevent misleading high rankings.
	- STEP 5 — Deterministic confidence computed from observable metadata and rubric agreement (not a free-text probability).
	- STEP 6 — Tier assignment (A/B/C/D) with clear numeric boundaries and output fields for `paper_type` and `policy_relevance`.

- Output schema updates:
	- The EQR result object now includes `paper_type`, `policy_relevance` (0–20), `confidence` (deterministic), `score` (0–100), and `tier`.

- Rationale: These changes reduce false positives from textbooks, methods handbooks, and unrelated literature while emphasizing whether the paper actually evaluates the mechanism the user cares about.

## Detailed changes: `retriever/multi_source_retriever.py`

- Consolidated query expansion logic into `build_query_expansions()` inside `multi_source_retriever.py` (removed `retriever/analyze_query.py`).
- Enriched expansion prompt to produce high-quality academic search queries:
	- Produces N=10 academically-oriented variants per user query.
	- Varies specificity and scope (national vs subnational vs city), includes jurisdiction synonyms, and targeted policy mechanisms (RCT, DID, regression discontinuity, propensity score matching, natural experiments).
	- Generates multiple analytical angles: equity, cost-effectiveness, implementation barriers, unintended consequences, gendered impacts.
	- Adds useful search tokens (population descriptors, measurement endpoints, date ranges) and synonyms to improve recall across Semantic Scholar, OpenAlex, CrossRef.
	- Enforces a strict, machine-parseable output format for downstream pipelines.

- Implementation notes:
	- The function calls the configured LLM (`gemini-2.0-flash-exp`) to generate expansions with a fixed generation config.
	- Prompts are designed for reproducibility (low temperature, deterministic instruction framing) and to avoid hallucinated citations.

---

If you'd like, I can also:
- Run the test suite now and fix any failing tests related to these modules.
- Add a compact `reports/runs.md` listing recent run timestamps and links to files for quick navigation.

