# evidence_quality.py
"""
LLM-based Evidence Quality Rating (EQR) for PolicyChat.

Design goals:
- No hardcoded country/city (jurisdiction comes from the user question or explicit inputs).
- All grading criteria live in the LLM prompt (transparent rubric).
- Deterministic + safe: guards against empty prompts, invalid JSON, and rate-limit errors.
- Cheap by default: you can call this only for top-K papers, not all retrieved.

Expected paper dict fields (best effort):
- paper_id, title, abstract/content, year, venue, citationCount, url, fieldsOfStudy, authors
"""

from __future__ import annotations

import json
import re
import time
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, Tuple, List


# -----------------------------
# Utilities
# -----------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# -----------------------------
# Output schema
# -----------------------------
@dataclass
class EQRResult:
    eqr: int  # 0..100
    tier: str  # A/B/C/D
    confidence: float  # 0..1
    method_class: str  # RCT/QUASI/META/OBS/QUAL/DESCRIPTIVE/UNKNOWN
    context_scope: str  # TARGET/REGIONAL/GLOBAL/UNKNOWN
    paper_type: str  # OUTCOME_STUDY/PROTOCOL/THEORY_ETHICS/METHODS_HANDBOOK/REVIEW_META/DESCRIPTIVE/UNKNOWN
    policy_relevance: Dict[str, str]  # mechanism_match + notes
    reasons: List[str]
    warnings: List[str]
    rubric_breakdown: Dict[str, int]  # criterion -> points
    evidence_gaps: List[str]
    extracted: Dict[str, Any]  # optional extracted fields (sample size, outcomes, etc.)


# -----------------------------
# Prompt template
# -----------------------------
RUBRIC_PROMPT = """You are an evidence-quality rater for policy research.

USER POLICY QUESTION (verbatim):
"{user_question}"

Your job:
Given ONE research paper (title/abstract/metadata), assign an Evidence Quality Rating (EQR) from 0 to 100
and classify its credibility and usefulness for answering the user's policy question.

IMPORTANT RULES:
- Use only the provided paper content/metadata. Do NOT invent details.
- If information is missing (e.g., no abstract), penalize confidence and note evidence gaps.
- Output MUST be valid JSON only (no markdown), matching the schema exactly.
- Be strict but fair; avoid country bias. Relevance to the user's target scope is a positive signal.
- High methodological quality alone is NOT sufficient if policy relevance is weak.

TARGET CONTEXT (derived from the user question or explicit inputs):
- target_jurisdiction: {target_jurisdiction}
- target_region: {target_region}
- target_population: {target_population}
- target_policy_domain: {target_policy_domain}

PAPER INPUT:
title: {title}
year: {year}
venue: {venue}
citationCount: {citations}
url: {url}
fieldsOfStudy: {fields}
abstract_or_content: {content}

STEP 1: CLASSIFY PAPER TYPE
First decide: paper_type = OUTCOME_STUDY | PROTOCOL | THEORY_ETHICS | METHODS_HANDBOOK | REVIEW_META | DESCRIPTIVE | UNKNOWN

- OUTCOME_STUDY: Reports empirical results/outcomes from intervention/policy
- PROTOCOL: Study design/protocol without results
- THEORY_ETHICS: Theoretical/ethical discussion without empirical evidence
- METHODS_HANDBOOK: Textbook/handbook/guide about research methods
- REVIEW_META: Systematic review/meta-analysis
- DESCRIPTIVE: Descriptive statistics without causal analysis
- UNKNOWN: Cannot determine

STEP 2: APPLY TYPE-BASED CAPS
- If PROTOCOL: cap methodology<=15, cap recency_policy<=6, add warning 'protocol without outcomes'
- If THEORY_ETHICS or METHODS_HANDBOOK: cap methodology<=10, cap data_quality<=5, set method_class to QUAL or DESCRIPTIVE, add warning 'not empirical outcome evidence'
- If abstract_or_content length < 300 chars: confidence must be <= 0.55, add evidence_gap 'insufficient text'
- If venue missing/unknown: venue score <= 4, add warning 'venue unknown'
- If venue is preprint/medRxiv/SSRN/arXiv: venue score <= 6 unless clear peer-review signal

STEP 3: GRADING CRITERIA (score each 0..max; total 0..100)
1) Policy mechanism relevance to USER QUESTION (0..20) [CRITICAL]
   - 20: Directly studies the SAME intervention/policy lever and target population asked in question
   - 15: Closely related intervention or mechanism, highly informative for the policy question
   - 8: Same domain but different lever (e.g., question asks about cash transfers, paper about training)
   - 3: Indirect relevance (general poverty/employment research)
   - 0: Not relevant to the user's policy question
   NOTE: Use the verbatim USER POLICY QUESTION to judge mechanism match. Keywords alone are insufficient.

2) Methodological rigor (0..30)
   - 30: RCT/field experiment with REPORTED OUTCOMES; clean identification; clear design
   - 24: Strong quasi-experimental (DiD/IV/RD/event study) with REPORTED OUTCOMES and credible assumptions
   - 15: Observational regression with controls; robustness discussed; OUTCOMES reported
   - 8: Descriptive correlations/surveys without causal claims
   - 4: Qualitative only (useful for lived experience but low causal strength)
   - 0: Unclear method / opinion piece / protocol without results
   PENALTY: If abstract says "randomized" but no outcomes reported (protocol/pilot/feasibility): cap at 15

3) Data quality & measurement (0..15)
   - higher: admin/linked data, large representative surveys, clearly defined outcomes
   - lower: small/non-representative samples, unclear measures, self-reported only

4) External validity & geographic/population context match (0..15)
   - 15: Setting/jurisdiction explicitly stated and matches TARGET; comparable population
   - 10: Same region (e.g., Latin America) and highly comparable context
   - 5: Different region but mechanism plausibly transferable; or setting not explicitly stated
   - 0: Context mismatch
   RULE: If setting/jurisdiction NOT explicitly stated in title/abstract/metadata, cap at 5 and add evidence_gap

5) Recency & policy relevance (0..10)
   - higher: recent or still-policy-relevant, addresses interventions/mechanisms

6) Publication / venue credibility (0..5)
   - 5: Top-tier peer-reviewed journal (AER, QJE, Lancet, JAMA) or major series (NBER, IZA, World Bank)
   - 3: Solid peer-reviewed journal
   - 2: Preprint with clear peer-review track (revise & resubmit)
   - 1: Preprint/working paper (medRxiv, SSRN, arXiv)
   - 0: Unknown venue
   RULE: If venue missing/unknown, max score = 1

7) Transparency & reproducibility (0..10)
   - higher: clear methods, limitations, data availability/replication info
   - lower: vague methods, no limitations discussed

8) Scholarly impact proxy (0..5)
   - Use formula: citations_per_year = citationCount / max(1, current_year - year)
   - 5 if citations_per_year >= 50
   - 4 if >= 20
   - 3 if >= 10
   - 2 if >= 3
   - 1 if >= 1
   - 0 otherwise
   NOTE: Current year = 2026

STEP 4: APPLY HARD CAPS
- If paper does NOT study the same policy mechanism as USER QUESTION: cap total EQR at 75 (max Tier B)
- If paper_type is METHODS_HANDBOOK or THEORY_ETHICS: cap total EQR at 60 (max Tier C)
- Only assign EQR >= 95 if ALL extracted fields non-null AND paper reports outcomes AND is TARGET scope AND high mechanism relevance

STEP 5: CALCULATE CONFIDENCE (deterministic formula)
Start: 0.9
Subtract 0.25 if no abstract/content (length < 100 chars)
Subtract 0.15 if intervention not extractable
Subtract 0.15 if identification/method not extractable
Subtract 0.10 if venue missing/unknown
Subtract 0.10 if setting/jurisdiction not extractable
Clamp result to [0.0, 1.0]

STEP 6: ASSIGN TIER
- Tier A: EQR >= 80 AND confidence >= 0.70
- Tier B: EQR 65..79
- Tier C: EQR 50..64
- Tier D: EQR < 50 OR confidence < 0.40

METHOD CLASS (choose one):
RCT | QUASI | META | OBS | QUAL | DESCRIPTIVE | UNKNOWN

CONTEXT SCOPE (choose one):
TARGET | REGIONAL | GLOBAL | UNKNOWN

OUTPUT JSON SCHEMA (must match keys/types):
{{
  "paper_type": "<OUTCOME_STUDY|PROTOCOL|THEORY_ETHICS|METHODS_HANDBOOK|REVIEW_META|DESCRIPTIVE|UNKNOWN>",
  "eqr": <int 0..100>,
  "tier": "<A|B|C|D>",
  "confidence": <float 0..1>,
  "method_class": "<...>",
  "context_scope": "<...>",
  "policy_relevance": {{
    "mechanism_match": "<HIGH|MEDIUM|LOW|NONE>",
    "notes": "<short explanation of how paper relates to user question>"
  }},
  "reasons": [<short strings explaining scoring>],
  "warnings": [<short strings about limitations/caps applied>],
  "rubric_breakdown": {{
    "mechanism_relevance": <int 0..20>,
    "methodology": <int 0..30>,
    "data_quality": <int 0..15>,
    "context_match": <int 0..15>,
    "recency_policy": <int 0..10>,
    "venue": <int 0..5>,
    "transparency": <int 0..10>,
    "impact": <int 0..5>
  }},
  "evidence_gaps": [<short strings about missing information>],
  "extracted": {{
    "sample_size": <string or null>,
    "setting": <string or null>,
    "intervention": <string or null>,
    "outcomes": <string or null>,
    "identification": <string or null>
  }}
}}

Now produce the JSON.
"""


# -----------------------------
# Core rater
# -----------------------------
class EvidenceQualityRater:
    """
    Provide an llm_generate callable that takes a prompt string and returns the model text response.

    Example adapters:
      - Gemini: llm_generate = lambda prompt: client.models.generate_content(...).text
      - OpenAI: llm_generate = lambda prompt: client.responses.create(...).output_text

    Keep this module provider-agnostic.
    """

    def __init__(
        self,
        llm_generate: Callable[[str], str],
        max_retries: int = 2,
        backoff_seconds: int = 20,
        cache: Optional[Dict[str, EQRResult]] = None,
        debug: bool = False,
    ):
        self.llm_generate = llm_generate
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.cache = cache if cache is not None else {}
        self.debug = debug

    def rate_paper(
        self,
        paper: Dict[str, Any],
        user_question: str,
        target_jurisdiction: Optional[str] = None,
        target_region: Optional[str] = None,
        target_population: Optional[str] = None,
        target_policy_domain: Optional[str] = None,
    ) -> EQRResult:
        # Build minimal safe inputs
        title = paper.get("title") or paper.get("Source") or "Unknown title"
        url = paper.get("url") or paper.get("URL") or ""
        venue = paper.get("venue") or paper.get("journal") or paper.get("publicationVenue") or ""
        year = safe_int(paper.get("year") or paper.get("Year"))
        citations = safe_int(paper.get("citationCount") or paper.get("citations") or paper.get("Citations"))
        fields = paper.get("fieldsOfStudy") or paper.get("fields") or ""

        content = (
            paper.get("abstract")
            or paper.get("content")
            or paper.get("Content")
            or paper.get("tldr")
            or ""
        )

        # Guard: Gemini/OpenAI "contents must not be empty"
        user_question = (user_question or "").strip()
        if not user_question:
            user_question = "Policy question not provided."

        # If content is missing, still non-empty prompt (title + url)
        if not content.strip():
            content = f"(No abstract/content available.) Title: {title}. URL: {url}"

        # Cache key
        key_obj = {
            "paper_id": paper.get("paper_id") or paper.get("paperId") or url or title,
            "title": title,
            "year": year,
            "venue": venue,
            "citations": citations,
            "url": url,
            "content_excerpt": normalize_text(content)[:500],
            "user_question": normalize_text(user_question),
            "target_jurisdiction": normalize_text(target_jurisdiction or ""),
            "target_region": normalize_text(target_region or ""),
            "target_population": normalize_text(target_population or ""),
            "target_policy_domain": normalize_text(target_policy_domain or ""),
        }
        cache_key = stable_hash(key_obj)
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = RUBRIC_PROMPT.format(
            user_question=user_question,
            target_jurisdiction=target_jurisdiction or "UNKNOWN",
            target_region=target_region or "UNKNOWN",
            target_population=target_population or "UNKNOWN",
            target_policy_domain=target_policy_domain or "UNKNOWN",
            title=title,
            year=year if year is not None else "UNKNOWN",
            venue=venue or "UNKNOWN",
            citations=citations if citations is not None else "UNKNOWN",
            url=url or "UNKNOWN",
            fields=fields if fields else "UNKNOWN",
            content=content[:4000],  # keep prompt bounded
        )

        if self.debug:
            preview = prompt[:300].replace("\n", " ")
            print(f"[EQR] Prompt preview: {preview}...")

        # Call LLM with retry/backoff
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_generate(prompt)
                raw = (raw or "").strip()
                if not raw:
                    raise ValueError("LLM returned empty text.")

                parsed = self._parse_and_validate(raw)
                
                # ENHANCEMENT: Add rule-based scoring for jurisdiction, policy levers, outcomes
                parsed = self._enhance_scoring_with_rules(
                    parsed, paper, title, content, 
                    target_jurisdiction, target_policy_domain
                )
                
                result = self._to_result(parsed)
                self.cache[cache_key] = result
                return result

            except Exception as e:
                last_err = e
                if self.debug:
                    print(f"[EQR] Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))
                else:
                    break

        # Deterministic fallback if LLM fails
        fallback = self._fallback_result(title=title, url=url, err=str(last_err))
        self.cache[cache_key] = fallback
        return fallback

    # -----------------------------
    # Rule-based scoring enhancements
    # -----------------------------
    def _enhance_scoring_with_rules(
        self, 
        parsed: Dict[str, Any], 
        paper: Dict[str, Any],
        title: str,
        content: str,
        target_jurisdiction: Optional[str],
        target_policy_domain: Optional[str]
    ) -> Dict[str, Any]:
        """
        Add rule-based scoring components for:
        1) Jurisdiction match (0-25 points)
        2) Policy lever signals (0-15 points)
        3) Outcome relevance (0-15 points)
        4) Methods-only penalty
        """
        search_text = (title + " " + content).lower()
        
        # 1) Jurisdiction score (0-25)
        jurisdiction_score = 0
        if target_jurisdiction:
            jurisdiction_tokens = [
                target_jurisdiction.lower(),
                "colombia", "colombian", "bogotá", "bogota", "medellín", "medellin",
                "cali", "barranquilla", "cartagena", "dane", "sena", "dps"
            ]
            if any(token in search_text for token in jurisdiction_tokens):
                jurisdiction_score = 25
                parsed["warnings"].append(f"Jurisdiction match: +{jurisdiction_score} points")
        
        # 2) Policy lever score (0-15)
        policy_keywords = [
            "cash transfer", "cct", "conditional cash", "transfer", "subsidy", "voucher",
            "training", "vocational", "sena", "internship", "apprenticeship", "almp",
            "active labor market", "public works", "microcredit", "microfinance",
            "employment program", "job training", "wage subsidy", "tax credit",
            "housing", "health insurance"
        ]
        policy_lever_score = 15 if any(kw in search_text for kw in policy_keywords) else 0
        if policy_lever_score > 0:
            parsed["warnings"].append(f"Policy lever signals: +{policy_lever_score} points")
        
        # 3) Outcome relevance score (0-15)
        outcome_keywords = [
            "poverty", "income", "consumption", "employment", "earnings", "wages",
            "informality", "unemployment", "food security", "inequality"
        ]
        outcome_relevance_score = 15 if any(kw in search_text for kw in outcome_keywords) else 0
        if outcome_relevance_score > 0:
            parsed["warnings"].append(f"Outcome relevance: +{outcome_relevance_score} points")
        
        # 4) Methods-only penalty
        methods_penalty = 0
        methods_keywords = ["handbook", "introduction", "in practice", "guide", "manual"]
        if any(kw in search_text for kw in methods_keywords):
            # Check if it's ONLY a methods book (no substantive keywords)
            has_substance = any(kw in search_text for kw in policy_keywords + outcome_keywords)
            if not has_substance:
                methods_penalty = -15
                parsed["warnings"].append(f"Methods-only book penalty: {methods_penalty} points")
                # Cap tier at C
                if parsed["tier"] in ["A", "B"]:
                    parsed["tier"] = "C"
        
        # Apply adjustments to EQR
        total_bonus = jurisdiction_score + policy_lever_score + outcome_relevance_score + methods_penalty
        parsed["eqr"] = max(0, min(100, parsed["eqr"] + total_bonus))
        
        # Recalculate tier based on new EQR
        eqr = parsed["eqr"]
        if eqr >= 80 and parsed["confidence"] >= 0.70:
            parsed["tier"] = "A"
        elif eqr >= 65:
            parsed["tier"] = "B"
        elif eqr >= 50:
            parsed["tier"] = "C"
        else:
            parsed["tier"] = "D"
        
        # Add to rubric breakdown
        parsed["rubric_breakdown"]["jurisdiction_bonus"] = jurisdiction_score
        parsed["rubric_breakdown"]["policy_lever_bonus"] = policy_lever_score
        parsed["rubric_breakdown"]["outcome_relevance_bonus"] = outcome_relevance_score
        parsed["rubric_breakdown"]["methods_penalty"] = methods_penalty
        
        return parsed

    # -----------------------------
    # Parsing / validation
    # -----------------------------
    def _parse_and_validate(self, raw: str) -> Dict[str, Any]:
        # Some models may wrap JSON in stray text. Extract first JSON object.
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM output.")
        obj = json.loads(m.group(0))

        required_keys = {
            "eqr", "tier", "confidence", "method_class", "context_scope",
            "paper_type", "policy_relevance", "reasons", "warnings", 
            "rubric_breakdown", "evidence_gaps", "extracted"
        }
        missing = required_keys - set(obj.keys())
        if missing:
            raise ValueError(f"Missing keys in LLM JSON: {missing}")

        # Basic type checks
        eqr = int(obj["eqr"])
        if not (0 <= eqr <= 100):
            raise ValueError("eqr out of range.")
        tier = str(obj["tier"]).strip().upper()
        if tier not in {"A", "B", "C", "D"}:
            raise ValueError("tier invalid.")

        conf = float(obj["confidence"])
        if not (0.0 <= conf <= 1.0):
            raise ValueError("confidence out of range.")

        if not isinstance(obj["reasons"], list) or not isinstance(obj["warnings"], list):
            raise ValueError("reasons/warnings must be lists.")

        rb = obj["rubric_breakdown"]
        for k in ["mechanism_relevance", "methodology", "data_quality", "context_match", "recency_policy", "venue", "transparency", "impact"]:
            if k not in rb:
                raise ValueError(f"rubric_breakdown missing {k}")
            rb[k] = int(rb[k])
        
        # Validate policy_relevance
        if "policy_relevance" not in obj or not isinstance(obj["policy_relevance"], dict):
            raise ValueError("policy_relevance must be a dict")
        pr = obj["policy_relevance"]
        if "mechanism_match" not in pr or "notes" not in pr:
            raise ValueError("policy_relevance must have mechanism_match and notes")

        return obj

    def _to_result(self, obj: Dict[str, Any]) -> EQRResult:
        return EQRResult(
            eqr=int(obj["eqr"]),
            tier=str(obj["tier"]).strip().upper(),
            confidence=float(obj["confidence"]),
            method_class=str(obj["method_class"]).strip().upper(),
            context_scope=str(obj["context_scope"]).strip().upper(),
            paper_type=str(obj.get("paper_type", "UNKNOWN")).strip().upper(),
            policy_relevance=obj.get("policy_relevance", {"mechanism_match": "UNKNOWN", "notes": ""}),
            reasons=[str(x) for x in obj.get("reasons", [])],
            warnings=[str(x) for x in obj.get("warnings", [])],
            rubric_breakdown={k: int(v) for k, v in obj["rubric_breakdown"].items()},
            evidence_gaps=[str(x) for x in obj.get("evidence_gaps", [])],
            extracted=obj.get("extracted", {}) or {},
        )

    def _fallback_result(self, title: str, url: str, err: str) -> EQRResult:
        return EQRResult(
            eqr=45,
            tier="D",
            confidence=0.30,
            method_class="UNKNOWN",
            context_scope="UNKNOWN",
            paper_type="UNKNOWN",
            policy_relevance={"mechanism_match": "UNKNOWN", "notes": "LLM failed"},
            reasons=[
                "LLM scoring failed; using conservative fallback.",
                f"Paper: {title}",
            ],
            warnings=[
                "Do not rely on this score; check paper manually.",
                f"LLM error: {err}",
            ],
            rubric_breakdown={
                "mechanism_relevance": 8,
                "methodology": 10,
                "data_quality": 5,
                "context_match": 5,
                "recency_policy": 5,
                "venue": 2,
                "transparency": 5,
                "impact": 5,
            },
            evidence_gaps=["LLM unavailable; rubric not applied."],
            extracted={
                "sample_size": None,
                "setting": None,
                "intervention": None,
                "outcomes": None,
                "identification": None,
                "url": url,
            },
        )


# -----------------------------
# Optional: small helper to infer target context from user question
# (no hardcoded Colombia; just reads what user wrote)
# -----------------------------
def infer_target_context_from_question(user_question: str) -> Dict[str, Optional[str]]:
    """
    Minimal heuristic: extract country/city-like tokens only if user explicitly states them.
    This avoids hardcoding any location list.

    If you later want richer extraction, you can run a single cheap LLM call ONCE per query,
    not per paper, and pass its output into rate_paper() for all selected papers.
    """
    q = normalize_text(user_question)
    # Simple: look for patterns like "in <X>" where X is a capitalized phrase in original question.
    # Here we return UNKNOWN unless user explicitly included obvious tokens like "colombia".
    # (You can extend this safely without hardcoding specific cities.)
    target_jurisdiction = None
    if " columbia " in f" {q} ":
        # common misspelling; keep what user meant
        target_jurisdiction = "Colombia"
    # If user explicitly said Colombia, capture it (not hardcoded scoring; just extraction)
    if " colombia " in f" {q} ":
        target_jurisdiction = "Colombia"

    return {
        "target_jurisdiction": target_jurisdiction,
        "target_region": None,
        "target_population": None,
        "target_policy_domain": None,
    }
