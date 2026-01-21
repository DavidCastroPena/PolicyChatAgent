import json
import os
import re
import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import numpy as np
from PyPDF2 import PdfReader
from retriever.Chunkenizer import Chunkenizer
from retriever.Embbedingator import Embbedingator
from retriever.PerformQuery import PerformQuery
from retriever.QuestionsAndAnswers.generateMemo import GenerateMemo
from retriever.evidence_quality import EvidenceQualityRater, infer_target_context_from_question
from retriever.multi_source_retriever import smart_retrieve, stable_paper_id
from pathlib import Path 
import time
import google.generativeai as genai
from dotenv import load_dotenv

# === CONFIGURATION ===
MAX_LLM_PAPERS = 8  # Maximum papers to send to Gemini (avoid 429 rate limits)

# === GLOBAL MODEL CACHE ===
_embedding_model_cache = None

def get_embedding_model():
    """Get or create a singleton Embbedingator instance to avoid reloading the model."""
    global _embedding_model_cache
    if _embedding_model_cache is None:
        _embedding_model_cache = Embbedingator()
    return _embedding_model_cache


def safe_llm_prompt(text: str, paper_title: str = "") -> str:
    """
    Ensure LLM prompt is never empty (prevents 'contents must not be empty' error).
    
    Args:
        text: Content to use in prompt
        paper_title: Paper title for fallback
    
    Returns:
        Safe non-empty prompt text
    """
    text = (text or "").strip()
    
    if not text:
        # Fallback: use title + explicit instruction
        if paper_title:
            return f"Paper title: {paper_title}\n\nNote: Full content not available. Please answer based on the title alone, acknowledging this limitation."
        else:
            return "Content unavailable. Please respond: 'Insufficient information to answer this question.'"
    
    return text


def normalize_eqr(eqr_scores):
    """
    Normalize EQR scores to [0, 1] range for ranking.
    
    Args:
        eqr_scores (list): List of EQR scores (0-100)
    
    Returns:
        list: Normalized scores [0, 1]
    """
    if not eqr_scores or len(eqr_scores) == 0:
        return []
    
    min_score = min(eqr_scores)
    max_score = max(eqr_scores)
    
    if max_score == min_score:
        return [1.0] * len(eqr_scores)
    
    return [(s - min_score) / (max_score - min_score) for s in eqr_scores]


def get_paper_id(paper_data):
    """
    Generate canonical paper ID with fallback priority:
    1. paperId (Semantic Scholar)
    2. DOI
    3. URL
    4. normalized title + year
    """
    # Priority 1: Semantic Scholar paperId
    if paper_data.get("paperId"):
        return paper_data["paperId"]
    
    # Priority 2: DOI
    external_ids = paper_data.get("externalIds", {})
    if external_ids and external_ids.get("DOI"):
        return external_ids["DOI"]
    
    # Priority 3: URL
    if paper_data.get("url"):
        return paper_data["url"]
    
    # Priority 4: normalized title + year
    title = paper_data.get("title", "unknown")
    year = paper_data.get("year", "")
    normalized = re.sub(r'[^a-z0-9]', '', title.lower())
    return f"{normalized}_{year}"


def load_user_inputs():
    """
    Load user_inputs.json from CWD first, then fallback to project root.
    This ensures compatibility across local dev, Docker, and cloud deployments.
    """
    cwd_path = os.path.join(os.getcwd(), "user_inputs.json")
    
    if os.path.exists(cwd_path):
        return cwd_path
    
    # Fallback to project root relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fallback_path = os.path.join(base_dir, "user_inputs.json")
    
    return fallback_path


class Coordinator:
    def __init__(self,  message_output=None):
        """
        Initialize the Coordinator with user inputs and pipeline components.
        Args:
            user_inputs_file (str): Path to the JSON file containing user inputs.
        """
        # Load user_inputs.json with robust path resolution
        file_path = load_user_inputs()

        # Open and load the JSON file
        with open(file_path, "r") as json_file:
            self.user_inputs = json.load(json_file)

        # Store message output function
        self.message_output = message_output or print

        # Read inputs with safe defaults
        self.query = self.user_inputs.get("query", "")
        self.papers_folder = self.user_inputs.get("papers_folder")
        # Optional override folder (from sidebar)
        self.optional_local_folder = self.user_inputs.get("optional_local_folder")
        self.local_papers = self.user_inputs.get("local_papers", [])
        self.option = self.user_inputs.get("option", "1")
        self.topic = self.user_inputs.get("topic", "unemployment")  # Default to unemployment if not specified
        
        # Topic validation guardrail
        inferred_topic = self._infer_topic_from_question(self.query)
        self.inferred_topic = None
        topic_lock = bool(self.user_inputs.get("topic_lock", False))
        if inferred_topic and inferred_topic != self.topic:
            self.inferred_topic = inferred_topic
            if topic_lock:
                self.message(f"⚠️ Sidebar topic='{self.topic}' but question looks like '{inferred_topic}'. Topic lock active; keeping sidebar topic.")
            else:
                self.message(f"⚠️ Sidebar topic='{self.topic}' but question looks like '{inferred_topic}'. Using inferred topic for retrieval/scoring. To keep the sidebar topic, set 'topic_lock': true in user_inputs.json")
                self.topic = inferred_topic
        
        self.external_search = bool(self.user_inputs.get("external_search", False))
        self.max_results = int(self.user_inputs.get("max_results", 10))
        self.semantic_scholar_api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # API key support: environment variable `SEMANTIC_SCHOLAR_API_KEY` or user_inputs
        self.api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or self.user_inputs.get("semantic_scholar_api_key")

        # Requests session with retries/backoff
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        # Basic UA and optional API key header
        self.session.headers.update({"User-Agent": "PolicyChatAgent/1.0"})
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})

        # If optional folder provided but no explicit local_papers, discover files
        if not self.local_papers and self.optional_local_folder:
            if os.path.isdir(self.optional_local_folder):
                found = [os.path.join(self.optional_local_folder, f) for f in os.listdir(self.optional_local_folder)
                         if f.lower().endswith('.pdf') or f.lower().endswith('.txt')]
                self.local_papers = found
                self.papers_folder = self.optional_local_folder

        # Initialize components
        self.chunkenizer = Chunkenizer(self.papers_folder if self.papers_folder else ".")
        self.embbedingator = get_embedding_model()  # Use cached model
        # PerformQuery will be initialized only if needed (not used in current pipeline)
        
        # Initialize Gemini for EQR scoring
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash-exp",
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 5,
                    "max_output_tokens": 2048,
                }
            )
            # Create EQR rater with Gemini adapter
            self.eqr_rater = EvidenceQualityRater(
                llm_generate=lambda prompt: model.generate_content(prompt).text,
                max_retries=2,
                backoff_seconds=10,
                debug=True
            )
            # Extract target context from user query
            self.target_context = infer_target_context_from_question(self.query)
        else:
            self.message("⚠️ GEMINI_API_KEY not found - EQR scoring will use fallback mode")
            self.eqr_rater = None
            self.target_context = {}
        
        # Initialize tracking
        self.retrieval_stats = {}
        self.evidence_summary = {}

    def _infer_topic_from_question(self, question):
        """
        Infer topic from question keywords.
        Returns topic name if confidence >= 2 keyword hits, else None.
        """
        if not question:
            return None
        
        q_lower = question.lower()
        
        # Poverty keywords
        poverty_keywords = ["poverty", "poor", "cash transfer", "inequality", "consumption", "pobreza"]
        poverty_count = sum(1 for kw in poverty_keywords if kw in q_lower)
        
        # Unemployment keywords
        unemployment_keywords = ["unemployment", "employment", "job", "labor market", "labour market", 
                                 "wage", "earnings", "empleo", "desempleo"]
        unemployment_count = sum(1 for kw in unemployment_keywords if kw in q_lower)
        
        # Decide based on counts (threshold: >= 2 hits)
        if poverty_count >= 2 and poverty_count > unemployment_count:
            return "poverty"
        elif unemployment_count >= 2 and unemployment_count > poverty_count:
            return "unemployment"
        
        return None

    def get_jurisdiction_tokens(self, user_question):
        """
        Extract jurisdiction-specific tokens for matching.
        Uses inferred target_jurisdiction (if available) or extracts place-like tokens from the question.
        """
        # If a target_jurisdiction was inferred by EQR/analysis, use that first
        if hasattr(self, 'target_context') and self.target_context:
            tj = self.target_context.get('target_jurisdiction')
            if tj:
                tokens = re.findall(r"[a-zA-Z0-9]+", tj.lower())
                return [t for t in tokens if len(t) > 2]

        # Otherwise, extract capitalized tokens from the question as place-name heuristics
        words = user_question.split()
        location_tokens = []
        for word in words:
            clean = re.sub(r'[^a-zA-Z]', '', word)
            if clean and clean[0].isupper() and len(clean) > 3:
                location_tokens.append(clean.lower())

        return location_tokens

    def match_jurisdiction(self, paper, tokens):
        """
        Check if paper matches any jurisdiction tokens.
        Searches in title, abstract, venue, and concepts.
        """
        if not tokens:
            return False
        
        # Combine searchable text
        search_text = " ".join([
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("content", ""),
            paper.get("venue", ""),
            " ".join(paper.get("concepts", []) if isinstance(paper.get("concepts"), list) else [])
        ]).lower()
        
        # Check for any token match
        return any(token in search_text for token in tokens)

    def message(self, text):
        """
        Utility method to output messages
        """
        if self.message_output:
            self.message_output(text)
    
    def deduplicate_papers(self, papers):
        """
        Deduplicate papers by URL, keeping the entry with longer content.
        """
        by_url = {}
        for p in papers:
            url = p.get("url", "")
            if not url:
                continue
            
            if url not in by_url:
                by_url[url] = p
            else:
                # Keep the entry with longer content or higher similarity
                if len(p.get("content", "")) > len(by_url[url].get("content", "")):
                    by_url[url] = p
                elif len(p.get("content", "")) == len(by_url[url].get("content", "")):
                    # If same length, keep higher similarity
                    if p.get("query_similarity", 0) > by_url[url].get("query_similarity", 0):
                        by_url[url] = p
        
        return list(by_url.values())
    
    def is_colombia_relevant(self, text):
        """Deprecated: replaced by generic jurisdiction check.
        Returns True if any jurisdiction token appears in the given text.
        """
        tokens = self.get_jurisdiction_tokens(text or "")
        if not tokens:
            return False
        text_lower = (text or "").lower()
        return any(t in text_lower for t in tokens)
    
    def detect_intervention_study(self, paper):
        """
        Detect if paper is an intervention study (RCT, quasi-experiment, etc.)
        """
        content = (paper.get("title", "") + " " + paper.get("content", "")).lower()
        
        intervention_keywords = [
            'randomized', 'rct', 'experiment', 'quasi-experiment',
            'difference-in-differences', 'did', 'regression discontinuity',
            'instrumental variable', 'impact evaluation', 'program evaluation',
            'treatment effect', 'causal', 'intervention'
        ]
        
        return any(kw in content for kw in intervention_keywords)

    def transform_query_for_academic_search(self, query, colombia_anchored=True):
        """
        Transform a conversational query into academic search terms.
        If Colombia-anchored, adds geographic constraints.
        """
        import re
        
        # Remove conversational phrases
        query_lower = query.lower()
        conversational_phrases = [
            r'\bi want to\b', r'\bhow can i\b', r'\bwhat can i do\b', r'\bhelp me\b',
            r'\bplease\b', r'\bcan you\b', r'\bwould like to\b', r'\bneed to\b',
            r'\bthe scope is\b', r'\bit is focused on\b', r'\bthe stakeholders are\b',
            r'\bsuch as\b', r'\band the\b'
        ]
        for phrase in conversational_phrases:
            query_lower = re.sub(phrase, '', query_lower)
        
        # Check for any jurisdiction tokens
        jurisdiction_tokens = self.get_jurisdiction_tokens(query)
        
        # Split on common sentence/clause boundaries
        parts = re.split(r'[.,;]', query_lower)
        core_query = parts[0].strip()
        
        # Extract key policy terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had'}
        words = core_query.split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Build base query
        academic_query = ' '.join(key_terms[:6])
        
        # Add intervention/policy terms
        policy_terms = ['evaluation', 'impact', 'program', 'policy', 'intervention']
        academic_query += ' ' + ' '.join(policy_terms[:2])
        
        # Add geographic constraint if jurisdiction tokens found, else default to Latin America
        if jurisdiction_tokens:
            academic_query = f"{jurisdiction_tokens[0].title()} {academic_query}"
        else:
            academic_query = f"Latin America {academic_query}"
        
        return academic_query.strip()
    
    def generate_scholarly_queries(self, user_query, topic):
        """
        Generate 3-4 targeted scholarly queries for multi-query retrieval.
        """
        queries = []
        
        # Detect any jurisdiction mention or inferred jurisdiction
        jurisdiction_tokens = self.get_jurisdiction_tokens(user_query)
        
        # Extract core policy terms
        policy_terms = []
        if 'unemployment' in user_query.lower() or topic in ['unemployment', 'female_unemployment']:
            policy_terms = ['unemployment', 'employment', 'labor market']
        elif 'education' in user_query.lower() or topic == 'education':
            policy_terms = ['education', 'training', 'vocational']
        elif 'poverty' in user_query.lower() or topic in ['poverty', 'poverty_reduction']:
            policy_terms = ['poverty', 'welfare', 'social protection']
        
        # Use jurisdiction token if present, otherwise default to Latin America
        prefix = jurisdiction_tokens[0].title() if jurisdiction_tokens else "Latin America"
        queries.append(f"{' '.join(policy_terms[:2])} {prefix}")
        queries.append(f"{policy_terms[0]} developing countries evaluation")
        queries.append(f"youth employment programs randomized {prefix}")
        
        # Add intervention-focused query
        queries.append(f"{' '.join(policy_terms[:1])} Latin America randomized evaluation")
        
        return queries[:4]  # Return top 4 queries

    def process_local_papers(self):
        """
        Process and chunk local papers selected by the user.
        Returns:
            list: List of chunks from local papers.
        """
        self.message("📂 Processing papers in local folder ...")
        chunks = []
        for paper in self.local_papers:
            file_path = paper
            paper_chunks = self.chunkenizer.process_file(file_path)
            for chunk in paper_chunks:
                chunks.append({"source": paper, "content": chunk})
        return chunks

    def fetch_external_papers(self):
        """
        Multi-source external paper retrieval with hardening.
        Uses smart_retrieve from multi_source_retriever module.
        
        Returns:
            list: Scored and ranked papers with stable IDs
        """
        if not self.external_search:
            self.message("🔎 External search disabled.")
            return []
        # Use new multi-source retriever
        self.message("🔎 Retrieving papers from multiple sources...")
        top10, top50, stats = smart_retrieve(self.query, target_pool=50)
        
        # Add stable_id and ensure required fields
        for paper in top50:
            if "stable_id" not in paper:
                paper["stable_id"] = stable_paper_id(paper)
            
            # Ensure required fields exist
            paper.setdefault("content", paper.get("abstract", ""))
            paper.setdefault("eqr", 0)
            paper.setdefault("tier", "D")
            paper.setdefault("eqr_reasons", [])
            paper.setdefault("eqr_warnings", [])
            paper.setdefault("full_text", None)
            paper.setdefault("pdf_path", None)
            paper.setdefault("rank_score", paper.get("retrieval_rank_score", 0.0))
            paper.setdefault("query_similarity", 0.5)  # Default similarity
            paper.setdefault("citation_count", paper.get("citationCount", 0))
            paper.setdefault("is_open_access", False)
        
        # Store stats for evidence summary
        self.retrieval_stats = stats
        
        self.message(f"\n📊 Multi-source retrieval stats:")
        self.message(f"   Total retrieved: {stats['retrieved_count']}")
        self.message(f"   Unique papers: {stats['deduped_count']}")
        self.message(f"   Abstracts present: {stats['abstracts_present']}/{len(top50)} ({stats['abstracts_present_rate']}%)")
        self.message(f"   Policy reports: {stats['policy_report_count']}")
        self.message(f"   API calls used: {stats['api_calls']}")
        
        external_papers = top50
        
        # Create stable_id-keyed map for tracking papers
        all_papers_by_id = {p["stable_id"]: p for p in external_papers}
        
        # Save all retrieved papers for traceability
        import datetime as dt
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("./reports/external_papers", exist_ok=True)
        all_papers_log = f"./reports/external_papers/all_retrieved_papers_{timestamp}.json"
        with open(all_papers_log, "w", encoding='utf-8') as f:
            json.dump([{
                "stable_id": p.get("stable_id", ""),
                "title": p.get("title", ""),
                "year": p.get("year", "N/A"),
                "authors": p.get("authors", ""),
                "url": p.get("url", ""),
                "citation_count": p.get("citation_count", 0),
                "source": p.get("source", "unknown"),
                "source_type": p.get("source_type", "academic_paper"),
                "venue": p.get("venue", ""),
                "retrieval_rank_score": p.get("retrieval_rank_score", 0)
            } for p in external_papers], f, indent=2, ensure_ascii=False)
        self.message(f"📋 Saved full list of {len(external_papers)} papers to {all_papers_log}")
        
        # === SCORE PAPERS WITH EQR ===
        self.message(f"📊 Scoring {len(external_papers)} papers for evidence quality...")
        for paper in external_papers:
            # HARDEN: Ensure content never empty (prevent Gemini "contents must not be empty" error)
            safe_content = safe_llm_prompt(
                paper.get("content") or paper.get("abstract") or "",
                paper.get("title", "Unknown")
            )
            paper["content"] = safe_content
            
            if self.eqr_rater:
                try:
                    eqr_result = self.eqr_rater.rate_paper(
                        paper=paper,
                        user_question=self.query,
                        target_jurisdiction=self.target_context.get("target_jurisdiction"),
                        target_region=self.target_context.get("target_region"),
                        target_population=self.target_context.get("target_population"),
                        target_policy_domain=self.topic
                    )
                    paper["eqr"] = eqr_result.eqr
                    paper["tier"] = eqr_result.tier
                    paper["eqr_reasons"] = eqr_result.reasons
                    paper["eqr_warnings"] = eqr_result.warnings
                    paper["method_class"] = eqr_result.method_class
                    paper["context_scope"] = eqr_result.context_scope
                except Exception as e:
                    self.message(f"⚠️ EQR scoring failed for {paper['title'][:60]}: {e}")
                    # Use fallback scoring
                    paper["eqr"] = 45
                    paper["tier"] = "D"
                    paper["eqr_reasons"] = ["EQR scoring failed"]
                    paper["eqr_warnings"] = [str(e)]
            else:
                # No Gemini API key - use conservative fallback
                paper["eqr"] = 45
                paper["tier"] = "D"
                paper["eqr_reasons"] = ["GEMINI_API_KEY not configured"]
                paper["eqr_warnings"] = ["Using fallback scoring"]
        
        # Log tier breakdown
        tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        filtered_papers = external_papers
        for p in filtered_papers:
            tier_counts[p.get("tier", "D")] += 1
        self.message(f"   📊 Tier distribution: A={tier_counts['A']}, B={tier_counts['B']}, C={tier_counts['C']}, D={tier_counts['D']}")
        
        # === JURISDICTION MATCHING (Colombia-first gating) ===
        jurisdiction_tokens = self.get_jurisdiction_tokens(self.query)
        for paper in filtered_papers:
            paper["jurisdiction_match"] = self.match_jurisdiction(paper, jurisdiction_tokens)

        jurisdiction_count = sum(1 for p in filtered_papers if p.get("jurisdiction_match", False))
        self.message(f"   🌍 Jurisdiction-matching papers: {jurisdiction_count}/{len(filtered_papers)}")
        
        # === RERANK USING EQR + RELEVANCE ===
        # Normalize EQR scores to [0, 1]
        eqr_scores = [p["eqr"] for p in filtered_papers]
        eqr_normalized = normalize_eqr(eqr_scores)
        
        # Calculate final rank score: 55% EQR + 45% query similarity + jurisdiction boost
        for i, paper in enumerate(filtered_papers):
            eqr_norm = eqr_normalized[i] if i < len(eqr_normalized) else 0
            jurisdiction_boost = 0.3 if paper.get("jurisdiction_match", False) else 0
            paper["rank_score"] = 0.55 * eqr_norm + 0.45 * paper["query_similarity"] + jurisdiction_boost
        
        # Sort by combined rank score (descending)
        filtered_papers.sort(key=lambda x: x["rank_score"], reverse=True)
        
        # === JURISDICTION-FIRST GATING: SELECT MAX 8 PAPERS FOR LLM ===
        # Separate jurisdiction-matching from non-matching
        jurisdiction_papers = [p for p in filtered_papers if p.get("jurisdiction_match", False)]
        non_jurisdiction_papers = [p for p in filtered_papers if not p.get("jurisdiction_match", False)]
        
        # Select papers for LLM processing
        selected_for_llm = []
        
        # Priority 1: Jurisdiction-matching papers (fill up to MAX_LLM_PAPERS)
        selected_for_llm.extend(jurisdiction_papers[:MAX_LLM_PAPERS])
        
        # Priority 2: Backfill with non-jurisdiction papers if needed
        backfill_needed = MAX_LLM_PAPERS - len(selected_for_llm)
        if backfill_needed > 0:
            # Prioritize tier A/B from non-jurisdiction
            tier_ab_backfill = [p for p in non_jurisdiction_papers if p["tier"] in ["A", "B"]]
            for paper in tier_ab_backfill[:backfill_needed]:
                paper["backfill"] = True  # Mark as backfill
                selected_for_llm.append(paper)
            
            # If still not enough, add tier C
            if len(selected_for_llm) < 3:
                tier_c_backfill = [p for p in non_jurisdiction_papers if p["tier"] == "C"]
                remaining_slots = min(MAX_LLM_PAPERS - len(selected_for_llm), 3)
                for paper in tier_c_backfill[:remaining_slots]:
                    paper["context_only"] = True
                    paper["backfill"] = True
                    selected_for_llm.append(paper)
        
        # Cap at MAX_LLM_PAPERS
        selected_for_llm = selected_for_llm[:MAX_LLM_PAPERS]
        
        # Calculate evidence quality summary
        jurisdiction_selected = sum(1 for p in selected_for_llm if p.get("jurisdiction_match", False))
        backfill_selected = sum(1 for p in selected_for_llm if p.get("backfill", False))
        
        selected_tier_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for p in selected_for_llm:
            selected_tier_counts[p.get("tier", "D")] += 1
        
        self.message(f"\n🎯 Selected {len(selected_for_llm)}/{len(filtered_papers)} papers for LLM processing (max {MAX_LLM_PAPERS})")
        self.message(f"   Jurisdiction-matching: {jurisdiction_selected}, Backfill: {backfill_selected}")
        self.message(f"   Selected tiers: A={selected_tier_counts['A']}, B={selected_tier_counts['B']}, C={selected_tier_counts['C']}")
        
        # Store evidence summary for memo
        self.evidence_summary = {
            "retrieved_total": len(all_papers_by_id),
            "deduped_count": len(external_papers),
            "scored_count": len(filtered_papers),
            "selected_for_llm": len(selected_for_llm),
            "tier_counts_all": tier_counts,
            "tier_counts_selected": selected_tier_counts,
            "jurisdiction_selected": jurisdiction_selected,
            "backfill_selected": backfill_selected,
            "jurisdiction_total": jurisdiction_count
        }
        
        # Show evidence quality summary
        self.message(f"\n📊 Evidence Quality Summary:")
        self.message(f"   Retrieved: {len(all_papers_by_id)} → Deduped: {len(external_papers)} → Scored: {len(filtered_papers)}")
        self.message(f"   Selected for LLM: {len(selected_for_llm)} (Tier A={selected_tier_counts['A']}, B={selected_tier_counts['B']}, C={selected_tier_counts['C']})")
        self.message(f"   Jurisdiction match: {jurisdiction_selected} selected from {jurisdiction_count} total")
        
        # Debug: reasons for exclusion (helps trace why papers were not selected)
        excluded = [p for p in external_papers if p not in selected_for_llm]
        reason_counts = {}
        for p in excluded:
            reasons = []
            if not p.get('jurisdiction_match'):
                reasons.append('no_jurisdiction')
            if p.get('tier', 'D') in ['C', 'D']:
                reasons.append('low_tier')
            if p.get('query_similarity', 0) < 0.2:
                reasons.append('low_similarity')
            key = ','.join(reasons) if reasons else 'other'
            reason_counts[key] = reason_counts.get(key, 0) + 1

        if reason_counts:
            self.message(f"   🔎 Exclusion reasons summary: {reason_counts}")

        # Widen selection if too few papers selected (backstop)
        min_required = min(4, MAX_LLM_PAPERS)
        if len(selected_for_llm) < min_required:
            self.message(f"   ⚠️ Only {len(selected_for_llm)} selected for LLM; widening selection to at least {min_required}.")
            non_selected = [p for p in filtered_papers if p not in selected_for_llm]
            for p in non_selected:
                if len(selected_for_llm) >= min_required:
                    break
                p['expanded_backfill'] = True
                selected_for_llm.append(p)

        # Use selected_for_llm for downstream processing
        filtered_papers = selected_for_llm
        
        # === UPDATE PDF DOWNLOAD CRITERIA ===
        # Download PDFs for high-quality papers: tier A/B OR eqr >= 65
        for paper in filtered_papers:
                if paper.get("pdf_path"):  # Already downloaded
                    continue
                    
                should_download = (
                    paper["tier"] in {"A", "B"} or 
                    paper["eqr"] >= 65
                ) and paper.get("open_pdf")
                
                if should_download and not paper.get("full_text"):
                    try:
                        self.message(f"⬇️ Downloading high-quality paper (Tier {paper['tier']}, EQR {paper['eqr']}): '{paper['title'][:60]}...'")
                        os.makedirs("./downloaded_papers", exist_ok=True)
                        pdf_fname = re.sub(r"[^a-zA-Z0-9_\-]", "_", paper['title'])[:120] + ".pdf"
                        pdf_path = os.path.join("./downloaded_papers", pdf_fname)
                        pdf_resp = self.session.get(paper["open_pdf"], timeout=30)
                        pdf_resp.raise_for_status()
                        with open(pdf_path, "wb") as pf:
                            pf.write(pdf_resp.content)
                        paper["pdf_path"] = pdf_path

                        # Extract text
                        try:
                            reader = PdfReader(pdf_path)
                            text_pages = []
                            for page in reader.pages:
                                try:
                                    text_pages.append(page.extract_text() or "")
                                except Exception:
                                    text_pages.append("")
                            full_text = "\n\n".join(text_pages).strip()
                            paper["full_text"] = full_text if full_text else None
                            # Update content with full_text
                            if paper["full_text"]:
                                paper["content"] = paper["full_text"][:20000]
                                # Re-score with full text
                                if self.eqr_rater:
                                    try:
                                        eqr_result = self.eqr_rater.rate_paper(
                                            paper=paper,
                                            user_question=self.query,
                                            target_jurisdiction=self.target_context.get("target_jurisdiction"),
                                            target_region=self.target_context.get("target_region"),
                                            target_population=self.target_context.get("target_population"),
                                            target_policy_domain=self.topic
                                        )
                                        paper["eqr"] = eqr_result.eqr
                                        paper["tier"] = eqr_result.tier
                                        paper["eqr_reasons"] = eqr_result.reasons
                                        paper["eqr_warnings"] = eqr_result.warnings
                                    except Exception as e:
                                        self.message(f"⚠️ Re-scoring failed: {e}")
                        except Exception as e:
                            self.message(f"⚠️ Failed extracting PDF: {e}")
                    except requests.exceptions.RequestException as e:
                        self.message(f"⚠️ Failed to download PDF: {e}")
        
        # Log filtering results
        if len(filtered_papers) < len(external_papers):
            self.message(f"🔍 Filtered to {len(filtered_papers)} most relevant papers (from {len(external_papers)}) based on semantic similarity")
        
        # Show top papers with EQR scores
        self.message(f"\n🏆 Top papers by evidence quality:")
        for i, p in enumerate(filtered_papers[:5], 1):
            self.message(f"{i}. [{p['tier']}] EQR {p['eqr']}: {p['title'][:80]}...")
            
        # Update all_contents and content_by_title to only include filtered papers
        all_contents = [p["content"] for p in filtered_papers]
        content_by_title = {p["title"]: p["content"] for p in filtered_papers}
        
        self.message(f"⛳️ I retrieved {len(filtered_papers)} relevant papers from Semantic Scholar API.")
        return filtered_papers, all_contents, content_by_title


    def process_external_papers(self, external_papers):
        """
        Chunk external papers retrieved from Genie API.
        Args:
            external_papers (list): List of external paper contents.

        Returns:
            list: List of chunks from external papers.
        """
        self.message("⚙️ Processing papers retrieved form Genie API ...")
        chunks = []
        for paper in external_papers:
            paper_chunks = self.chunkenizer.chunk_text(paper["content"])
            for chunk in paper_chunks:
                chunks.append({
                    "source": paper["title"],
                    "content": chunk,
                    "url": paper["url"]
                })
        return chunks

    def calculate_similarities(self, chunks):
        """
        Calculate similarity scores between the query and each chunk.
        Args:
            chunks (list): List of chunks to compare.

        Returns:
            list: List of chunks with similarity scores.
        """
        import numpy as np
        
        query_embedding = self.embbedingator.embed_text(self.query)
        results = []
        for chunk in chunks:
            chunk_embedding = self.embbedingator.embed_text(chunk["content"])
            # Calculate cosine similarity directly
            dot_product = np.dot(query_embedding, chunk_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_chunk = np.linalg.norm(chunk_embedding)
            similarity = dot_product / (norm_query * norm_chunk)
            results.append({
                "source": chunk["source"],
                "content": chunk["content"],
                "similarity": float(similarity),  # Ensure JSON serialization compatibility
                "url": chunk.get("url")  # Include URL if available
            })
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

    def save_results(self, results, report_name, include_url=False):
        """
        Save results to a JSONL report file.
        Args:
            results (list): List of similarity results.
            report_name (str): Name of the report file.
            include_url (bool): Whether to include URL in the report (for external papers).

        Returns:
            str: Path to the saved report.
        """
        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", report_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine subdirectory based on report type
        if "local" in report_name.lower():
            report_dir = "./reports/local_papers"
        elif "external" in report_name.lower():
            report_dir = "./reports/external_papers"
        elif "combined" in report_name.lower():
            report_dir = "./reports/combined"
        else:
            report_dir = "./reports"
        
        # Create directory if it doesn't exist
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = f"{report_dir}/{sanitized_name}_{timestamp}.jsonl"

        with open(report_path, "w") as f:
            for result in results:
                # Prepare the JSON line
                report_line = {
                    "Source": result["source"],
                    "Content": result["content"],
                    "Similarity Score": result["similarity"]
                }
                # Add URL for external papers
                if include_url and result.get("url"):
                    report_line["URL"] = result["url"]

                json.dump(report_line, f)
                f.write("\n")

        print(f"Report saved to {report_path}")
        return report_path

    def run_pipeline(self):
        """
        Execute the pipeline based on user inputs.
        Returns a dict with top_papers for UI display.
        """
        self.message("🚀 Starting research pipeline ...")

        all_external_contents = []
        content_by_title = {}
        self.top_papers = []  # Store top papers for UI

        # Process local papers only if any were provided or discovered
        local_results = []
        if self.local_papers:
            local_chunks = self.process_local_papers()
            print("Calculating similarities for local papers...")
            local_results = self.calculate_similarities(local_chunks)
            self.save_results(local_results, "local_papers_report")
        else:
            self.message("ℹ️ No local papers provided; skipping local processing.")

        # External search if requested
        external_results = []
        if self.external_search:
            print("Fetching and processing external papers...")
            external_papers, all_external_contents, content_by_title = self.fetch_external_papers()

            # Only proceed if we actually got papers
            if external_papers:
                # Store top papers for UI (these already have EQR scores from fetch_external_papers)
                self.top_papers = external_papers[:10]  # Top 10 papers with EQR scores
                
                external_chunks = self.process_external_papers(external_papers)

                print("Calculating similarities for external papers...")
                external_results = self.calculate_similarities(external_chunks)
                self.save_results(external_results, "external_papers_report", include_url=True)
            else:
                self.message("⚠️ No external papers retrieved. The pipeline will stop here.")
                return {"top_papers": []}  # Return empty for UI
        else:
            self.message("ℹ️ External search not enabled.")

        # Combine results and save
        combined_results = local_results + external_results
        if combined_results:
            self.save_results(combined_results, "combined_report", include_url=True)
        else:
            self.message("⚠️ No papers to analyze. Please check your settings or try a different query.")
            return {"top_papers": []}  # Return empty for UI
        
        # === STEP 5: SEND SELECTED PAPERS TO GEMINI ===
        # Papers already filtered to MAX_LLM_PAPERS in fetch_external_papers()
        if hasattr(self, 'top_papers') and self.top_papers:
            papers_for_memo = self.top_papers  # Already gated at max 8
            
            # Prepare content for memo
            all_external_contents = [p.get("content", "") for p in papers_for_memo]
            content_by_paper_id = {p.get("paper_id", p.get("title", "")): p.get("content", "") for p in papers_for_memo}
            
            # Debug: Check content lengths
            for paper_id, content in content_by_paper_id.items():
                self.message(f"DEBUG: Paper '{paper_id[:60]}...' has {len(content)} chars of content")
            
            self.message(f"\n📝 Generating memo from {len(papers_for_memo)} selected papers (max {MAX_LLM_PAPERS})...")
        else:
            all_external_contents = []
            content_by_paper_id = {}
        
        # Generate memo with evidence summary
        memo = GenerateMemo(message_output=self.message_output)
        
        # Pass evidence summary to memo generator
        if hasattr(self, 'evidence_summary'):
            memo.evidence_summary = self.evidence_summary

        # Extract the 'query' field
        user_query = self.user_inputs.get("query")
        memo_path = memo.run(all_external_contents, content_by_paper_id, user_query)
        
        return {"top_papers": self.top_papers if hasattr(self, 'top_papers') else [], "memo_path": memo_path}


if __name__ == "__main__":
    # Run the pipeline
    coordinator = Coordinator()
    coordinator.run_pipeline()
