"""
Multi-Source Evidence Retrieval for PolicyChat

Retrieves academic papers and policy reports from:
- Semantic Scholar (academic papers with stable IDs)
- OpenAlex (broader social science coverage)
- Crossref (DOI enrichment)
- World Bank (policy reports and evaluations)

Design principles:
- No hardcoded jurisdictions (extract from user question)
- Stable paper IDs (DOI > paperId > hash)
- Graceful degradation (missing abstracts/PDFs don't crash)
- Budget-aware (limit API calls)
"""

import re
import hashlib
import os
import time
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote_plus
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# ============================================================================
# UTILITIES
# ============================================================================

def normalize_text(s: str) -> str:
    """Normalize text for matching: lowercase, remove accents, collapse whitespace."""
    if not s:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    return s


def stable_paper_id(paper: Dict[str, Any]) -> str:
    """
    Generate stable paper ID with priority:
    1. DOI (normalized)
    2. Semantic Scholar paperId
    3. OpenAlex id
    4. Hash of normalized title + year
    """
    # Priority 1: DOI
    doi = paper.get("doi") or (paper.get("externalIds") or {}).get("DOI")
    if doi:
        return f"doi:{normalize_text(doi)}"
    
    # Priority 2: Semantic Scholar paperId
    paper_id = paper.get("paperId")
    if paper_id:
        return f"s2:{paper_id}"
    
    # Priority 3: OpenAlex id
    openalex_id = paper.get("openalex_id")
    if openalex_id:
        # Extract short ID from URL
        if "/" in openalex_id:
            openalex_id = openalex_id.split("/")[-1]
        return f"oa:{openalex_id}"
    
    # Priority 4: Hash of title + year
    title = normalize_text(paper.get("title", "unknown"))
    year = str(paper.get("year", ""))
    hash_input = f"{title}_{year}"
    hash_digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:12]
    return f"hash:{hash_digest}"


def create_session() -> requests.Session:
    """Create requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "PolicyChat/1.0 (Research Tool)"})
    return session


# ============================================================================
# QUERY EXPANSION (Pure LLM)
# ============================================================================

def build_query_expansions(user_question: str) -> Dict[str, Any]:
    """
    Generate diverse academic search queries using LLM.
    
    **REQUIRES LLM** - Will fail if GEMINI_API_KEY not configured.
    
    Returns:
        {
            "queries": [str, ...],  # 10 LLM-generated query variants
            "context_tokens": {}     # Empty dict (LLM handles all context)
        }
    """
    print("🤖 Generating LLM-powered query expansions...")
    try:
        import google.generativeai as genai
        import json
        
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError("No Gemini API key found in environment")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Enriched prompt for high-quality query variants
        prompt = (
            f"You are a policy research expert. Generate 10 diverse academic search queries to find research papers "
            f"that answer this policy question.\n\n"
            f"Question: \"{user_question}\"\n\n"
            f"REQUIREMENTS:\n"
            f"1. Vary specificity: Include both jurisdiction-specific and general comparative queries\n"
            f"2. Use research terminology: Include terms like 'RCT', 'evaluation', 'impact assessment', 'quasi-experimental', "
            f"'difference-in-differences', 'regression discontinuity', 'systematic review', 'meta-analysis'\n"
            f"3. Cover different angles: policy design, implementation, outcomes, cost-effectiveness, equity impacts\n"
            f"4. Use synonyms and alternative phrasings for key concepts\n"
            f"5. Mix broad literature reviews with specific intervention types\n"
            f"6. Include both academic phrasing and policy terminology\n"
            f"7. If jurisdiction is mentioned, create variants with and without it\n"
            f"8. Each query should be 3-10 words, optimized for academic search engines\n\n"
            f"Return ONLY a JSON array of 10 query strings, nothing else.\n"
            f"Example: [\"unemployment reduction RCT developing countries\", \"active labor market policies impact evaluation\", "
            f"\"job training programs systematic review\"]"
        )
        
        response = model.generate_content(prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Extract JSON array
        m = re.search(r'\[.*?\]', text, flags=re.DOTALL)
        if m:
            variants = json.loads(m.group(0))
            variants = [v for v in variants if isinstance(v, str)][:10]
        else:
            # Fallback: split by lines
            variants = [line.strip().strip('"\'') for line in text.split('\n') if line.strip()][:10]
        
        if len(variants) < 3:
            raise ValueError(f"LLM returned insufficient variants: {len(variants)}")
        
        print(f"✅ Generated {len(variants)} LLM query variants")
        return {
            "queries": variants,
            "context_tokens": {}  # LLM handles all context
        }
        
    except Exception as e:
        error_msg = f"❌ LLM query expansion failed: {e}"
        print(error_msg)
        print("⚠️ LLM expansion is REQUIRED. Please check:")
        print("   1. GEMINI_API_KEY is set")
        print("   2. Network connectivity to Gemini API")
        print("   3. API has sufficient quota")
        raise RuntimeError(f"LLM query expansion failed - cannot proceed without expansions. {e}") from e


# ============================================================================
# SOURCE FETCHERS
# ============================================================================

def fetch_semantic_scholar(query: str, n: int = 20, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    """
    Fetch papers from Semantic Scholar (unauthenticated).
    
    Returns list of papers in unified schema.
    """
    if session is None:
        session = create_session()
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(n, 100),
        "fields": "title,abstract,year,venue,citationCount,url,externalIds,fieldsOfStudy,authors,paperId"
    }
    
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        papers = []
        for item in data.get("data", []):
            paper = {
                "title": item.get("title", ""),
                "abstract": item.get("abstract") or "",
                "year": item.get("year"),
                "venue": item.get("venue") or "",
                "citationCount": item.get("citationCount") or 0,
                "url": item.get("url") or "",
                "paperId": item.get("paperId"),
                "externalIds": item.get("externalIds", {}),
                "fieldsOfStudy": item.get("fieldsOfStudy") or [],
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "source": "semantic_scholar",
                "source_type": "academic_paper"
            }
            # Extract DOI
            if paper["externalIds"] and "DOI" in paper["externalIds"]:
                paper["doi"] = paper["externalIds"]["DOI"]
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        print(f"⚠️ Semantic Scholar fetch failed: {e}")
        return []


def fetch_openalex(query: str, n: int = 20, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    """
    Fetch papers from OpenAlex (unauthenticated).
    Decodes inverted abstract index when present.
    
    Returns list of papers in unified schema.
    """
    if session is None:
        session = create_session()
    
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(n, 200),
        "mailto": "policychat@research.org"  # Polite pool
    }
    
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        # Handle empty or malformed responses
        if not data or "results" not in data:
            return []
        
        papers = []
        for item in data.get("results", []):
            if not item:  # Skip None items
                continue
            # Decode inverted abstract if present
            abstract_text = ""
            abstract_inverted = item.get("abstract_inverted_index")
            if abstract_inverted:
                try:
                    # Reconstruct abstract from inverted index
                    words = [""] * 1000  # Pre-allocate
                    for word, positions in abstract_inverted.items():
                        for pos in positions:
                            if pos < len(words):
                                words[pos] = word
                    abstract_text = " ".join(w for w in words if w).strip()
                except Exception:
                    abstract_text = ""
            
            paper = {
                "title": item.get("title", ""),
                "abstract": abstract_text,
                "year": item.get("publication_year"),
                "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name", ""),
                "citationCount": item.get("cited_by_count") or 0,
                "url": item.get("id") or "",
                "openalex_id": item.get("id", ""),
                "doi": item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None,
                "fieldsOfStudy": [c.get("display_name", "") for c in item.get("concepts", [])[:5]],
                "authors": [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])],
                "source": "openalex",
                "source_type": "academic_paper"
            }
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        print(f"⚠️ OpenAlex fetch failed: {e}")
        return []


def fetch_crossref_by_title(title: str, n: int = 1, session: Optional[requests.Session] = None) -> Optional[Dict[str, Any]]:
    """
    Enrich paper metadata using Crossref (DOI registry).
    Use sparingly for enrichment, not bulk discovery.
    
    Returns single enriched paper dict or None.
    """
    if session is None:
        session = create_session()
    
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": title,
        "rows": n
    }
    
    try:
        resp = session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        items = data.get("message", {}).get("items", [])
        if not items:
            return None
        
        item = items[0]
        return {
            "doi": item.get("DOI"),
            "publisher": item.get("publisher"),
            "venue": (item.get("container-title") or [""])[0],
            "year": (item.get("published-print") or item.get("published-online") or {}).get("date-parts", [[None]])[0][0],
            "url": item.get("URL"),
            "source": "crossref"
        }
    
    except Exception as e:
        print(f"⚠️ Crossref enrichment failed for '{title[:50]}': {e}")
        return None


def fetch_worldbank(query: str, n: int = 20, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    """
    Fetch policy reports from World Bank Documents API.
    
    Returns list of papers in unified schema with source_type="policy_report".
    """
    if session is None:
        session = create_session()
    
    url = "https://search.worldbank.org/api/v2/wds"
    params = {
        "format": "json",
        "qterm": query,
        "rows": min(n, 100)
    }
    
    try:
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        papers = []
        for item in data.get("documents", {}).get("", []):  # API has weird nesting
            paper = {
                "title": item.get("display_title") or item.get("title", ""),
                "abstract": item.get("abstract") or item.get("abstracten") or "",
                "year": int(item.get("docdt", "")[:4]) if item.get("docdt") and len(item.get("docdt", "")) >= 4 else None,
                "venue": "World Bank",
                "citationCount": 0,  # World Bank doesn't track citations
                "url": item.get("pdfurl") or item.get("url_friendly_title") or "",
                "doi": None,
                "fieldsOfStudy": [item.get("doctype", "Policy Report")],
                "authors": [item.get("author", "World Bank")],
                "source": "worldbank",
                "source_type": "policy_report"
            }
            papers.append(paper)
        
        return papers
    
    except Exception as e:
        print(f"⚠️ World Bank fetch failed: {e}")
        return []


# ============================================================================
# DEDUPLICATION & RANKING
# ============================================================================

def unify_and_dedupe(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate papers by stable ID and merge metadata from multiple sources.
    
    Priority for fields:
    - DOI/publisher: Crossref > Semantic Scholar > OpenAlex
    - Abstract: Semantic Scholar > OpenAlex > World Bank
    - Citations: max across sources
    
    Returns deduplicated list with sources_seen tracking.
    """
    by_id = {}
    
    for paper in papers:
        sid = stable_paper_id(paper)
        
        if sid not in by_id:
            # First occurrence
            paper["stable_id"] = sid
            paper["sources_seen"] = [paper["source"]]
            by_id[sid] = paper
        else:
            # Merge with existing entry
            existing = by_id[sid]
            existing["sources_seen"].append(paper["source"])
            
            # Prefer abstract from Semantic Scholar
            if paper["source"] == "semantic_scholar" and paper.get("abstract"):
                existing["abstract"] = paper["abstract"]
            elif not existing.get("abstract") and paper.get("abstract"):
                existing["abstract"] = paper["abstract"]
            
            # Prefer DOI from Crossref or earliest source
            if paper.get("doi") and not existing.get("doi"):
                existing["doi"] = paper["doi"]
            
            # Prefer venue from Crossref
            if paper["source"] == "crossref" and paper.get("venue"):
                existing["venue"] = paper["venue"]
            elif not existing.get("venue") and paper.get("venue"):
                existing["venue"] = paper["venue"]
            
            # Max citations
            existing["citationCount"] = max(
                existing.get("citationCount", 0),
                paper.get("citationCount", 0)
            )
            
            # Merge fieldsOfStudy
            existing_fields = set(existing.get("fieldsOfStudy", []))
            new_fields = set(paper.get("fieldsOfStudy", []))
            existing["fieldsOfStudy"] = list(existing_fields | new_fields)
    
    return list(by_id.values())


def lightweight_rank(papers: List[Dict[str, Any]], context_tokens: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rank papers by retrieval_rank_score (no LLM needed).
    
    Scoring factors:
    - Method signals (RCT, DiD, IV, RD, meta-analysis, impact eval)
    - Jurisdiction match
    - Policy report boost (World Bank)
    - Citations (log-scaled)
    - Penalties for non-substantive papers
    
    Returns sorted list with retrieval_rank_score added.
    """
    jurisdiction_terms = [normalize_text(t) for t in context_tokens.get("jurisdiction_terms", [])]
    topic_terms = [normalize_text(t) for t in context_tokens.get("topic_terms", [])]
    
    # Method signal keywords
    method_keywords = {
        "rct": 10,
        "randomized": 10,
        "randomised": 10,
        "field experiment": 10,
        "difference in differences": 8,
        "diff-in-diff": 8,
        "instrumental variable": 8,
        "regression discontinuity": 8,
        "meta-analysis": 7,
        "systematic review": 7,
        "impact evaluation": 6,
        "quasi-experimental": 6,
        "propensity score": 5,
        "event study": 5,
        "natural experiment": 5
    }
    
    penalty_keywords = {
        "replication data": -5,
        "dataset": -3,
        "supplement": -3,
        "appendix": -3,
        "corrigendum": -5,
        "erratum": -5
    }
    
    # Book/textbook detection (these are NOT policy-relevant research)
    book_indicators = {
        "textbook": -10,
        "handbook": -8,
        "manual": -8,
        "coursebook": -10,
        "introduction to": -6,
        "principles of": -6,
        "economics: ": -7,  # Title pattern like "Economics: Theory and Practice"
        "companion to": -7,
        "encyclopedia": -8,
        "dictionary": -8,
        "reader": -6,
        "essentials of": -6
    }
    
    for paper in papers:
        score = 0.0
        text = normalize_text(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        
        # CRITICAL: Query coverage boost (multi-variant matches = higher relevance)
        query_coverage = paper.get("query_coverage", 0)
        if query_coverage >= 3:
            score += 15  # Matched 3+ query variants = very relevant
        elif query_coverage == 2:
            score += 8   # Matched 2 variants = relevant
        elif query_coverage == 1:
            score += 2   # Matched 1 variant = baseline
        # query_coverage == 0 should not happen after dedup, but gets 0 boost
        
        # Method signals
        for keyword, points in method_keywords.items():
            if keyword in text:
                score += points
        
        # Jurisdiction match
        for jterm in jurisdiction_terms:
            if jterm in text:
                score += 8
        
        # Topic match
        for tterm in topic_terms:
            if tterm in text:
                score += 3
        
        # Policy report boost
        if paper.get("source_type") == "policy_report":
            score += 5
        
        # Citations (log-scaled, don't punish new papers)
        citations = paper.get("citationCount", 0)
        if citations > 0:
            import math
            score += math.log10(citations + 1) * 2
        
        # Penalties
        for keyword, points in penalty_keywords.items():
            if keyword in text:
                score += points
        
        # Abstract presence bonus
        if paper.get("abstract") and len(paper["abstract"]) > 100:
            score += 3
        
        # Venue quality heuristic
        venue = normalize_text(paper.get("venue", ""))
        if any(kw in venue for kw in ["nber", "world bank", "iza", "economic review", "econometrica", "journal of political economy"]):
            score += 4
        
        paper["retrieval_rank_score"] = round(score, 2)
    
    # Sort by score descending
    papers.sort(key=lambda p: p.get("retrieval_rank_score", 0), reverse=True)
    
    return papers


# ============================================================================
# ORCHESTRATION
# ============================================================================

def smart_retrieve(
    user_question: str,
    target_pool: int = 50,
    session: Optional[requests.Session] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Multi-source retrieval with smart orchestration.
    
    Process:
    1. Build query expansions
    2. Fetch from World Bank (1-2 queries)
    3. Fetch from OpenAlex (all queries)
    4. Fetch from Semantic Scholar (all queries)
    5. Deduplicate
    6. Rank with lightweight scorer
    7. Enrich top 20 with Crossref (if missing DOI/venue)
    8. Return top 10, top 50, and stats
    
    Returns:
        (top10, top50, stats)
    """
    if session is None:
        session = create_session()
    
    print(f"\n📚 Multi-source retrieval for: '{user_question}'\n")
    
    # Step 1: Query expansion
    expansion = build_query_expansions(user_question)
    queries = expansion["queries"]
    context_tokens = expansion["context_tokens"]
    
    print(f"🔍 Generated {len(queries)} queries:")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")
    print(f"📍 Jurisdiction: {context_tokens['jurisdiction_terms'] or 'GLOBAL'}")
    print(f"📊 Topics: {', '.join(context_tokens['topic_terms']) or 'General'}\n")
    
    # Step 2-4: Fetch from all sources
    # Track which queries matched each paper (for relevance boosting)
    all_papers = []
    query_match_tracker = {}  # stable_id -> set of query indices
    api_calls = 0
    
    # World Bank (first 2 queries only, policy focus)
    print("🏦 Fetching from World Bank...")
    for q_idx, query in enumerate(queries[:2]):
        wb_papers = fetch_worldbank(query, n=10, session=session)
        # Track query matches
        for paper in wb_papers:
            sid = stable_paper_id(paper)
            if sid not in query_match_tracker:
                query_match_tracker[sid] = set()
            query_match_tracker[sid].add(q_idx)
        all_papers.extend(wb_papers)
        api_calls += 1
        print(f"   Q{q_idx+1}: {len(wb_papers)} policy reports")
        time.sleep(0.5)  # Rate limit courtesy
    
    # OpenAlex (all queries)
    print("🔬 Fetching from OpenAlex...")
    for q_idx, query in enumerate(queries):
        oa_papers = fetch_openalex(query, n=15, session=session)
        # Track query matches
        for paper in oa_papers:
            sid = stable_paper_id(paper)
            if sid not in query_match_tracker:
                query_match_tracker[sid] = set()
            query_match_tracker[sid].add(q_idx)
        all_papers.extend(oa_papers)
        api_calls += 1
        print(f"   Q{q_idx+1}: {len(oa_papers)} papers")
        time.sleep(0.5)
    
    # Semantic Scholar (all queries)
    print("📖 Fetching from Semantic Scholar...")
    for q_idx, query in enumerate(queries):
        s2_papers = fetch_semantic_scholar(query, n=15, session=session)
        # Track query matches
        for paper in s2_papers:
            sid = stable_paper_id(paper)
            if sid not in query_match_tracker:
                query_match_tracker[sid] = set()
            query_match_tracker[sid].add(q_idx)
        all_papers.extend(s2_papers)
        api_calls += 1
        print(f"   Q{q_idx+1}: {len(s2_papers)} papers")
        time.sleep(1.0)  # S2 has stricter rate limits
    
    print(f"\n📦 Retrieved {len(all_papers)} total items (before dedup)")
    
    # Step 5: Deduplicate and attach query coverage
    deduped = unify_and_dedupe(all_papers)
    print(f"🔗 Deduplicated to {len(deduped)} unique papers")
    
    # Attach query coverage to each paper (how many variants matched it)
    for paper in deduped:
        sid = paper.get("stable_id")
        matched_queries = query_match_tracker.get(sid, set())
        paper["query_coverage"] = len(matched_queries)
        paper["matched_query_indices"] = sorted(list(matched_queries))
    
    # Show query coverage distribution
    coverage_dist = {}
    for paper in deduped:
        cov = paper.get("query_coverage", 0)
        coverage_dist[cov] = coverage_dist.get(cov, 0) + 1
    print(f"📊 Query coverage: {dict(sorted(coverage_dist.items(), reverse=True))}")
    
    # Step 6: Rank (now uses query_coverage for relevance boost)
    ranked = lightweight_rank(deduped, context_tokens)
    
    # Step 7: Enrich top 20 with Crossref (if missing DOI/venue)
    print("🔖 Enriching top papers with Crossref...")
    enriched_count = 0
    for paper in ranked[:20]:
        if not paper.get("doi") or not paper.get("venue"):
            crossref_data = fetch_crossref_by_title(paper["title"], session=session)
            if crossref_data:
                if not paper.get("doi") and crossref_data.get("doi"):
                    paper["doi"] = crossref_data["doi"]
                if not paper.get("venue") and crossref_data.get("venue"):
                    paper["venue"] = crossref_data["venue"]
                enriched_count += 1
                api_calls += 1
                time.sleep(0.3)
    print(f"   → Enriched {enriched_count} papers")
    
    # Step 8: Return results
    top10 = ranked[:10]
    top50 = ranked[:target_pool]
    
    # Calculate stats
    abstracts_present = sum(1 for p in top50 if p.get("abstract") and len(p["abstract"]) > 50)
    policy_reports = sum(1 for p in top50 if p.get("source_type") == "policy_report")
    multi_match_count = sum(1 for p in top50 if p.get("query_coverage", 0) >= 2)
    avg_coverage_top10 = sum(p.get("query_coverage", 0) for p in top10) / len(top10) if top10 else 0
    
    stats = {
        "retrieved_count": len(all_papers),
        "deduped_count": len(deduped),
        "abstracts_present": abstracts_present,
        "abstracts_present_rate": round(abstracts_present / len(top50) * 100, 1) if top50 else 0,
        "policy_report_count": policy_reports,
        "api_calls": api_calls,
        "queries_used": len(queries),
        "multi_match_papers": multi_match_count,
        "avg_coverage_top10": round(avg_coverage_top10, 1)
    }
    
    print(f"\n✅ Retrieval complete:")
    print(f"   📊 API calls: {api_calls}")
    print(f"   📄 Abstracts present: {abstracts_present}/{len(top50)} ({stats['abstracts_present_rate']}%)")
    print(f"   🏛️ Policy reports: {policy_reports}")
    print(f"   🎯 Multi-match papers (2+ variants): {multi_match_count}/{len(top50)}")
    print(f"   ⭐ Avg query coverage (top 10): {stats['avg_coverage_top10']:.1f} variants/paper")
    
    return top10, top50, stats
