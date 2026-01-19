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
from pathlib import Path 



class Coordinator:
    def __init__(self,  message_output=None):
        """
        Initialize the Coordinator with user inputs and pipeline components.
        Args:
            user_inputs_file (str): Path to the JSON file containing user inputs.
        """
        # Get the current script's directory
        current_dir = Path(__file__).resolve().parent

        # Navigate to the parent directory and then to the target file
        file_path = current_dir.parent / "user_inputs.json"

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
        self.embbedingator = Embbedingator()
        # PerformQuery will be initialized only if needed (not used in current pipeline)

    def message(self, text):
        """
        Utility method to output messages
        """
        if self.message_output:
            self.message_output(text)

    def transform_query_for_academic_search(self, query):
        """
        Transform a conversational query into academic search terms.
        Extract key nouns, locations, and policy-relevant terms.
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
        
        # Split on common sentence/clause boundaries
        parts = re.split(r'[.,;]', query_lower)
        # Take the first sentence/clause which usually contains the core question
        core_query = parts[0].strip()
        
        # Extract key policy terms and keep important words
        # Remove common filler words but keep substantive terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had'}
        words = core_query.split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Build academic query - limit to 5-6 most important terms
        academic_query = ' '.join(key_terms[:6])
        
        # If the query is too short after processing, use more words
        if len(academic_query) < 10 and len(key_terms) > 0:
            academic_query = ' '.join(key_terms[:8])
        elif len(academic_query) < 5:
            # Fallback to cleaned original
            academic_query = ' '.join(words[:8])
        
        return academic_query.strip()

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
        Fetch external papers from Semantic Scholar API.
        Returns:
            list: List of external paper data including content.
            list: List of all content.
            dict: Dictionary where keys are document titles and values are the concatenated contents.
        """
        if not self.external_search:
            self.message("🔎 External search disabled. Skipping Semantic Scholar API.")
            return [], [], {}
        self.message("🔎 Retrieving external papers from Semantic Scholar API ...")
        
        # Transform conversational query to academic search terms
        academic_query = self.transform_query_for_academic_search(self.query)
        self.message(f"🔍 Transformed query: '{academic_query}'")
        
        # Semantic Scholar API parameters
        params = {
            "query": academic_query,
            "limit": int(self.max_results),  # Number of results to retrieve
            # Request richer metadata and open access PDF info
            "fields": "title,abstract,year,authors,url,openAccessPdf,fieldsOfStudy,citationCount,referenceCount,venue,isOpenAccess"
        }

        try:
            self.message(f"📡 Querying Semantic Scholar API...")
            response = self.session.get(self.semantic_scholar_api_url, params=params, timeout=15)
            self.message(f"📊 API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                error_detail = response.text
                self.message(f"❌ API Error: {error_detail}")
                response.raise_for_status()
            
            data = response.json()
            self.message(f"📚 API returned {len(data.get('data', []))} papers")
            external_papers = []
            all_contents = []
            content_by_title = {}

            # Process Semantic Scholar response
            for paper in data.get("data", []):
                title = paper.get("title", "Unknown Title")
                abstract = paper.get("abstract", "")
                year = paper.get("year", "N/A")
                authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
                url = paper.get("url", "")
                citation_count = paper.get("citationCount", 0)
                is_open = paper.get("isOpenAccess", False)
                open_pdf = None
                if paper.get("openAccessPdf") and isinstance(paper.get("openAccessPdf"), dict):
                    open_pdf = paper.get("openAccessPdf").get("url")

                # Combine title and abstract as content (fallback to title)
                content = f"{title}. {abstract}" if abstract else title

                # Basic relevance estimate: cosine between query and abstract/title embeddings
                try:
                    q_emb = self.embbedingator.embed_text(self.query)
                    p_emb = self.embbedingator.embed_text(content)
                    sim = float(np.dot(q_emb, p_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(p_emb)))
                    
                    # Boost score for papers with geographic relevance
                    geo_keywords = ['colombia', 'bogota', 'latin america', 'south america', 
                                   'brazil', 'argentina', 'mexico', 'peru', 'chile', 
                                   'developing country', 'developing countries', 'emerging economy']
                    content_lower = content.lower()
                    geo_boost = sum(0.05 for kw in geo_keywords if kw in content_lower)
                    sim = min(1.0, sim + geo_boost)  # Cap at 1.0
                except Exception:
                    sim = 0.0

                paper_entry = {
                    "title": title,
                    "content": content,
                    "url": url,
                    "year": year,
                    "authors": authors,
                    "citation_count": citation_count,
                    "is_open_access": is_open,
                    "open_pdf": open_pdf,
                    "query_similarity": sim,
                    "full_text": None,
                    "pdf_path": None
                }

                # Decide whether to attempt full-text retrieval
                # Conditions: high semantic similarity OR high citation count
                try_full_text = (sim >= 0.70) or (citation_count >= 50)

                if try_full_text and open_pdf:
                    try:
                        self.message(f"⬇️ Downloading PDF for '{title}'")
                        os.makedirs("./downloaded_papers", exist_ok=True)
                        pdf_fname = re.sub(r"[^a-zA-Z0-9_\-]", "_", title)[:120] + ".pdf"
                        pdf_path = os.path.join("./downloaded_papers", pdf_fname)
                        pdf_resp = self.session.get(open_pdf, timeout=30)
                        pdf_resp.raise_for_status()
                        with open(pdf_path, "wb") as pf:
                            pf.write(pdf_resp.content)
                        paper_entry["pdf_path"] = pdf_path

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
                            paper_entry["full_text"] = full_text if full_text else None
                            # Replace content with full_text for downstream processing if available
                            if paper_entry["full_text"]:
                                paper_entry["content"] = paper_entry["full_text"][:20000]
                        except Exception as e:
                            self.message(f"⚠️ Failed extracting PDF for '{title}': {e}")
                    except requests.exceptions.RequestException as e:
                        self.message(f"⚠️ Failed to download PDF for '{title}': {e}")

                external_papers.append(paper_entry)
                all_contents.append(paper_entry["content"])

                if title in content_by_title:
                    content_by_title[title] += f" {paper_entry['content']}"
                else:
                    content_by_title[title] = paper_entry["content"]

            # Filter papers by relevance threshold and sort by similarity
            relevance_threshold = 0.45  # Keep papers with at least 45% similarity (includes geo-boost)
            filtered_papers = [p for p in external_papers if p["query_similarity"] >= relevance_threshold]
            
            # Sort by similarity descending
            filtered_papers.sort(key=lambda x: x["query_similarity"], reverse=True)
            
            # Log filtering results
            if len(filtered_papers) < len(external_papers):
                self.message(f"🔍 Filtered to {len(filtered_papers)} most relevant papers (from {len(external_papers)}) based on semantic similarity")
            
            # Update all_contents and content_by_title to only include filtered papers
            all_contents = [p["content"] for p in filtered_papers]
            content_by_title = {p["title"]: p["content"] for p in filtered_papers}
            
            self.message(f"⛳️ I retrieved {len(filtered_papers)} relevant papers from Semantic Scholar API.")
            return filtered_papers, all_contents, content_by_title

        except requests.exceptions.RequestException as e:
            self.message(f"❌ Error fetching external papers from Semantic Scholar: {e}")
            print(f"Error fetching external papers from Semantic Scholar: {e}")
            return [], [], {}


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
        report_path = f"./{sanitized_name}_{timestamp}.jsonl"

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
        """
        self.message("🚀 Starting research pipeline ...")

        all_external_contents = []
        content_by_title = {}

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
                external_chunks = self.process_external_papers(external_papers)

                print("Calculating similarities for external papers...")
                external_results = self.calculate_similarities(external_chunks)
                self.save_results(external_results, "external_papers_report", include_url=True)
            else:
                self.message("⚠️ No external papers retrieved. The pipeline will stop here.")
                return  # Exit early if no papers were found
        else:
            self.message("ℹ️ External search not enabled.")

        # Combine results and save
        combined_results = local_results + external_results
        if combined_results:
            self.save_results(combined_results, "combined_report", include_url=True)
        else:
            self.message("⚠️ No papers to analyze. Please check your settings or try a different query.")
            return  # Exit if no papers at all
        
        
        # Generate memo
        memo = GenerateMemo(message_output=self.message_output)



        # Extract the 'query' field
        user_query = self.user_inputs.get("query")
        memo.run(all_external_contents, content_by_title, user_query)


if __name__ == "__main__":
    # Run the pipeline
    coordinator = Coordinator()
    coordinator.run_pipeline()
