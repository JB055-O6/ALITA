"""
Optional Web Learning System

Implements Task 27 requirements:
- DuckDuckGo search API (FREE, no limits)
- Information extraction and summarization
- Local knowledge base caching
- Multi-source verification
- Offline-first mode with explicit web toggle

Uses FREE services:
- DuckDuckGo Instant Answer API (no API key needed)
- BeautifulSoup for web scraping
- Transformers for summarization
- ChromaDB for local caching

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import asyncio
from dataclasses import dataclass, asdict

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import chromadb
except ImportError:
    chromadb = None


@dataclass
class SearchResult:
    """Search result from web."""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: datetime
    relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class LearnedKnowledge:
    """Knowledge learned from web."""
    query: str
    summary: str
    sources: List[str]
    confidence: float
    verified: bool
    cached_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['cached_date'] = self.cached_date.isoformat()
        return data


class WebSearcher:
    """Search web using DuckDuckGo (FREE, no API key)."""
    
    def __init__(self):
        self.ddg_available = DDGS is not None
        self.requests_available = requests is not None
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search web using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of search results
        """
        if not self.ddg_available:
            logging.warning("DuckDuckGo search not available")
            return []
        
        try:
            results = []
            
            # Use DuckDuckGo search (FREE, no API key needed)
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for i, result in enumerate(search_results):
                    results.append(SearchResult(
                        title=result.get('title', ''),
                        url=result.get('href', ''),
                        snippet=result.get('body', ''),
                        source='duckduckgo',
                        timestamp=datetime.now(),
                        relevance_score=1.0 - (i * 0.1)  # Decreasing relevance
                    ))
            
            logging.info(f"Found {len(results)} results for: {query}")
            return results
            
        except Exception as e:
            logging.error(f"Web search failed: {e}")
            return []
    
    async def fetch_page_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch page content.
        
        Args:
            url: Page URL
            timeout: Request timeout
            
        Returns:
            Page text content or None
        """
        if not self.requests_available or not BeautifulSoup:
            return None
        
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text[:5000]  # Limit to 5000 chars
            
            return None
            
        except Exception as e:
            logging.warning(f"Failed to fetch {url}: {e}")
            return None


class InformationExtractor:
    """Extract and summarize information."""
    
    def __init__(self):
        self.summarizer = None
        self.summarizer_loaded = False
        
        if pipeline:
            try:
                # Use FREE summarization model
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=-1  # CPU
                )
                self.summarizer_loaded = True
                logging.info("Summarization model loaded")
            except Exception as e:
                logging.warning(f"Summarization model loading failed: {e}")
    
    def extract_key_information(self, text: str, max_length: int = 500) -> str:
        """Extract key information from text.
        
        Args:
            text: Input text
            max_length: Maximum output length
            
        Returns:
            Extracted information
        """
        if not text:
            return ""
        
        # Simple extraction: first few sentences
        sentences = text.split('. ')
        key_info = '. '.join(sentences[:5])
        
        if len(key_info) > max_length:
            key_info = key_info[:max_length] + "..."
        
        return key_info
    
    def summarize(self, text: str, max_length: int = 150) -> str:
        """Summarize text using AI.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Summary
        """
        if not text:
            return ""
        
        # Use AI summarization if available
        if self.summarizer_loaded:
            try:
                # Limit input length
                input_text = text[:1024]
                
                summary = self.summarizer(
                    input_text,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                
                return summary[0]['summary_text']
                
            except Exception as e:
                logging.warning(f"AI summarization failed: {e}")
        
        # Fallback to simple extraction
        return self.extract_key_information(text, max_length)
    
    def combine_summaries(self, summaries: List[str]) -> str:
        """Combine multiple summaries.
        
        Args:
            summaries: List of summaries
            
        Returns:
            Combined summary
        """
        if not summaries:
            return ""
        
        # Remove duplicates
        unique_summaries = list(set(summaries))
        
        # Combine
        combined = " ".join(unique_summaries)
        
        # Summarize again if too long
        if len(combined) > 500:
            combined = self.summarize(combined, max_length=200)
        
        return combined


class KnowledgeCache:
    """Cache learned knowledge locally."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("data/web_knowledge")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = self.cache_dir / "knowledge_cache.json"
        self.cache: Dict[str, LearnedKnowledge] = {}
        
        # Vector database for semantic search
        self.vector_db = None
        if chromadb:
            try:
                self.vector_db = chromadb.PersistentClient(
                    path=str(self.cache_dir / "chroma")
                )
                self.collection = self.vector_db.get_or_create_collection(
                    name="web_knowledge"
                )
                logging.info("Knowledge cache vector DB initialized")
            except Exception as e:
                logging.warning(f"Vector DB initialization failed: {e}")
        
        self.load_cache()
    
    def load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    for query, knowledge_data in data.items():
                        knowledge_data['cached_date'] = datetime.fromisoformat(
                            knowledge_data['cached_date']
                        )
                        self.cache[query] = LearnedKnowledge(**knowledge_data)
            except Exception as e:
                logging.error(f"Failed to load knowledge cache: {e}")
    
    def save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self.cache.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logging.error(f"Failed to save knowledge cache: {e}")
    
    def add_knowledge(self, knowledge: LearnedKnowledge):
        """Add knowledge to cache.
        
        Args:
            knowledge: Learned knowledge
        """
        self.cache[knowledge.query] = knowledge
        
        # Add to vector DB
        if self.vector_db:
            try:
                self.collection.add(
                    documents=[knowledge.summary],
                    metadatas=[{
                        "query": knowledge.query,
                        "sources": json.dumps(knowledge.sources),
                        "confidence": knowledge.confidence,
                        "verified": knowledge.verified
                    }],
                    ids=[knowledge.query]
                )
            except Exception as e:
                logging.warning(f"Failed to add to vector DB: {e}")
        
        self.save_cache()
    
    def get_knowledge(self, query: str) -> Optional[LearnedKnowledge]:
        """Get cached knowledge.
        
        Args:
            query: Query string
            
        Returns:
            Cached knowledge or None
        """
        return self.cache.get(query)
    
    def search_similar(self, query: str, max_results: int = 3) -> List[LearnedKnowledge]:
        """Search for similar cached knowledge.
        
        Args:
            query: Query string
            max_results: Maximum results
            
        Returns:
            List of similar knowledge
        """
        if not self.vector_db:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results
            )
            
            similar = []
            for metadata in results['metadatas'][0]:
                cached_query = metadata['query']
                if cached_query in self.cache:
                    similar.append(self.cache[cached_query])
            
            return similar
            
        except Exception as e:
            logging.warning(f"Similar search failed: {e}")
            return []
    
    def is_fresh(self, knowledge: LearnedKnowledge, max_age_days: int = 30) -> bool:
        """Check if knowledge is still fresh.
        
        Args:
            knowledge: Learned knowledge
            max_age_days: Maximum age in days
            
        Returns:
            True if fresh
        """
        age = datetime.now() - knowledge.cached_date
        return age.days < max_age_days


class SourceVerifier:
    """Verify information from multiple sources."""
    
    def verify_information(self, summaries: List[str], sources: List[str]) -> Tuple[bool, float]:
        """Verify information consistency.
        
        Args:
            summaries: List of summaries from different sources
            sources: List of source URLs
            
        Returns:
            Tuple of (verified, confidence)
        """
        if len(summaries) < 2:
            return False, 0.5
        
        # Simple verification: check for common keywords
        all_words = set()
        word_counts = {}
        
        for summary in summaries:
            words = set(summary.lower().split())
            all_words.update(words)
            
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate agreement
        common_words = [w for w, count in word_counts.items() if count >= len(summaries) / 2]
        agreement_ratio = len(common_words) / len(all_words) if all_words else 0
        
        # Verified if agreement > 30%
        verified = agreement_ratio > 0.3
        confidence = min(agreement_ratio * 2, 1.0)
        
        return verified, confidence


class WebAgent:
    """Main web learning system."""
    
    def __init__(self):
        self.searcher = WebSearcher()
        self.extractor = InformationExtractor()
        self.cache = KnowledgeCache()
        self.verifier = SourceVerifier()
        
        # Web access control
        self.web_enabled = False  # Offline-first
        self.explicit_web_toggle = False
        
        logging.info("Web Agent initialized (offline-first mode)")
    
    def enable_web_access(self, enabled: bool = True):
        """Enable or disable web access.
        
        Args:
            enabled: True to enable web access
        """
        self.web_enabled = enabled
        self.explicit_web_toggle = True
        logging.info(f"Web access: {'enabled' if enabled else 'disabled'}")
    
    async def learn(self, query: str, force_web: bool = False) -> Optional[LearnedKnowledge]:
        """Learn from web or cache.
        
        Args:
            query: Query to learn about
            force_web: Force web search even if cached
            
        Returns:
            Learned knowledge or None
        """
        # Check cache first (offline-first)
        if not force_web:
            cached = self.cache.get_knowledge(query)
            if cached and self.cache.is_fresh(cached):
                logging.info(f"Using cached knowledge for: {query}")
                return cached
            
            # Check similar cached knowledge
            similar = self.cache.search_similar(query, max_results=1)
            if similar:
                logging.info(f"Using similar cached knowledge for: {query}")
                return similar[0]
        
        # Check if web access allowed
        if not self.web_enabled and not force_web:
            logging.info("Web access disabled, using cache only")
            return None
        
        # Search web
        logging.info(f"Searching web for: {query}")
        search_results = await self.searcher.search(query, max_results=5)
        
        if not search_results:
            return None
        
        # Extract information from top results
        summaries = []
        sources = []
        
        for result in search_results[:3]:  # Top 3 results
            # Use snippet
            summary = self.extractor.summarize(result.snippet, max_length=150)
            if summary:
                summaries.append(summary)
                sources.append(result.url)
            
            # Optionally fetch full page (commented out for speed)
            # content = await self.searcher.fetch_page_content(result.url)
            # if content:
            #     summary = self.extractor.summarize(content, max_length=200)
            #     summaries.append(summary)
        
        if not summaries:
            return None
        
        # Combine summaries
        combined_summary = self.extractor.combine_summaries(summaries)
        
        # Verify information
        verified, confidence = self.verifier.verify_information(summaries, sources)
        
        # Create learned knowledge
        knowledge = LearnedKnowledge(
            query=query,
            summary=combined_summary,
            sources=sources,
            confidence=confidence,
            verified=verified,
            cached_date=datetime.now()
        )
        
        # Cache for future offline use
        self.cache.add_knowledge(knowledge)
        
        logging.info(f"Learned and cached knowledge for: {query}")
        return knowledge
    
    async def answer_question(self, question: str) -> Optional[str]:
        """Answer question using web learning.
        
        Args:
            question: Question to answer
            
        Returns:
            Answer or None
        """
        knowledge = await self.learn(question)
        
        if knowledge:
            answer = f"{knowledge.summary}\n\n"
            answer += f"Sources: {', '.join(knowledge.sources[:2])}\n"
            answer += f"Confidence: {knowledge.confidence:.0%}\n"
            answer += f"Verified: {'Yes' if knowledge.verified else 'No'}"
            return answer
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            "cached_queries": len(self.cache.cache),
            "web_enabled": self.web_enabled,
            "explicit_toggle": self.explicit_web_toggle,
            "cache_directory": str(self.cache.cache_dir)
        }


# Global instance
_web_agent: Optional[WebAgent] = None


def get_web_agent() -> WebAgent:
    """Get global web agent instance."""
    global _web_agent
    
    if _web_agent is None:
        _web_agent = WebAgent()
    
    return _web_agent
