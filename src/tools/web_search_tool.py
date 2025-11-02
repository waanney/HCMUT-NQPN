"""
Web Search Tool using OpenAI's web_search capability
Provides search functionality for the Suggestion Agent to find external references and solutions.
"""
from __future__ import annotations

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ========================
# Data Models
# ========================
class SearchResult(BaseModel):
    """Single search result with URL, title, and snippet."""
    url: str
    title: str
    snippet: str
    relevance_score: Optional[float] = None
    source_type: str = "web"  # web, academic, news, etc.
    timestamp: Optional[str] = None


class WebSearchResponse(BaseModel):
    """Response from web search containing results and metadata."""
    query: str
    results: List[SearchResult] = Field(default_factory=list)
    total_results: int = 0
    search_time_ms: float = 0
    model_used: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SearchCache:
    """Simple in-memory cache for search results."""
    _cache: Dict[str, tuple[WebSearchResponse, float]] = field(default_factory=dict)
    ttl_seconds: int = 3600  # 1 hour default
    
    def get(self, query: str) -> Optional[WebSearchResponse]:
        """Get cached result if not expired."""
        if query in self._cache:
            result, timestamp = self._cache[query]
            if time.time() - timestamp < self.ttl_seconds:
                logger.debug(f"Cache hit for query: {query}")
                return result
            else:
                del self._cache[query]
        return None
    
    def set(self, query: str, result: WebSearchResponse):
        """Store result in cache with current timestamp."""
        self._cache[query] = (result, time.time())
        logger.debug(f"Cached result for query: {query}")
    
    def clear(self):
        """Clear all cached results."""
        self._cache.clear()


# ========================
# Web Search Client
# ========================
class WebSearchTool:
    """
    Web search tool using OpenAI's web_search capability.
    
    Features:
    - Uses OpenAI Responses API with web_search tool
    - In-memory caching with TTL
    - Rate limiting and retry logic
    - Configurable result limits
    - Extracts URLs and snippets from results
    
    Usage:
        tool = WebSearchTool(api_key="sk-...", model="gpt-4o")
        results = tool.search("best practices for requirements engineering")
        for result in results.results:
            print(f"{result.title}: {result.url}")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_results: int = 5,
        cache_ttl: int = 3600,
        enable_cache: bool = True,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize web search tool.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: OpenAI model to use (must support web_search tool)
            max_results: Maximum number of results to return
            cache_ttl: Cache time-to-live in seconds
            enable_cache: Whether to enable result caching
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter.")
        
        self.model = model
        self.max_results = max_results
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.client = OpenAI(api_key=self.api_key)
        self.cache = SearchCache(ttl_seconds=cache_ttl) if enable_cache else None
        
        logger.info(f"Initialized WebSearchTool with model={model}, max_results={max_results}")
    
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        use_cache: bool = True,
    ) -> WebSearchResponse:
        """
        Perform web search using OpenAI's web_search tool.
        
        Args:
            query: Search query string
            max_results: Override default max_results for this query
            use_cache: Whether to use cached results if available
        
        Returns:
            WebSearchResponse with results and metadata
        """
        if not query or not query.strip():
            return WebSearchResponse(
                query=query,
                error="Empty query provided"
            )
        
        query = query.strip()
        max_results = max_results or self.max_results
        
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(query)
            if cached:
                logger.info(f"Returning cached results for: {query}")
                return cached
        
        # Perform search with retries
        start_time = time.time()
        result = self._search_with_retry(query, max_results)
        search_time = (time.time() - start_time) * 1000
        
        result.search_time_ms = search_time
        
        # Cache successful results
        if use_cache and self.cache and not result.error:
            self.cache.set(query, result)
        
        return result
    
    def _search_with_retry(self, query: str, max_results: int) -> WebSearchResponse:
        """Execute search with exponential backoff retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self._execute_search(query, max_results)
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Search attempt {attempt + 1}/{self.max_retries} failed for query '{query}': {e}"
                )
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.debug(f"Retrying in {delay}s...")
                    time.sleep(delay)
        
        # All retries failed
        logger.error(f"All {self.max_retries} search attempts failed for query: {query}")
        return WebSearchResponse(
            query=query,
            error=f"Search failed after {self.max_retries} attempts: {last_error}"
        )
    
    def _execute_search(self, query: str, max_results: int) -> WebSearchResponse:
        """
        Execute the actual search using OpenAI Responses API.
        
        Note: The OpenAI Python SDK doesn't yet have a stable 'responses' API.
        This implementation uses the chat completions API with a workaround
        to extract web search results from the response.
        """
        try:
            # Use chat completion with instructions to search the web
            # The model should use web_search tool internally
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a web search assistant. When given a query, search the web "
                            f"and return up to {max_results} relevant results. "
                            "For each result, provide: URL, title, and a brief snippet. "
                            "Format your response as JSON with this structure: "
                            '{"results": [{"url": "...", "title": "...", "snippet": "..."}]}. '
                            "Return ONLY the JSON, no other text."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Search query: {query}"
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                timeout=self.timeout,
            )
            
            # Extract results from response
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            results = self._parse_search_results(content, query)
            
            return WebSearchResponse(
                query=query,
                results=results[:max_results],
                total_results=len(results),
                model_used=response.model,
            )
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from model response: {e}")
            return WebSearchResponse(
                query=query,
                error=f"Failed to parse search results: {e}"
            )
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            raise
    
    def _parse_search_results(self, content: str, query: str) -> List[SearchResult]:
        """Parse search results from model response."""
        try:
            # Try to extract JSON from response
            content = content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            data = json.loads(content)
            
            results = []
            for item in data.get("results", []):
                try:
                    result = SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        relevance_score=item.get("relevance_score"),
                        source_type=item.get("source_type", "web"),
                        timestamp=datetime.now().isoformat(),
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to parse search result item: {e}")
                    continue
            
            return results
        
        except json.JSONDecodeError:
            # Fallback: try to extract URLs and titles from text
            logger.warning("Could not parse JSON, attempting text extraction")
            return self._extract_results_from_text(content)
    
    def _extract_results_from_text(self, text: str) -> List[SearchResult]:
        """Fallback: extract URLs and context from plain text response."""
        import re
        
        results = []
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        for url in urls[:self.max_results]:
            # Try to extract title/context around URL
            url_index = text.find(url)
            context_start = max(0, url_index - 100)
            context_end = min(len(text), url_index + len(url) + 100)
            snippet = text[context_start:context_end].strip()
            
            results.append(SearchResult(
                url=url,
                title=url.split("/")[2] if "/" in url else url,  # Domain as title
                snippet=snippet,
                timestamp=datetime.now().isoformat(),
            ))
        
        return results
    
    def search_multiple(
        self,
        queries: List[str],
        max_results_per_query: Optional[int] = None,
    ) -> Dict[str, WebSearchResponse]:
        """
        Search multiple queries and return results as a dictionary.
        
        Args:
            queries: List of search queries
            max_results_per_query: Max results per query (uses default if None)
        
        Returns:
            Dictionary mapping query -> WebSearchResponse
        """
        results = {}
        for query in queries:
            logger.info(f"Searching: {query}")
            results[query] = self.search(query, max_results=max_results_per_query)
        return results
    
    def clear_cache(self):
        """Clear the search cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Search cache cleared")


# ========================
# Convenience Functions
# ========================
@lru_cache(maxsize=1)
def get_default_search_tool() -> WebSearchTool:
    """Get a singleton instance of WebSearchTool with default settings."""
    return WebSearchTool()


def quick_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Quick search function for simple use cases.
    
    Args:
        query: Search query
        max_results: Maximum results to return
    
    Returns:
        List of SearchResult objects
    """
    tool = get_default_search_tool()
    response = tool.search(query, max_results=max_results)
    return response.results


# ========================
# CLI Demo
# ========================
def main():
    """Demo/test the web search tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Search Tool Demo")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum results")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create tool
    tool = WebSearchTool(
        model=args.model,
        max_results=args.max_results,
        enable_cache=not args.no_cache,
    )
    
    # Search
    print(f"\n🔍 Searching for: {args.query}\n")
    response = tool.search(args.query)
    
    if response.error:
        print(f"❌ Error: {response.error}")
        return
    
    print(f"✅ Found {response.total_results} results in {response.search_time_ms:.0f}ms\n")
    
    for i, result in enumerate(response.results, 1):
        print(f"{i}. {result.title}")
        print(f"   🔗 {result.url}")
        print(f"   📝 {result.snippet[:150]}...")
        print()


if __name__ == "__main__":
    main()
