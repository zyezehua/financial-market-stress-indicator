"""
Fetch recent macro/financial news to ground LLM narratives in current events.

Uses Tavily Search API (free tier: 1000 searches/month).
Sign up at https://tavily.com — set TAVILY_API_KEY in .env.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def _build_query(attribution: List[dict]) -> str:
    """Turn top SHAP factor labels into a focused search query."""
    if not attribution:
        return "financial market stress volatility credit spreads"
    top_labels = [f["label"] for f in attribution[:3]]
    return " ".join(top_labels) + " financial market stress"


def fetch_macro_news(
    attribution: List[dict],
    n: int = 5,
    api_key: Optional[str] = None,
    days_back: int = 7,
) -> List[dict]:
    """
    Search for recent macro news relevant to the top stress drivers.

    Parameters
    ----------
    attribution : list of factor attribution dicts (from factor_attribution.py)
    n           : number of articles to return
    api_key     : Tavily API key (falls back to TAVILY_API_KEY env var)
    days_back   : how many days back to search

    Returns
    -------
    List of dicts with keys: title, snippet, url, published_date
    """
    key = api_key or os.getenv("TAVILY_API_KEY", "")
    if not key:
        logger.warning("TAVILY_API_KEY not set — skipping news fetch.")
        return []

    try:
        from tavily import TavilyClient
    except ImportError:
        logger.warning("tavily-python not installed. Run: pip install tavily-python")
        return []

    query = _build_query(attribution)
    logger.info("Fetching news: %r", query)

    try:
        client = TavilyClient(api_key=key)
        response = client.search(
            query=query,
            search_depth="basic",
            topic="finance",
            days=days_back,
            max_results=n,
            include_answer=False,
        )
        articles = []
        for r in response.get("results", []):
            articles.append({
                "title":          r.get("title", ""),
                "snippet":        r.get("content", "")[:300],
                "url":            r.get("url", ""),
                "published_date": r.get("published_date", ""),
            })
        logger.info("Fetched %d news articles.", len(articles))
        return articles
    except Exception as exc:
        logger.warning("Tavily search failed (%s) — continuing without news.", exc)
        return []
