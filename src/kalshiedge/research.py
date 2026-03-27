"""News research pipeline with caching."""

import time
from dataclasses import dataclass, field
from urllib.parse import quote

import feedparser
import httpx
import structlog

from kalshiedge._observe import SpanType, observe

logger = structlog.get_logger()

CACHE_TTL_SECONDS = 1800  # 30 minutes
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


@dataclass
class NewsItem:
    title: str
    source: str = ""
    published: str = ""
    description: str = ""


@dataclass
class _CacheEntry:
    items: list[NewsItem]
    timestamp: float = field(default_factory=time.time)


_cache: dict[str, _CacheEntry] = {}


@observe(span_type=SpanType.RETRIEVAL)
async def gather_news(query: str) -> list[NewsItem]:
    """Gather news via Google News RSS, fall back to GDELT."""
    cache_key = query.lower().strip()
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached.timestamp) < CACHE_TTL_SECONDS:
        logger.debug("news_cache_hit", query=query)
        return cached.items

    items = await _try_google_rss(query)
    if not items:
        items = await _try_gdelt(query)

    _cache[cache_key] = _CacheEntry(items=items)
    logger.info("news_gathered", query=query, count=len(items))
    return items


async def _try_google_rss(query: str) -> list[NewsItem]:
    """Fetch Google News RSS directly with feedparser."""
    try:
        url = GOOGLE_NEWS_RSS.format(query=quote(query))
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        feed = feedparser.parse(resp.text)
        entries = feed.get("entries", [])[:5]
        return [
            NewsItem(
                title=e.get("title", ""),
                source=e.get("source", {}).get("title", ""),
                published=e.get("published", ""),
            )
            for e in entries
        ]
    except Exception:
        logger.debug("google_rss_failed", query=query)
        return []


async def _try_gdelt(query: str) -> list[NewsItem]:
    """Fall back to GDELT API for news."""
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": "5",
            "format": "json",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("articles", [])
        return [
            NewsItem(
                title=a.get("title", ""),
                source=a.get("domain", ""),
                published=a.get("seendate", ""),
                description=a.get("title", ""),
            )
            for a in articles[:5]
        ]
    except Exception:
        logger.debug("gdelt_failed", query=query)
        return []


def format_news_context(items: list[NewsItem]) -> str:
    """Format news items into a string for the forecaster prompt."""
    if not items:
        return "No recent news found."
    lines = []
    for i, item in enumerate(items, 1):
        line = f"{i}. {item.title}"
        if item.source:
            line += f" ({item.source})"
        if item.published:
            line += f" [{item.published}]"
        lines.append(line)
    return "\n".join(lines)
