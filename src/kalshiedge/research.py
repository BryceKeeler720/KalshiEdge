"""News research pipeline with caching."""

import time
from dataclasses import dataclass, field

import structlog

from kalshiedge._observe import SpanType, observe

logger = structlog.get_logger()

CACHE_TTL_SECONDS = 1800  # 30 minutes


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
    """Gather news for a market title. Tries pygooglenews, falls back to GDELT."""
    cache_key = query.lower().strip()
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached.timestamp) < CACHE_TTL_SECONDS:
        logger.debug("news_cache_hit", query=query)
        return cached.items

    items = await _try_googlenews(query)
    if not items:
        items = await _try_gdelt(query)

    _cache[cache_key] = _CacheEntry(items=items)
    logger.info("news_gathered", query=query, count=len(items))
    return items


async def _try_googlenews(query: str) -> list[NewsItem]:
    try:
        from pygooglenews import GoogleNews

        gn = GoogleNews(lang="en", country="US")
        results = gn.search(query, when="7d")
        entries = results.get("entries", [])[:5]
        return [
            NewsItem(
                title=e.get("title", ""),
                source=e.get("source", {}).get("title", ""),
                published=e.get("published", ""),
            )
            for e in entries
        ]
    except Exception:
        logger.debug("googlenews_failed", query=query)
        return []


async def _try_gdelt(query: str) -> list[NewsItem]:
    try:
        from gdeltdoc import Filters, GdeltDoc

        gd = GdeltDoc()
        f = Filters(keyword=query, timespan="7d")
        articles = gd.article_search(f)
        if articles.empty:
            return []
        records = articles.head(5).to_dict("records")
        return [
            NewsItem(
                title=r.get("title", ""),
                source=r.get("domain", ""),
                published=r.get("seendate", ""),
                description=r.get("title", ""),
            )
            for r in records
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
