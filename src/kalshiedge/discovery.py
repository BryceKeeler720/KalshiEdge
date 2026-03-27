"""Market discovery — event-driven approach to find tradeable markets."""

import asyncio
import datetime

import structlog
from pydantic import BaseModel

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.kalshi_client import KalshiClient

logger = structlog.get_logger()

MIN_VOLUME = 50
MIN_HOURS_TO_EXPIRY = 2
MAX_DAYS_TO_EXPIRY = 90  # Don't trade markets expiring more than 90 days out
PAGE_DELAY = 2.5  # seconds between API calls to avoid rate limits

# Categories where LLMs have strongest forecasting edge
PRIORITY_CATEGORIES = {
    "Economics",
    "Elections",
    "Politics",
    "World",
    "Climate and Weather",
    "Science and Technology",
    "Financial",
}


class Market(BaseModel):
    ticker: str
    title: str
    yes_bid: int  # cents
    yes_ask: int
    no_bid: int
    no_ask: int
    last_price: int  # cents
    volume: int
    volume_24h: int
    open_interest: int
    close_time: str
    category: str = ""
    event_ticker: str = ""
    series_ticker: str = ""


def _dollars_to_cents(val: str | int | float | None) -> int:
    """Convert dollar string like '0.6200' or legacy int cents to int cents."""
    if val is None:
        return 0
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(round(val * 100))
    try:
        return int(round(float(val) * 100))
    except (ValueError, TypeError):
        return 0


def _parse_volume(val: str | int | float | None) -> int:
    """Parse volume from either int (legacy) or float-string (new API)."""
    if val is None:
        return 0
    if isinstance(val, int):
        return val
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


@observe(span_type=SpanType.RETRIEVAL)
async def discover_markets(client: KalshiClient) -> list[Market]:
    """Fetch events, then get markets per series. Filters by category, volume, spread."""
    # Step 1: Get events in priority categories
    events_data = await client.get_events()
    events = events_data.get("events", [])

    series_tickers: list[tuple[str, str]] = []  # (series_ticker, category)
    for e in events:
        cat = e.get("category", "")
        series = e.get("series_ticker", "")
        if cat in PRIORITY_CATEGORIES and series:
            series_tickers.append((series, cat))

    # Deduplicate series
    seen: set[str] = set()
    unique_series: list[tuple[str, str]] = []
    for st, cat in series_tickers:
        if st not in seen:
            seen.add(st)
            unique_series.append((st, cat))

    # Cap to avoid excessive API calls per cycle
    max_series = 30
    unique_series = unique_series[:max_series]

    logger.info(
        "events_scanned",
        total_events=len(events),
        series_count=len(unique_series),
    )

    # Step 2: Fetch markets for each series
    markets: list[Market] = []
    for series_ticker, category in unique_series:
        await asyncio.sleep(PAGE_DELAY)
        try:
            data = await client.get_markets(limit=50, series_ticker=series_ticker)
            for m in data.get("markets", []):
                parsed = _parse_market(m, category)
                if parsed and _passes_filters(parsed):
                    markets.append(parsed)
        except Exception:
            logger.warning("series_fetch_failed", series=series_ticker)

    logger.info("markets_discovered", total=len(markets))
    return markets


def _parse_market(raw: dict, category: str = "") -> Market | None:
    try:
        # Skip multi-leg combo markets
        ticker = raw["ticker"]
        if ticker.startswith("KXMVE"):
            return None

        yes_bid = _dollars_to_cents(raw.get("yes_bid_dollars") or raw.get("yes_bid"))
        yes_ask = _dollars_to_cents(raw.get("yes_ask_dollars") or raw.get("yes_ask"))
        no_bid = _dollars_to_cents(raw.get("no_bid_dollars") or raw.get("no_bid"))
        no_ask = _dollars_to_cents(raw.get("no_ask_dollars") or raw.get("no_ask"))
        last_price = _dollars_to_cents(
            raw.get("last_price_dollars") or raw.get("last_price")
        )
        volume = _parse_volume(raw.get("volume_fp") or raw.get("volume"))
        volume_24h = _parse_volume(raw.get("volume_24h_fp") or raw.get("volume_24h"))
        open_interest = _parse_volume(
            raw.get("open_interest_fp") or raw.get("open_interest")
        )

        return Market(
            ticker=ticker,
            title=raw.get("title", ""),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_price=last_price if last_price > 0 else yes_bid,
            volume=volume,
            volume_24h=volume_24h,
            open_interest=open_interest,
            close_time=raw.get("close_time", ""),
            category=category or raw.get("category", ""),
            event_ticker=raw.get("event_ticker", ""),
            series_ticker=raw.get("series_ticker", ""),
        )
    except Exception:
        logger.warning("market_parse_failed", ticker=raw.get("ticker"))
        return None


def _passes_filters(market: Market) -> bool:
    # Must have some volume
    if market.volume < MIN_VOLUME:
        return False

    # Must have time remaining — but not too much
    if market.close_time:
        try:
            close = datetime.datetime.fromisoformat(
                market.close_time.replace("Z", "+00:00")
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            hours_left = (close - now).total_seconds() / 3600
            if hours_left < MIN_HOURS_TO_EXPIRY:
                return False
            if hours_left > MAX_DAYS_TO_EXPIRY * 24:
                return False
        except ValueError:
            pass

    # Must have a meaningful price (not 0 or 100)
    if market.last_price <= 2 or market.last_price >= 98:
        return False

    # Must have some bid/ask (not completely empty book)
    if market.yes_bid <= 0 and market.yes_ask <= 0:
        return False

    return True


async def check_orderbook_depth(client: KalshiClient, ticker: str) -> bool:
    """Check that the orderbook has sufficient depth."""
    try:
        ob = await client.get_orderbook(ticker)
        # New API: orderbook_fp with yes_dollars/no_dollars
        # Legacy: orderbook with yes/no
        book = ob.get("orderbook_fp") or ob.get("orderbook", {})
        yes_levels = len(book.get("yes_dollars") or book.get("yes") or [])
        no_levels = len(book.get("no_dollars") or book.get("no") or [])
        min_depth = settings.orderbook_min_depth
        return yes_levels >= min_depth and no_levels >= min_depth
    except Exception:
        logger.warning("orderbook_check_failed", ticker=ticker)
        return False
