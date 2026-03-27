"""Economic event calendar — boosts priority for markets near scheduled events.

Uses free public sources for upcoming Fed, CPI, GDP, and jobs report dates.
"""

import datetime

import httpx
import structlog

logger = structlog.get_logger()

# Known recurring US economic events (approximate schedule)
# These are static fallbacks — FRED API provides live dates
RECURRING_EVENTS = [
    {"name": "FOMC Decision", "keywords": ["fed", "fomc", "rate cut", "rate hike"]},
    {"name": "CPI Release", "keywords": ["cpi", "inflation"]},
    {"name": "Jobs Report", "keywords": ["jobs report", "nonfarm", "payroll", "unemployment"]},
    {"name": "GDP Report", "keywords": ["gdp"]},
    {"name": "PCE Release", "keywords": ["pce", "personal consumption"]},
]

# Cache fetched dates
_event_cache: dict[str, list[datetime.date]] = {}
_cache_time: float = 0
CACHE_TTL = 86400  # 24 hours


async def get_upcoming_events(days_ahead: int = 14) -> list[dict]:
    """Return upcoming economic events within the next N days."""
    import time

    global _cache_time

    now = datetime.date.today()
    end = now + datetime.timedelta(days=days_ahead)

    # Try FRED API for real dates
    if time.time() - _cache_time > CACHE_TTL:
        await _fetch_fred_dates()
        _cache_time = time.time()

    events = []
    for name, dates in _event_cache.items():
        for d in dates:
            if now <= d <= end:
                event_info = next(
                    (e for e in RECURRING_EVENTS if e["name"] == name), None
                )
                events.append({
                    "name": name,
                    "date": d.isoformat(),
                    "days_until": (d - now).days,
                    "keywords": event_info["keywords"] if event_info else [],
                })

    events.sort(key=lambda e: e["days_until"])
    logger.info("upcoming_events", count=len(events))
    return events


async def _fetch_fred_dates() -> None:
    """Fetch upcoming economic release dates from FRED calendar."""
    try:
        # FRED releases calendar (public, no API key needed)
        url = "https://api.stlouisfed.org/fred/releases/dates"
        today = datetime.date.today()
        params = {
            "realtime_start": today.isoformat(),
            "realtime_end": (today + datetime.timedelta(days=30)).isoformat(),
            "file_type": "json",
            "api_key": "DEMO_KEY",  # FRED allows limited demo access
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.debug("fred_api_unavailable", status=resp.status_code)
                _populate_static_dates()
                return
            data = resp.json()

        # Parse release dates
        for release_date in data.get("release_dates", []):
            date_str = release_date.get("date", "")
            try:
                d = datetime.date.fromisoformat(date_str)
                for event in RECURRING_EVENTS:
                    name_lower = release_date.get("release_name", "").lower()
                    if any(kw in name_lower for kw in event["keywords"]):
                        if event["name"] not in _event_cache:
                            _event_cache[event["name"]] = []
                        _event_cache[event["name"]].append(d)
            except ValueError:
                continue

        logger.info("fred_dates_fetched", events=len(_event_cache))
    except Exception:
        logger.debug("fred_fetch_failed_using_static")
        _populate_static_dates()


def _populate_static_dates() -> None:
    """Fallback: generate approximate event dates based on typical schedules."""
    today = datetime.date.today()
    _event_cache.clear()

    for event in RECURRING_EVENTS:
        dates = []
        # Generate approximate monthly dates for the next 60 days
        for month_offset in range(3):
            m = today.month + month_offset
            y = today.year + (m - 1) // 12
            m = ((m - 1) % 12) + 1
            # Approximate: most releases happen in first 2 weeks of month
            try:
                dates.append(datetime.date(y, m, 10))
            except ValueError:
                pass
        _event_cache[event["name"]] = [d for d in dates if d >= today]


def boost_event_markets(
    markets: list,
    upcoming_events: list[dict],
) -> list:
    """Score boost for markets matching upcoming events. Closer = bigger boost."""
    if not upcoming_events:
        return markets

    all_keywords = set()
    keyword_boost: dict[str, float] = {}
    for event in upcoming_events:
        days = max(event["days_until"], 1)
        boost = 10.0 / days  # Closer events get bigger boost
        for kw in event["keywords"]:
            kw_lower = kw.lower()
            all_keywords.add(kw_lower)
            keyword_boost[kw_lower] = max(keyword_boost.get(kw_lower, 0), boost)

    for market in markets:
        title_lower = market.title.lower()
        event_score = sum(
            boost for kw, boost in keyword_boost.items() if kw in title_lower
        )
        if event_score > 0:
            current = getattr(market, "_score", 0)
            market._score = current * (1 + event_score)  # type: ignore[attr-defined]

    return markets
