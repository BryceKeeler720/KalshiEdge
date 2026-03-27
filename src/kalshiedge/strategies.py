"""Trading strategies beyond the primary calibration edge.

Strategy 1 (calibration_edge) lives in main.py's _process_market.
Strategies 2-4 are implemented here and run each cycle.
"""

import asyncio
import datetime

import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.discovery import Market, _dollars_to_cents
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore
from kalshiedge.risk import ProposedTrade, RiskManager
from kalshiedge.trader import place_limit_order

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Strategy 2: Near-Expiry Convergence
# Buy contracts priced 93-97c when resolution is imminent and outcome is near
# certain. Pure arbitrage — no forecasting needed.
# ---------------------------------------------------------------------------

CONVERGENCE_MIN_PRICE = 93
CONVERGENCE_MAX_PRICE = 97
CONVERGENCE_MAX_HOURS = 24  # Only within 24h of close


@observe(span_type=SpanType.CHAIN)
async def run_near_expiry_convergence(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
) -> int:
    """Find near-expiry markets where outcome is nearly certain. Returns trade count."""
    trades = 0

    try:
        data = await kalshi.get_markets(limit=100)
        raw_markets = data.get("markets", [])
    except Exception:
        logger.warning("convergence_market_fetch_failed")
        return 0

    now = datetime.datetime.now(datetime.timezone.utc)

    for m in raw_markets:
        ticker = m.get("ticker", "")
        if ticker.startswith("KXMVE"):
            continue

        close_str = m.get("close_time", "")
        if not close_str:
            continue

        try:
            close = datetime.datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            hours_left = (close - now).total_seconds() / 3600
        except ValueError:
            continue

        if hours_left < 0.5 or hours_left > CONVERGENCE_MAX_HOURS:
            continue

        yes_bid = _dollars_to_cents(m.get("yes_bid_dollars") or m.get("yes_bid"))
        no_bid = _dollars_to_cents(m.get("no_bid_dollars") or m.get("no_bid"))

        # Check if YES side is near certain
        if CONVERGENCE_MIN_PRICE <= yes_bid <= CONVERGENCE_MAX_PRICE:
            edge = (100 - yes_bid) / 100
            trades += await _convergence_trade(
                kalshi, store, risk, ticker, "yes", yes_bid, edge
            )

        # Check if NO side is near certain
        if CONVERGENCE_MIN_PRICE <= no_bid <= CONVERGENCE_MAX_PRICE:
            edge = (100 - no_bid) / 100
            trades += await _convergence_trade(
                kalshi, store, risk, ticker, "no", no_bid, edge
            )

    logger.info("convergence_scan_complete", trades=trades)
    return trades


async def _convergence_trade(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    ticker: str,
    side: str,
    price: int,
    edge: float,
) -> int:
    bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
    max_cost = int(bankroll * settings.max_position_pct)
    count = max_cost // price
    if count <= 0:
        return 0

    trade = ProposedTrade(
        ticker=ticker, side=side, action="buy",
        price_cents=price, count=count, edge=edge,
    )
    if not await risk.can_trade(trade):
        return 0

    if settings.dry_run:
        logger.info(
            "dry_run_convergence",
            ticker=ticker, side=side, count=count,
            price_cents=price, edge=f"{edge:.1%}",
        )
        return 0

    order = await place_limit_order(kalshi, ticker, side, count, price)
    if order:
        await store.record_trade(
            ticker=ticker, side=side, action="buy",
            price_cents=price, count=count,
            order_id=order.get("order_id", ""),
            strategy="near_expiry_convergence",
        )
        logger.info(
            "convergence_trade",
            ticker=ticker, side=side, count=count, price=price,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Strategy 3: Event-Driven Positioning
# Look for markets tied to upcoming scheduled events (Fed, CPI, earnings)
# and position when model has strong conviction before the event.
# ---------------------------------------------------------------------------

EVENT_KEYWORDS = [
    "Fed", "FOMC", "rate cut", "rate hike", "CPI", "inflation",
    "GDP", "jobs report", "unemployment", "nonfarm", "payroll",
    "earnings", "court ruling", "Supreme Court", "verdict",
    "election", "vote", "ballot", "primary",
]


@observe(span_type=SpanType.CHAIN)
async def find_event_driven_markets(
    all_markets: list[Market],
) -> list[Market]:
    """Filter markets that match event-driven keywords. Higher edge threshold."""
    matches = []
    for m in all_markets:
        title_lower = m.title.lower()
        if any(kw.lower() in title_lower for kw in EVENT_KEYWORDS):
            matches.append(m)
    logger.info("event_driven_candidates", count=len(matches))
    return matches


# ---------------------------------------------------------------------------
# Strategy 4: Intra-Event Arbitrage
# Within a mutually-exclusive event, probabilities should sum to ~100%.
# If they sum to significantly more or less, there's an arb opportunity.
# ---------------------------------------------------------------------------

ARB_THRESHOLD = 0.05  # 5% deviation from 100% to trigger


@observe(span_type=SpanType.CHAIN)
async def run_intra_event_arbitrage(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
) -> int:
    """Find events where market probabilities don't sum to ~100%."""
    trades = 0

    try:
        events_data = await kalshi.get_events()
        events = events_data.get("events", [])
    except Exception:
        logger.warning("arb_events_fetch_failed")
        return 0

    for event in events[:20]:  # Cap to avoid rate limits
        if not event.get("mutually_exclusive", False):
            continue

        event_ticker = event.get("event_ticker", "")
        await asyncio.sleep(1.5)  # Rate limit courtesy

        try:
            data = await kalshi.get_markets(
                limit=50, event_ticker=event_ticker
            )
            markets = data.get("markets", [])
        except Exception:
            continue

        if len(markets) < 2:
            continue

        # Sum up implied probabilities
        total_prob = 0.0
        market_prices: list[tuple[str, int]] = []
        for m in markets:
            ticker = m.get("ticker", "")
            if ticker.startswith("KXMVE"):
                continue
            price = _dollars_to_cents(
                m.get("last_price_dollars") or m.get("last_price")
            )
            if price > 0:
                total_prob += price / 100
                market_prices.append((ticker, price))

        if len(market_prices) < 2:
            continue

        deviation = total_prob - 1.0

        if abs(deviation) > ARB_THRESHOLD:
            logger.info(
                "arb_opportunity",
                event=event_ticker,
                total_prob=f"{total_prob:.1%}",
                deviation=f"{deviation:+.1%}",
                markets=len(market_prices),
            )

            if deviation > ARB_THRESHOLD:
                # Overpriced — sell the most overpriced (buy NO on highest)
                market_prices.sort(key=lambda x: x[1], reverse=True)
                ticker, price = market_prices[0]
                edge = deviation / len(market_prices)
                trades += await _arb_trade(
                    kalshi, store, risk, ticker, "no", price, edge, event_ticker
                )
            elif deviation < -ARB_THRESHOLD:
                # Underpriced — buy the cheapest YES
                market_prices.sort(key=lambda x: x[1])
                ticker, price = market_prices[0]
                edge = abs(deviation) / len(market_prices)
                trades += await _arb_trade(
                    kalshi, store, risk, ticker, "yes", price, edge, event_ticker
                )

    logger.info("arb_scan_complete", trades=trades)
    return trades


async def _arb_trade(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    ticker: str,
    side: str,
    price_cents: int,
    edge: float,
    event_ticker: str,
) -> int:
    if edge < settings.min_edge_threshold:
        return 0

    buy_price = price_cents if side == "yes" else (100 - price_cents)
    bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
    max_cost = int(bankroll * settings.max_position_pct)
    count = max_cost // buy_price if buy_price > 0 else 0
    if count <= 0:
        return 0

    trade = ProposedTrade(
        ticker=ticker, side=side, action="buy",
        price_cents=buy_price, count=count, edge=edge,
    )
    if not await risk.can_trade(trade):
        return 0

    if settings.dry_run:
        logger.info(
            "dry_run_arb",
            ticker=ticker, side=side, count=count,
            price_cents=buy_price, edge=f"{edge:.1%}",
            event=event_ticker,
        )
        return 0

    order = await place_limit_order(kalshi, ticker, side, count, buy_price)
    if order:
        await store.record_trade(
            ticker=ticker, side=side, action="buy",
            price_cents=buy_price, count=count,
            order_id=order.get("order_id", ""),
            strategy="intra_event_arbitrage",
        )
        logger.info(
            "arb_trade",
            ticker=ticker, side=side, count=count,
            event=event_ticker,
        )
        return 1
    return 0
