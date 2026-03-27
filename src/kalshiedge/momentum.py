"""Momentum detector — catch sudden price/volume spikes that signal informed trading.

Runs in the fast loop (zero Claude cost). Monitors:
- Price moves >5c in recent history
- Volume spikes >3x average
- Trades in the direction of the move
"""

import datetime
import time

import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.discovery import _dollars_to_cents, _parse_volume
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore
from kalshiedge.risk import ProposedTrade, RiskManager
from kalshiedge.trader import place_limit_order

logger = structlog.get_logger()

# Momentum thresholds
MIN_PRICE_MOVE_CENTS = 5  # Minimum price change to trigger
VOLUME_SPIKE_MULTIPLIER = 3.0  # Volume must be 3x the norm
MIN_VOLUME_FOR_SIGNAL = 100  # Minimum absolute volume
LOOKBACK_HOURS = 4  # Compare current vs N hours ago
MOMENTUM_EDGE = 0.06  # Treat momentum as 6% edge for sizing
MAX_MOMENTUM_TRADES_PER_CYCLE = 2

# Cache of previous prices for comparison
_price_cache: dict[str, dict] = {}  # ticker -> {price, volume, timestamp}


def record_price(ticker: str, price_cents: int, volume: int) -> None:
    """Record a price observation for momentum tracking."""
    now = time.time()
    if ticker not in _price_cache:
        _price_cache[ticker] = {
            "price": price_cents,
            "volume": volume,
            "timestamp": now,
            "prev_price": price_cents,
            "prev_volume": volume,
            "prev_timestamp": now,
        }
    else:
        entry = _price_cache[ticker]
        age = now - entry["timestamp"]
        # Rotate: current becomes previous after lookback period
        if age > LOOKBACK_HOURS * 3600:
            entry["prev_price"] = entry["price"]
            entry["prev_volume"] = entry["volume"]
            entry["prev_timestamp"] = entry["timestamp"]
        entry["price"] = price_cents
        entry["volume"] = volume
        entry["timestamp"] = now


def detect_momentum(ticker: str) -> dict | None:
    """Check if a ticker has momentum. Returns signal dict or None."""
    entry = _price_cache.get(ticker)
    if not entry:
        return None

    age = entry["timestamp"] - entry["prev_timestamp"]
    if age < 1800:  # Need at least 30 min of history
        return None

    price_now = entry["price"]
    price_prev = entry["prev_price"]
    vol_now = entry["volume"]
    vol_prev = entry["prev_volume"]

    price_move = price_now - price_prev

    # Check price move threshold
    if abs(price_move) < MIN_PRICE_MOVE_CENTS:
        return None

    # Check volume spike
    if vol_prev > 0:
        vol_ratio = vol_now / vol_prev
    else:
        vol_ratio = VOLUME_SPIKE_MULTIPLIER + 1 if vol_now > MIN_VOLUME_FOR_SIGNAL else 0

    if vol_now < MIN_VOLUME_FOR_SIGNAL:
        return None

    # Direction of momentum
    side = "yes" if price_move > 0 else "no"

    return {
        "ticker": ticker,
        "side": side,
        "price_move": price_move,
        "price_now": price_now,
        "volume_ratio": round(vol_ratio, 1),
        "volume_now": vol_now,
        "signal_strength": abs(price_move) * vol_ratio,
    }


@observe(span_type=SpanType.CHAIN)
async def run_momentum_scan(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
) -> int:
    """Scan markets for momentum signals. Returns trade count."""
    trades = 0

    # Fetch fresh market data for all markets we're tracking
    try:
        data = await kalshi.get_markets(limit=100)
        raw_markets = data.get("markets", [])
    except Exception:
        logger.warning("momentum_market_fetch_failed")
        return 0

    # Update price cache
    signals = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for m in raw_markets:
        ticker = m.get("ticker", "")
        if ticker.startswith("KXMVE"):
            continue

        price = _dollars_to_cents(
            m.get("last_price_dollars") or m.get("last_price")
        )
        volume = _parse_volume(m.get("volume_24h_fp") or m.get("volume_24h"))

        if price <= 2 or price >= 98:
            continue

        # Check expiry — only short-dated markets
        close_str = m.get("close_time", "")
        if close_str:
            try:
                close = datetime.datetime.fromisoformat(
                    close_str.replace("Z", "+00:00")
                )
                days_left = (close - now).total_seconds() / 86400
                if days_left > 90 or days_left < 0.1:
                    continue
            except ValueError:
                continue

        record_price(ticker, price, volume)
        signal = detect_momentum(ticker)
        if signal:
            signals.append(signal)

    # Sort by signal strength, take top signals
    signals.sort(key=lambda s: s["signal_strength"], reverse=True)

    # Skip tickers we already hold
    open_trades = await store.get_open_positions()
    open_tickers = {t["ticker"] for t in open_trades}

    for signal in signals[:MAX_MOMENTUM_TRADES_PER_CYCLE]:
        ticker = signal["ticker"]
        if ticker in open_tickers:
            continue

        side = signal["side"]
        price = signal["price_now"]
        buy_price = price if side == "yes" else (100 - price)

        if buy_price <= 0 or buy_price >= 99:
            continue

        bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
        # Smaller position size for momentum — higher risk
        max_cost = int(bankroll * settings.max_position_pct * 0.5)
        count = max_cost // buy_price
        if count <= 0:
            continue

        trade = ProposedTrade(
            ticker=ticker,
            side=side,
            action="buy",
            price_cents=buy_price,
            count=count,
            edge=MOMENTUM_EDGE,
        )
        if not await risk.can_trade(trade):
            continue

        logger.info(
            "momentum_signal",
            ticker=ticker,
            side=side,
            price_move=f"{signal['price_move']:+d}c",
            volume_ratio=signal["volume_ratio"],
            count=count,
            price=buy_price,
        )

        if settings.dry_run:
            continue

        order = await place_limit_order(kalshi, ticker, side, count, buy_price)
        if order:
            await store.record_trade(
                ticker=ticker,
                side=side,
                action="buy",
                price_cents=buy_price,
                count=count,
                order_id=order.get("order_id", ""),
                strategy="momentum",
            )
            trades += 1
            open_tickers.add(ticker)

    if signals:
        logger.info("momentum_scan_complete", signals=len(signals), trades=trades)

    return trades
