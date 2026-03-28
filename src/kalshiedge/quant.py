"""Quant engine — pure math, zero Claude cost, high-frequency trading strategies.

Runs on a 30-60 second cycle alongside the fast loop. All strategies use only
price, volume, orderbook, and time data — no LLM calls.

Strategies:
  1. Mean Reversion  — fade price moves that deviate from rolling VWAP
  2. Orderbook Imbalance — trade in direction of bid/ask pressure
  3. Wide Convergence — broader near-expiry band (85-97c, up to 72h)
  4. Cross-Event Relative Value — related markets that diverge
  5. Theta Decay Seller — sell expensive YES on low-prob markets further from expiry
  6. Spread Capture — place passive maker orders to collect bid-ask spread
"""

import datetime
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.discovery import _dollars_to_cents, _parse_volume
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore
from kalshiedge.risk import ProposedTrade, RiskManager
from kalshiedge.trader import place_limit_order

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Quant configuration — lower thresholds, smaller sizes, more trades
# ---------------------------------------------------------------------------

QUANT_MIN_EDGE = settings.quant_min_edge
QUANT_POSITION_PCT = settings.quant_position_pct
QUANT_MAX_TRADES_PER_CYCLE = settings.quant_max_trades_per_cycle

# Mean Reversion
MR_MIN_DEVIATION_CENTS = 4  # Price must deviate 4c+ from VWAP
MR_VWAP_WINDOW_SECONDS = 3600  # 1-hour VWAP window
MR_MIN_OBSERVATIONS = 5  # Need enough data points
MR_EDGE = 0.04  # 4% assumed edge on mean reversion

# Orderbook Imbalance
OB_IMBALANCE_RATIO = 2.0  # Bid volume must be 2x ask (or vice versa)
OB_MIN_DEPTH_CENTS = 200  # Minimum total depth in cents on each side
OB_EDGE = 0.035  # 3.5% assumed edge

# Wide Convergence (relaxed version of Strategy 2)
WC_MIN_PRICE = 85  # Wider band: 85-97c (vs 93-97c)
WC_MAX_PRICE = 97
WC_MAX_HOURS = 72  # 72h window (vs 24h)
WC_MIN_HOURS = 0.25  # At least 15 min to expiry

# Cross-Event Relative Value
XEVENT_MIN_DIVERGENCE = 0.06  # 6% price divergence between related markets

# Theta Decay
THETA_MAX_YES_PRICE = 12  # YES priced 3-12c (wider than safe compounder's 1-7c)
THETA_MIN_YES_PRICE = 3
THETA_MAX_DAYS = 45  # Up to 45 days (vs 30)
THETA_MIN_HOURS = 4

# Spread Capture
SPREAD_MIN_WIDTH = 3  # Minimum 3c spread to capture
SPREAD_EDGE = 0.03


# ---------------------------------------------------------------------------
# Rolling price tracker for VWAP / mean reversion
# ---------------------------------------------------------------------------


@dataclass
class PriceObservation:
    price_cents: int
    volume: int
    timestamp: float


@dataclass
class MarketSnapshot:
    ticker: str
    title: str
    yes_bid: int
    yes_ask: int
    no_bid: int
    no_ask: int
    last_price: int
    volume_24h: int
    close_time: str
    event_ticker: str = ""
    series_ticker: str = ""


# Global price history for VWAP calculation
_price_history: dict[str, list[PriceObservation]] = defaultdict(list)
_MAX_HISTORY_PER_TICKER = 200


def record_observation(ticker: str, price_cents: int, volume: int) -> None:
    """Record a price/volume observation for VWAP tracking."""
    now = time.time()
    history = _price_history[ticker]
    history.append(PriceObservation(price_cents, volume, now))
    # Trim old observations
    cutoff = now - MR_VWAP_WINDOW_SECONDS * 2
    _price_history[ticker] = [
        obs for obs in history[-_MAX_HISTORY_PER_TICKER:] if obs.timestamp > cutoff
    ]


def compute_vwap(ticker: str, window_seconds: int = MR_VWAP_WINDOW_SECONDS) -> float | None:
    """Compute volume-weighted average price over the window."""
    now = time.time()
    cutoff = now - window_seconds
    observations = [obs for obs in _price_history.get(ticker, []) if obs.timestamp > cutoff]
    if len(observations) < MR_MIN_OBSERVATIONS:
        return None
    total_value = sum(obs.price_cents * max(obs.volume, 1) for obs in observations)
    total_volume = sum(max(obs.volume, 1) for obs in observations)
    if total_volume == 0:
        return None
    return total_value / total_volume


# ---------------------------------------------------------------------------
# Strategy Q1: Mean Reversion — fade deviations from VWAP
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.CHAIN)
async def run_mean_reversion(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    snapshots: list[MarketSnapshot],
    open_tickers: set[str],
) -> int:
    """Trade when price deviates significantly from rolling VWAP."""
    trades = 0

    for snap in snapshots:
        if snap.ticker in open_tickers:
            continue
        if snap.last_price <= 5 or snap.last_price >= 95:
            continue

        vwap = compute_vwap(snap.ticker)
        if vwap is None:
            continue

        deviation = snap.last_price - vwap
        if abs(deviation) < MR_MIN_DEVIATION_CENTS:
            continue

        # Price above VWAP → expect reversion down → buy NO
        # Price below VWAP → expect reversion up → buy YES
        if deviation > 0:
            side = "no"
            buy_price = 100 - snap.last_price
        else:
            side = "yes"
            buy_price = snap.last_price

        if buy_price <= 2 or buy_price >= 98:
            continue

        # Edge scales with deviation size
        edge = min(0.08, MR_EDGE + abs(deviation) * 0.005)

        result = await _place_quant_trade(
            kalshi, store, risk, snap.ticker, side, buy_price, edge, "quant_mean_reversion",
        )
        if result:
            trades += 1
            open_tickers.add(snap.ticker)

    return trades


# ---------------------------------------------------------------------------
# Strategy Q2: Orderbook Imbalance — trade in direction of pressure
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.CHAIN)
async def run_orderbook_imbalance(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    snapshots: list[MarketSnapshot],
    open_tickers: set[str],
) -> int:
    """Trade when orderbook shows strong directional pressure."""
    trades = 0

    for snap in snapshots:
        if snap.ticker in open_tickers:
            continue
        if snap.last_price <= 5 or snap.last_price >= 95:
            continue

        try:
            ob = await kalshi.get_orderbook(snap.ticker)
            book = ob.get("orderbook_fp") or ob.get("orderbook", {})
        except Exception:
            continue

        yes_levels = book.get("yes_dollars") or book.get("yes") or []
        no_levels = book.get("no_dollars") or book.get("no") or []

        if not yes_levels or not no_levels:
            continue

        # Calculate total depth on each side
        yes_depth = sum(
            _parse_volume(level[1]) if isinstance(level, (list, tuple)) else 0
            for level in yes_levels
        )
        no_depth = sum(
            _parse_volume(level[1]) if isinstance(level, (list, tuple)) else 0
            for level in no_levels
        )

        if yes_depth < OB_MIN_DEPTH_CENTS or no_depth < OB_MIN_DEPTH_CENTS:
            continue

        # Determine imbalance direction
        if yes_depth > no_depth * OB_IMBALANCE_RATIO:
            # Heavy YES-side buying pressure → price likely going up → buy YES
            side = "yes"
            buy_price = snap.yes_ask if snap.yes_ask > 0 else snap.last_price
            imbalance = yes_depth / no_depth
        elif no_depth > yes_depth * OB_IMBALANCE_RATIO:
            # Heavy NO-side buying pressure → price likely going down → buy NO
            side = "no"
            buy_price = snap.no_ask if snap.no_ask > 0 else (100 - snap.last_price)
            imbalance = no_depth / yes_depth
        else:
            continue

        if buy_price <= 2 or buy_price >= 98:
            continue

        # Edge scales with imbalance strength
        edge = min(0.07, OB_EDGE + (imbalance - OB_IMBALANCE_RATIO) * 0.005)

        result = await _place_quant_trade(
            kalshi, store, risk, snap.ticker, side, buy_price, edge, "quant_ob_imbalance",
        )
        if result:
            trades += 1
            open_tickers.add(snap.ticker)

    return trades


# ---------------------------------------------------------------------------
# Strategy Q3: Wide Convergence — broader near-expiry band
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.CHAIN)
async def run_wide_convergence(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    snapshots: list[MarketSnapshot],
    open_tickers: set[str],
) -> int:
    """Wider convergence band: 85-97c with up to 72h to expiry."""
    trades = 0
    now = datetime.datetime.now(datetime.timezone.utc)

    for snap in snapshots:
        if snap.ticker in open_tickers:
            continue

        if not snap.close_time:
            continue

        try:
            close = datetime.datetime.fromisoformat(snap.close_time.replace("Z", "+00:00"))
            hours_left = (close - now).total_seconds() / 3600
        except ValueError:
            continue

        if hours_left < WC_MIN_HOURS or hours_left > WC_MAX_HOURS:
            continue

        # Check YES side convergence
        if WC_MIN_PRICE <= snap.yes_bid <= WC_MAX_PRICE:
            edge = (100 - snap.yes_bid) / 100
            # Scale edge by time proximity — closer to expiry = more certain
            time_bonus = max(0, (1 - hours_left / WC_MAX_HOURS) * 0.02)
            effective_edge = edge + time_bonus

            if effective_edge >= QUANT_MIN_EDGE:
                result = await _place_quant_trade(
                    kalshi, store, risk, snap.ticker, "yes", snap.yes_bid,
                    effective_edge, "quant_wide_convergence",
                )
                if result:
                    trades += 1
                    open_tickers.add(snap.ticker)
                    continue

        # Check NO side convergence
        if WC_MIN_PRICE <= snap.no_bid <= WC_MAX_PRICE:
            edge = (100 - snap.no_bid) / 100
            time_bonus = max(0, (1 - hours_left / WC_MAX_HOURS) * 0.02)
            effective_edge = edge + time_bonus

            if effective_edge >= QUANT_MIN_EDGE:
                result = await _place_quant_trade(
                    kalshi, store, risk, snap.ticker, "no", snap.no_bid,
                    effective_edge, "quant_wide_convergence",
                )
                if result:
                    trades += 1
                    open_tickers.add(snap.ticker)

    return trades


# ---------------------------------------------------------------------------
# Strategy Q4: Cross-Event Relative Value
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.CHAIN)
async def run_cross_event_relative_value(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    snapshots: list[MarketSnapshot],
    open_tickers: set[str],
) -> int:
    """Find related markets within the same series/event that have diverged."""
    trades = 0

    # Group markets by event_ticker
    event_groups: dict[str, list[MarketSnapshot]] = defaultdict(list)
    for snap in snapshots:
        if snap.event_ticker:
            event_groups[snap.event_ticker].append(snap)

    for event_ticker, group in event_groups.items():
        if len(group) < 2:
            continue

        # Sum implied probabilities — should be ~100% for mutually exclusive events
        total_implied = sum(s.last_price for s in group)
        if total_implied == 0:
            continue

        deviation_pct = abs(total_implied - 100) / 100

        if deviation_pct < XEVENT_MIN_DIVERGENCE:
            continue

        if total_implied > 100 + (XEVENT_MIN_DIVERGENCE * 100):
            # Overpriced — buy NO on the most overpriced market
            group.sort(key=lambda s: s.last_price, reverse=True)
            target = group[0]
            if target.ticker in open_tickers:
                continue
            buy_price = 100 - target.last_price
            if buy_price <= 2 or buy_price >= 98:
                continue
            edge = deviation_pct / len(group)
            if edge >= QUANT_MIN_EDGE:
                result = await _place_quant_trade(
                    kalshi, store, risk, target.ticker, "no", buy_price,
                    edge, "quant_cross_event_rv",
                )
                if result:
                    trades += 1
                    open_tickers.add(target.ticker)

        elif total_implied < 100 - (XEVENT_MIN_DIVERGENCE * 100):
            # Underpriced — buy YES on the cheapest market
            group.sort(key=lambda s: s.last_price)
            target = group[0]
            if target.ticker in open_tickers:
                continue
            buy_price = target.last_price
            if buy_price <= 2 or buy_price >= 98:
                continue
            edge = deviation_pct / len(group)
            if edge >= QUANT_MIN_EDGE:
                result = await _place_quant_trade(
                    kalshi, store, risk, target.ticker, "yes", buy_price,
                    edge, "quant_cross_event_rv",
                )
                if result:
                    trades += 1
                    open_tickers.add(target.ticker)

    return trades


# ---------------------------------------------------------------------------
# Strategy Q5: Theta Decay — sell expensive YES on low-prob markets
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.CHAIN)
async def run_theta_decay(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    snapshots: list[MarketSnapshot],
    open_tickers: set[str],
) -> int:
    """Buy NO on markets where YES is cheap but not dirt-cheap — time decay earns."""
    trades = 0
    now = datetime.datetime.now(datetime.timezone.utc)

    for snap in snapshots:
        if snap.ticker in open_tickers:
            continue

        yes_price = snap.last_price
        if yes_price < THETA_MIN_YES_PRICE or yes_price > THETA_MAX_YES_PRICE:
            continue

        if not snap.close_time:
            continue

        try:
            close = datetime.datetime.fromisoformat(snap.close_time.replace("Z", "+00:00"))
            hours_left = (close - now).total_seconds() / 3600
        except ValueError:
            continue

        if hours_left < THETA_MIN_HOURS or hours_left > THETA_MAX_DAYS * 24:
            continue

        # Estimate true NO probability from price + time
        days_left = hours_left / 24
        # Low YES price + shorter time = higher NO certainty
        # Use a simple model: p_no = (1 - yes/100) * decay_factor
        time_decay = 1 - (days_left / (THETA_MAX_DAYS * 1.5))
        time_decay = max(0.5, min(1.0, time_decay))
        p_no_est = (1 - yes_price / 100) * time_decay

        no_price = 100 - yes_price
        edge = p_no_est - (no_price / 100)

        if edge < QUANT_MIN_EDGE:
            continue

        result = await _place_quant_trade(
            kalshi, store, risk, snap.ticker, "no", no_price, edge, "quant_theta_decay",
        )
        if result:
            trades += 1
            open_tickers.add(snap.ticker)

    return trades


# ---------------------------------------------------------------------------
# Strategy Q6: Spread Capture — passive maker orders
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.CHAIN)
async def run_spread_capture(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    snapshots: list[MarketSnapshot],
    open_tickers: set[str],
) -> int:
    """Place maker orders inside the spread to capture bid-ask."""
    trades = 0

    for snap in snapshots:
        if snap.ticker in open_tickers:
            continue

        # Need both bid and ask to compute spread
        if snap.yes_bid <= 0 or snap.yes_ask <= 0:
            continue

        spread = snap.yes_ask - snap.yes_bid
        if spread < SPREAD_MIN_WIDTH:
            continue

        # Place a YES buy 1c above the bid (maker order = free)
        mid = (snap.yes_bid + snap.yes_ask) // 2
        buy_price = snap.yes_bid + 1

        # Edge is half the spread minus expected taker exit fee
        half_spread = spread / 2 / 100
        # Approximate taker fee at mid price
        p_mid = mid / 100
        taker_fee = 0.07 * 4 * p_mid * (1 - p_mid)
        edge = half_spread - taker_fee

        if edge < SPREAD_EDGE:
            continue

        if buy_price <= 2 or buy_price >= 98:
            continue

        result = await _place_quant_trade(
            kalshi, store, risk, snap.ticker, "yes", buy_price, edge, "quant_spread_capture",
        )
        if result:
            trades += 1
            open_tickers.add(snap.ticker)

    return trades


# ---------------------------------------------------------------------------
# Shared: place a quant trade with smaller sizing
# ---------------------------------------------------------------------------


async def _place_quant_trade(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    ticker: str,
    side: str,
    price_cents: int,
    edge: float,
    strategy: str,
) -> bool:
    """Size and place a quant trade. Returns True if order placed."""
    if edge < QUANT_MIN_EDGE:
        return False

    bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
    size_mult = await risk.size_adjustment()

    # Quant uses smaller position sizing — 2% of bankroll
    max_cost = int(bankroll * QUANT_POSITION_PCT * size_mult)
    count = max_cost // price_cents
    if count <= 0:
        return False

    trade = ProposedTrade(
        ticker=ticker,
        side=side,
        action="buy",
        price_cents=price_cents,
        count=count,
        edge=edge,
    )

    # Use quant-specific edge check (bypass the 8% risk manager check)
    if not await _can_trade_quant(risk, trade):
        return False

    if settings.dry_run:
        logger.info(
            "dry_run_quant",
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            edge=f"{edge:.1%}",
            strategy=strategy,
        )
        return True  # Count as "trade" for logging in dry run

    order = await place_limit_order(kalshi, ticker, side, count, price_cents)
    if order:
        await store.record_trade(
            ticker=ticker,
            side=side,
            action="buy",
            price_cents=price_cents,
            count=count,
            order_id=order.get("order_id", ""),
            strategy=strategy,
        )
        logger.info(
            "quant_trade",
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            edge=f"{edge:.1%}",
            strategy=strategy,
        )
        return True
    return False


async def _can_trade_quant(risk: RiskManager, trade: ProposedTrade) -> bool:
    """Risk check for quant trades — uses lower edge threshold than agent track."""
    bankroll = await risk.store.get_bankroll_cents()
    if bankroll is None:
        bankroll = risk.initial_bankroll_cents

    # Quant-specific: 3% edge minimum (not 8%)
    if trade.edge < QUANT_MIN_EDGE:
        return False

    # Still respect position limits
    open_positions = await risk.store.get_open_positions()
    if len(open_positions) >= settings.max_concurrent_positions:
        return False

    # Per-trade size limit (use quant's 2% instead of 5%)
    trade_cost = trade.count * trade.price_cents
    max_per_trade = int(bankroll * QUANT_POSITION_PCT)
    if trade_cost > max_per_trade:
        return False

    # Total exposure limit (still 50%)
    total_exposure = sum(p["count"] * p["price_cents"] for p in open_positions)
    if total_exposure + trade_cost > int(bankroll * settings.max_exposure_pct):
        return False

    # Daily loss limit
    daily_pnl = await risk.store.get_daily_pnl_cents()
    if daily_pnl < 0 and abs(daily_pnl) >= int(risk.initial_bankroll_cents * 0.10):
        return False

    # Drawdown hard stop
    drawdown = risk._drawdown_pct(bankroll)
    if drawdown >= 0.40:
        return False

    return True


# ---------------------------------------------------------------------------
# Main quant engine entry point
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.AGENT)
async def run_quant_cycle(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
) -> int:
    """Run all quant strategies. Returns total trade count."""
    logger.info("quant_cycle_start")
    total_trades = 0

    # Fetch market data (single API call, shared across all strategies)
    try:
        data = await kalshi.get_markets(limit=200)
        raw_markets = data.get("markets", [])
    except Exception:
        logger.warning("quant_market_fetch_failed")
        return 0

    # Parse into snapshots and record VWAP observations
    snapshots: list[MarketSnapshot] = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for m in raw_markets:
        ticker = m.get("ticker", "")

        # Skip sports
        skip_prefixes = (
            "KXMVE", "KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXSOCCER",
            "KXMMA", "KXUFC", "KXBOXING", "KXTENNIS", "KXGOLF",
            "KXCOLLEGE", "KXNCAA", "KXWWE", "KXESPORTS", "KXMARCH",
            "KXNASCAR", "KXF1RACE", "KXPGA",
        )
        if any(ticker.startswith(p) for p in skip_prefixes):
            continue

        last_price = _dollars_to_cents(m.get("last_price_dollars") or m.get("last_price"))
        if last_price <= 0:
            continue

        volume_24h = _parse_volume(m.get("volume_24h_fp") or m.get("volume_24h"))
        yes_bid = _dollars_to_cents(m.get("yes_bid_dollars") or m.get("yes_bid"))
        yes_ask = _dollars_to_cents(m.get("yes_ask_dollars") or m.get("yes_ask"))
        no_bid = _dollars_to_cents(m.get("no_bid_dollars") or m.get("no_bid"))
        no_ask = _dollars_to_cents(m.get("no_ask_dollars") or m.get("no_ask"))

        snap = MarketSnapshot(
            ticker=ticker,
            title=m.get("title", ""),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_price=last_price,
            volume_24h=volume_24h,
            close_time=m.get("close_time", ""),
            event_ticker=m.get("event_ticker", ""),
            series_ticker=m.get("series_ticker", ""),
        )
        snapshots.append(snap)

        # Feed VWAP tracker
        record_observation(ticker, last_price, volume_24h)

    logger.info("quant_markets_loaded", count=len(snapshots))

    # Get open tickers to avoid duplicates
    open_trades = await store.get_open_positions()
    open_tickers = {t["ticker"] for t in open_trades}

    # Run each strategy, respecting per-cycle cap
    strategies = [
        ("mean_reversion", run_mean_reversion),
        ("ob_imbalance", run_orderbook_imbalance),
        ("wide_convergence", run_wide_convergence),
        ("cross_event_rv", run_cross_event_relative_value),
        ("theta_decay", run_theta_decay),
        ("spread_capture", run_spread_capture),
    ]

    for name, strategy_fn in strategies:
        if total_trades >= QUANT_MAX_TRADES_PER_CYCLE:
            logger.info("quant_cycle_cap_reached", trades=total_trades)
            break

        try:
            count = await strategy_fn(kalshi, store, risk, snapshots, open_tickers)
            total_trades += count
            if count > 0:
                logger.info(f"quant_{name}_trades", count=count)
        except Exception:
            logger.exception(f"quant_{name}_failed")

    logger.info("quant_cycle_complete", total_trades=total_trades, markets_scanned=len(snapshots))
    return total_trades
