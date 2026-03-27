"""Async entry point — main trading loop."""

import asyncio
import atexit
import signal

import anthropic
import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.alerts import AlertManager
from kalshiedge.config import settings
from kalshiedge.discovery import Market, check_orderbook_depth, discover_markets
from kalshiedge.edge import compute_edge, net_edge, quarter_kelly
from kalshiedge.forecaster import forecast_market
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore
from kalshiedge.positions import (
    check_settlements,
    evaluate_exits,
    execute_exits,
    sync_positions,
)
from kalshiedge.research import gather_news
from kalshiedge.risk import ProposedTrade, RiskManager
from kalshiedge.strategies import (
    find_event_driven_markets,
    run_intra_event_arbitrage,
    run_near_expiry_convergence,
)
from kalshiedge.trader import monitor_fill, place_limit_order

logger = structlog.get_logger()


@observe(span_type=SpanType.AGENT)
async def run_cycle(
    kalshi: KalshiClient,
    claude: anthropic.Anthropic,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
) -> None:
    """Single discovery -> forecast -> trade cycle."""
    logger.info("cycle_start")

    # Cancel stale resting orders from previous cycles so we can re-price
    try:
        await _cancel_stale_resting_orders(kalshi, store)
    except Exception:
        logger.exception("stale_order_cancel_failed")

    # Phase 4: Sync positions, check settlements, evaluate exits
    try:
        positions = await sync_positions(kalshi, store)
        await check_settlements(kalshi, store)
        if positions:
            exits = await evaluate_exits(kalshi, store, positions)
            if exits:
                await execute_exits(kalshi, store, exits)
    except Exception:
        logger.exception("position_management_failed")

    # Strategy 2: Near-Expiry Convergence (no Claude calls needed)
    try:
        await run_near_expiry_convergence(kalshi, store, risk)
    except Exception:
        logger.exception("convergence_strategy_failed")

    # Strategy 4: Intra-Event Arbitrage (no Claude calls needed)
    try:
        await run_intra_event_arbitrage(kalshi, store, risk)
    except Exception:
        logger.exception("arbitrage_strategy_failed")

    # Discover markets for Strategy 1 (Calibration Edge) + Strategy 3 (Event-Driven)
    markets = await discover_markets(kalshi)
    if not markets:
        logger.info("no_markets_found")
        return

    # Strategy 3: Prioritize event-driven markets
    event_markets = await find_event_driven_markets(markets)

    # Combine: event-driven first, then top by volume, deduplicated
    seen_tickers: set[str] = set()
    prioritized: list[Market] = []
    for m in event_markets:
        if m.ticker not in seen_tickers:
            seen_tickers.add(m.ticker)
            prioritized.append(m)
    markets.sort(key=lambda m: m.volume, reverse=True)
    for m in markets:
        if m.ticker not in seen_tickers:
            seen_tickers.add(m.ticker)
            prioritized.append(m)

    cap = settings.max_forecasts_per_cycle
    selected = prioritized[:cap]
    logger.info("markets_selected", count=len(selected), cap=cap)

    # Skip tickers we already have open trades on
    open_trades = await store.get_open_positions()
    open_tickers = {t["ticker"] for t in open_trades}

    event_tickers = {m.ticker for m in event_markets}
    for market in selected:
        if market.ticker in open_tickers:
            logger.debug("skipping_existing_position", ticker=market.ticker)
            continue
        strategy = "event_driven" if market.ticker in event_tickers else "calibration_edge"
        try:
            await _process_market(
                kalshi, claude, store, risk, alerts, market, strategy=strategy
            )
        except Exception:
            logger.exception("market_processing_failed", ticker=market.ticker)
            await alerts.notify_error(
                f"Failed processing {market.ticker}", context="run_cycle"
            )

    logger.info("cycle_complete", markets_processed=len(selected))


async def _process_market(
    kalshi: KalshiClient,
    claude: anthropic.Anthropic,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
    market: Market,
    strategy: str = "calibration_edge",
) -> None:
    has_depth = await check_orderbook_depth(kalshi, market.ticker)
    if not has_depth:
        logger.debug("skipping_thin_orderbook", ticker=market.ticker)
        return

    news = await gather_news(market.title)

    result = await forecast_market(
        client=claude,
        title=market.title,
        price_cents=market.last_price,
        close_time=market.close_time,
        news_items=news,
    )
    if result is None:
        return

    side, edge = compute_edge(result.probability, market.last_price)
    effective_edge = net_edge(result.probability, market.last_price)

    await store.record_forecast(
        ticker=market.ticker,
        title=market.title,
        market_price_cents=market.last_price,
        model_probability=result.probability,
        edge=edge,
        confidence_low=result.confidence_low,
        confidence_high=result.confidence_high,
        reasoning=result.reasoning,
        strategy=strategy,
    )

    if side == "none" or effective_edge < settings.min_edge_threshold:
        logger.info("no_edge", ticker=market.ticker, edge=edge, net_edge=effective_edge)
        return

    # Calculate position size
    bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
    size_mult = await risk.size_adjustment()

    # Price at the current ask to maximize fill probability
    # For YES: buy at yes_ask. For NO: buy at no_ask (= 100 - yes_bid).
    if side == "yes":
        price = market.yes_ask if market.yes_ask > 0 else market.last_price
        budget = int(quarter_kelly(result.probability, price, bankroll) * size_mult)
    else:
        price = market.no_ask if market.no_ask > 0 else (100 - market.last_price)
        p_no = 1 - result.probability
        budget = int(quarter_kelly(p_no, price, bankroll) * size_mult)

    if budget <= 0 or price <= 0:
        return

    count = budget // price
    if count <= 0:
        return

    # Risk check
    trade = ProposedTrade(
        ticker=market.ticker,
        side=side,
        action="buy",
        price_cents=price,
        count=count,
        edge=effective_edge,
    )
    if not await risk.can_trade(trade):
        return

    # Dry run — log the signal but don't execute
    if settings.dry_run:
        logger.info(
            "dry_run_signal",
            ticker=market.ticker,
            side=side,
            count=count,
            price_cents=price,
            edge=f"{effective_edge:.1%}",
            model_prob=f"{result.probability:.1%}",
            market_price=market.last_price,
            reasoning=result.reasoning[:120] if result.reasoning else "",
        )
        return

    # Execute
    order = await place_limit_order(kalshi, market.ticker, side, count, price)
    if order is None:
        return

    order_id = order.get("order_id", "")
    await store.record_trade(
        ticker=market.ticker,
        side=side,
        action="buy",
        price_cents=price,
        count=count,
        order_id=order_id,
        strategy=strategy,
    )
    await alerts.notify_trade(
        ticker=market.ticker,
        side=side,
        action="buy",
        count=count,
        price_cents=price,
        edge=effective_edge,
    )

    # Non-blocking fill monitor
    asyncio.create_task(_track_fill(kalshi, store, order_id, market.ticker))


async def _track_fill(
    kalshi: KalshiClient, store: PortfolioStore, order_id: str, ticker: str
) -> None:
    status = await monitor_fill(kalshi, order_id, ticker)
    await store.update_trade_status(order_id, status)


async def _cancel_stale_resting_orders(
    kalshi: KalshiClient, store: PortfolioStore
) -> None:
    """Cancel all resting orders so we can re-evaluate and re-price."""
    try:
        orders_data = await kalshi.get_orders(status="resting")
        orders = orders_data.get("orders", [])
        for order in orders:
            order_id = order.get("order_id")
            if order_id:
                await kalshi.cancel_order(order_id)
                await store.update_trade_status(order_id, "canceled")
                logger.info("stale_order_cancelled", order_id=order_id)
        if orders:
            logger.info("stale_orders_cancelled", count=len(orders))
    except Exception:
        logger.warning("stale_order_cancel_failed")


async def main() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    logger.info(
        "kalshiedge_starting",
        env=settings.kalshi_env,
        dry_run=settings.dry_run,
        bankroll_usd=settings.bankroll_usd,
        cycle_interval=settings.cycle_interval_seconds,
    )
    if settings.dry_run:
        logger.info("dry_run_mode — forecasts only, no orders will be placed")

    # Initialize 2signal if configured
    ts = None
    try:
        if settings.twosignal_api_key:
            from twosignal import TwoSignal
            from twosignal.wrappers.anthropic import wrap_anthropic

            ts = TwoSignal(
                api_key=settings.twosignal_api_key,
                base_url=settings.twosignal_base_url or None,
                deployment_id="kalshiedge-v0.1.0",
            )
            claude = wrap_anthropic(
                anthropic.Anthropic(api_key=settings.anthropic_api_key)
            )
            atexit.register(ts.shutdown)
            logger.info("twosignal_initialized")
        else:
            claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    except ImportError:
        logger.warning("twosignal_not_installed")
        claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    kalshi = KalshiClient()
    store = PortfolioStore()
    await store.initialize()

    # Sync real bankroll from Kalshi on startup
    try:
        bal = await kalshi.get_balance()
        real_bankroll = bal.get("balance", 0)
        await store.set_bankroll_cents(real_bankroll)
        logger.info("bankroll_from_kalshi", cents=real_bankroll)
    except Exception:
        await store.set_bankroll_cents(settings.bankroll_cents)
        logger.warning("using_config_bankroll")

    risk = RiskManager(store)
    alerts = AlertManager()

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_requested")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        while not shutdown_event.is_set():
            try:
                await run_cycle(kalshi, claude, store, risk, alerts)
            except Exception:
                logger.exception("cycle_failed")
                await alerts.notify_error("Cycle failed", context="main_loop")

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=settings.cycle_interval_seconds
                )
            except asyncio.TimeoutError:
                pass  # Normal — time to run next cycle
    finally:
        await kalshi.close()
        await store.close()
        if ts:
            ts.shutdown()
        logger.info("kalshiedge_shutdown")


if __name__ == "__main__":
    asyncio.run(main())
