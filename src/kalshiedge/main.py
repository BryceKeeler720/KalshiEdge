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
from kalshiedge.edge import compute_edge, kelly_no, net_edge, quarter_kelly
from kalshiedge.forecaster import forecast_market
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore
from kalshiedge.research import gather_news
from kalshiedge.risk import ProposedTrade, RiskManager
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

    markets = await discover_markets(kalshi)
    if not markets:
        logger.info("no_markets_found")
        return

    # Sort by volume descending, cap to control API costs
    markets.sort(key=lambda m: m.volume, reverse=True)
    cap = settings.max_forecasts_per_cycle
    markets = markets[:cap]
    logger.info("markets_selected", count=len(markets), cap=cap)

    for market in markets:
        try:
            await _process_market(kalshi, claude, store, risk, alerts, market)
        except Exception:
            logger.exception("market_processing_failed", ticker=market.ticker)
            await alerts.notify_error(
                f"Failed processing {market.ticker}", context="run_cycle"
            )

    logger.info("cycle_complete", markets_processed=len(markets))


async def _process_market(
    kalshi: KalshiClient,
    claude: anthropic.Anthropic,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
    market: Market,
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
    )

    if side == "none" or effective_edge < settings.min_edge_threshold:
        logger.info("no_edge", ticker=market.ticker, edge=edge, net_edge=effective_edge)
        return

    # Calculate position size
    bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
    size_mult = await risk.size_adjustment()

    if side == "yes":
        budget = int(quarter_kelly(result.probability, market.last_price, bankroll) * size_mult)
        price = market.last_price
    else:
        budget = int(kelly_no(result.probability, market.last_price, bankroll) * size_mult)
        price = 100 - market.last_price

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
    await store.set_bankroll_cents(settings.bankroll_cents)

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
