"""Async entry point — dual-loop trading architecture.

Fast loop (2 min): position sync, exits, repricing, convergence, arbitrage
Slow loop (10 min): market discovery, news, Claude forecasting, batch API
"""

import asyncio
import atexit
import datetime
import signal

import anthropic
import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.alerts import AlertManager
from kalshiedge.batch_forecaster import BatchForecaster
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
from kalshiedge.websocket import KalshiWebSocket

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Fast loop — runs every 2 minutes, zero Claude calls
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.AGENT)
async def run_fast_cycle(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
    ws: KalshiWebSocket | None = None,
) -> None:
    """Fast cycle: position management, repricing, convergence, arbitrage."""
    logger.info("fast_cycle_start")

    # Sync positions and bankroll from Kalshi
    try:
        positions = await sync_positions(kalshi, store)
        await check_settlements(kalshi, store)
        if positions:
            exits = await evaluate_exits(kalshi, store, positions)
            if exits:
                await execute_exits(kalshi, store, exits)
    except Exception:
        logger.exception("position_management_failed")

    # Cancel stale resting orders (older than configured threshold)
    try:
        await _cancel_stale_resting_orders(kalshi, store)
    except Exception:
        logger.exception("stale_order_cancel_failed")

    # Strategy 2: Near-Expiry Convergence (no Claude calls)
    try:
        await run_near_expiry_convergence(kalshi, store, risk)
    except Exception:
        logger.exception("convergence_strategy_failed")

    # Strategy 4: Intra-Event Arbitrage (no Claude calls)
    try:
        await run_intra_event_arbitrage(kalshi, store, risk)
    except Exception:
        logger.exception("arbitrage_strategy_failed")

    logger.info("fast_cycle_complete")


# ---------------------------------------------------------------------------
# Slow loop — runs every 10 minutes, uses Claude for forecasting
# ---------------------------------------------------------------------------


@observe(span_type=SpanType.AGENT)
async def run_slow_cycle(
    kalshi: KalshiClient,
    claude: anthropic.Anthropic,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
    batch: BatchForecaster | None = None,
    ws: KalshiWebSocket | None = None,
) -> None:
    """Slow cycle: discovery, news, forecasting, batch submit/collect."""
    logger.info("slow_cycle_start")

    # Collect batch results from previous slow cycle
    if batch and batch.has_pending_batch:
        try:
            batch_results = batch.collect_results()
            if batch_results:
                await _process_batch_results(
                    kalshi, store, risk, alerts, batch_results
                )
        except Exception:
            logger.exception("batch_collection_failed")

    # Discover markets
    markets = await discover_markets(kalshi)
    if not markets:
        logger.info("no_markets_found")
        return

    # Prioritize event-driven markets (Strategy 3)
    event_markets = await find_event_driven_markets(markets)

    # Score markets: prefer high volume + shorter expiry (faster capital turnover)
    now = datetime.datetime.now(datetime.timezone.utc)
    for m in markets:
        days_to_expiry = 365.0  # default for missing close_time
        if m.close_time:
            try:
                close = datetime.datetime.fromisoformat(
                    m.close_time.replace("Z", "+00:00")
                )
                days_to_expiry = max((close - now).total_seconds() / 86400, 0.1)
            except ValueError:
                pass
        # Score: volume / sqrt(days) — rewards high volume + short duration
        m._score = m.volume / (days_to_expiry ** 0.5)  # type: ignore[attr-defined]

    # Combine: event-driven first, then scored, deduplicated
    markets.sort(key=lambda m: getattr(m, "_score", 0), reverse=True)
    seen_tickers: set[str] = set()
    prioritized: list[Market] = []
    for m in event_markets:
        if m.ticker not in seen_tickers:
            seen_tickers.add(m.ticker)
            prioritized.append(m)
    for m in markets:
        if m.ticker not in seen_tickers:
            seen_tickers.add(m.ticker)
            prioritized.append(m)

    cap = settings.max_forecasts_per_cycle
    selected = prioritized[:cap]

    # Skip tickers we already have open trades on
    open_trades = await store.get_open_positions()
    open_tickers = {t["ticker"] for t in open_trades}
    selected = [m for m in selected if m.ticker not in open_tickers]
    logger.info("markets_selected", count=len(selected), cap=cap)

    if not selected:
        logger.info("slow_cycle_complete", markets_processed=0)
        return

    # Subscribe WebSocket to selected tickers
    if ws:
        await ws.subscribe_tickers([m.ticker for m in selected])

    # Gather news for all selected markets
    markets_with_news = []
    for market in selected:
        news = await gather_news(market.title)
        markets_with_news.append((market, news))

    # Submit batch forecast for next cycle (50% cheaper)
    if batch and not batch.has_pending_batch and markets_with_news:
        try:
            batch.submit_batch([
                {
                    "ticker": m.ticker,
                    "title": m.title,
                    "price_cents": m.last_price,
                    "close_time": m.close_time,
                    "news_items": news,
                }
                for m, news in markets_with_news
            ])
        except Exception:
            logger.exception("batch_submit_failed")

    # Run synchronous forecasts for immediate trading
    event_tickers = {m.ticker for m in event_markets}
    for market, _news in markets_with_news:
        strategy = "event_driven" if market.ticker in event_tickers else "calibration_edge"
        try:
            await _process_market(
                kalshi, claude, store, risk, alerts, market, strategy=strategy
            )
        except Exception:
            logger.exception("market_processing_failed", ticker=market.ticker)
            await alerts.notify_error(
                f"Failed processing {market.ticker}", context="slow_cycle"
            )

    logger.info("slow_cycle_complete", markets_processed=len(selected))


# ---------------------------------------------------------------------------
# Market processing (shared by sync and batch paths)
# ---------------------------------------------------------------------------


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

    await _execute_signal(
        kalshi, store, risk, alerts, market.ticker, side,
        result.probability, market, effective_edge, strategy,
    )


async def _execute_signal(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
    ticker: str,
    side: str,
    model_prob: float,
    market: Market,
    effective_edge: float,
    strategy: str,
) -> None:
    """Size, risk-check, and place an order."""
    bankroll = await store.get_bankroll_cents() or settings.bankroll_cents
    size_mult = await risk.size_adjustment()

    if side == "yes":
        price = market.yes_ask if market.yes_ask > 0 else market.last_price
        budget = int(quarter_kelly(model_prob, price, bankroll) * size_mult)
    else:
        price = market.no_ask if market.no_ask > 0 else (100 - market.last_price)
        p_no = 1 - model_prob
        budget = int(quarter_kelly(p_no, price, bankroll) * size_mult)

    if budget <= 0 or price <= 0:
        return

    count = budget // price
    if count <= 0:
        return

    trade = ProposedTrade(
        ticker=ticker, side=side, action="buy",
        price_cents=price, count=count, edge=effective_edge,
    )
    if not await risk.can_trade(trade):
        return

    if settings.dry_run:
        logger.info(
            "dry_run_signal",
            ticker=ticker, side=side, count=count, price_cents=price,
            edge=f"{effective_edge:.1%}", model_prob=f"{model_prob:.1%}",
            strategy=strategy,
        )
        return

    order = await place_limit_order(kalshi, ticker, side, count, price)
    if order is None:
        return

    order_id = order.get("order_id", "")
    await store.record_trade(
        ticker=ticker, side=side, action="buy",
        price_cents=price, count=count,
        order_id=order_id, strategy=strategy,
    )
    await alerts.notify_trade(
        ticker=ticker, side=side, action="buy",
        count=count, price_cents=price,
        edge=effective_edge, strategy=strategy,
    )
    asyncio.create_task(_track_fill(kalshi, store, order_id, ticker))


async def _process_batch_results(
    kalshi: KalshiClient,
    store: PortfolioStore,
    risk: RiskManager,
    alerts: AlertManager,
    results: dict[str, dict],
) -> None:
    """Process forecast results from a completed batch and execute trades."""
    from kalshiedge.discovery import Market as _Market
    from kalshiedge.discovery import _dollars_to_cents

    for ticker, forecast in results.items():
        prob = forecast["probability"]
        price_cents = forecast["price_cents"]
        title = forecast["title"]

        side, edge = compute_edge(prob, price_cents)
        effective_edge = net_edge(prob, price_cents)

        await store.record_forecast(
            ticker=ticker, title=title,
            market_price_cents=price_cents,
            model_probability=prob, edge=edge,
            confidence_low=forecast.get("confidence_low"),
            confidence_high=forecast.get("confidence_high"),
            reasoning=forecast.get("reasoning"),
            strategy="batch_calibration_edge",
        )

        if side == "none" or effective_edge < settings.min_edge_threshold:
            continue

        # Get fresh price from Kalshi
        try:
            market_data = await kalshi.get_market(ticker)
            m = market_data.get("market", market_data)
            yes_ask = _dollars_to_cents(m.get("yes_ask_dollars"))
            no_ask = _dollars_to_cents(m.get("no_ask_dollars"))
            last = _dollars_to_cents(
                m.get("last_price_dollars") or m.get("last_price")
            )
        except Exception:
            yes_ask = price_cents
            no_ask = 100 - price_cents
            last = price_cents

        # Build a minimal Market for _execute_signal
        market_obj = _Market(
            ticker=ticker, title=title,
            yes_bid=0, yes_ask=yes_ask or last,
            no_bid=0, no_ask=no_ask or (100 - last),
            last_price=last, volume=0, volume_24h=0,
            open_interest=0, close_time="",
        )
        await _execute_signal(
            kalshi, store, risk, alerts, ticker, side,
            prob, market_obj, effective_edge, "batch_calibration_edge",
        )

    logger.info("batch_results_processed", count=len(results))


async def _track_fill(
    kalshi: KalshiClient, store: PortfolioStore, order_id: str, ticker: str
) -> None:
    status = await monitor_fill(kalshi, order_id, ticker)
    await store.update_trade_status(order_id, status)


async def _cancel_stale_resting_orders(
    kalshi: KalshiClient, store: PortfolioStore
) -> None:
    """Cancel resting orders older than stale_order_minutes."""
    try:
        orders_data = await kalshi.get_orders(status="resting")
        orders = orders_data.get("orders", [])
        now = datetime.datetime.now(datetime.timezone.utc)
        threshold = datetime.timedelta(minutes=settings.stale_order_minutes)
        cancelled = 0

        for order in orders:
            created = order.get("created_time", "")
            if created:
                try:
                    order_time = datetime.datetime.fromisoformat(
                        created.replace("Z", "+00:00")
                    )
                    if (now - order_time) < threshold:
                        continue  # Not stale yet — let it fill
                except ValueError:
                    pass

            order_id = order.get("order_id")
            if order_id:
                await kalshi.cancel_order(order_id)
                await store.update_trade_status(order_id, "canceled")
                cancelled += 1

        if cancelled:
            logger.info("stale_orders_cancelled", count=cancelled)
    except Exception:
        logger.warning("stale_order_cancel_failed")


# ---------------------------------------------------------------------------
# Entry point — dual loop scheduler
# ---------------------------------------------------------------------------


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
        fast_interval=settings.fast_cycle_seconds,
        slow_interval=settings.cycle_interval_seconds,
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

    # Batch forecaster for 50% cost reduction
    batch_claude = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    batch = BatchForecaster(batch_claude)
    logger.info("batch_forecaster_initialized")

    # WebSocket for real-time data
    ws: KalshiWebSocket | None = KalshiWebSocket()
    try:
        await ws.start()
    except Exception:
        logger.warning("websocket_start_failed")
        ws = None

    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("shutdown_requested")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Track when slow cycle last ran
    last_slow_cycle = 0.0

    try:
        while not shutdown_event.is_set():
            now = asyncio.get_event_loop().time()

            # Always run fast cycle
            try:
                await run_fast_cycle(kalshi, store, risk, alerts, ws=ws)
            except Exception:
                logger.exception("fast_cycle_failed")

            # Run slow cycle if enough time has passed
            if (now - last_slow_cycle) >= settings.cycle_interval_seconds:
                try:
                    await run_slow_cycle(
                        kalshi, claude, store, risk, alerts,
                        batch=batch, ws=ws,
                    )
                except Exception:
                    logger.exception("slow_cycle_failed")
                    await alerts.notify_error("Slow cycle failed", context="main_loop")
                last_slow_cycle = now

            # Wait for fast cycle interval
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(), timeout=settings.fast_cycle_seconds
                )
            except asyncio.TimeoutError:
                pass
    finally:
        if ws:
            await ws.stop()
        await kalshi.close()
        await store.close()
        if ts:
            ts.shutdown()
        logger.info("kalshiedge_shutdown")


if __name__ == "__main__":
    asyncio.run(main())
