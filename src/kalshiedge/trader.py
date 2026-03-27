"""Order execution via Kalshi API — limit orders only."""

import asyncio

import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.kalshi_client import KalshiClient

logger = structlog.get_logger()

FILL_CHECK_INTERVAL = 10  # seconds
FILL_CHECK_MAX_ATTEMPTS = 30


@observe(span_type=SpanType.TOOL)
async def place_limit_order(
    client: KalshiClient,
    ticker: str,
    side: str,
    count: int,
    price_cents: int,
) -> dict | None:
    """Place a limit buy order. Returns order response or None on failure."""
    try:
        result = await client.create_order(
            ticker=ticker,
            action="buy",
            side=side,
            type="limit",
            count=count,
            yes_price=price_cents if side == "yes" else (100 - price_cents),
        )
        order = result.get("order", result)
        logger.info(
            "order_placed",
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            order_id=order.get("order_id"),
        )
        return order
    except Exception:
        logger.exception("order_placement_failed", ticker=ticker, side=side)
        return None


async def monitor_fill(
    client: KalshiClient,
    order_id: str,
    ticker: str,
) -> str:
    """Poll order status until filled, cancelled, or timeout. Returns final status."""
    for _ in range(FILL_CHECK_MAX_ATTEMPTS):
        try:
            orders_data = await client.get_orders(ticker=ticker, status=None)
            orders = orders_data.get("orders", [])
            for order in orders:
                if order.get("order_id") == order_id:
                    status = order.get("status", "")
                    if status in ("executed", "canceled"):
                        logger.info("order_final_status", order_id=order_id, status=status)
                        return status
        except Exception:
            logger.warning("fill_check_failed", order_id=order_id)

        await asyncio.sleep(FILL_CHECK_INTERVAL)

    logger.warning("fill_check_timeout", order_id=order_id)
    return "timeout"


async def cancel_stale_orders(client: KalshiClient, ticker: str) -> int:
    """Cancel all resting orders for a ticker. Returns count cancelled."""
    cancelled = 0
    try:
        orders_data = await client.get_orders(ticker=ticker, status="resting")
        for order in orders_data.get("orders", []):
            order_id = order.get("order_id")
            if order_id:
                await client.cancel_order(order_id)
                cancelled += 1
                logger.info("order_cancelled", order_id=order_id, ticker=ticker)
    except Exception:
        logger.exception("cancel_orders_failed", ticker=ticker)
    return cancelled
