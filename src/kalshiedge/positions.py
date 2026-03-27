"""Position management — sync with Kalshi, evaluate exits, track settlements."""

import structlog

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.discovery import _dollars_to_cents
from kalshiedge.edge import compute_edge, net_edge
from kalshiedge.kalshi_client import KalshiClient
from kalshiedge.portfolio import PortfolioStore

logger = structlog.get_logger()

# Exit when edge has flipped or shrunk below this threshold
EXIT_EDGE_THRESHOLD = 0.03


STOP_LOSS_PCT = settings.stop_loss_pct
TAKE_PROFIT_PCT = settings.take_profit_pct


@observe(span_type=SpanType.RETRIEVAL)
async def sync_positions(kalshi: KalshiClient, store: PortfolioStore) -> list[dict]:
    """Fetch live positions from Kalshi and sync bankroll."""
    try:
        bal = await kalshi.get_balance()
        balance_cents = bal.get("balance", 0)
        await store.set_bankroll_cents(balance_cents)
        logger.info("bankroll_synced", balance_cents=balance_cents)
    except Exception:
        logger.warning("bankroll_sync_failed")

    try:
        pos_data = await kalshi.get_positions()
        positions = pos_data.get("market_positions", pos_data.get("positions", []))
        live = []
        for p in positions:
            yes_qty = _safe_int(p.get("yes_contracts", 0))
            no_qty = _safe_int(p.get("no_contracts", 0))
            if yes_qty > 0 or no_qty > 0:
                live.append({
                    "ticker": p.get("ticker", ""),
                    "yes_contracts": yes_qty,
                    "no_contracts": no_qty,
                    "market_exposure": p.get("market_exposure", 0),
                })
        logger.info("positions_synced", count=len(live))
        return live
    except Exception:
        logger.warning("position_sync_failed")
        return []


@observe(span_type=SpanType.CHAIN)
async def evaluate_exits(
    kalshi: KalshiClient,
    store: PortfolioStore,
    positions: list[dict],
    forecaster_fn=None,
    claude=None,
) -> list[dict]:
    """Re-evaluate each open position. Return list of exit signals."""
    exits = []

    for pos in positions:
        ticker = pos["ticker"]
        yes_qty = pos["yes_contracts"]
        no_qty = pos["no_contracts"]

        try:
            market_data = await kalshi.get_market(ticker)
            market = market_data.get("market", market_data)

            # Get current market price
            last_price = _dollars_to_cents(
                market.get("last_price_dollars") or market.get("last_price")
            )
            if last_price <= 0:
                continue

            status = market.get("status", "")
            result_val = market.get("result", "")

            # Check if market has settled
            if result_val in ("yes", "no") or status in ("settled", "finalized"):
                outcome = 1 if result_val == "yes" else 0
                pnl = _calc_settlement_pnl(pos, outcome)
                exits.append({
                    "ticker": ticker,
                    "action": "settled",
                    "outcome": result_val,
                    "pnl_cents": pnl,
                })
                await store.mark_resolved(ticker, outcome)
                logger.info(
                    "market_settled",
                    ticker=ticker,
                    outcome=result_val,
                    pnl_cents=pnl,
                )
                continue

            # Re-evaluate edge with current price
            # We need the model's probability — check latest forecast
            latest = await store.execute_fetchall(
                "SELECT model_probability FROM forecasts "
                "WHERE ticker = ? ORDER BY created_at DESC LIMIT 1",
                (ticker,),
            )
            if not latest:
                continue

            model_prob = latest[0][0]
            side, edge = compute_edge(model_prob, last_price)
            eff_edge = net_edge(model_prob, last_price)

            # Determine if we should exit
            holding_side = "yes" if yes_qty > 0 else "no"
            should_exit = False
            exit_reason = ""

            # Check stop-loss / take-profit
            entry = await store.execute_fetchall(
                "SELECT price_cents FROM trades "
                "WHERE ticker = ? AND action = 'buy' AND side = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (ticker, holding_side),
            )
            if entry:
                entry_price = entry[0][0]
                if holding_side == "yes":
                    current_value = last_price
                else:
                    current_value = 100 - last_price
                if entry_price > 0:
                    unrealized_pct = (current_value - entry_price) / entry_price
                    if unrealized_pct <= STOP_LOSS_PCT:
                        should_exit = True
                        exit_reason = f"stop_loss_{unrealized_pct:+.0%}"
                    elif unrealized_pct >= TAKE_PROFIT_PCT:
                        should_exit = True
                        exit_reason = f"take_profit_{unrealized_pct:+.0%}"

            # Check edge-based exits
            if not should_exit:
                if holding_side == "yes" and side != "yes":
                    should_exit = True
                    exit_reason = "edge_flipped_to_no"
                elif holding_side == "no" and side != "no" and side != "none":
                    should_exit = True
                    exit_reason = "edge_flipped_to_yes"
                elif abs(eff_edge) < EXIT_EDGE_THRESHOLD:
                    should_exit = True
                    exit_reason = "edge_evaporated"

            if should_exit:
                exits.append({
                    "ticker": ticker,
                    "action": "exit",
                    "reason": exit_reason,
                    "side": holding_side,
                    "quantity": yes_qty if holding_side == "yes" else no_qty,
                    "current_price": last_price,
                    "model_prob": model_prob,
                    "edge": eff_edge,
                })
                logger.info(
                    "exit_signal",
                    ticker=ticker,
                    reason=exit_reason,
                    side=holding_side,
                    edge=f"{eff_edge:.1%}",
                )

        except Exception:
            logger.warning("exit_eval_failed", ticker=ticker)

    return exits


@observe(span_type=SpanType.TOOL)
async def execute_exits(
    kalshi: KalshiClient,
    store: PortfolioStore,
    exits: list[dict],
) -> None:
    """Execute exit orders for positions that should be closed."""
    for exit_signal in exits:
        if exit_signal["action"] == "settled":
            logger.info(
                "settlement_recorded",
                ticker=exit_signal["ticker"],
                outcome=exit_signal["outcome"],
                pnl_cents=exit_signal["pnl_cents"],
            )
            continue

        if exit_signal["action"] != "exit":
            continue

        ticker = exit_signal["ticker"]
        side = exit_signal["side"]
        qty = exit_signal["quantity"]
        price = exit_signal["current_price"]

        if settings.dry_run:
            logger.info(
                "dry_run_exit",
                ticker=ticker,
                side=side,
                quantity=qty,
                reason=exit_signal["reason"],
            )
            continue

        # Sell by placing opposite order
        # If we hold YES, sell YES. If we hold NO, sell NO.
        sell_price = price if side == "yes" else (100 - price)
        try:
            result = await kalshi.create_order(
                ticker=ticker,
                action="sell",
                side=side,
                type="limit",
                count=qty,
                yes_price=sell_price if side == "yes" else (100 - sell_price),
            )
            order = result.get("order", result)
            order_id = order.get("order_id", "")
            await store.record_trade(
                ticker=ticker,
                side=side,
                action="sell",
                price_cents=sell_price,
                count=qty,
                order_id=order_id,
            )
            logger.info(
                "exit_order_placed",
                ticker=ticker,
                side=side,
                count=qty,
                price_cents=sell_price,
                reason=exit_signal["reason"],
            )
        except Exception:
            logger.exception("exit_order_failed", ticker=ticker)


async def check_settlements(
    kalshi: KalshiClient, store: PortfolioStore
) -> None:
    """Check settlement history and update resolved forecasts."""
    try:
        settlements = await kalshi._request(
            "GET", "/portfolio/settlements", authenticated=True
        )
        for s in settlements.get("settlements", []):
            ticker = s.get("ticker", "")
            outcome = s.get("result", "")
            if ticker and outcome in ("yes", "no"):
                outcome_int = 1 if outcome == "yes" else 0
                await store.mark_resolved(ticker, outcome_int)

                # Update trade P&L
                pnl = s.get("revenue", 0)
                if isinstance(pnl, str):
                    pnl = int(float(pnl) * 100)
                order_id = s.get("order_id", "")
                if order_id:
                    await store.update_trade_status(
                        order_id, "settled", pnl_cents=pnl
                    )
    except Exception:
        logger.debug("settlement_check_failed")


def _calc_settlement_pnl(position: dict, outcome: int) -> int:
    """Calculate P&L in cents for a settled position."""
    yes_qty = position.get("yes_contracts", 0)
    no_qty = position.get("no_contracts", 0)

    if outcome == 1:  # YES wins
        # YES holders get 100c per contract, NO holders get 0
        return yes_qty * 100
    else:  # NO wins
        # NO holders get 100c per contract, YES holders get 0
        return no_qty * 100


def _safe_int(val) -> int:
    if val is None:
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0
