"""Risk manager with circuit breakers and position limits."""

import structlog
from pydantic import BaseModel

from kalshiedge._observe import SpanType, observe
from kalshiedge.config import settings
from kalshiedge.portfolio import PortfolioStore

logger = structlog.get_logger()

DAILY_LOSS_LIMIT = 0.10
DRAWDOWN_REDUCE = 0.30
DRAWDOWN_STOP = 0.40


class ProposedTrade(BaseModel):
    ticker: str
    side: str
    action: str
    price_cents: int
    count: int
    edge: float


class RiskManager:
    """Enforces risk limits from CLAUDE.md.

    Decorated with @observe(span_type=SpanType.GUARDRAIL) in main.py integration.
    """

    def __init__(self, store: PortfolioStore, initial_bankroll_cents: int | None = None):
        self.store = store
        self.initial_bankroll_cents = initial_bankroll_cents or settings.bankroll_cents

    @observe(span_type=SpanType.GUARDRAIL)
    async def can_trade(self, trade: ProposedTrade) -> bool:
        """Check all risk limits. Returns True if trade is allowed."""
        bankroll = await self.store.get_bankroll_cents()
        if bankroll is None:
            bankroll = self.initial_bankroll_cents

        # Edge threshold
        if trade.edge < settings.min_edge_threshold:
            logger.info("risk_rejected_low_edge", ticker=trade.ticker, edge=trade.edge)
            return False

        # Max concurrent positions
        open_positions = await self.store.get_open_positions()
        if len(open_positions) >= settings.max_concurrent_positions:
            logger.info("risk_rejected_max_positions", count=len(open_positions))
            return False

        # Per-trade size limit (5% of bankroll)
        trade_cost = trade.count * trade.price_cents
        max_per_trade = int(bankroll * settings.max_position_pct)
        if trade_cost > max_per_trade:
            logger.info(
                "risk_rejected_position_size",
                ticker=trade.ticker,
                trade_cost=trade_cost,
                max_per_trade=max_per_trade,
            )
            return False

        # Total exposure limit (50% of bankroll)
        total_exposure = sum(p["count"] * p["price_cents"] for p in open_positions)
        if total_exposure + trade_cost > int(bankroll * settings.max_exposure_pct):
            logger.info(
                "risk_rejected_total_exposure",
                current_exposure=total_exposure,
                proposed_cost=trade_cost,
            )
            return False

        # Daily loss limit (10%)
        daily_pnl = await self.store.get_daily_pnl_cents()
        if daily_pnl < 0 and abs(daily_pnl) >= int(self.initial_bankroll_cents * DAILY_LOSS_LIMIT):
            logger.warning("risk_rejected_daily_loss_limit", daily_pnl_cents=daily_pnl)
            return False

        # Drawdown checks
        drawdown = self._drawdown_pct(bankroll)
        if drawdown >= DRAWDOWN_STOP:
            logger.warning("risk_rejected_hard_stop", drawdown=drawdown)
            return False

        logger.info("risk_approved", ticker=trade.ticker, edge=trade.edge)
        return True

    def _drawdown_pct(self, current_bankroll_cents: int) -> float:
        if self.initial_bankroll_cents <= 0:
            return 0.0
        return 1 - (current_bankroll_cents / self.initial_bankroll_cents)

    async def size_adjustment(self) -> float:
        """Returns a multiplier for position sizing based on drawdown."""
        bankroll = await self.store.get_bankroll_cents()
        if bankroll is None:
            bankroll = self.initial_bankroll_cents

        drawdown = self._drawdown_pct(bankroll)
        if drawdown >= DRAWDOWN_REDUCE:
            logger.info("risk_size_halved", drawdown=drawdown)
            return 0.5
        return 1.0
