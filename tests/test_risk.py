"""Tests for RiskManager limits."""

import pytest

from kalshiedge.portfolio import PortfolioStore
from kalshiedge.risk import ProposedTrade, RiskManager


@pytest.fixture
async def store(tmp_path):
    s = PortfolioStore(db_path=str(tmp_path / "test.db"))
    await s.initialize()
    await s.set_bankroll_cents(10000)  # $100
    yield s
    await s.close()


@pytest.fixture
async def risk(store):
    return RiskManager(store, initial_bankroll_cents=10000)


def _trade(
    ticker: str = "TEST-MKT",
    price_cents: int = 50,
    count: int = 5,
    edge: float = 0.10,
) -> ProposedTrade:
    return ProposedTrade(
        ticker=ticker, side="yes", action="buy",
        price_cents=price_cents, count=count, edge=edge,
    )


@pytest.mark.asyncio
async def test_basic_trade_approved(risk):
    assert await risk.can_trade(_trade()) is True


@pytest.mark.asyncio
async def test_low_edge_rejected(risk):
    assert await risk.can_trade(_trade(edge=0.05)) is False


@pytest.mark.asyncio
async def test_position_size_limit(risk):
    # 5% of 10000 = 500 cents max. 20 contracts @ 50c = 1000 > 500
    assert await risk.can_trade(_trade(count=20)) is False


@pytest.mark.asyncio
async def test_max_concurrent_positions(risk, store):
    # Fill up to max positions
    for i in range(12):
        await store.record_trade(
            ticker=f"MKT-{i}", side="yes", action="buy",
            price_cents=50, count=1, order_id=f"order-{i}", status="executed",
        )
    assert await risk.can_trade(_trade()) is False


@pytest.mark.asyncio
async def test_total_exposure_limit(risk, store):
    # Use 45% of bankroll in existing positions
    await store.record_trade(
        ticker="BIG-POS", side="yes", action="buy",
        price_cents=50, count=90, order_id="big-order", status="executed",
    )
    # Try to add more that would exceed 50%
    assert await risk.can_trade(_trade(count=20)) is False


@pytest.mark.asyncio
async def test_drawdown_hard_stop(risk, store):
    # Set bankroll to 60% of initial (40% drawdown)
    await store.set_bankroll_cents(6000)
    assert await risk.can_trade(_trade()) is False


@pytest.mark.asyncio
async def test_drawdown_reduces_size(risk, store):
    await store.set_bankroll_cents(7000)  # 30% drawdown
    mult = await risk.size_adjustment()
    assert mult == 0.5
