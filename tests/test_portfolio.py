"""Tests for async SQLite portfolio store."""

import pytest

from kalshiedge.portfolio import PortfolioStore


@pytest.fixture
async def store(tmp_path):
    s = PortfolioStore(db_path=str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_bankroll_roundtrip(store):
    await store.set_bankroll_cents(15000)
    assert await store.get_bankroll_cents() == 15000


@pytest.mark.asyncio
async def test_bankroll_update(store):
    await store.set_bankroll_cents(10000)
    await store.set_bankroll_cents(12000)
    assert await store.get_bankroll_cents() == 12000


@pytest.mark.asyncio
async def test_record_trade(store):
    tid = await store.record_trade("TEST-MKT", "yes", "buy", 50, 10, "ord-1")
    assert tid > 0
    positions = await store.get_open_positions()
    assert len(positions) == 1
    assert positions[0]["ticker"] == "TEST-MKT"


@pytest.mark.asyncio
async def test_record_forecast(store):
    fid = await store.record_forecast(
        "TEST-MKT", "Test market?", 50, 0.65, 0.15, strategy="calibration_edge"
    )
    assert fid > 0
    rows = await store.execute_fetchall("SELECT * FROM forecasts")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_trade_status_update(store):
    await store.record_trade("TEST-MKT", "yes", "buy", 50, 5, "ord-1")
    await store.update_trade_status("ord-1", "executed", pnl_cents=200)
    rows = await store.execute_fetchall(
        "SELECT status, pnl_cents FROM trades WHERE order_id = 'ord-1'"
    )
    assert rows[0][0] == "executed"
    assert rows[0][1] == 200


@pytest.mark.asyncio
async def test_mark_resolved(store):
    await store.record_forecast("TEST-MKT", "Test?", 50, 0.7, 0.2)
    await store.mark_resolved("TEST-MKT", 1)
    rows = await store.execute_fetchall(
        "SELECT actual_outcome FROM forecasts WHERE ticker = 'TEST-MKT'"
    )
    assert rows[0][0] == 1


@pytest.mark.asyncio
async def test_daily_snapshot(store):
    await store.record_daily_snapshot(10000, 500, 3, 5, 10)
    rows = await store.execute_fetchall("SELECT * FROM daily_snapshots")
    assert len(rows) == 1
    assert rows[0][1] == 10000  # balance_cents


@pytest.mark.asyncio
async def test_strategy_column_migration(store):
    await store.record_trade("TEST", "yes", "buy", 50, 5, strategy="convergence")
    rows = await store.execute_fetchall(
        "SELECT strategy FROM trades WHERE ticker = 'TEST'"
    )
    assert rows[0][0] == "convergence"
