"""Async SQLite portfolio state management with WAL mode."""

import datetime

import aiosqlite
import structlog

logger = structlog.get_logger()

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    count INTEGER NOT NULL,
    order_id TEXT,
    status TEXT DEFAULT 'pending',
    strategy TEXT DEFAULT 'calibration_edge',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    filled_at TEXT,
    pnl_cents INTEGER
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    market_price_cents INTEGER NOT NULL,
    model_probability REAL NOT NULL,
    confidence_low REAL,
    confidence_high REAL,
    edge REAL NOT NULL,
    reasoning TEXT,
    strategy TEXT DEFAULT 'calibration_edge',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    actual_outcome INTEGER
);

CREATE TABLE IF NOT EXISTS portfolio (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

MIGRATIONS = [
    "ALTER TABLE trades ADD COLUMN strategy TEXT DEFAULT 'calibration_edge'",
    "ALTER TABLE forecasts ADD COLUMN strategy TEXT DEFAULT 'calibration_edge'",
    """CREATE TABLE IF NOT EXISTS daily_snapshots (
        date TEXT PRIMARY KEY,
        balance_cents INTEGER NOT NULL,
        pnl_cents INTEGER DEFAULT 0,
        open_positions INTEGER DEFAULT 0,
        trades INTEGER DEFAULT 0,
        forecasts INTEGER DEFAULT 0
    )""",
]


class PortfolioStore:
    """Async SQLite portfolio state store."""

    def __init__(self, db_path: str = "kalshiedge.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        await self._run_migrations()
        logger.info("portfolio_db_initialized", path=self.db_path)

    async def _run_migrations(self) -> None:
        for sql in MIGRATIONS:
            try:
                await self.db.execute(sql)
                await self.db.commit()
            except Exception:
                pass  # Column already exists

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("PortfolioStore not initialized — call initialize() first")
        return self._db

    async def record_trade(
        self,
        ticker: str,
        side: str,
        action: str,
        price_cents: int,
        count: int,
        order_id: str | None = None,
        status: str = "pending",
        strategy: str = "calibration_edge",
    ) -> int:
        cursor = await self.db.execute(
            """INSERT INTO trades
               (ticker, side, action, price_cents, count, order_id, status, strategy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, side, action, price_cents, count, order_id, status, strategy),
        )
        await self.db.commit()
        logger.info(
            "trade_recorded",
            ticker=ticker,
            side=side,
            action=action,
            price_cents=price_cents,
            count=count,
        )
        return cursor.lastrowid  # type: ignore[return-value]

    async def record_forecast(
        self,
        ticker: str,
        title: str,
        market_price_cents: int,
        model_probability: float,
        edge: float,
        confidence_low: float | None = None,
        confidence_high: float | None = None,
        reasoning: str | None = None,
        strategy: str = "calibration_edge",
    ) -> int:
        cursor = await self.db.execute(
            """INSERT INTO forecasts
               (ticker, title, market_price_cents, model_probability, edge,
                confidence_low, confidence_high, reasoning, strategy)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, title, market_price_cents, model_probability, edge,
             confidence_low, confidence_high, reasoning, strategy),
        )
        await self.db.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_open_positions(self) -> list[dict]:
        cursor = await self.db.execute(
            """SELECT ticker, side, action, price_cents, count, order_id, status
               FROM trades
               WHERE status IN ('pending', 'resting', 'executed')
                 AND pnl_cents IS NULL"""
        )
        rows = await cursor.fetchall()
        return [
            {
                "ticker": r[0],
                "side": r[1],
                "action": r[2],
                "price_cents": r[3],
                "count": r[4],
                "order_id": r[5],
                "status": r[6],
            }
            for r in rows
        ]

    async def get_bankroll_cents(self) -> int | None:
        cursor = await self.db.execute(
            "SELECT value FROM portfolio WHERE key = 'bankroll_cents'"
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else None

    async def set_bankroll_cents(self, cents: int) -> None:
        await self.db.execute(
            """INSERT INTO portfolio (key, value, updated_at) VALUES ('bankroll_cents', ?, ?)
               ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?""",
            (str(cents), _now(), str(cents), _now()),
        )
        await self.db.commit()

    async def get_daily_pnl_cents(self) -> int:
        today = datetime.date.today().isoformat()
        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(pnl_cents), 0) FROM trades WHERE date(filled_at) = ?",
            (today,),
        )
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def record_daily_snapshot(
        self,
        balance_cents: int,
        pnl_cents: int,
        open_positions: int,
        trades: int,
        forecasts: int,
    ) -> None:
        today = datetime.date.today().isoformat()
        await self.db.execute(
            """INSERT INTO daily_snapshots
               (date, balance_cents, pnl_cents, open_positions, trades, forecasts)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 balance_cents = ?, pnl_cents = ?, open_positions = ?,
                 trades = ?, forecasts = ?""",
            (today, balance_cents, pnl_cents, open_positions, trades, forecasts,
             balance_cents, pnl_cents, open_positions, trades, forecasts),
        )
        await self.db.commit()

    async def mark_resolved(self, ticker: str, outcome: int) -> None:
        await self.db.execute(
            "UPDATE forecasts SET actual_outcome = ? WHERE ticker = ? AND actual_outcome IS NULL",
            (outcome, ticker),
        )
        await self.db.commit()
        logger.info("market_resolved", ticker=ticker, outcome=outcome)

    async def update_trade_status(
        self, order_id: str, status: str, pnl_cents: int | None = None
    ) -> None:
        if pnl_cents is not None:
            await self.db.execute(
                "UPDATE trades SET status = ?, filled_at = ?, pnl_cents = ? WHERE order_id = ?",
                (status, _now(), pnl_cents, order_id),
            )
        else:
            await self.db.execute(
                "UPDATE trades SET status = ? WHERE order_id = ?",
                (status, order_id),
            )
        await self.db.commit()

    async def execute_fetchall(self, sql: str, params: tuple = ()) -> list:
        cursor = await self.db.execute(sql, params)
        return await cursor.fetchall()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
