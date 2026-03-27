"""Telegram notification sender. No-op if TELEGRAM_BOT_TOKEN is not set."""

import structlog
import telegram

from kalshiedge.config import settings

logger = structlog.get_logger()


class AlertManager:
    """Async Telegram alert sender."""

    def __init__(self) -> None:
        self._bot: telegram.Bot | None = None
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self._bot = telegram.Bot(token=settings.telegram_bot_token)
            logger.info("telegram_alerts_enabled")
        else:
            logger.info("telegram_alerts_disabled")

    async def _send(self, text: str) -> None:
        if self._bot is None:
            return
        try:
            await self._bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception:
            logger.exception("telegram_send_failed")

    async def notify_trade(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        edge: float,
    ) -> None:
        cost_usd = count * price_cents / 100
        msg = (
            f"*Trade Executed*\n"
            f"Ticker: `{ticker}`\n"
            f"Action: {action.upper()} {side.upper()}\n"
            f"Contracts: {count} @ {price_cents}\u00a2\n"
            f"Cost: ${cost_usd:.2f}\n"
            f"Edge: {edge:.1%}"
        )
        await self._send(msg)
        logger.info(
            "trade_alert_sent",
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            price_cents=price_cents,
        )

    async def notify_error(self, error: str, context: str = "") -> None:
        msg = f"*Error*\n{error}"
        if context:
            msg += f"\nContext: {context}"
        await self._send(msg)
        logger.warning("error_alert_sent", error=error, context=context)

    async def notify_daily_summary(
        self,
        balance_cents: int,
        daily_pnl_cents: int,
        open_positions: int,
        trades_today: int,
    ) -> None:
        msg = (
            f"*Daily Summary*\n"
            f"Balance: ${balance_cents / 100:.2f}\n"
            f"Daily P&L: ${daily_pnl_cents / 100:+.2f}\n"
            f"Open Positions: {open_positions}\n"
            f"Trades Today: {trades_today}"
        )
        await self._send(msg)
        logger.info(
            "daily_summary_sent", balance_cents=balance_cents, daily_pnl_cents=daily_pnl_cents
        )
