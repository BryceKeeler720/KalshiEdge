"""Alert manager — Discord webhooks + Telegram. No-op if not configured."""

import httpx
import structlog
import telegram

from kalshiedge.config import settings

logger = structlog.get_logger()


class AlertManager:
    """Async alert sender via Discord webhook and/or Telegram."""

    def __init__(self) -> None:
        # Telegram
        self._bot: telegram.Bot | None = None
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self._bot = telegram.Bot(token=settings.telegram_bot_token)
            logger.info("telegram_alerts_enabled")

        # Discord
        self._discord_url: str | None = settings.discord_webhook_url or None
        if self._discord_url:
            logger.info("discord_alerts_enabled")

        if not self._bot and not self._discord_url:
            logger.info("alerts_disabled")

    async def _send_telegram(self, text: str) -> None:
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

    async def _send_discord(self, content: str, embed: dict | None = None) -> None:
        if not self._discord_url:
            return
        try:
            payload: dict = {}
            if embed:
                payload["embeds"] = [embed]
            else:
                payload["content"] = content
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._discord_url, json=payload)
                resp.raise_for_status()
        except Exception:
            logger.exception("discord_send_failed")

    async def _send(self, text: str, embed: dict | None = None) -> None:
        await self._send_telegram(text)
        await self._send_discord(text, embed=embed)

    async def notify_trade(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        price_cents: int,
        edge: float,
        strategy: str = "calibration_edge",
    ) -> None:
        cost_usd = count * price_cents / 100
        text = (
            f"**Trade Executed**\n"
            f"Ticker: `{ticker}`\n"
            f"Action: {action.upper()} {side.upper()}\n"
            f"Contracts: {count} @ {price_cents}\u00a2\n"
            f"Cost: ${cost_usd:.2f}\n"
            f"Edge: {edge:.1%}\n"
            f"Strategy: {strategy}"
        )
        embed = {
            "title": f"\u2705 {action.upper()} {side.upper()} — {ticker}",
            "color": 0x00B894 if side == "yes" else 0xE17055,
            "fields": [
                {"name": "Contracts", "value": f"{count} @ {price_cents}\u00a2", "inline": True},
                {"name": "Cost", "value": f"${cost_usd:.2f}", "inline": True},
                {"name": "Edge", "value": f"{edge:.1%}", "inline": True},
                {"name": "Strategy", "value": strategy.replace("_", " ").title(), "inline": True},
            ],
        }
        await self._send(text, embed=embed)
        logger.info(
            "trade_alert_sent", ticker=ticker, side=side,
            action=action, count=count, strategy=strategy,
        )

    async def notify_error(self, error: str, context: str = "") -> None:
        text = f"**Error**\n{error}"
        if context:
            text += f"\nContext: {context}"
        embed = {
            "title": "\u26a0\ufe0f Error",
            "description": error,
            "color": 0xE17055,
        }
        if context:
            embed["footer"] = {"text": context}
        await self._send(text, embed=embed)
        logger.warning("error_alert_sent", error=error, context=context)

    async def notify_daily_summary(
        self,
        balance_cents: int,
        daily_pnl_cents: int,
        open_positions: int,
        trades_today: int,
    ) -> None:
        text = (
            f"**Daily Summary**\n"
            f"Balance: ${balance_cents / 100:.2f}\n"
            f"Daily P&L: ${daily_pnl_cents / 100:+.2f}\n"
            f"Open Positions: {open_positions}\n"
            f"Trades Today: {trades_today}"
        )
        pnl_color = 0x00B894 if daily_pnl_cents >= 0 else 0xE17055
        embed = {
            "title": "\U0001f4ca Daily Summary",
            "color": pnl_color,
            "fields": [
                {"name": "Balance", "value": f"${balance_cents / 100:.2f}", "inline": True},
                {"name": "Daily P&L", "value": f"${daily_pnl_cents / 100:+.2f}", "inline": True},
                {"name": "Positions", "value": str(open_positions), "inline": True},
                {"name": "Trades", "value": str(trades_today), "inline": True},
            ],
        }
        await self._send(text, embed=embed)
        logger.info(
            "daily_summary_sent",
            balance_cents=balance_cents,
            daily_pnl_cents=daily_pnl_cents,
        )

    async def notify_exit(
        self, ticker: str, side: str, reason: str, edge: float
    ) -> None:
        text = (
            f"**Position Exit**\n"
            f"Ticker: `{ticker}`\n"
            f"Side: {side.upper()}\n"
            f"Reason: {reason}\n"
            f"Edge: {edge:.1%}"
        )
        embed = {
            "title": f"\U0001f6aa EXIT — {ticker}",
            "color": 0xFDCB6E,
            "fields": [
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Reason", "value": reason.replace("_", " ").title(), "inline": True},
                {"name": "Edge", "value": f"{edge:.1%}", "inline": True},
            ],
        }
        await self._send(text, embed=embed)

    async def notify_settlement(
        self, ticker: str, outcome: str, pnl_cents: int
    ) -> None:
        text = (
            f"**Settlement**\n"
            f"Ticker: `{ticker}`\n"
            f"Outcome: {outcome.upper()}\n"
            f"P&L: ${pnl_cents / 100:+.2f}"
        )
        color = 0x00B894 if pnl_cents >= 0 else 0xE17055
        embed = {
            "title": f"\U0001f3c1 SETTLED — {ticker}",
            "color": color,
            "fields": [
                {"name": "Outcome", "value": outcome.upper(), "inline": True},
                {"name": "P&L", "value": f"${pnl_cents / 100:+.2f}", "inline": True},
            ],
        }
        await self._send(text, embed=embed)
