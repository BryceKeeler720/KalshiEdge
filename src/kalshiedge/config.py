"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Kalshi API
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_env: str = "demo"

    # Anthropic
    anthropic_api_key: str = ""

    # 2signal
    twosignal_api_key: str = ""
    twosignal_base_url: str = ""

    # News
    newsapi_key: str = ""
    news_cache_ttl_seconds: int = 1800

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_webhook_url: str = ""

    # Mode
    dry_run: bool = True

    # Model configuration
    forecast_model: str = "claude-sonnet-4-6"
    forecast_temperatures: str = "0.3,0.5,0.7"  # Comma-separated
    extremization_factor: float = 1.3
    prompt_caching: bool = True

    # Trading parameters
    bankroll_usd: float = 100.0
    min_edge_threshold: float = 0.08
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    max_exposure_pct: float = 0.50
    max_concurrent_positions: int = 12
    cycle_interval_seconds: int = 600
    fast_cycle_seconds: int = 120
    max_screens_per_cycle: int = 200  # Cheap Haiku screens — scan everything
    max_sonnet_per_cycle: int = 10  # Expensive full forecasts after screening
    stale_order_minutes: int = 10

    # Strategy tuning
    convergence_min_price: int = 93
    convergence_max_price: int = 97
    convergence_max_hours: int = 24
    orderbook_min_depth: int = 2
    stop_loss_pct: float = -0.15
    take_profit_pct: float = 0.30

    # Daily summary
    daily_summary_hour: int = 23  # UTC hour to send daily summary (11pm)

    # Logging
    log_level: str = "INFO"

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_env == "prod":
            return "https://api.elections.kalshi.com/trade-api/v2"
        return "https://demo-api.kalshi.co/trade-api/v2"

    @property
    def bankroll_cents(self) -> int:
        return int(self.bankroll_usd * 100)

    @property
    def private_key_pem(self) -> str:
        return Path(self.kalshi_private_key_path).read_text()

    @property
    def temperatures(self) -> list[float]:
        return [float(t.strip()) for t in self.forecast_temperatures.split(",")]

    @field_validator("kelly_fraction")
    @classmethod
    def validate_kelly_fraction(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("KELLY_FRACTION must be between 0 and 1")
        return v

    @field_validator("min_edge_threshold")
    @classmethod
    def validate_min_edge(cls, v: float) -> float:
        if not 0 <= v <= 0.5:
            raise ValueError("MIN_EDGE_THRESHOLD must be between 0 and 0.5")
        return v

    @field_validator("bankroll_usd")
    @classmethod
    def validate_bankroll(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("BANKROLL_USD must be positive")
        return v

    @field_validator("max_position_pct", "max_exposure_pct")
    @classmethod
    def validate_pct(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Percentage values must be positive")
        return v


settings = Settings()
