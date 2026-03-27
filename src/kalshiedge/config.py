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

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Mode
    dry_run: bool = True  # Forecast only, no order execution

    # Trading parameters
    bankroll_usd: float = 100.0
    min_edge_threshold: float = 0.08
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.05
    max_exposure_pct: float = 0.50
    max_concurrent_positions: int = 12
    cycle_interval_seconds: int = 300
    max_forecasts_per_cycle: int = 5

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
