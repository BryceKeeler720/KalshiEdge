"""Edge detection, Kelly criterion sizing, and fee calculations."""

import math


def compute_edge(p_model: float, price_cents: int) -> tuple[str, float]:
    """Compare model probability to market price, return (side, edge_magnitude)."""
    p_market = price_cents / 100
    edge_yes = p_model - p_market
    edge_no = p_market - p_model

    if edge_yes > edge_no and edge_yes > 0:
        return "yes", edge_yes
    elif edge_no > 0:
        return "no", edge_no
    return "none", 0.0


def estimated_taker_fee_cents(price_cents: int) -> float:
    """Approximate taker fee per contract in cents.

    Fees peak at 50c (~1.7c) and drop toward extremes.
    Maker (resting limit) orders are FREE.
    """
    p = price_cents / 100
    base_rate = 0.07
    fee = base_rate * 4 * p * (1 - p)
    return round(fee * 100, 2)


def net_edge(p_model: float, price_cents: int) -> float:
    """Edge after accounting for worst-case taker exit fee and spread."""
    side, raw_edge = compute_edge(p_model, price_cents)
    if side == "none":
        return 0.0
    exit_price = int(p_model * 100)
    exit_price = max(1, min(99, exit_price))
    fee_cents = estimated_taker_fee_cents(exit_price)
    spread_cents = 1  # typical 1c spread
    cost_pct = (fee_cents + spread_cents) / 100
    return raw_edge - cost_pct


def quarter_kelly(p_model: float, price_cents: int, bankroll_cents: int) -> int:
    """Calculate quarter-Kelly position size in cents.

    Returns the cost in cents to buy the optimal number of contracts.
    """
    p_market = price_cents / 100
    if p_model <= p_market:
        return 0

    b = (100 - price_cents) / price_cents
    q = 1 - p_model
    full_kelly = (b * p_model - q) / b

    if full_kelly <= 0:
        return 0

    quarter = full_kelly * 0.25
    max_size = int(bankroll_cents * 0.05)  # 5% cap
    position_cents = min(int(quarter * bankroll_cents), max_size)

    num_contracts = position_cents // price_cents
    return num_contracts * price_cents


def kelly_no(p_model: float, price_cents: int, bankroll_cents: int) -> int:
    """Quarter-Kelly for buying NO contracts."""
    no_price = 100 - price_cents
    p_no = 1 - p_model
    return quarter_kelly(p_no, no_price, bankroll_cents)


def contracts_from_budget(budget_cents: int, price_cents: int) -> int:
    """How many contracts can we buy with this budget."""
    return budget_cents // price_cents


def extremize(p: float, factor: float | None = None) -> float:
    """Push probability away from 0.5 by multiplying log-odds."""
    if factor is None:
        from kalshiedge.config import settings

        factor = settings.extremization_factor
    if p <= 0.01 or p >= 0.99:
        return p
    log_odds = math.log(p / (1 - p))
    adjusted = log_odds * factor
    return 1 / (1 + math.exp(-adjusted))


def aggregate_forecasts(probabilities: list[float]) -> float:
    """Trimmed mean: drop highest and lowest, average rest, then extremize."""
    if len(probabilities) <= 2:
        raw = sum(probabilities) / len(probabilities)
    else:
        sorted_p = sorted(probabilities)
        trimmed = sorted_p[1:-1]
        raw = sum(trimmed) / len(trimmed)
    return extremize(raw)
