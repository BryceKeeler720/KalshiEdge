"""Tests for edge detection, Kelly sizing, extremization, and fee calculations."""

from kalshiedge.edge import (
    aggregate_forecasts,
    compute_edge,
    contracts_from_budget,
    estimated_taker_fee_cents,
    extremize,
    kelly_no,
    net_edge,
    quarter_kelly,
)


class TestComputeEdge:
    def test_yes_edge(self):
        side, edge = compute_edge(0.75, 60)
        assert side == "yes"
        assert abs(edge - 0.15) < 1e-9

    def test_no_edge(self):
        side, edge = compute_edge(0.40, 60)
        assert side == "no"
        assert abs(edge - 0.20) < 1e-9

    def test_no_edge_at_market(self):
        side, edge = compute_edge(0.60, 60)
        assert side == "none"
        assert edge == 0.0

    def test_extreme_prices(self):
        side, edge = compute_edge(0.99, 5)
        assert side == "yes"
        assert edge > 0.9


class TestQuarterKelly:
    def test_basic_kelly(self):
        result = quarter_kelly(0.70, 50, 10000)
        assert result > 0
        assert result <= 500  # 5% cap of 10000

    def test_no_edge_returns_zero(self):
        assert quarter_kelly(0.50, 50, 10000) == 0

    def test_model_below_market_returns_zero(self):
        assert quarter_kelly(0.40, 50, 10000) == 0

    def test_five_percent_cap(self):
        # Very high edge should still be capped at 5%
        result = quarter_kelly(0.99, 10, 100000)
        assert result <= 5000

    def test_result_is_whole_contracts(self):
        result = quarter_kelly(0.70, 55, 10000)
        # Result should be divisible by price (whole number of contracts)
        assert result % 55 == 0

    def test_small_bankroll(self):
        result = quarter_kelly(0.70, 50, 100)
        # With 100 cents bankroll, very limited
        assert result >= 0


class TestKellyNo:
    def test_basic_no_kelly(self):
        # Model says 30% chance of YES, market at 60 cents
        # So model says 70% NO, no_price = 40 cents
        result = kelly_no(0.30, 60, 10000)
        assert result > 0

    def test_no_edge_no_side(self):
        result = kelly_no(0.60, 60, 10000)
        assert result == 0


class TestExtremize:
    def test_pushes_away_from_half(self):
        result = extremize(0.6, 1.3)
        assert result > 0.6

    def test_pushes_low_further_low(self):
        result = extremize(0.3, 1.3)
        assert result < 0.3

    def test_identity_at_extreme(self):
        assert extremize(0.005) == 0.005
        assert extremize(0.995) == 0.995

    def test_fifty_stays_fifty(self):
        result = extremize(0.5, 1.3)
        assert abs(result - 0.5) < 1e-9

    def test_factor_one_is_identity(self):
        result = extremize(0.7, 1.0)
        assert abs(result - 0.7) < 1e-9


class TestAggregateForecasts:
    def test_trimmed_mean_three(self):
        # With 3 values, drops highest and lowest, uses middle
        result = aggregate_forecasts([0.5, 0.6, 0.7])
        expected = extremize(0.6)
        assert abs(result - expected) < 1e-9

    def test_two_values_uses_mean(self):
        result = aggregate_forecasts([0.5, 0.7])
        expected = extremize(0.6)
        assert abs(result - expected) < 1e-9

    def test_single_value(self):
        result = aggregate_forecasts([0.6])
        expected = extremize(0.6)
        assert abs(result - expected) < 1e-9


class TestFees:
    def test_fee_at_fifty_cents(self):
        fee = estimated_taker_fee_cents(50)
        # At 50c: 0.07 * 4 * 0.5 * 0.5 * 100 = 7.0c
        assert fee == 7.0

    def test_fee_at_extreme(self):
        fee_95 = estimated_taker_fee_cents(95)
        fee_50 = estimated_taker_fee_cents(50)
        assert fee_95 < fee_50

    def test_fee_symmetric(self):
        fee_20 = estimated_taker_fee_cents(20)
        fee_80 = estimated_taker_fee_cents(80)
        assert abs(fee_20 - fee_80) < 0.01


class TestNetEdge:
    def test_net_edge_less_than_raw(self):
        raw_side, raw_edge = compute_edge(0.75, 60)
        n_edge = net_edge(0.75, 60)
        assert n_edge < raw_edge
        assert n_edge > 0

    def test_no_edge_returns_zero(self):
        assert net_edge(0.60, 60) == 0.0


class TestContractsFromBudget:
    def test_basic(self):
        assert contracts_from_budget(500, 50) == 10

    def test_remainder_dropped(self):
        assert contracts_from_budget(120, 55) == 2

    def test_insufficient(self):
        assert contracts_from_budget(10, 50) == 0


class TestCentsProbabilityConversion:
    """Verify cents ↔ probability conversions are consistent."""

    def test_price_to_probability(self):
        assert 65 / 100 == 0.65

    def test_probability_to_price(self):
        assert int(0.65 * 100) == 65

    def test_no_price_complement(self):
        yes_price = 65
        no_price = 100 - yes_price
        assert no_price == 35
        assert (yes_price + no_price) == 100
