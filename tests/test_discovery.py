"""Tests for market discovery parsing and filtering."""

from kalshiedge.discovery import Market, _dollars_to_cents, _parse_market, _passes_filters


class TestDollarsToCents:
    def test_string_dollar(self):
        assert _dollars_to_cents("0.6200") == 62

    def test_string_small(self):
        assert _dollars_to_cents("0.0300") == 3

    def test_string_one(self):
        assert _dollars_to_cents("1.0000") == 100

    def test_int_passthrough(self):
        assert _dollars_to_cents(62) == 62

    def test_float(self):
        assert _dollars_to_cents(0.62) == 62

    def test_none(self):
        assert _dollars_to_cents(None) == 0

    def test_empty_string(self):
        assert _dollars_to_cents("") == 0


class TestParseMarket:
    def test_new_api_format(self):
        raw = {
            "ticker": "TEST-MKT",
            "title": "Test Market",
            "yes_bid_dollars": "0.5000",
            "yes_ask_dollars": "0.5500",
            "no_bid_dollars": "0.4500",
            "no_ask_dollars": "0.5000",
            "last_price_dollars": "0.5200",
            "volume_fp": "1000.00",
            "volume_24h_fp": "200.00",
            "open_interest_fp": "500.00",
            "close_time": "2026-04-01T00:00:00Z",
        }
        m = _parse_market(raw)
        assert m is not None
        assert m.ticker == "TEST-MKT"
        assert m.yes_bid == 50
        assert m.yes_ask == 55
        assert m.last_price == 52
        assert m.volume == 1000

    def test_skips_mve_tickers(self):
        raw = {"ticker": "KXMVE-SOMETHING", "title": "Combo"}
        m = _parse_market(raw)
        assert m is None


class TestPassesFilters:
    def _market(self, **kwargs) -> Market:
        defaults = {
            "ticker": "TEST", "title": "Test", "yes_bid": 40, "yes_ask": 45,
            "no_bid": 55, "no_ask": 60, "last_price": 42, "volume": 1000,
            "volume_24h": 200, "open_interest": 500,
            "close_time": "2026-05-01T00:00:00Z",  # ~30 days out
        }
        defaults.update(kwargs)
        return Market(**defaults)

    def test_passes_good_market(self):
        assert _passes_filters(self._market()) is True

    def test_rejects_low_volume(self):
        assert _passes_filters(self._market(volume=10)) is False

    def test_rejects_extreme_price(self):
        assert _passes_filters(self._market(last_price=1)) is False
        assert _passes_filters(self._market(last_price=99)) is False

    def test_rejects_no_bids(self):
        assert _passes_filters(self._market(yes_bid=0, yes_ask=0)) is False
