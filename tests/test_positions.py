"""Tests for position management — stop-loss, take-profit, settlement."""


from kalshiedge.positions import STOP_LOSS_PCT, TAKE_PROFIT_PCT, _calc_settlement_pnl


class TestSettlementPnl:
    def test_yes_wins_holding_yes(self):
        pos = {"yes_contracts": 10, "no_contracts": 0}
        assert _calc_settlement_pnl(pos, 1) == 1000  # 10 * 100c

    def test_no_wins_holding_no(self):
        pos = {"yes_contracts": 0, "no_contracts": 5}
        assert _calc_settlement_pnl(pos, 0) == 500  # 5 * 100c

    def test_yes_wins_holding_no(self):
        pos = {"yes_contracts": 0, "no_contracts": 5}
        assert _calc_settlement_pnl(pos, 1) == 0  # NO holders get nothing

    def test_no_wins_holding_yes(self):
        pos = {"yes_contracts": 10, "no_contracts": 0}
        assert _calc_settlement_pnl(pos, 0) == 0  # YES holders get nothing


class TestStopLossTakeProfit:
    def test_stop_loss_threshold(self):
        assert STOP_LOSS_PCT < 0
        assert STOP_LOSS_PCT >= -1.0

    def test_take_profit_threshold(self):
        assert TAKE_PROFIT_PCT > 0
        assert TAKE_PROFIT_PCT <= 1.0

    def test_stop_loss_triggers(self):
        # Entry at 50c, current value at 40c = -20% < -15% stop loss
        entry_price = 50
        current_value = 40
        unrealized_pct = (current_value - entry_price) / entry_price
        assert unrealized_pct <= STOP_LOSS_PCT

    def test_take_profit_triggers(self):
        # Entry at 50c, current value at 70c = +40% > 30% take profit
        entry_price = 50
        current_value = 70
        unrealized_pct = (current_value - entry_price) / entry_price
        assert unrealized_pct >= TAKE_PROFIT_PCT

    def test_no_trigger_in_range(self):
        entry_price = 50
        current_value = 52
        unrealized_pct = (current_value - entry_price) / entry_price
        assert STOP_LOSS_PCT < unrealized_pct < TAKE_PROFIT_PCT
