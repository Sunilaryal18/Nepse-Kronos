import pytest

from nepse_kronos.costs import CostModel


def test_commission_tiers():
    c = CostModel()
    assert c.commission(50_000) == pytest.approx(180.0)            # 0.36%
    assert c.commission(50_001) == pytest.approx(50_001 * 0.0033)  # 0.33%
    assert c.commission(2_000_000) == pytest.approx(6_200.0)       # 0.31%
    assert c.commission(20_000_000) == pytest.approx(48_000.0)     # 0.24%


def test_trade_cost_adds_sebon_slippage_and_dp():
    c = CostModel(slippage=0.001)
    value = 100_000
    assert c.trade_cost(value) == pytest.approx(value * 0.0033 + value * 0.00015 + value * 0.001 + 25)
    assert c.trade_cost(0) == 0.0


def test_trade_cost_grows_with_size_relative_to_daily_turnover():
    c = CostModel(slippage=0.001, impact=0.1)
    value = 100_000
    # trading 10% of a normal day's turnover adds 0.1 * 0.1 = 1% slippage
    assert c.trade_cost(value, daily_turnover=1_000_000) == pytest.approx(
        value * 0.0033 + value * 0.00015 + value * (0.001 + 0.01) + 25)
    # extra slippage is capped at 5% in total
    assert c.trade_cost(value, daily_turnover=1.0) == pytest.approx(value * 0.0033 + value * 0.00015 + value * 0.05 + 25)


def test_capital_gains_tax_only_on_profit():
    c = CostModel(cgt_rate=0.10)
    assert c.capital_gains_tax(12_000, 10_000) == pytest.approx(200.0)
    assert c.capital_gains_tax(9_000, 10_000) == 0.0


def test_capital_gains_tax_is_lower_after_one_year():
    c = CostModel()
    assert c.capital_gains_tax(12_000, 10_000, held_days=365) == pytest.approx(200.0)   # 10%: a year or less
    assert c.capital_gains_tax(12_000, 10_000, held_days=366) == pytest.approx(150.0)   # 7.5%: more than a year
