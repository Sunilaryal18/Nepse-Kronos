"""NEPSE trading costs for a resident individual (rates checked 2026-10-04)."""
from dataclasses import dataclass

# (upper trade value in NPR, broker commission rate); the rate of the bracket applies to the whole trade
COMMISSION_TIERS = (
    (50_000, 0.0036),
    (500_000, 0.0033),
    (2_000_000, 0.0031),
    (10_000_000, 0.0027),
    (float("inf"), 0.0024),
)
MAX_SLIPPAGE = 0.05


@dataclass(frozen=True)
class CostModel:
    commission_tiers: tuple = COMMISSION_TIERS
    sebon_rate: float = 0.00015   # SEBON fee, each buy and sell
    dp_charge: float = 25.0       # CDSC DP charge per stock per trade
    slippage: float = 0.0025      # spread / price impact on a small trade, each buy and sell
    impact: float = 0.1           # extra slippage per unit of (trade value / normal daily turnover)
    cgt_rate: float = 0.10        # capital gains tax, holdings of one year or less (FY 2083/84)
    cgt_rate_long: float = 0.075  # capital gains tax, holdings of more than one year
    long_term_days: int = 365
    dividend_tax: float = 0.05    # withheld on cash dividends
    interest_tax: float = 0.06    # withheld on bank interest for individuals

    def commission(self, value):
        for limit, rate in self.commission_tiers:
            if value <= limit:
                return value * rate
        raise ValueError(f"no commission tier for {value}")

    def slippage_rate(self, value, daily_turnover=None):
        rate = self.slippage
        if daily_turnover:
            rate += self.impact * value / daily_turnover
        return min(rate, MAX_SLIPPAGE)

    def trade_cost(self, value, daily_turnover=None):
        """Everything paid on one buy or one sell of `value` NPR in one stock, except tax."""
        if value <= 0:
            return 0.0
        return (self.commission(value) + value * self.sebon_rate
                + value * self.slippage_rate(value, daily_turnover) + self.dp_charge)

    def capital_gains_tax(self, proceeds, cost_basis, held_days=0):
        rate = self.cgt_rate_long if held_days > self.long_term_days else self.cgt_rate
        return rate * max(0.0, proceeds - cost_basis)
