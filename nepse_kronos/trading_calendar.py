from pathlib import Path

import pandas as pd

NEPSE_WEEKMASK = "Mon Tue Wed Thu Fri"
DEFAULT_HOLIDAYS_PATH = Path(__file__).parent / "holidays.csv"


def load_holidays(path=DEFAULT_HOLIDAYS_PATH):
    """Read market holidays from a CSV with a 'date' column. Missing file means no holidays."""
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path, comment="#")
    return sorted(pd.to_datetime(df["date"]).dt.normalize().tolist())


def next_trading_days(last_date, n, holidays=(), weekmask=NEPSE_WEEKMASK):
    """The n NEPSE trading days strictly after last_date."""
    start = pd.Timestamp(last_date).normalize() + pd.Timedelta(days=1)
    days = pd.bdate_range(start=start, periods=n, freq="C", weekmask=weekmask, holidays=list(holidays))
    return pd.Series(days, name="timestamps")
