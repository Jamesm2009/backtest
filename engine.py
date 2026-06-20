"""
Portfolio backtest engine.
- Weekly time steps (Friday close convention).
- Quarterly rebalancing (every 13 weeks, plus week 0).
- Any ticker with no price data yet at a rebalance date has its target
  weight redirected to CASH (0% return) until the ticker's data begins,
  at which point it's picked back up at the *next* rebalance.
- CASH itself can also be an explicit user-selected weight.
"""

import numpy as np
import pandas as pd

REBALANCE_WEEKS = 13  # quarterly on a weekly grid


def run_backtest(weights: dict, price_panel: pd.DataFrame, start_value: float = 10000.0,
                  rebalance_weeks: int = REBALANCE_WEEKS) -> pd.Series:
    """
    weights: {ticker: target_weight}, must sum to 1.0. May include 'CASH'.
    price_panel: DataFrame, index = weekly dates (ascending), columns = tickers
                 (NOT including CASH). NaN where the ticker has no data yet
                 (i.e. before its inception / before our fetch window).
    Returns: pd.Series of portfolio value, same index as price_panel.
    """
    dates = price_panel.index
    n = len(dates)
    tickers = [t for t in weights.keys() if t != "CASH"]
    cash_target = weights.get("CASH", 0.0)

    # weekly simple returns per ticker (NaN-safe)
    rets = price_panel[tickers].pct_change()

    alloc = {t: 0.0 for t in tickers}
    alloc["CASH"] = 0.0
    values = np.zeros(n)

    for i, d in enumerate(dates):
        is_rebalance = (i == 0) or (i % rebalance_weeks == 0)

        if is_rebalance:
            total = start_value if i == 0 else sum(alloc.values())
            new_alloc = {t: 0.0 for t in tickers}
            new_alloc["CASH"] = 0.0
            for t in tickers:
                has_price = not pd.isna(price_panel.loc[d, t])
                if has_price:
                    new_alloc[t] = total * weights[t]
                else:
                    new_alloc["CASH"] += total * weights[t]
            new_alloc["CASH"] += total * cash_target
            alloc = new_alloc
        else:
            for t in tickers:
                r = rets.loc[d, t]
                if pd.isna(r):
                    # ticker has no price this week (pre-inception) — stays in cash,
                    # contributes nothing here since alloc[t] should be 0 in that case
                    continue
                alloc[t] *= (1.0 + r)
            # CASH: flat, 0% return

        values[i] = sum(alloc.values())

    return pd.Series(values, index=dates, name="portfolio_value")


def max_drawdown_alltime(values: pd.Series) -> float:
    running_max = values.cummax()
    dd = (values - running_max) / running_max
    return dd.min()


def max_drawdown_rolling_12mo(values: pd.Series, window_weeks: int = 52) -> float:
    """Worst drawdown observed within any trailing 12-month (52-week) window."""
    worst = 0.0
    arr = values.values
    n = len(arr)
    for i in range(n):
        lo = max(0, i - window_weeks + 1)
        window = arr[lo:i + 1]
        peak = window.max()
        dd = (arr[i] - peak) / peak
        if dd < worst:
            worst = dd
    return worst


def cagr(values: pd.Series) -> float:
    n_years = (values.index[-1] - values.index[0]).days / 365.25
    if n_years <= 0:
        return None
    return (values.iloc[-1] / values.iloc[0]) ** (1 / n_years) - 1


def annualized_vol(values: pd.Series) -> float:
    weekly_ret = values.pct_change().dropna()
    return weekly_ret.std() * np.sqrt(52)


def total_return(values: pd.Series) -> float:
    return values.iloc[-1] / values.iloc[0] - 1
