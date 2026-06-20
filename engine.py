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

        if i > 0:
            # Always apply this week's price movement to the existing allocation
            # FIRST — including on rebalance weeks. Skipping this on rebalance
            # weeks would silently drop that week's return for every 13th week.
            for t in tickers:
                r = rets.loc[d, t]
                if pd.isna(r):
                    continue
                alloc[t] *= (1.0 + r)
            # CASH: flat, 0% return

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


def sharpe_ratio(values: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Annualized Sharpe ratio from weekly returns. Risk-free rate defaults to 0%
    (a simplifying assumption — swap in a T-bill series later for more rigor).
    """
    weekly_ret = values.pct_change().dropna()
    if weekly_ret.empty:
        return None
    weekly_rf = (1.0 + risk_free_annual) ** (1 / 52) - 1.0
    excess = weekly_ret - weekly_rf
    std = excess.std()
    if std == 0 or pd.isna(std):
        return None
    return (excess.mean() / std) * np.sqrt(52)


def sortino_ratio(values: pd.Series, risk_free_annual: float = 0.0) -> float:
    """
    Annualized Sortino ratio: like Sharpe, but only penalizes downside volatility
    (weeks where the return fell short of the risk-free rate). Same 0% risk-free
    assumption as sharpe_ratio.
    """
    weekly_ret = values.pct_change().dropna()
    if weekly_ret.empty:
        return None
    weekly_rf = (1.0 + risk_free_annual) ** (1 / 52) - 1.0
    excess = weekly_ret - weekly_rf
    downside = excess[excess < 0]
    if downside.empty:
        return None
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return None
    return (excess.mean() / downside_dev) * np.sqrt(52)


def beta_and_correlation(values: pd.Series, benchmark_prices: pd.Series):
    """
    Beta and correlation of `values` (e.g. portfolio value series) against a
    benchmark price series (e.g. SPY), both at weekly resolution. Returns
    (beta, correlation), either of which may be None if there's not enough
    overlapping data.
    """
    port_ret = values.pct_change().dropna()
    bench_ret = benchmark_prices.pct_change().dropna()
    aligned = pd.concat([port_ret, bench_ret], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return None, None
    p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    var_b = b.var()
    if var_b == 0 or pd.isna(var_b):
        return None, None
    beta = p.cov(b) / var_b
    corr = p.corr(b)
    return (None if pd.isna(beta) else beta), (None if pd.isna(corr) else corr)


def rolling_mean(values: pd.Series, window: int = 63) -> pd.Series:
    """Simple trailing moving average, NaN until `window` points are available."""
    return values.rolling(window=window, min_periods=window).mean()
