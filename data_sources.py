"""
Data layer: Yahoo Finance fetch + Upstash Redis cache.

Uses the Upstash *pipeline* REST endpoint (POST /pipeline) rather than the
path-based /set/{key} endpoint, to avoid the double-JSON-encoding issue seen
on other dashboards in this project.
"""

import json
import os
import requests
import pandas as pd
import yfinance as yf

REDIS_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

CACHE_EX_SECONDS = 60 * 60 * 30  # 30 hours — daily cron keeps it fresh
MAX_YEARS_FETCH  = 16            # fetch a bit more than the 15yr max lookback

PRICE_KEY_PREFIX = "backtest_price_weekly:"


def _pipeline(commands):
    """commands: list of command-arrays, e.g. [["SET","k","v","EX","100"]]"""
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    try:
        r = requests.post(
            f"{REDIS_URL}/pipeline",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}",
                     "Content-Type": "application/json"},
            data=json.dumps(commands),
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  Redis pipeline error: HTTP {r.status_code} {r.text[:200]}")
            return None
        return r.json()
    except Exception as e:
        print(f"  Redis pipeline exception: {e}")
        return None


def redis_set_json(key, value, ex_seconds=CACHE_EX_SECONDS):
    payload = json.dumps(value)
    result = _pipeline([["SET", key, payload, "EX", str(ex_seconds)]])
    if result is None:
        return False
    return result[0].get("result") == "OK"


def redis_get_json(key):
    result = _pipeline([["GET", key]])
    if result is None:
        return None
    raw = result[0].get("result")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def redis_del(key):
    _pipeline([["DEL", key]])


# ── Yahoo Finance fetch ───────────────────────────────────────────────────────

def fetch_weekly_series_live(ticker: str, years: int = MAX_YEARS_FETCH):
    """
    Fetch daily adjusted-close history from Yahoo Finance, resample to weekly
    (Friday) close. auto_adjust=True bakes dividend reinvestment + splits into
    the price series, so no separate distribution-reinvestment logic is needed.
    Returns a pandas Series indexed by weekly Friday dates, or None on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=f"{years}y", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        close = hist["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        weekly = close.resample("W-FRI").last().ffill()
        weekly = weekly.dropna()
        if weekly.empty:
            return None
        return weekly
    except Exception as e:
        print(f"  Yahoo Finance fetch error for {ticker}: {e}")
        return None


def cache_ticker(ticker: str):
    """Fetch live data for a ticker and store it in Redis. Returns the Series or None."""
    weekly = fetch_weekly_series_live(ticker)
    if weekly is None:
        return None
    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
        "close": [round(float(v), 4) for v in weekly.values],
    }
    redis_set_json(PRICE_KEY_PREFIX + ticker, payload)
    return weekly


def get_weekly_series(ticker: str, years: int = MAX_YEARS_FETCH, allow_live_fetch=True):
    """
    Return a pandas Series (weekly Friday closes) for `ticker`, sliced to the
    most recent `years` years. Tries Redis cache first; falls back to a live
    Yahoo Finance fetch (and re-caches) if allowed and cache is empty/stale.
    """
    cached = redis_get_json(PRICE_KEY_PREFIX + ticker)
    weekly = None
    if cached and cached.get("dates"):
        idx = pd.to_datetime(cached["dates"])
        weekly = pd.Series(cached["close"], index=idx)
    elif allow_live_fetch:
        weekly = cache_ticker(ticker)

    if weekly is None:
        return None

    cutoff = weekly.index[-1] - pd.DateOffset(years=years)
    return weekly[weekly.index >= cutoff]


def build_price_panel(tickers: list, years: int):
    """
    Returns (panel, missing_tickers).
    panel: DataFrame indexed on the UNION of all weekly dates across tickers,
           forward-filled per ticker, columns = tickers. NaN before a
           ticker's first available date (used for cash-bridging upstream).
    """
    series_map = {}
    missing = []
    for t in tickers:
        s = get_weekly_series(t, years=years)
        if s is None or s.empty:
            missing.append(t)
            continue
        series_map[t] = s

    if not series_map:
        return None, missing

    panel = pd.DataFrame(series_map)
    panel = panel.sort_index()
    panel = panel.ffill()  # fills gaps where one ticker's calendar has an extra week
    return panel, missing
