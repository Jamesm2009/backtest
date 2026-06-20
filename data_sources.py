"""
Data layer: Yahoo Finance fetch + Upstash Redis cache.

Uses the Upstash *pipeline* REST endpoint (POST /pipeline) rather than the
path-based /set/{key} endpoint, to avoid the double-JSON-encoding issue seen
on other dashboards in this project.

Caching strategy:
- KNOWN tickers (the 35 MFs, 74 ETFs, and benchmark constituents) are SEEDED
  once with a full historical fetch, then kept warm by a WEEKLY cron hitting
  /refresh, which does a small incremental fetch (just the last few weeks)
  and merges it into the cached series rather than re-downloading 16 years
  of history every time. Cache TTL is long (9 days) since cron refreshes
  weekly and the data itself only changes weekly anyway (weekly resolution).
- CUSTOM tickers (free-text, not pre-warmed) are fetched on demand and cached
  with a short TTL (1 hour) — long enough to avoid hammering Yahoo Finance if
  the same custom backtest is re-run a few times in a row, short enough that
  stale/bad data doesn't linger.
"""

import json
import os
from datetime import datetime, timedelta
import requests
import pandas as pd
import yfinance as yf

REDIS_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")

KNOWN_TICKER_TTL_SECONDS  = 60 * 60 * 24 * 9   # 9 days — comfortably covers a weekly cron
CUSTOM_TICKER_TTL_SECONDS = 60 * 60 * 1        # 1 hour — ad hoc, not pre-warmed
MAX_YEARS_FETCH = 16  # fetch/retain a bit more than the 15yr max lookback
INCREMENTAL_OVERLAP_DAYS = 21  # re-fetch a small trailing overlap on each update

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


def redis_set_json(key, value, ex_seconds=KNOWN_TICKER_TTL_SECONDS):
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

def fetch_weekly_series_live(ticker: str, start: str = None, end: str = None,
                              years: float = MAX_YEARS_FETCH):
    """
    Fetch daily adjusted-close history from Yahoo Finance, resample to weekly
    (Friday) close. auto_adjust=True bakes dividend reinvestment + splits into
    the price series, so no separate distribution-reinvestment logic is needed.

    Uses explicit start/end dates rather than yfinance's `period=` shorthand —
    yfinance's documented period values are a fixed set (1d, 5d, 1mo, 3mo, 6mo,
    1y, 2y, 5y, 10y, ytd, max), not arbitrary "Ny" strings, so passing a custom
    value like "16y" isn't guaranteed to behave as intended. start/end dates
    also let the incremental updater fetch just a small recent window.

    Returns a pandas Series indexed by weekly Friday dates, or None on failure.
    """
    if start is None:
        start = (datetime.now() - timedelta(days=int(years * 365.25))).strftime("%Y-%m-%d")
    try:
        tk = yf.Ticker(ticker)
        kwargs = {"start": start, "interval": "1d", "auto_adjust": True}
        if end is not None:
            kwargs["end"] = end
        hist = tk.history(**kwargs)
        if hist is None or hist.empty:
            return None
        close = hist["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        weekly = close.resample("W-FRI").last().ffill()
        weekly = weekly.dropna()
        return _sanitize_series(weekly)
    except Exception as e:
        print(f"  Yahoo Finance fetch error for {ticker}: {e}")
        return None


def _sanitize_series(weekly: pd.Series):
    """
    Basic data-quality check before a series is trusted/cached:
    drop non-positive prices (bad ticks), require a minimum amount of history
    to be useful, and bail out entirely if nothing sane is left.
    """
    if weekly is None or weekly.empty:
        return None
    weekly = weekly[weekly > 0]
    if len(weekly) < 4:  # need at least ~a month of weekly closes to be useful
        return None
    return weekly


def seed_or_update_ticker(ticker: str, ttl: int = KNOWN_TICKER_TTL_SECONDS):
    """
    The core caching strategy: if nothing is cached yet, do a full historical
    fetch (seed). If something IS cached, fetch only a small recent overlap
    window and merge it in, rather than re-downloading the full history.
    Always re-caches with the given TTL (refreshing the expiry on every call).
    Returns the resulting Series, or None if no data could be obtained at all.
    """
    cached = redis_get_json(PRICE_KEY_PREFIX + ticker)

    if not cached or not cached.get("dates"):
        weekly = fetch_weekly_series_live(ticker, years=MAX_YEARS_FETCH)
        if weekly is None:
            return None
    else:
        existing = pd.Series(cached["close"], index=pd.to_datetime(cached["dates"])).sort_index()
        overlap_start = (existing.index[-1] - pd.Timedelta(days=INCREMENTAL_OVERLAP_DAYS)).strftime("%Y-%m-%d")
        recent = fetch_weekly_series_live(ticker, start=overlap_start)
        if recent is None or recent.empty:
            weekly = existing  # live fetch failed — keep what we already had
        else:
            merged = pd.concat([existing, recent])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()  # fresh data wins on overlap
            cutoff = merged.index[-1] - pd.DateOffset(years=MAX_YEARS_FETCH)
            weekly = merged[merged.index >= cutoff]

    payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
        "close": [round(float(v), 4) for v in weekly.values],
    }
    redis_set_json(PRICE_KEY_PREFIX + ticker, payload, ex_seconds=ttl)
    return weekly


def get_weekly_series(ticker: str, years: int = MAX_YEARS_FETCH, allow_live_fetch=True,
                       ttl: int = KNOWN_TICKER_TTL_SECONDS):
    """
    Return a pandas Series (weekly Friday closes) for `ticker`, sliced to the
    most recent `years` years. Tries Redis cache first; falls back to
    seed_or_update_ticker (and re-caches with `ttl`) if allowed and cache is
    empty/expired. Pass a short `ttl` for ad hoc/custom tickers that aren't
    pre-warmed by the cron job.
    """
    cached = redis_get_json(PRICE_KEY_PREFIX + ticker)
    weekly = None
    if cached and cached.get("dates"):
        idx = pd.to_datetime(cached["dates"])
        weekly = pd.Series(cached["close"], index=idx)
    elif allow_live_fetch:
        weekly = seed_or_update_ticker(ticker, ttl=ttl)

    if weekly is None:
        return None

    cutoff = weekly.index[-1] - pd.DateOffset(years=years)
    return weekly[weekly.index >= cutoff]


def build_price_panel(tickers: list, years: int, known_tickers: set = None):
    """
    Returns (panel, missing_tickers).
    panel: DataFrame indexed on the UNION of all weekly dates across tickers,
           forward-filled (capped) per ticker, columns = tickers. NaN before a
           ticker's first available date (used for cash-bridging upstream).

    known_tickers: optional set of symbols considered "known" (MF/ETF/benchmark
                   list, pre-warmed by the weekly cron) — these get the long
                   cache TTL on any live-fetch fallback. Anything not in this
                   set is treated as ad hoc/custom and gets the short TTL.
    """
    known_tickers = known_tickers or set()
    series_map = {}
    missing = []
    for t in tickers:
        ttl = KNOWN_TICKER_TTL_SECONDS if t in known_tickers else CUSTOM_TICKER_TTL_SECONDS
        s = get_weekly_series(t, years=years, ttl=ttl)
        if s is None or s.empty:
            missing.append(t)
            continue
        series_map[t] = s

    if not series_map:
        return None, missing

    panel = pd.DataFrame(series_map)
    panel = panel.sort_index()
    # Cap forward-fill to a couple of weeks — enough to absorb minor fetch-timing
    # misalignment between tickers, but NOT enough to mask a genuine fund closure.
    # A ticker that stops returning data for 3+ consecutive weeks is treated as
    # closed: it stays NaN, and the backtest engine routes its target weight to
    # Cash from the next rebalance onward (see engine.run_backtest's has_price
    # check). Without this cap, an unlimited ffill would freeze a closed fund's
    # last price forever and keep "reinvesting" in it at every rebalance instead.
    panel = panel.ffill(limit=2)
    return panel, missing
