"""
KCM Portfolio Backtest Tool
- Modes: Mutual Funds (35) / ETFs (74, incl. benchmark-only tickers) / Custom tickers
- Up to 10 holdings + Cash, default equal weight with optional override
- Quarterly rebalancing, cash-bridging for any ticker shorter than the lookback window
- Benchmarks: SPY Only, Dalio All Weather, 60/40 ETF, Growth ETFs
- Weekly series, $10,000 start, dividend reinvestment via Yahoo auto-adjusted close
- Metrics: total return, CAGR, annualized vol, all-time max DD, rolling-12mo max DD
- PDF export (chart + tables) via /api/pdf
"""

import json
import os
import threading
import time
from io import BytesIO
from flask import Flask, render_template, jsonify, request, send_file

import data_sources as ds
import engine as eng
from pdf_report import build_pdf

app = Flask(__name__)

with open("funds_mf.json") as f:
    FUNDS_MF = json.load(f)
with open("funds_etf.json") as f:
    FUNDS_ETF = json.load(f)
with open("benchmarks.json") as f:
    BENCHMARKS = json.load(f)

MF_BY_SYMBOL  = {f["symbol"]: f for f in FUNDS_MF}
ETF_BY_SYMBOL = {f["symbol"]: f for f in FUNDS_ETF}

MAX_HOLDINGS = 10
VALID_LOOKBACKS = (10, 15)

# All tickers that should be kept warm in Redis by the daily cron
_ALL_KNOWN_TICKERS = sorted(set(
    [f["symbol"] for f in FUNDS_MF] +
    [f["symbol"] for f in FUNDS_ETF] +
    [t for b in BENCHMARKS.values() for t in b["weights"].keys()]
))

_refresh_state = {"running": False, "done": 0, "total": 0, "last_run": None, "errors": []}
_refresh_lock = threading.Lock()


def morningstar_url_for(symbol, mode):
    if mode == "mf" and symbol in MF_BY_SYMBOL:
        return MF_BY_SYMBOL[symbol].get("morningstar_url")
    if mode == "etf" and symbol in ETF_BY_SYMBOL:
        return f"https://www.morningstar.com/etfs/arcx/{symbol.lower()}/quote"
    # custom / unknown — best-effort generic Morningstar search link
    return f"https://www.morningstar.com/search?query={symbol}"


def _validate_weighted_basket(tickers_in, weights_in, max_n, restrict_to=None, label_for_errors="tickers"):
    """
    Shared validator for both the main portfolio and the custom benchmark.
    restrict_to: optional dict of symbol->info to validate against (MF_BY_SYMBOL / ETF_BY_SYMBOL),
                 or None to allow any symbol (custom mode).
    Returns (tickers, weights_dict) or (None, error_message).
    """
    if not isinstance(tickers_in, list) or not (1 <= len(tickers_in) <= max_n):
        return None, f"{label_for_errors} must be a list of 1 to {max_n} symbols"
    tickers = [t.strip().upper() for t in tickers_in if t.strip()]

    if restrict_to is not None:
        bad = [t for t in tickers if t not in restrict_to]
        if bad:
            return None, f"Unknown symbol(s) in {label_for_errors}: {', '.join(bad)}"

    if weights_in:
        try:
            w = {k.strip().upper(): float(v) / 100.0 for k, v in weights_in.items()}
        except (TypeError, ValueError):
            return None, f"{label_for_errors} weights must be numeric percentages"
        total = sum(w.values())
        if abs(total - 1.0) > 0.01:
            return None, f"{label_for_errors} weights must sum to 100% (got {total*100:.1f}%)"
        unknown = [k for k in w.keys() if k != "CASH" and k not in tickers]
        if unknown:
            return None, f"{label_for_errors} weights reference symbols not in tickers: {', '.join(unknown)}"
    else:
        eq = 1.0 / len(tickers)
        w = {t: eq for t in tickers}

    return tickers, w


def _validate_request(body):
    mode = body.get("mode")
    if mode not in ("mf", "etf", "custom"):
        return None, "mode must be 'mf', 'etf', or 'custom'"

    lookback = body.get("lookback")
    if lookback not in VALID_LOOKBACKS:
        return None, "lookback must be 10 or 15"

    restrict = MF_BY_SYMBOL if mode == "mf" else (ETF_BY_SYMBOL if mode == "etf" else None)
    tickers, w = _validate_weighted_basket(
        body.get("tickers", []), body.get("weights"), MAX_HOLDINGS,
        restrict_to=restrict, label_for_errors="portfolio")
    if tickers is None:
        return None, w  # w holds the error message in this branch

    parsed = {"mode": mode, "lookback": lookback, "tickers": tickers, "weights": w}

    # Optional: which fixed benchmarks to include (default: all)
    include_benchmarks = body.get("include_benchmarks")
    if include_benchmarks is None:
        include_benchmarks = list(BENCHMARKS.keys())
    else:
        if not isinstance(include_benchmarks, list):
            return None, "include_benchmarks must be a list"
        unknown_b = [k for k in include_benchmarks if k not in BENCHMARKS]
        if unknown_b:
            return None, f"Unknown benchmark key(s): {', '.join(unknown_b)}"
    parsed["include_benchmarks"] = include_benchmarks

    # Optional: a single custom benchmark, same shape as the portfolio basket
    custom_bench_in = body.get("custom_benchmark")
    if custom_bench_in:
        cb_tickers, cb_w = _validate_weighted_basket(
            custom_bench_in.get("tickers", []), custom_bench_in.get("weights"),
            MAX_HOLDINGS, restrict_to=None, label_for_errors="custom_benchmark")
        if cb_tickers is None:
            return None, cb_w
        cb_label = (custom_bench_in.get("label") or "Custom Benchmark").strip()[:60]
        parsed["custom_benchmark"] = {"label": cb_label, "tickers": cb_tickers, "weights": cb_w}
    else:
        parsed["custom_benchmark"] = None

    return parsed, None


def compute_full_result(parsed):
    mode, lookback, tickers, weights = (parsed["mode"], parsed["lookback"],
                                         parsed["tickers"], parsed["weights"])
    include_benchmarks = parsed.get("include_benchmarks", list(BENCHMARKS.keys()))
    custom_benchmark = parsed.get("custom_benchmark")

    portfolio_tickers = list(tickers)
    bench_tickers_needed = set()
    for key in include_benchmarks:
        bench_tickers_needed |= set(BENCHMARKS[key]["weights"].keys())
    if custom_benchmark:
        bench_tickers_needed |= set(custom_benchmark["tickers"])

    all_tickers = sorted(set(portfolio_tickers) | bench_tickers_needed)
    panel, missing = ds.build_price_panel(all_tickers, lookback)
    if panel is None or panel.empty:
        return None, "Could not retrieve price data for any selected ticker."

    series = {}
    metrics = {}
    benchmark_labels = {}

    def run(weights_dict, label_key):
        cols = [t for t in weights_dict.keys() if t != "CASH" and t in panel.columns]
        cash_extra = sum(w for t, w in weights_dict.items() if t != "CASH" and t not in panel.columns)
        eff = {t: weights_dict[t] for t in cols}
        eff["CASH"] = weights_dict.get("CASH", 0.0) + cash_extra
        sub_panel = panel[cols] if cols else panel.iloc[:, 0:0]
        vals = eng.run_backtest(eff, sub_panel, 10000.0)
        series[label_key] = {
            "dates": [d.strftime("%Y-%m-%d") for d in vals.index],
            "values": [round(float(v), 2) for v in vals.values],
        }
        metrics[label_key] = {
            "total_return": eng.total_return(vals),
            "cagr": eng.cagr(vals),
            "annualized_vol": eng.annualized_vol(vals),
            "max_drawdown_alltime": eng.max_drawdown_alltime(vals),
            "max_drawdown_rolling_12mo": eng.max_drawdown_rolling_12mo(vals),
            "final_value": round(float(vals.iloc[-1]), 2),
        }

    run(weights, "portfolio")
    for key in include_benchmarks:
        run(BENCHMARKS[key]["weights"], key)
        benchmark_labels[key] = BENCHMARKS[key]["label"]
    if custom_benchmark:
        run(custom_benchmark["weights"], "custom_benchmark")
        benchmark_labels["custom_benchmark"] = custom_benchmark["label"]

    result = {
        "series": series,
        "metrics": metrics,
        "benchmark_labels": benchmark_labels,
        "portfolio_holdings": [
            {"symbol": t, "weight": round(weights.get(t, 0) * 100, 2),
             "morningstar_url": morningstar_url_for(t, mode)}
            for t in tickers
        ],
        "custom_benchmark_holdings": (
            [{"symbol": t, "weight": round(custom_benchmark["weights"].get(t, 0) * 100, 2),
              "morningstar_url": morningstar_url_for(t, "custom")}
             for t in custom_benchmark["tickers"]]
            if custom_benchmark else None
        ),
        "cash_weight": round(weights.get("CASH", 0) * 100, 2),
        "lookback_years": lookback,
        "missing_tickers": missing,
    }
    return result, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
        funds_mf=FUNDS_MF, funds_etf=FUNDS_ETF,
        benchmarks=BENCHMARKS, max_holdings=MAX_HOLDINGS)


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    body = request.get_json(force=True, silent=True) or {}
    parsed, err = _validate_request(body)
    if err:
        return jsonify({"error": err}), 400
    result, err2 = compute_full_result(parsed)
    if err2:
        return jsonify({"error": err2}), 422
    return jsonify(result)


@app.route("/api/pdf", methods=["POST"])
def api_pdf():
    body = request.get_json(force=True, silent=True) or {}
    parsed, err = _validate_request(body)
    if err:
        return jsonify({"error": err}), 400
    result, err2 = compute_full_result(parsed)
    if err2:
        return jsonify({"error": err2}), 422
    pdf_bytes = build_pdf(result)
    buf = BytesIO(pdf_bytes)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                      download_name="kcm_portfolio_backtest.pdf")


def _do_refresh():
    with _refresh_lock:
        _refresh_state["running"] = True
        _refresh_state["done"] = 0
        _refresh_state["total"] = len(_ALL_KNOWN_TICKERS)
        _refresh_state["errors"] = []

    for t in _ALL_KNOWN_TICKERS:
        try:
            ds.cache_ticker(t)
        except Exception as e:
            with _refresh_lock:
                _refresh_state["errors"].append(f"{t}: {e}")
        with _refresh_lock:
            _refresh_state["done"] += 1
        time.sleep(1)  # be polite to Yahoo Finance

    with _refresh_lock:
        _refresh_state["running"] = False
        _refresh_state["last_run"] = time.strftime("%Y-%m-%d %H:%M:%S")


@app.route("/refresh")
def refresh():
    """Cron-triggered: warm the Redis cache for every known MF/ETF/benchmark ticker."""
    with _refresh_lock:
        if _refresh_state["running"]:
            return jsonify({"status": "already running", "progress": _refresh_state})
    threading.Thread(target=_do_refresh, daemon=True).start()
    return jsonify({"status": "refresh started", "tickers": len(_ALL_KNOWN_TICKERS)})


@app.route("/status")
def status():
    with _refresh_lock:
        return jsonify(dict(_refresh_state))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
