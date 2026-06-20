"""
Builds a one-page-plus PDF report from a backtest result dict (same shape
returned by app.compute_full_result): chart of all 5 series, a performance
table, a drawdown table, and the portfolio holdings with Morningstar links.
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

LABELS = {
    "portfolio": "Your Portfolio",
}


def _chart_image(result):
    fig, ax = plt.subplots(figsize=(9.5, 4.2), dpi=150)
    colors_map = {"portfolio": "#2563eb", "spy_only": "#6b7280",
                  "dalio_all_weather": "#16a34a", "sixty_forty": "#b45309",
                  "growth": "#dc2626", "custom_benchmark": "#9333ea"}
    for key, s in result["series"].items():
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in s["dates"]]
        label = LABELS.get(key, result["benchmark_labels"].get(key, key))
        ax.plot(dates, s["values"], label=label, linewidth=1.6,
                color=colors_map.get(key, None))
    ax.set_title(f"Portfolio Value Over Time ({result['lookback_years']}-Year Backtest, $10,000 Start)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def _pct(v, decimals=1):
    if v is None:
        return "—"
    return f"{v*100:.{decimals}f}%"


def build_pdf(result):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                             topMargin=0.5*inch, bottomMargin=0.5*inch,
                             leftMargin=0.5*inch, rightMargin=0.5*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=16)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8.5)

    elements = []
    elements.append(Paragraph("KCM Portfolio Backtest Report", title_style))
    elements.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
        f"{result['lookback_years']}-year lookback &nbsp;|&nbsp; $10,000 starting value &nbsp;|&nbsp; "
        f"Quarterly rebalancing &nbsp;|&nbsp; Dividends reinvested",
        small))
    elements.append(Spacer(1, 10))

    # Holdings table
    elements.append(Paragraph("Portfolio Holdings", styles["Heading3"]))
    hold_rows = [["Symbol", "Weight"]]
    for h in result["portfolio_holdings"]:
        link = h.get("morningstar_url")
        sym_cell = Paragraph(f'<link href="{link}">{h["symbol"]}</link>' if link else h["symbol"], small)
        hold_rows.append([sym_cell, f'{h["weight"]:.1f}%'])
    if result.get("cash_weight"):
        hold_rows.append(["Cash", f'{result["cash_weight"]:.1f}%'])
    hold_table = Table(hold_rows, colWidths=[3*inch, 1.2*inch])
    hold_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e2d45")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    elements.append(hold_table)
    elements.append(Spacer(1, 14))

    if result.get("custom_benchmark_holdings"):
        cb_label = result["benchmark_labels"].get("custom_benchmark", "Custom Benchmark")
        elements.append(Paragraph(f"{cb_label} Holdings", styles["Heading3"]))
        cb_rows = [["Symbol", "Weight"]]
        for h in result["custom_benchmark_holdings"]:
            link = h.get("morningstar_url")
            sym_cell = Paragraph(f'<link href="{link}">{h["symbol"]}</link>' if link else h["symbol"], small)
            cb_rows.append([sym_cell, f'{h["weight"]:.1f}%'])
        cb_table = Table(cb_rows, colWidths=[3*inch, 1.2*inch])
        cb_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6b21a8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e3")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(cb_table)
        elements.append(Spacer(1, 14))

    # Chart
    chart_buf = _chart_image(result)
    elements.append(Image(chart_buf, width=7.2*inch, height=3.2*inch))
    elements.append(Spacer(1, 12))

    # Performance table
    elements.append(Paragraph("Performance Summary", styles["Heading3"]))
    perf_rows = [["Strategy", "Final Value", "Total\nReturn", "CAGR", "Ann.\nVol",
                  "Max DD\n(All-Time)", "Max DD\n(12-Mo Roll)"]]
    order = ["portfolio"] + [k for k in result["metrics"] if k != "portfolio"]
    for key in order:
        m = result["metrics"][key]
        label = LABELS.get(key, result["benchmark_labels"].get(key, key))
        perf_rows.append([
            label,
            f"${m['final_value']:,.0f}",
            _pct(m["total_return"]),
            _pct(m["cagr"]),
            _pct(m["annualized_vol"]),
            _pct(m["max_drawdown_alltime"]),
            _pct(m["max_drawdown_rolling_12mo"]),
        ])
    perf_table = Table(perf_rows, colWidths=[1.5*inch, 0.95*inch, 0.95*inch, 0.8*inch, 0.8*inch, 1.1*inch, 1.2*inch])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e2d45")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e3")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dbeafe")),  # highlight portfolio row
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 8))

    if result.get("missing_tickers"):
        elements.append(Paragraph(
            "Note: no data could be retrieved for: " + ", ".join(result["missing_tickers"]) +
            ". These were excluded from the calculation.", small))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()
