"""Two-page PDF factsheet (A4), matplotlib only so it builds anywhere."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

NAVY = "#1f3b5a"
GREY = "#8a94a3"
RED = "#b5473a"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def _pct(x, d=1):
    return "" if pd.isna(x) else f"{x * 100:.{d}f}%"


def _table(ax, rows, col_labels=None, col_widths=None):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=col_labels, loc="upper left", cellLoc="left",
                 colWidths=col_widths, edges="horizontal")
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d0d4da")
        if r == 0 and col_labels:
            cell.set_text_props(weight="bold", color=NAVY)
    return t


def build_factsheet(path, cfg, level, net, bench, stats, rel, cal_ret, cal_ret_b,
                    weights_last, sectors, rebal_costs, asof_label):
    b = bench.reindex(level.index).ffill()
    b = b / b.iloc[0] * level.iloc[0]
    with PdfPages(path) as pdf:
        # ---------------- page 1
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.965, cfg.name, fontsize=16, weight="bold", color=NAVY)
        fig.text(0.06, 0.945, f"{cfg.ticker}  |  Price return, INR  |  Data as of {asof_label}",
                 fontsize=8.5, color=GREY)
        fig.text(0.06, 0.915,
                 "Rules-based index of 30 liquid NSE stocks selected on a 50/50 blend of risk-adjusted "
                 "momentum and low volatility,\ninverse-volatility weighted with 8% stock and 30% sector caps. "
                 "Reviewed quarterly (Mar, Jun, Sep, Dec).", fontsize=8)

        ax = fig.add_axes([0.07, 0.62, 0.88, 0.26])
        ax.plot(level.index, level, color=NAVY, lw=1.3, label=cfg.ticker)
        ax.plot(net.index, net, color=NAVY, lw=0.8, ls="--", alpha=0.7, label=f"{cfg.ticker} net of rebalance costs")
        ax.plot(b.index, b, color=GREY, lw=1.1, label="NIFTY 50 (rebased)")
        ax.set_title("Performance since base date (base = 1000)")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(alpha=0.25)

        ax = fig.add_axes([0.07, 0.47, 0.88, 0.11])
        dd = level / level.cummax() - 1
        ddb = b / b.cummax() - 1
        ax.fill_between(dd.index, dd, 0, color=NAVY, alpha=0.35, lw=0, label=cfg.ticker)
        ax.plot(ddb.index, ddb, color=GREY, lw=0.8, label="NIFTY 50")
        ax.set_title("Drawdown")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.legend(frameon=False, loc="lower left")

        rows = [
            ["Annualised return", _pct(stats["cagr"]), _pct(stats["cagr"] - rel["excess_cagr"])],
            ["Annualised volatility", _pct(stats["vol"]), _pct(stats.get("bench_vol"))],
            ["Return / vol", f"{stats['sharpe']:.2f}", f"{stats.get('bench_sharpe', np.nan):.2f}"],
            ["Max drawdown", _pct(stats["max_drawdown"]), _pct(stats.get("bench_mdd"))],
            ["Beta to NIFTY 50", f"{rel['beta']:.2f}", "1.00"],
            ["Tracking error", _pct(rel["tracking_error"]), ""],
            ["Information ratio", f"{rel['info_ratio']:.2f}", ""],
            ["Downside capture", _pct(rel["downside_capture"], 0), ""],
        ]
        ax = fig.add_axes([0.07, 0.25, 0.42, 0.18])
        _table(ax, rows, ["Risk / return", cfg.ticker, "NIFTY 50"], [0.5, 0.25, 0.25])

        yrs = cal_ret.index
        rows = [[str(y), _pct(cal_ret[y]), _pct(cal_ret_b.get(y)), _pct(cal_ret[y] - cal_ret_b.get(y, np.nan))]
                for y in yrs]
        ax = fig.add_axes([0.54, 0.25, 0.41, 0.18])
        _table(ax, rows, ["Calendar year", cfg.ticker, "NIFTY 50", "Diff"], [0.31, 0.23, 0.23, 0.23])

        fig.text(0.07, 0.205, "Notes", weight="bold", color=NAVY, fontsize=8)
        fig.text(0.07, 0.195,
                 "First and last calendar years are partial. Index levels before launch are back-tested using the rules in\n"
                 "docs/methodology.md applied to historical data; they are not live performance. The stock universe is built from\n"
                 "the current NIFTY 500 membership list, so the back-test carries survivorship bias (see methodology section 10).\n"
                 "Net series deducts estimated half-spread and square-root market impact at each review for "
                 f"INR {cfg.costs.tracking_aum_cr:,.0f} cr tracking AUM.\n"
                 "Calculated with official-style closing prices (volume-weighted average of the last 30 minutes of trading).",
                 fontsize=6.8, color="#444", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------- page 2
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.06, 0.965, f"{cfg.ticker} - portfolio and maintenance", fontsize=13, weight="bold", color=NAVY)

        w = weights_last[weights_last > 0].sort_values(ascending=False)
        sec = w.groupby(sectors.reindex(w.index).fillna("Unclassified")).sum().sort_values()
        ax = fig.add_axes([0.25, 0.66, 0.68, 0.26])
        ax.barh(sec.index, sec.values, color=NAVY)
        ax.axvline(cfg.weighting.sector_cap, color=RED, lw=0.8, ls=":")
        ax.text(cfg.weighting.sector_cap, len(sec) - 0.4, " sector cap", color=RED, fontsize=7, va="bottom")
        ax.set_xlim(0, cfg.weighting.sector_cap * 1.15)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title("Sector weights")

        rows = [[t, sectors.get(t, ""), _pct(x, 2)] for t, x in w.head(10).items()]
        ax = fig.add_axes([0.07, 0.40, 0.5, 0.22])
        _table(ax, rows, ["Top 10 constituents", "Sector", "Weight"], [0.45, 0.35, 0.2])

        ax = fig.add_axes([0.62, 0.40, 0.33, 0.22])
        _table(ax, [["Constituents", str(len(w))],
                    ["Effective N", f"{1 / (w ** 2).sum():.1f}"],
                    ["Largest weight", _pct(w.max(), 2)],
                    ["Smallest weight", _pct(w.min(), 2)],
                    ["Avg one-way turnover / review", _pct(rebal_costs['one_way_turnover'].mean())],
                    ["Avg cost / review", f"{rebal_costs['cost_bps'].mean():.1f} bps"],
                    ["Reviews since base", str(len(rebal_costs))]],
               ["Characteristics", ""], [0.7, 0.3])

        ax = fig.add_axes([0.07, 0.12, 0.88, 0.22])
        ax.bar(rebal_costs.index, rebal_costs["one_way_turnover"], width=40, color=NAVY, label="One-way turnover")
        ax2 = ax.twinx()
        ax2.plot(rebal_costs.index, rebal_costs["cost_bps"], color=RED, marker="o", ms=2.5, lw=0.8, label="Cost (bps)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title("Turnover and estimated cost at each review")
        ax2.spines["right"].set_visible(True)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
        pdf.savefig(fig)
        plt.close(fig)
