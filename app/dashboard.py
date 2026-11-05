"""IMLV30 dashboard.

    streamlit run app/dashboard.py

Everything is computed from data/panel.parquet the first time the app loads (about
10-15 seconds) and then cached. Nothing is read from output/, so the numbers on screen
are always what the code produces today.
"""
from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idxf import analytics as an  # noqa: E402
from idxf.ablation import with_overrides  # noqa: E402
from idxf.config import load_config  # noqa: E402
from idxf.corpactions import load_ca  # noqa: E402
from idxf.costs import capacity, net_of_cost, rebalance_costs  # noqa: E402
from idxf.data import load_market  # noqa: E402
from idxf.engine import calculate, prepare  # noqa: E402
from idxf.pipeline import parent_equal_weight  # noqa: E402
from idxf.pricing import bs  # noqa: E402
from idxf.pricing.products import (PPN, RangeAccrual, ppn_greeks, range_accrual_closed_form,  # noqa: E402
                                   solve_participation, solve_range_coupon)
from idxf.pricing.vol import ewma_vol, fit_garch, garch_term_vol  # noqa: E402

# set this to your repository once it is on GitHub; the link only shows when it is a real URL
REPO_URL = "https://github.com/Kaustubh3124/index-factory"

DATA = ROOT / "data"
NAVY, LIGHT, GREEN, GREY, RED = "#1f3b5a", "#6f8aa8", "#3d7a5a", "#8a94a3", "#b5473a"
PAGES = ["Overview", "Holdings and reviews", "Rule lab", "Note pricer", "Data quality"]

st.set_page_config(page_title="IMLV30 index dashboard", layout="wide")


# ----------------------------------------------------------------------------- data

@st.cache_resource(show_spinner="Building the index from 8 years of NSE data (first load only)...")
def load() -> dict:
    cfg = load_config(ROOT / "config" / "imlv30.yaml")
    mkt = load_market(DATA / "panel.parquet", DATA / "sectors.csv", data_dir=DATA)
    ca = load_ca(DATA / "corporate_actions.csv")
    _, _, rets, adj = prepare(mkt, ca)
    run = calculate(mkt, cfg, ca)
    costs = rebalance_costs(run, mkt, rets, cfg)
    bench = mkt.benchmark.reindex(run.level.index).ffill()
    levels = pd.DataFrame({
        "IMLV30": run.level,
        "IMLV30 net of costs": net_of_cost(run.level, costs),
        "Eligible universe, equal weight": parent_equal_weight(run, rets),
        "NIFTY 50": bench / bench.iloc[0] * run.level.iloc[0],
    })
    return {"cfg": cfg, "mkt": mkt, "ca": ca, "rets": rets, "adj": adj, "run": run, "costs": costs,
            "capacity": capacity(run, mkt, cfg), "levels": levels,
            "garch": fit_garch(run.level.pct_change().dropna())}


@st.cache_data(show_spinner=False, max_entries=30)
def lab_run(params: tuple) -> dict:
    d = load()
    cfg = with_overrides(d["cfg"], dict(params))
    run = calculate(d["mkt"], cfg, d["ca"])
    costs = rebalance_costs(run, d["mkt"], d["rets"], cfg)
    sectors = d["mkt"].sectors
    max_sector = max(r.weights.groupby(sectors.reindex(r.weights.index)).sum().max() for r in run.reviews)
    years = (run.level.index[-1] - run.level.index[0]).days / 365.25
    # per year, not per review, so quarterly and monthly rules can be compared
    return {"level": run.level, "net": net_of_cost(run.level, costs),
            "turnover": float(costs["one_way_turnover"].sum() / years),
            "cost_bps": float(costs["cost_bps"].sum() / years),
            "max_sector": float(max_sector), "reviews": len(run.reviews)}


@st.cache_data(show_spinner=False, max_entries=60)
def mc_note(T, prot, cap, avg, fees, S0, r, q, sigma):
    note = solve_participation(PPN(T=T, protection=prot, cap=cap, avg_months=avg, fees=fees),
                               S0, r, q, sigma, n_paths=60_000, seed=7)
    g = ppn_greeks(note, S0, r, q, sigma, n_paths=60_000, seed=7)
    return note.participation, g


# ----------------------------------------------------------------------------- helpers

def pct(x, d=1):
    return "-" if x is None or pd.isna(x) else f"{x * 100:.{d}f}%"


def stats_table(levels: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for c in levels.columns:
        s = an.perf_stats(levels[c].dropna())
        rows[c] = {"Annual return": pct(s["cagr"]), "Volatility": pct(s["vol"]),
                   "Return / vol": f"{s['sharpe']:.2f}", "Max drawdown": pct(s["max_drawdown"])}
    return pd.DataFrame(rows)


def line_chart(df: pd.DataFrame, log: bool = False, height: int = 360, palette=None):
    names = list(df.columns)
    long = df.reset_index(names="date").melt("date", var_name="series", value_name="level")
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    colors = palette or [NAVY, LIGHT, GREEN, GREY][: len(names)]
    lo, hi = float(long["level"].min()), float(long["level"].max())
    # log axes get "niced" out to the next power of ten, which squashes the lines; pin the domain
    yscale = (alt.Scale(type="log", domain=[lo * 0.95, hi * 1.05], nice=False) if log
              else alt.Scale(zero=False))
    return (alt.Chart(long).mark_line(strokeWidth=1.6)
            .encode(x=alt.X("date:T", title=None),
                    y=alt.Y("level:Q", title=None, scale=yscale),
                    color=alt.Color("series:N", scale=alt.Scale(domain=names, range=colors),
                                    legend=alt.Legend(orient="top", title=None, labelLimit=320)),
                    tooltip=[alt.Tooltip("date:T"), "series:N", alt.Tooltip("level:Q", format=",.1f")])
            .properties(height=height).add_params(zoom))


def sector_bars(w: pd.Series, sectors: pd.Series, cap: float | None = None, height: int = 260):
    sw = (w.groupby(sectors.reindex(w.index).fillna("Unclassified")).sum()
          .sort_values(ascending=False).rename("weight").rename_axis("sector").reset_index())
    bars = (alt.Chart(sw).mark_bar(color=NAVY)
            .encode(x=alt.X("weight:Q", axis=alt.Axis(format="%"), title=None),
                    y=alt.Y("sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
                    tooltip=["sector:N", alt.Tooltip("weight:Q", format=".1%")])
            .properties(height=height))
    if cap is not None and cap < 1:
        rule = alt.Chart(pd.DataFrame({"cap": [cap]})).mark_rule(color=RED, strokeDash=[4, 3]).encode(x="cap:Q")
        return bars + rule
    return bars


def weights_table(w: pd.Series, sectors: pd.Series) -> pd.DataFrame:
    w = w[w > 0].sort_values(ascending=False)
    return pd.DataFrame({"Stock": w.index, "Sector": sectors.reindex(w.index).values,
                         "Weight": [pct(x, 2) for x in w.values]})


# ----------------------------------------------------------------------------- pages

def page_overview(d):
    run, lv = d["run"], d["levels"]
    st.title("India Momentum Low-Vol 30 (IMLV30)")
    st.write(
        "A rules-based index of 30 liquid NSE stocks, picked every quarter on a 50/50 blend of "
        "risk-adjusted momentum and low volatility, weighted by inverse volatility with an 8% stock cap "
        "and a 30% sector cap. Calculated the way index providers do it: index shares, a divisor, and "
        "corporate actions applied on their ex-dates. All levels are a back-test on historical data."
    )
    s, b = an.perf_stats(run.level), an.perf_stats(lv["NIFTY 50"])
    rel = an.relative_stats(run.level, lv["NIFTY 50"])
    c = st.columns(4)
    c[0].metric("Annual return", pct(s["cagr"]), f"{(s['cagr'] - b['cagr']) * 100:+.1f} pts vs NIFTY 50")
    c[1].metric("Volatility", pct(s["vol"]), f"{(s['vol'] - b['vol']) * 100:+.1f} pts", delta_color="inverse")
    c[2].metric("Max drawdown", pct(s["max_drawdown"]), f"{(s['max_drawdown'] - b['max_drawdown']) * 100:+.1f} pts")
    c[3].metric("Beta to NIFTY 50", f"{rel['beta']:.2f}")

    log = st.toggle("Log scale", value=False)
    st.altair_chart(line_chart(lv, log=log), width="stretch")
    st.caption(f"Base 1000 on {run.level.index[0]:%d %b %Y}. Drag to pan, scroll to zoom, double-click to reset.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Risk and return")
        st.dataframe(stats_table(lv))
        st.markdown(
            "**How to read this.** Most of the gap to the NIFTY 50 is not the factors: an equal-weighted "
            "basket of the same 150 liquid stocks did even better, because broad liquid stocks beat the 50 "
            "largest over this period. What the momentum / low-vol tilt bought is a smoother ride: about the "
            "same return per unit of risk as its parent universe, with a much shallower drawdown."
        )
    with right:
        st.subheader("Calendar years")
        cy = pd.DataFrame({c: an.calendar_returns(lv[c]) for c in ["IMLV30", "NIFTY 50", "Eligible universe, equal weight"]})
        cy.columns = ["IMLV30", "NIFTY 50", "Equal weight"]
        cy.index = [f"{y}{' (part)' if y in (cy.index.min(), cy.index.max()) else ''}" for y in cy.index]
        st.dataframe(cy.map(pct))

    st.subheader("Drawdown")
    dd = (lv[["IMLV30", "NIFTY 50"]] / lv[["IMLV30", "NIFTY 50"]].cummax() - 1)
    long = dd.reset_index(names="date").melt("date", var_name="series", value_name="dd")
    st.altair_chart(
        alt.Chart(long).mark_area(opacity=0.45, line=True)
        .encode(x=alt.X("date:T", title=None), y=alt.Y("dd:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("series:N", scale=alt.Scale(domain=["IMLV30", "NIFTY 50"], range=[NAVY, GREY]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["date:T", "series:N", alt.Tooltip("dd:Q", format=".1%")])
        .properties(height=200), width="stretch")


def page_holdings(d):
    run, mkt, costs, cfg = d["run"], d["mkt"], d["costs"], d["cfg"]
    st.header("Holdings and reviews")
    t1, t2 = st.tabs(["Current portfolio", "Review history"])
    with t1:
        last = run.weights.index[-1]
        w = run.weights.loc[last]
        st.caption(f"Weights at the close of {last:%d %b %Y} (they drift with prices between reviews).")
        a, b = st.columns([3, 2])
        a.dataframe(weights_table(w, mkt.sectors), hide_index=True, height=560)
        b.altair_chart(sector_bars(w[w > 0], mkt.sectors, cfg.weighting.sector_cap, height=320), width="stretch")
        b.caption("Dashed line: 30% sector cap.")
    with t2:
        reviews = run.reviews[::-1]
        labels = [f"{r.review.effective:%d %b %Y}{' (initial build)' if r.review.initial else ''}" for r in reviews]
        i = st.selectbox("Review", range(len(reviews)), format_func=lambda k: labels[k])
        rv = reviews[i]
        r = rv.review
        st.write(f"Determination **{r.determination:%d %b %Y}** · announcement **{r.announcement:%d %b %Y}** · "
                 f"effective after the close of **{r.effective:%d %b %Y}** · {int(rv.universe['eligible'].sum())} eligible stocks")
        if r.effective in costs.index:
            c = costs.loc[r.effective]
            m = st.columns(4)
            m[0].metric("One-way turnover", pct(c["one_way_turnover"], 0))
            m[1].metric("Estimated cost", f"{c['cost_bps']:.1f} bps")
            m[2].metric("Hardest trade at ₹2,000 cr AUM", f"{c['max_days_to_trade']:.1f} days", help=f"{c['worst_name']}, trading 20% of daily volume")
            m[3].metric("Capacity (1 day at 20% of ADV)", f"₹{d['capacity'].loc[r.effective]:,.0f} cr")
        if not r.initial:
            rows = []
            for t in rv.adds:
                rows.append({"Action": "Add", "Stock": t, "Sector": mkt.sectors.get(t, ""), "Why": f"rank {int(rv.scores.loc[t, 'rank'])}"})
            for t in rv.deletes:
                rk = rv.scores["rank"].get(t)
                if pd.notna(rk):
                    why = f"rank {int(rk)}"
                else:
                    u = rv.universe.loc[t] if t in rv.universe.index else None
                    why = (f"not eligible: liquidity rank {int(u['liq_rank'])}" if u is not None and u["reason"] == "outside_top_n"
                           else f"not eligible: {u['reason']}" if u is not None else "no data")
                rows.append({"Action": "Delete", "Stock": t, "Sector": mkt.sectors.get(t, ""), "Why": why})
            st.subheader(f"{len(rv.adds)} additions, {len(rv.deletes)} deletions")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        a, b = st.columns([3, 2])
        a.subheader("Target weights")
        a.dataframe(weights_table(rv.weights, mkt.sectors), hide_index=True, height=420)
        b.subheader("By sector")
        b.altair_chart(sector_bars(rv.weights, mkt.sectors, cfg.weighting.sector_cap, height=300), width="stretch")


def page_lab(d):
    st.header("Rule lab")
    st.write("Change the rules and re-run the full back-test on the same data (every review since 2019). "
             "One run takes a few seconds.")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Number of stocks", 15, 50, 30, step=5)
    mom = c1.select_slider("Score mix", options=[0.0, 0.25, 0.5, 0.75, 1.0], value=0.5,
                           format_func=lambda v: f"{v:.0%} momentum / {1 - v:.0%} low vol")
    buf = c2.toggle("Selection buffer", value=True,
                    help="Top 2/3 always in; current members stay down to rank 4/3 x N (20/40 for 30 stocks)")
    scheme = c2.radio("Weighting", ["Inverse volatility", "Equal weight"], horizontal=True)
    cap = c3.select_slider("Sector cap", options=[0.2, 0.25, 0.3, 0.4, 0.5, 1.0], value=0.3,
                           format_func=lambda v: "none" if v >= 1 else f"{v:.0%}")
    freq = c3.radio("Reviews", ["Quarterly", "Semi-annual", "Monthly"], horizontal=True)

    months = {"Quarterly": (3, 6, 9, 12), "Semi-annual": (6, 12), "Monthly": tuple(range(1, 13))}[freq]
    params = {
        "selection.n_constituents": n,
        "selection.buffer_in": round(n * 2 / 3) if buf else n,
        "selection.buffer_out": round(n * 4 / 3) if buf else n,
        "scores.mom_weight": mom, "scores.lowvol_weight": 1 - mom,
        "weighting.scheme": "inverse_vol" if scheme == "Inverse volatility" else "equal",
        "weighting.sector_cap": cap,
        "weighting.stock_cap": max(0.08, round(1 / n + 0.01, 4)),
        "schedule.months": months,
    }
    if st.button("Run back-test", type="primary"):
        st.session_state["lab"] = tuple(sorted(params.items()))
    key = st.session_state.get("lab")
    if key is None:
        st.info("Pick some rules and press **Run back-test**. The published IMLV30 rules are the defaults.")
        return
    with st.spinner("Running reviews and calculating the index..."):
        res = lab_run(key)
    base = lab_run(())
    bench = d["levels"]["NIFTY 50"]

    def row(level, net, extra):
        s = an.perf_stats(level)
        return {"Annual return": pct(s["cagr"]), "Net of costs": pct(an.perf_stats(net)["cagr"]) if net is not None else "-",
                "Volatility": pct(s["vol"]), "Return / vol": f"{s['sharpe']:.2f}", "Max drawdown": pct(s["max_drawdown"]),
                "One-way turnover per year": pct(extra.get("turnover"), 0) if extra else "-",
                "Trading cost per year": f"{extra['cost_bps']:.0f} bps" if extra else "-",
                "Largest sector": pct(extra.get("max_sector"), 0) if extra else "-"}

    table = pd.DataFrame({"Your rules": row(res["level"], res["net"], res),
                          "IMLV30 (published rules)": row(base["level"], base["net"], base),
                          "NIFTY 50": row(bench, None, None)})
    st.dataframe(table)
    chart = pd.DataFrame({"Your rules": res["level"], "IMLV30": base["level"], "NIFTY 50": bench})
    st.altair_chart(line_chart(chart, log=True, palette=[RED, NAVY, GREY]), width="stretch")
    st.caption("Log scale. Picking the best-looking variant after seeing its back-test is overfitting; "
               "methodology section 11 explains which rules were kept and why.")


def page_note(d):
    run, g = d["run"], d["garch"]
    S0 = float(run.level.iloc[-1])
    st.header("Note pricer")
    st.write("A principal-protected note on IMLV30, built like an Indian market-linked debenture: a zero-coupon "
             "bond that pays back the protected amount, plus a call option on the index bought with what is "
             f"left. Priced off today's index level ({S0:,.2f}).")
    left, right = st.columns([1, 2])
    with left:
        T = st.slider("Tenor (years)", 1, 5, 3)
        prot = st.slider("Principal protection (%)", 90, 100, 100) / 100
        fees = st.slider("Upfront fees (%)", 0.0, 3.0, 1.5, 0.25) / 100
        r = st.slider("INR rate, continuous (%)", 4.0, 9.0, 6.75, 0.25) / 100
        q = st.slider("Dividend yield of the stocks (%)", 0.0, 3.0, 1.2, 0.1) / 100
        gv = garch_term_vol(g, T)
        if st.toggle(f"Use GARCH volatility for {T}y ({gv:.1%})", value=True):
            sigma = gv
        else:
            sigma = st.slider("Volatility (%)", 8, 40, int(round(gv * 100))) / 100
        cap = st.selectbox("Cap on the index return", [None, 0.2, 0.3, 0.5, 0.75, 1.0],
                           format_func=lambda c: "No cap" if c is None else f"{c:.0%}")
        avg = st.selectbox("Final index level", [0, 3, 6, 12],
                           format_func=lambda m: "Point to point" if m == 0 else f"Average of last {m} monthly closes")

    zcb = 100 * prot * np.exp(-r * T)
    budget = 100 * (1 - fees) - zcb
    unit = bs.price(S0, S0, T, r, q, sigma) / S0 * 100
    if cap is not None:
        unit -= bs.price(S0, S0 * (1 + cap), T, r, q, sigma) / S0 * 100
    with right:
        if budget <= 0:
            st.error("The bond alone costs more than the money left after fees. Lower the protection or the fees, "
                     "or raise the rate.")
            return
        if avg == 0:
            part = budget / unit
            greeks = _closed_form_greeks(S0, T, r, q, sigma, prot, part, cap)
        else:
            with st.spinner("Monte Carlo (60,000 paths)..."):
                part, greeks = mc_note(T, prot, cap, avg, fees, S0, r, q, sigma)
        m = st.columns(3)
        m[0].metric("Participation", pct(part, 1))
        m[1].metric("Bond (per 100)", f"{zcb:.2f}")
        m[2].metric("Left for the option", f"{budget:.2f}")
        st.caption(f"Investor pays 100. Fees {fees * 100:.2f}, the bond costs {zcb:.2f}, so {budget:.2f} buys "
                   f"options. 100% participation would cost {unit:.2f}" + (" (point to point)." if avg else ".")
                   + f" Volatility used: {sigma:.1%}; EWMA volatility today: {ewma_vol(run.level.pct_change().dropna()).iloc[-1]:.1%}.")

        x = np.linspace(-0.6, 1.5, 211)
        up = np.maximum(x, 0) if cap is None else np.minimum(np.maximum(x, 0), cap)
        pay = pd.DataFrame({"index return": x, "Note": prot - 1 + part * up, "Index (price return)": x})
        long = pay.melt("index return", var_name="series", value_name="return")
        st.altair_chart(
            alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=alt.X("index return:Q", axis=alt.Axis(format="%"),
                            title="Index return at maturity" + (" (average of the fixings)" if avg else "")),
                    y=alt.Y("return:Q", axis=alt.Axis(format="%"), title="Return to the investor"),
                    color=alt.Color("series:N", scale=alt.Scale(domain=["Note", "Index (price return)"], range=[NAVY, GREY]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("index return:Q", format=".0%"), "series:N", alt.Tooltip("return:Q", format=".1%")])
            .properties(height=300), width="stretch")

        st.subheader("Greeks, per 100 notional")
        gcols = st.columns(4)
        gcols[0].metric("Delta (1% index move)", f"{greeks['delta_1pct']:.3f}")
        gcols[1].metric("Vega per vol point", f"{greeks['vega_1vol']:.3f}")
        gcols[2].metric("Rho per bp", f"{greeks['rho_1bp']:.4f}")
        gcols[3].metric("Theta per day", f"{greeks['theta_1d']:.4f}")
        st.caption("Theta is usually positive here: the bond leg accretes faster than the option decays. "
                   "Rho is negative: higher rates cut the bond's value by more than they add to the call.")

        st.subheader("Participation vs volatility and rates (point to point)")
        vols = [sigma - 0.04, sigma - 0.02, sigma, sigma + 0.02, sigma + 0.04]
        rates = [r - 0.01, r, r + 0.01]
        grid = {}
        for rr in rates:
            col = []
            for v in vols:
                b_ = 100 * (1 - fees) - 100 * prot * np.exp(-rr * T)
                u_ = bs.price(S0, S0, T, rr, q, v) / S0 * 100
                if cap is not None:
                    u_ -= bs.price(S0, S0 * (1 + cap), T, rr, q, v) / S0 * 100
                col.append(pct(b_ / u_, 0) if b_ > 0 else "n/a")
            grid[f"rate {rr * 100:.2f}%"] = col
        st.dataframe(pd.DataFrame(grid, index=[f"vol {v * 100:.1f}%" for v in vols]))

    with st.expander("Range accrual: coupon for each day the index stays in a band"):
        a, b = st.columns(2)
        lo = a.slider("Lower bound (% of today's level)", 70, 99, 90)
        hi = b.slider("Upper bound (% of today's level)", 101, 130, 110)
        ra = solve_range_coupon(RangeAccrual(T=1.0, lower=lo / 100, upper=hi / 100), S0, r, q, sigma)
        cf = range_accrual_closed_form(ra, S0, r, q, sigma)
        st.write(f"1-year note, 100% protected, 1% fees: fair coupon **{ra.coupon:.2%} a year** if the index closes "
                 f"inside {lo}%-{hi}% every day. Expected days in range: **{cf['expected_days_in_range']:.0f} of 252**, "
                 f"so the expected coupon is about {ra.coupon * cf['expected_days_in_range'] / 252:.1%}.")


def _closed_form_greeks(S0, T, r, q, sigma, prot, part, cap):
    def leg(K):
        return bs.greeks(S0, K, T, r, q, sigma)
    g = leg(S0)
    keys = ["delta", "vega", "rho", "theta"]
    opt = {k: g[k] for k in keys}
    if cap is not None:
        gc = leg(S0 * (1 + cap))
        opt = {k: opt[k] - gc[k] for k in keys}
    per100 = part * 100 / S0
    bond = 100 * prot * np.exp(-r * T)
    return {
        "delta_1pct": per100 * opt["delta"] * 0.01 * S0,
        "vega_1vol": per100 * opt["vega"] * 0.01,
        "rho_1bp": (-T * bond + per100 * opt["rho"]) * 1e-4,
        "theta_1d": (r * bond + per100 * opt["theta"]) / 365,
    }


def page_data(d):
    mkt, ca, run, adj = d["mkt"], d["ca"], d["run"], d["adj"]
    st.header("Data quality")
    st.write("The source is raw 1-minute bars, which is good because corporate actions show up as price gaps. "
             "It also had problems a real feed can have. Every gap the scanner flagged has a written decision.")
    log = pd.read_csv(DATA / "ca_review_log.csv")
    m = st.columns(4)
    m[0].metric("Gaps flagged by the scanner", f"{(log['suggested'] != '-').sum():,}")
    m[1].metric("Adjustments applied", f"{(log['decision'] == 'applied').sum()}")
    m[2].metric("Dismissed with a reason", f"{(log['decision'] == 'no_action').sum():,}")
    m[3].metric("Still open", f"{(log['decision'] == 'OPEN').sum()}")
    st.markdown(
        "- **A mock trading session.** Saturday 8 Nov 2025 shows the NIFTY 50 down 10% and about 100 stocks gapping "
        "20%, then reverting on Monday. NSE runs mock sessions on some Saturdays; this one leaked into the data and "
        "was removed. Real weekend sessions (Budget days, Muhurat trading) are kept.\n"
        "- **A ticker with someone else's history.** IDFCFIRSTB before Oct 2024 is IDFC Ltd, which merged into the "
        "bank at 155:100. Handled as a share conversion.\n"
        "- **Splits half-applied by the vendor.** The 2026 file pre-divides closes for 2026 splits but leaves the "
        "open raw, so Kotak Bank's 1:5 split appears as x3.0 on 1 Jan and x1.67 on 12 Jan. Pick KOTAKBANK below.\n"
        "- **Scale flips and bad bars.** Persistent Systems switches between half-scaled and unscaled prices four "
        "times; six single bars sit at about 2x or 0.5x their neighbours.")

    st.subheader("Raw vs adjusted prices around a corporate action")
    tickers = sorted(ca["ticker"].unique())
    a, b = st.columns([1, 3])
    t = a.selectbox("Stock", tickers, index=tickers.index("KOTAKBANK") if "KOTAKBANK" in tickers else 0)
    ev = ca[ca["ticker"] == t]
    a.dataframe(ev[["ex_date", "action", "factor"]].assign(ex_date=ev["ex_date"].dt.date), hide_index=True)
    e0 = ev["ex_date"].min()
    lo_, hi_ = e0 - pd.Timedelta(days=60), ev["ex_date"].max() + pd.Timedelta(days=60)
    px = pd.DataFrame({"Raw close": mkt.close[t], "Adjusted close": adj[t]}).loc[lo_:hi_].dropna()
    rules = alt.Chart(ev.assign(ex=ev["ex_date"])).mark_rule(color=RED, strokeDash=[4, 3]).encode(x="ex:T")
    b.altair_chart(line_chart(px, height=300, palette=[GREY, NAVY]) + rules, width="stretch")
    b.caption("Red lines: ex-dates in corporate_actions.csv. The adjusted series is what the momentum and "
              "volatility calculations see; the index itself adjusts its share count on the ex-date instead.")

    st.subheader("All applied adjustments")
    kinds = sorted(ca["action"].unique())
    pick = st.multiselect("Type", kinds, default=kinds)
    show = ca[ca["action"].isin(pick)][["ex_date", "ticker", "action", "factor", "amount", "note"]]
    st.dataframe(show.assign(ex_date=show["ex_date"].dt.date), hide_index=True)

    st.subheader("Events that hit index constituents")
    evs = pd.DataFrame(run.events)
    if evs.empty:
        st.write("None.")
    else:
        st.dataframe(evs.assign(date=pd.to_datetime(evs["date"]).dt.date), hide_index=True)


# ----------------------------------------------------------------------------- main

def main():
    d = load()
    with st.sidebar:
        st.markdown("### IMLV30")
        page = st.radio("Page", PAGES, label_visibility="collapsed")
        st.caption(f"Data to {d['run'].level.index[-1]:%d %b %Y}. Back-tested index; not investment advice.")
        if "<" not in REPO_URL:
            st.markdown(f"[Source code on GitHub]({REPO_URL})")
    {"Overview": page_overview, "Holdings and reviews": page_holdings, "Rule lab": page_lab,
     "Note pricer": page_note, "Data quality": page_data}[page](d)


main()
