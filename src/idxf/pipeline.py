"""End-to-end run: data -> reviews -> index -> recon -> costs -> analytics -> pricing -> reports."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics as an
from .corpactions import load_ca
from .costs import capacity, net_of_cost, rebalance_costs
from .data import load_market
from .engine import calculate, prepare
from .monitor import proforma
from .pricing import bs
from .pricing.products import (PPN, RangeAccrual, ppn_closed_form, ppn_greeks, ppn_mc,
                               range_accrual_closed_form, range_accrual_mc, solve_participation,
                               solve_range_coupon)
from .pricing.vol import ewma_vol, fit_garch, garch_term_vol
from .recon import reconcile
from .report.documents import term_sheet_docx, write_notices
from .report.excel import build_workbook
from .report.factsheet import build_factsheet
from .shadow import calculate_shadow

log = logging.getLogger("idxf")


def run_all(cfg, data_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    mkt = load_market(data_dir / "panel.parquet", data_dir / "sectors.csv", data_dir=data_dir)
    ca = load_ca(data_dir / "corporate_actions.csv")
    log.info("market: %d tickers, %s -> %s", len(mkt.tickers), mkt.calendar[0].date(), mkt.calendar[-1].date())

    factors, divs, rets, adj_close = prepare(mkt, ca)
    run = calculate(mkt, cfg, ca)
    log.info("index: %d days, %d reviews, final level %.2f", len(run.level), len(run.reviews), run.level.iloc[-1])

    # --- reconciliation
    shadow = calculate_shadow(mkt, cfg, rets, {r.review.effective: r.weights for r in run.reviews}, factors, divs)
    rec = reconcile(run.level, shadow)
    log.info("recon: max rel diff %.2e, breaks %d", rec["max_rel_diff"], rec["n_breaks"])

    # --- costs / capacity
    costs = rebalance_costs(run, mkt, rets, cfg)
    net = net_of_cost(run.level, costs)
    cap = capacity(run, mkt, cfg)

    # --- performance
    bench = mkt.benchmark.reindex(run.level.index).ffill()
    stats = an.perf_stats(run.level)
    bstats = an.perf_stats(bench)
    stats.update({"bench_vol": bstats["vol"], "bench_sharpe": bstats["sharpe"], "bench_mdd": bstats["max_drawdown"],
                  "bench_cagr": bstats["cagr"], "net_cagr": an.perf_stats(net)["cagr"]})
    rel = an.relative_stats(run.level, bench)
    cal_ret, cal_ret_b = an.calendar_returns(run.level), an.calendar_returns(bench)

    # --- attribution vs equal-weighted parent universe, and factor regression
    attr, parent = _attribution(run, mkt, rets)
    factors_daily = _factor_returns(run, mkt, rets)
    reg = an.factor_regression(run.level.pct_change(), factors_daily)

    # --- pro-forma monitor as of the last date
    asof = mkt.calendar[-1]
    pf = proforma(mkt, cfg, rets, adj_close, asof, run.weights.loc[asof])

    # --- pricing
    pricing = _pricing(cfg, run, out_dir)

    # --- outputs
    levels = pd.DataFrame({cfg.ticker: run.level, f"{cfg.ticker}_net": net, "shadow": shadow,
                           "divisor": run.divisor, "NIFTY50": bench, "parent_ew": parent})
    levels.to_csv(out_dir / "levels.csv", float_format="%.8f")
    pd.DataFrame(run.events).to_csv(out_dir / "events.csv", index=False)
    costs.join(cap).to_csv(out_dir / "rebalance_costs.csv", float_format="%.6f")
    attr.to_csv(out_dir / "attribution_by_sector.csv", float_format="%.6f")
    reg.to_csv(out_dir / "factor_regression.csv", float_format="%.6f")
    pf["trades"].to_csv(out_dir / f"proforma_{asof.date()}.csv", float_format="%.4f")
    pf["watchlist"].to_csv(out_dir / f"watchlist_{asof.date()}.csv", float_format="%.4f")
    _review_log(run).to_csv(out_dir / "reviews.csv", index=False)
    write_notices(out_dir / "notices", cfg, run, mkt.sectors)

    build_factsheet(out_dir / f"{cfg.ticker}_factsheet.pdf", cfg, run.level, net, bench, stats, rel,
                    cal_ret, cal_ret_b, run.weights.loc[asof], mkt.sectors, costs, str(asof.date()))
    build_workbook(out_dir / f"{cfg.ticker}_workbook.xlsx", cfg, run, mkt, stats, rel, costs, rec,
                   pricing["excel_inputs"])
    term_sheet_docx(out_dir / f"{cfg.ticker}_PPN_termsheet.docx", cfg, pricing["note_obj"], pricing["ts_pricing"],
                    pricing["ts_greeks"], run.level.iloc[-1], asof)

    summary = {
        "asof": str(asof.date()), "tickers_in_data": int(len(mkt.tickers)),
        "reviews": len(run.reviews), "final_level": float(run.level.iloc[-1]),
        "stats": stats, "relative": rel,
        "calendar_returns": {int(k): float(v) for k, v in cal_ret.items()},
        "calendar_returns_nifty": {int(k): float(v) for k, v in cal_ret_b.items()},
        "recon": {k: v for k, v in rec.items() if k != "breaks"},
        "costs": {"avg_one_way_turnover": float(costs["one_way_turnover"].mean()),
                  "avg_cost_bps": float(costs["cost_bps"].mean()),
                  "max_cost_bps": float(costs["cost_bps"].max()),
                  "median_capacity_cr": float(cap.median()), "min_capacity_cr": float(cap.min())},
        "events": pd.DataFrame(run.events)["type"].value_counts().to_dict() if run.events else {},
        "factor_regression": {k: {"coef": float(v["coef"]), "t": float(v["t_stat"])} for k, v in reg.iterrows()},
        "factor_r2": float(reg.attrs["r2"]),
        "attribution_total": attr.sum().to_dict(),
        "proforma": {"asof": str(asof.date()), "one_way_turnover": pf["one_way_turnover"],
                     "adds": pf["trades"].query("status == 'ADD'").index.tolist(),
                     "deletes": pf["trades"].query("status == 'DELETE'").index.tolist()},
        "pricing": pricing["summary"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _review_log(run) -> pd.DataFrame:
    rows = []
    for rv in run.reviews:
        rows.append({"effective": rv.review.effective.date(), "determination": rv.review.determination.date(),
                     "eligible": int(rv.universe["eligible"].sum()), "adds": " ".join(rv.adds),
                     "deletes": " ".join(rv.deletes), **{k: v for k, v in rv.checks.items()}})
    return pd.DataFrame(rows)


def _period_returns(rets, start, end, names):
    win = rets.loc[start:end, names].iloc[1:]
    return (1 + win.fillna(0)).prod() - 1


def parent_equal_weight(run, rets) -> pd.Series:
    """Equal-weighted eligible universe, reset at each review and held until the next one.
    Same start value as the index, so the two can be plotted together."""
    eff = [r.review.effective for r in run.reviews] + [run.level.index[-1]]
    parent = pd.Series(1.0, index=run.level.index)
    for rv, s, e in zip(run.reviews, eff[:-1], eff[1:]):
        uni = rv.universe.index[rv.universe["eligible"]]
        daily = rets.loc[s:e, uni].iloc[1:].fillna(0)
        lvl = (1 + daily).cumprod().mean(axis=1)
        parent.loc[lvl.index] = parent.loc[s] * lvl
    return parent * run.level.iloc[0]


def _attribution(run, mkt, rets):
    eff = [r.review.effective for r in run.reviews] + [run.level.index[-1]]
    periods = []
    for rv, s, e in zip(run.reviews, eff[:-1], eff[1:]):
        uni = rv.universe.index[rv.universe["eligible"]]
        names = sorted(set(uni) | set(rv.weights.index))
        R = _period_returns(rets, s, e, names)
        wb = pd.Series(1.0 / len(uni), index=uni)
        periods.append((s, e, rv.weights, wb, R))
    attr = an.link_attribution(an.brinson_fachler(periods, mkt.sectors))
    return attr, parent_equal_weight(run, rets)


def _factor_returns(run, mkt, rets) -> pd.DataFrame:
    """Daily factor returns built from the same eligible universe and the same review dates.

        NIFTY50  market
        SIZE     equal-weighted eligible universe minus NIFTY 50 (the "broad liquid
                 universe beat the large-cap index" effect, which is most of the story
                 vs NIFTY in 2019-2026)
        MOM      top third minus bottom third on the momentum z-score, equal weighted
        LOWVOL   top third minus bottom third on the low-vol z-score, equal weighted
    """
    eff = [r.review.effective for r in run.reviews] + [run.level.index[-1]]
    out = []
    for rv, s, e in zip(run.reviews, eff[:-1], eff[1:]):
        sc = rv.scores
        k = len(sc) // 3
        daily = rets.loc[s:e].iloc[1:]

        def ls(col):
            o = sc.sort_values(col, ascending=False)
            return daily[o.index[:k]].mean(axis=1) - daily[o.index[-k:]].mean(axis=1)

        uni = rv.universe.index[rv.universe["eligible"]]
        # buy-and-hold equal weight within the period (same as the parent level series)
        growth = (1 + daily[uni].fillna(0)).cumprod()
        ew = growth.mean(axis=1)
        ew_ret = ew / ew.shift(1).fillna(1.0) - 1
        out.append(pd.DataFrame({"EW": ew_ret, "MOM": ls("z_mom"), "LOWVOL": ls("z_lowvol")}))
    f = pd.concat(out)
    f["NIFTY50"] = mkt.benchmark.pct_change().reindex(f.index)
    f["SIZE"] = f["EW"] - f["NIFTY50"]
    return f[["NIFTY50", "SIZE", "MOM", "LOWVOL"]]


def _pricing(cfg, run, out_dir: Path) -> dict:
    pc = cfg.pricing
    S0 = float(run.level.iloc[-1])
    r_idx = run.level.pct_change().dropna()
    g = fit_garch(r_idx)
    sigma = garch_term_vol(g, pc.note_tenor_years)
    ewma_now = float(ewma_vol(r_idx).iloc[-1])
    r, q, T = pc.rate, pc.div_yield, pc.note_tenor_years
    n, seed = pc.mc_paths, pc.seed

    p2p = solve_participation(PPN(T=T, fees=pc.note_fees), S0, r, q, sigma, n, seed)
    cf = ppn_closed_form(p2p, S0, r, q, sigma)
    mc = ppn_mc(p2p, S0, r, q, sigma, n, seed)
    capped = solve_participation(PPN(T=T, fees=pc.note_fees, cap=0.50), S0, r, q, sigma, n, seed)
    asian = solve_participation(PPN(T=T, fees=pc.note_fees, avg_months=6), S0, r, q, sigma, n, seed)
    greeks = ppn_greeks(asian, S0, r, q, sigma, n, seed)
    unit = ppn_mc(PPN(T=T, participation=1.0, avg_months=6, protection=0.0), S0, r, q, sigma, n, seed)

    # participation that the closed form gives - the Excel sheet should reproduce this number
    budget = 100 * (1 - pc.note_fees) - cf["zcb"]
    part_cf = budget / ppn_closed_form(PPN(T=T, participation=1.0), S0, r, q, sigma)["option_per_100pct"]

    ra = solve_range_coupon(RangeAccrual(T=1.0, lower=0.90, upper=1.10), S0, r, q, sigma)
    ra_cf = range_accrual_closed_form(ra, S0, r, q, sigma)
    ra_mc = range_accrual_mc(ra, S0, r, q, sigma, 100_000, seed)

    # sensitivity of participation to vol and rates - the two things a client always asks
    sens = []
    for v in (sigma - 0.04, sigma - 0.02, sigma, sigma + 0.02, sigma + 0.04):
        for rr in (r - 0.01, r, r + 0.01):
            bud = 100 * (1 - pc.note_fees) - 100 * np.exp(-rr * T)
            opt = bs.price(S0, S0, T, rr, q, v) / S0 * 100
            sens.append({"vol": v, "rate": rr, "participation": bud / opt})
    pd.DataFrame(sens).pivot(index="vol", columns="rate", values="participation") \
        .to_csv(out_dir / "participation_sensitivity.csv", float_format="%.4f")

    # greeks profile of the point-to-point note across spot and time (closed form)
    rows = []
    for tau in (3.0, 2.0, 1.0, 0.25):
        for m in np.linspace(0.6, 1.5, 37):
            S = S0 * m
            gk = bs.greeks(S, S0, tau, r, q, sigma)
            rows.append({"tau": tau, "spot_pct": m,
                         "value": 100 * np.exp(-r * tau) + p2p.participation * bs.price(S, S0, tau, r, q, sigma) / S0 * 100,
                         "delta_1pct": p2p.participation * gk["delta"] * S / S0,
                         "gamma_1pct": p2p.participation * gk["gamma"] * (0.01 * S) ** 2 / S0 * 100,
                         "vega_1vol": p2p.participation * gk["vega"] / S0})
    pd.DataFrame(rows).to_csv(out_dir / "note_greeks_profile.csv", index=False, float_format="%.6f")
    g["cond_vol"].to_csv(out_dir / "garch_conditional_vol.csv", float_format="%.6f")

    # back-test of the note on the index's own history (rolling start dates)
    bt = []
    lv = run.level
    days = int(round(T * 252))
    for i in range(0, len(lv) - days, 5):
        s, e = lv.iloc[i], lv.iloc[i + days]
        perf = e / s - 1
        bt.append({"start": lv.index[i], "index_return": perf,
                   "note_return": p2p.protection + p2p.participation * max(perf, 0) - 1})
    bt = pd.DataFrame(bt)
    bt.to_csv(out_dir / "note_backtest.csv", index=False, float_format="%.6f")

    summary = {
        "spot": S0, "rate": r, "div_yield": q,
        "garch": {k: float(v) for k, v in g.items() if k not in ("cond_vol",)},
        "pricing_vol_3y": sigma, "ewma_vol_now": ewma_now,
        "p2p": {"participation": p2p.participation, "closed_form_value": cf["value"],
                "mc_value": mc["value"], "mc_std_err": mc["std_err"], "zcb": cf["zcb"],
                "call_per_100": cf["option_per_100pct"]},
        "p2p_cap50": {"participation": capped.participation},
        "asian6": {"participation": asian.participation, "greeks": greeks},
        "range_accrual_1y_90_110": {"coupon": ra.coupon, "cf_value": ra_cf["value"], "mc_value": ra_mc["value"],
                                    "mc_std_err": ra_mc["std_err"],
                                    "expected_days_in_range": ra_cf["expected_days_in_range"]},
        "note_backtest": {"windows": int(len(bt)),
                          "pct_above_protection": float((bt["note_return"] > 0).mean()) if len(bt) else None,
                          "median_note_return": float(bt["note_return"].median()) if len(bt) else None},
    }
    return {
        "summary": summary,
        "note_obj": asian,
        "ts_pricing": {"r": r, "q": q, "sigma": sigma, "zcb": cf["zcb"],
                       "option_unit": unit["value"], "value": ppn_mc(asian, S0, r, q, sigma, n, seed)["value"],
                       "std_err": ppn_mc(asian, S0, r, q, sigma, n, seed)["std_err"]},
        "ts_greeks": greeks,
        "excel_inputs": {"S0": S0, "T": T, "r": r, "q": q, "sigma": sigma, "protection": 1.0,
                         "fees": pc.note_fees, "participation_cf": part_cf},
    }
