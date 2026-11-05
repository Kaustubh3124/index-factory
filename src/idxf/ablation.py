"""Run the earlier / alternative versions of the rulebook on the same data.

Each variant is the production config with a few parameters overridden. This is
how the final rule set was justified: every rule in the methodology has to earn
its place against the version without it.
"""
from __future__ import annotations

import copy

import pandas as pd

from . import analytics as an
from .costs import net_of_cost, rebalance_costs
from .engine import calculate, prepare

VARIANTS = {
    "A_mom_only_ew_nobuffer": {"scores.mom_weight": 1.0, "scores.lowvol_weight": 0.0,
                               "selection.buffer_in": 30, "selection.buffer_out": 30,
                               "weighting.scheme": "equal", "weighting.sector_cap": 1.0},
    "B_blend_ew_nobuffer": {"selection.buffer_in": 30, "selection.buffer_out": 30,
                            "weighting.scheme": "equal", "weighting.sector_cap": 1.0},
    "C_blend_ivol_nobuffer": {"selection.buffer_in": 30, "selection.buffer_out": 30,
                              "weighting.sector_cap": 1.0},
    "D_blend_ivol_buffer_nosectorcap": {"weighting.sector_cap": 1.0},
    "E_final_IMLV30": {},
    "F_lowvol_only": {"scores.mom_weight": 0.0, "scores.lowvol_weight": 1.0},
}


def with_overrides(cfg, overrides: dict):
    """Copy of the config with dotted-path overrides, e.g. {"selection.n_constituents": 20}."""
    c = copy.deepcopy(cfg)
    for k, v in overrides.items():
        obj = c
        *path, last = k.split(".")
        for p in path:
            obj = getattr(obj, p)
        setattr(obj, last, v)
    return c


def run_variants(mkt, cfg, ca, variants=VARIANTS) -> tuple[pd.DataFrame, dict]:
    _, _, rets, _ = prepare(mkt, ca)
    rows, levels = [], {}
    for name, ov in variants.items():
        c = with_overrides(cfg, ov)
        run = calculate(mkt, c, ca)
        costs = rebalance_costs(run, mkt, rets, c)
        net = net_of_cost(run.level, costs)
        st = an.perf_stats(run.level)
        rel = an.relative_stats(run.level, mkt.benchmark)
        sec_max = max(
            run.weights.loc[r.review.effective, r.weights.index].groupby(mkt.sectors.reindex(r.weights.index)).sum().max()
            for r in run.reviews)
        rows.append({
            "variant": name, "cagr": st["cagr"], "net_cagr": an.perf_stats(net)["cagr"], "vol": st["vol"],
            "sharpe": st["sharpe"], "max_dd": st["max_drawdown"], "beta": rel["beta"],
            "avg_turnover": costs["one_way_turnover"].mean(), "avg_cost_bps": costs["cost_bps"].mean(),
            "max_sector_weight": sec_max,
            "worst_quarter": run.level.resample("QE").last().pct_change().min(),
        })
        levels[name] = run.level
    return pd.DataFrame(rows).set_index("variant"), levels
