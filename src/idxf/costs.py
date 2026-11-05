"""Turnover, trading cost and capacity at each rebalance.

Cost model per name (as a fraction of traded value):
    half_spread + k * sigma_daily * sqrt(trade_value / ADV)
i.e. the usual square-root impact model. With k around 0.1 and participation
below 10% of ADV this lands at a few bps for NIFTY-100 type names, which is in
line with what desks quote for large/mid cap India.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_costs(run, mkt, rets: pd.DataFrame, cfg) -> pd.DataFrame:
    c = cfg.costs
    aum = c.tracking_aum_cr * 1e7
    rows = []
    for d, tr in run.rebalance_trades.items():
        if d == run.level.index[0]:
            continue  # initial build, not a rebalance
        dw = (tr["w_after"] - tr["w_before"]).abs()
        win = mkt.window(d, 63)
        adv = mkt.value.loc[win, tr.index].mean()
        sig = rets.loc[win, tr.index].std()
        trade_val = dw * aum
        part = (trade_val / adv).replace([np.inf], np.nan)
        cost_frac = (c.half_spread_bps / 1e4) + c.impact_coef * sig * np.sqrt(part.fillna(0))
        cost = float((dw * cost_frac).sum())
        days_to_trade = part / c.max_participation
        rows.append({
            "date": d,
            "one_way_turnover": float(0.5 * dw.sum()),
            "n_adds": int(((tr["w_before"] == 0) & (tr["w_after"] > 0)).sum()),
            "n_deletes": int(((tr["w_before"] > 0) & (tr["w_after"] == 0)).sum()),
            "cost_bps": cost * 1e4,
            "max_participation": float(part.max()),
            "max_days_to_trade": float(days_to_trade.max()),
            "worst_name": str(days_to_trade.idxmax()) if days_to_trade.notna().any() else "",
        })
    return pd.DataFrame(rows).set_index("date")


def net_of_cost(level: pd.Series, costs: pd.DataFrame) -> pd.Series:
    """Apply each rebalance's cost as a one-off drag on the effective date close."""
    r = level.pct_change().fillna(0.0)
    drag = pd.Series(0.0, index=level.index)
    drag.loc[costs.index] = costs["cost_bps"] / 1e4
    net = (1 + r) * (1 - drag)
    net.iloc[0] = 1.0
    return level.iloc[0] * net.cumprod()


def capacity(run, mkt, cfg, participation: float | None = None) -> pd.Series:
    """AUM (INR cr) at which the largest trade of each rebalance needs one full day
    at the given participation rate."""
    part = participation or cfg.costs.max_participation
    out = {}
    for d, tr in run.rebalance_trades.items():
        if d == run.level.index[0]:
            continue
        dw = (tr["w_after"] - tr["w_before"]).abs()
        adv = mkt.value.loc[mkt.window(d, 63), tr.index].mean()
        ratio = (adv * part / dw.replace(0, np.nan)).min()
        out[d] = ratio / 1e7
    return pd.Series(out, name="capacity_cr")
