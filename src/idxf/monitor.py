"""Pro-forma review monitor.

Runs the full selection on any date as if it were the determination date, so you
can see ahead of time who is likely to come in / go out, how close names are to the
buffer thresholds, and how big the trades would be for a given tracking AUM.
This is the kind of sheet an index desk (or the people trading against index
flows) look at in the weeks before a review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .selection import factor_scores, select
from .universe import eligible_universe
from .weighting import target_weights


def proforma(mkt, cfg, rets, adj_close, asof: pd.Timestamp, current_weights: pd.Series,
             aum_cr: float | None = None) -> dict:
    aum = (aum_cr or cfg.costs.tracking_aum_cr) * 1e7
    uni = eligible_universe(mkt, asof, cfg)
    names = uni.index[uni["eligible"]]
    sc = factor_scores(adj_close, rets, asof, names, cfg)
    incumbents = set(current_weights[current_weights > 0].index)
    chosen = select(sc, incumbents, cfg)

    vol = rets.loc[:asof, chosen].iloc[-cfg.weighting.vol_window:].std() * np.sqrt(252)
    w_new = target_weights(vol, mkt.sectors, cfg)

    allnames = sorted(set(chosen) | incumbents)
    w_old = current_weights.reindex(allnames).fillna(0.0)
    w_tgt = w_new.reindex(allnames).fillna(0.0)
    adv = mkt.value.loc[mkt.window(asof, 63), allnames].mean()

    t = pd.DataFrame({
        "rank": sc["rank"].reindex(allnames),
        "score": sc["score"].reindex(allnames),
        "w_current": w_old, "w_proforma": w_tgt,
        "trade_cr": (w_tgt - w_old) * aum / 1e7,
        "adv_cr": adv / 1e7,
    })
    t["days_of_adv"] = (t["trade_cr"].abs() / t["adv_cr"]).round(3)
    t["status"] = np.select(
        [(t.w_current == 0) & (t.w_proforma > 0), (t.w_current > 0) & (t.w_proforma == 0)],
        ["ADD", "DELETE"], default="")

    # names sitting near the thresholds - the ones that can flip before the real cut-off
    sel = cfg.selection
    edge = sc[(sc["rank"] > sel.buffer_in - 5) & (sc["rank"] <= sel.buffer_out + 5)].copy()
    edge["incumbent"] = edge.index.isin(incumbents)
    edge["in_proforma"] = edge.index.isin(chosen)

    return {"asof": asof, "trades": t.sort_values("trade_cr"), "watchlist": edge,
            "one_way_turnover": float(0.5 * (w_tgt - w_old).abs().sum())}
