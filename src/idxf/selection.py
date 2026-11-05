"""Factor scores and constituent selection (methodology sections 4 and 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _z(x: pd.Series, clip: float) -> pd.Series:
    z = (x - x.mean()) / x.std(ddof=0)
    return z.clip(-clip, clip)


def factor_scores(adj_close: pd.DataFrame, rets: pd.DataFrame, d: pd.Timestamp,
                  names: pd.Index, cfg) -> pd.DataFrame:
    s = cfg.scores
    px = adj_close.loc[:d, names]
    r = rets.loc[:d, names]
    if len(px) <= s.mom_long:
        raise ValueError(f"need more than {s.mom_long} days of prices before {d.date()}, have {len(px)}")

    p_skip = px.iloc[-1 - s.mom_skip]
    mom12 = p_skip / px.iloc[-1 - s.mom_long] - 1.0
    mom6 = p_skip / px.iloc[-1 - s.mom_short] - 1.0
    vol = r.iloc[-s.vol_window:].std() * np.sqrt(252)

    # risk-adjusted momentum: average of the 12m and 6m z-scores of return/vol
    ram = 0.5 * _z(mom12 / vol, s.winsor) + 0.5 * _z(mom6 / vol, s.winsor)
    z_mom = _z(ram, s.winsor)
    z_lv = _z(-vol, s.winsor)
    composite = s.mom_weight * z_mom + s.lowvol_weight * z_lv

    out = pd.DataFrame({"mom12": mom12, "mom6": mom6, "vol": vol,
                        "z_mom": z_mom, "z_lowvol": z_lv, "score": composite})
    # ties broken by liquidity rank upstream, so a stable sort is enough here
    out["rank"] = out["score"].rank(ascending=False, method="first")
    return out.sort_values("rank")


def select(scores: pd.DataFrame, incumbents: set[str], cfg) -> list[str]:
    """Buffer rule: top `buffer_in` always in, incumbents kept while rank <= buffer_out,
    remaining slots filled in rank order."""
    sel = cfg.selection
    n = sel.n_constituents
    ranked = scores["rank"].sort_values()

    chosen = [t for t, rk in ranked.items() if rk <= sel.buffer_in]
    keep = [t for t, rk in ranked.items()
            if t in incumbents and sel.buffer_in < rk <= sel.buffer_out]
    for t in keep:
        if len(chosen) >= n:
            break
        chosen.append(t)
    for t in ranked.index:
        if len(chosen) >= n:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen[:n]
