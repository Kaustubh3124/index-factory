"""Target weights (methodology section 6).

Inverse-volatility weights, then a stock cap / floor and a sector cap. Caps are
applied iteratively: clip the offenders, hand the excess to the uncapped names
pro rata, repeat until nothing breaches. With 30 names and an 8% cap it converges
in a handful of passes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_vol(vol: pd.Series) -> pd.Series:
    w = 1.0 / vol
    return w / w.sum()


def cap_weights(w: pd.Series, cap: float, floor: float = 0.0, max_iter: int = 100) -> pd.Series:
    w = w / w.sum()
    n = len(w)
    if cap * n < 1.0 - 1e-12:
        raise ValueError(f"cap {cap} infeasible for {n} names")
    if floor * n > 1.0 + 1e-12:
        raise ValueError(f"floor {floor} infeasible for {n} names")

    fixed = pd.Series(False, index=w.index)
    for _ in range(max_iter):
        over = (w > cap + 1e-12) & ~fixed
        under = (w < floor - 1e-12) & ~fixed
        if not over.any() and not under.any():
            break
        w[over] = cap
        w[under] = floor
        fixed |= over | under
        free = ~fixed
        target_free = 1.0 - w[fixed].sum()
        if free.any():
            w[free] = w[free] / w[free].sum() * target_free
    return w


def cap_sectors(w: pd.Series, sectors: pd.Series, cap: float, stock_cap: float,
                max_iter: int = 50) -> pd.Series:
    """Scale down sectors above `cap`, push the excess to other sectors pro rata,
    then re-apply the stock cap. Alternate until both hold."""
    w = w.copy()
    sec = sectors.reindex(w.index).fillna("Unclassified")
    for _ in range(max_iter):
        sw = w.groupby(sec).sum()
        bad = sw[sw > cap + 1e-9]
        if bad.empty and (w <= stock_cap + 1e-9).all():
            break
        capped_secs = set(bad.index)
        for s in capped_secs:
            m = sec == s
            w[m] *= cap / sw[s]
        others = ~sec.isin(capped_secs)
        if others.any():
            w[others] *= (1.0 - w[~others].sum()) / w[others].sum()
        w = _stock_cap_keep_sector(w, sec, stock_cap, capped_secs)
    return w / w.sum()


def _stock_cap_keep_sector(w, sec, cap, frozen_secs):
    # stock cap applied only inside sectors that are not pinned at the sector cap,
    # otherwise we'd undo the sector constraint we just enforced
    free = ~sec.isin(frozen_secs)
    if free.sum() == 0:
        return w
    budget = w[free].sum()
    sub = w[free] / budget
    eff_cap = min(1.0, cap / budget)
    if eff_cap * free.sum() >= 1.0:
        sub = cap_weights(sub, eff_cap)
    w[free] = sub * budget
    return w


def target_weights(vol: pd.Series, sectors: pd.Series, cfg) -> pd.Series:
    wc = cfg.weighting
    if wc.scheme == "equal":
        w = pd.Series(1.0 / len(vol), index=vol.index)
    else:
        w = inverse_vol(vol)
    w = cap_weights(w, wc.stock_cap, wc.stock_floor)
    if wc.sector_cap < 1.0:
        w = cap_sectors(w, sectors, wc.sector_cap, wc.stock_cap)
    # floor can be broken by the sector step if a sector is heavily scaled;
    # we accept that and report it rather than loop forever
    return w.sort_values(ascending=False)


def check_constraints(w: pd.Series, sectors: pd.Series, cfg, tol: float = 1e-6) -> dict:
    wc = cfg.weighting
    sw = w.groupby(sectors.reindex(w.index).fillna("Unclassified")).sum()
    return {
        "sum": float(w.sum()),
        "max_stock": float(w.max()),
        "min_stock": float(w.min()),
        "max_sector": float(sw.max()),
        "stock_cap_ok": bool(w.max() <= wc.stock_cap + tol),
        "sector_cap_ok": bool(sw.max() <= wc.sector_cap + tol),
        "n": int((w > 0).sum()),
        "effective_n": float(1.0 / np.sum(w.values ** 2)),
    }
