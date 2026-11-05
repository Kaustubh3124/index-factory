"""Shadow calculator: chains CA-adjusted constituent returns with drifting weights.

Deliberately shares no code with engine.py beyond the input data. See recon.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Market, stale_run


def calculate_shadow(mkt: Market, cfg, rets: pd.DataFrame, target_weights: dict,
                     factors: pd.DataFrame | None = None, divs: pd.DataFrame | None = None) -> pd.Series:
    """target_weights: {effective_date: pd.Series of weights}

    Special dividends: a price-return index treats the cash as reinvested across the
    whole index at ex-prices (that is what the divisor change in the production calc
    does). In return space that means the paying stock's weight is cut to its
    theoretical ex-value before the day's returns are applied.
    """
    cal = mkt.calendar[mkt.calendar >= cfg.base_date]
    R = np.nan_to_num(rets.loc[cal].to_numpy(dtype=float))
    if divs is not None and factors is not None:
        prev = mkt.close.shift(1)
        exm = (1.0 - divs * factors / prev).loc[cal].fillna(1.0).to_numpy()
    else:
        exm = np.ones_like(R)
    S = stale_run(mkt.traded).loc[cal].to_numpy()
    cols = {t: i for i, t in enumerate(rets.columns)}
    n = R.shape[1]

    w = np.zeros(n)
    lvl = np.empty(len(cal))
    for k, d in enumerate(cal):
        if k == 0:
            lvl[k] = cfg.base_value
        else:
            if (exm[k] != 1.0).any() and (w[exm[k] != 1.0] > 0).any():
                w = w * exm[k]
                w = w / w.sum()
            g = float(w @ R[k])
            lvl[k] = lvl[k - 1] * (1.0 + g)
            w = w * (1.0 + R[k]) / (1.0 + g)
            dead = (w > 0) & (S[k] >= cfg.data_quality.stale_days)
            if dead.any():
                w[dead] = 0.0
                w = w / w.sum()
        if d in target_weights:
            w = np.zeros(n)
            for t, x in target_weights[d].items():
                w[cols[t]] = x
    return pd.Series(lvl, index=cal, name="shadow")
