"""Performance stats, Brinson attribution and a simple factor regression."""
from __future__ import annotations

import numpy as np
import pandas as pd

TD = 252


def perf_stats(level: pd.Series, rf: float = 0.0) -> dict:
    r = level.pct_change().dropna()
    years = (level.index[-1] - level.index[0]).days / 365.25
    cagr = (level.iloc[-1] / level.iloc[0]) ** (1 / years) - 1
    vol = r.std() * np.sqrt(TD)
    dd = level / level.cummax() - 1
    trough = dd.idxmin()
    peak = level.loc[:trough].idxmax()
    return {
        "start": str(level.index[0].date()), "end": str(level.index[-1].date()),
        "total_return": float(level.iloc[-1] / level.iloc[0] - 1),
        "cagr": float(cagr), "vol": float(vol),
        "sharpe": float((cagr - rf) / vol) if vol else np.nan,
        "max_drawdown": float(dd.min()),
        "dd_peak": str(peak.date()), "dd_trough": str(trough.date()),
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else np.nan,
        "best_day": float(r.max()), "worst_day": float(r.min()),
        "hit_rate_monthly": float((level.resample("ME").last().pct_change().dropna() > 0).mean()),
    }


def relative_stats(level: pd.Series, bench: pd.Series) -> dict:
    a, b = level.align(bench, join="inner")
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    active = ra - rb
    te = active.std() * np.sqrt(TD)
    beta = np.cov(ra, rb)[0, 1] / rb.var()
    years = (a.index[-1] - a.index[0]).days / 365.25
    ann_a = (a.iloc[-1] / a.iloc[0]) ** (1 / years) - 1
    ann_b = (b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1
    up = rb > 0
    return {
        "excess_cagr": float(ann_a - ann_b),
        "tracking_error": float(te),
        "info_ratio": float((ann_a - ann_b) / te),
        "beta": float(beta),
        "correlation": float(np.corrcoef(ra, rb)[0, 1]),
        "upside_capture": float(ra[up].mean() / rb[up].mean()),
        "downside_capture": float(ra[~up].mean() / rb[~up].mean()),
    }


def calendar_returns(level: pd.Series) -> pd.Series:
    y = level.resample("YE").last()
    first = pd.Series([level.iloc[0]], index=[level.index[0]])
    return pd.concat([first, y]).pct_change().dropna().rename(lambda d: d.year)


def brinson_fachler(periods: list[tuple], sectors: pd.Series) -> pd.DataFrame:
    """Sector Brinson-Fachler attribution, one row per (period, sector).

    periods: list of (start, end, w_port, w_bench, stock_returns) where the weights
    are start-of-period stock weights and stock_returns are the period returns.
    """
    rows = []
    for start, end, wp, wb, r in periods:
        sec = sectors.reindex(r.index).fillna("Unclassified")
        wp = wp.reindex(r.index).fillna(0.0)
        wb = wb.reindex(r.index).fillna(0.0)
        Wp, Wb = wp.groupby(sec).sum(), wb.groupby(sec).sum()
        Rp = (wp * r).groupby(sec).sum() / Wp.replace(0, np.nan)
        Rb = (wb * r).groupby(sec).sum() / Wb.replace(0, np.nan)
        Rb_tot = float((wb * r).sum())
        idx = Wp.index.union(Wb.index)
        Wp, Wb = Wp.reindex(idx).fillna(0), Wb.reindex(idx).fillna(0)
        Rp, Rb = Rp.reindex(idx), Rb.reindex(idx)
        alloc = (Wp - Wb) * (Rb.fillna(Rb_tot) - Rb_tot)
        select_ = Wb * (Rp - Rb).fillna(0)
        inter = (Wp - Wb) * (Rp - Rb).fillna(0)
        # sectors held by the portfolio but not the benchmark: all of it is selection+allocation
        only_p = (Wb == 0) & (Wp > 0)
        inter[only_p] = 0.0
        select_[only_p] = (Wp * (Rp - Rb_tot))[only_p]
        alloc[only_p] = 0.0
        for s in idx:
            rows.append({"start": start, "end": end, "sector": s,
                         "w_port": Wp[s], "w_bench": Wb[s],
                         "allocation": alloc[s], "selection": select_[s], "interaction": inter[s]})
    return pd.DataFrame(rows)


def link_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Simple additive summary over periods (no Carino smoothing - good enough for quarterly
    periods; the residual vs geometric active return is reported separately)."""
    g = df.groupby("sector")[["allocation", "selection", "interaction"]].sum()
    g["total"] = g.sum(axis=1)
    return g.sort_values("total", ascending=False)


def factor_regression(index_ret: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """OLS of daily index returns on factor returns, with Newey-West (5 lag) t-stats."""
    df = pd.concat([index_ret.rename("y"), factors], axis=1).dropna()
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df.drop(columns="y").to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    L = 5
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, L + 1):
        wl = 1 - l / (L + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += wl * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    names = ["alpha"] + list(df.columns[1:])
    out = pd.DataFrame({"coef": beta, "t_stat": beta / se}, index=names)
    out.loc["alpha", "coef_annual"] = beta[0] * TD
    out.attrs["r2"] = 1 - e.var() / y.var()
    return out
