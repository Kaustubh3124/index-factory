"""Volatility estimates for the index: EWMA and GARCH(1,1) fitted by MLE.

We do not have a listed options market on our own index, so the pricing vol has to
come from the history. GARCH gives a term structure (mean-reverting forecast),
which is what we actually need for a 3y note - a 1m realised vol number is not it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TD = 252


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    v = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v * TD)


def fit_garch(r: pd.Series) -> dict:
    x = (r - r.mean()).dropna().to_numpy()
    var0 = x.var()

    def nll(p):
        omega, alpha, beta = p
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        h = np.empty_like(x)
        h[0] = var0
        for t in range(1, len(x)):
            h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]
        return 0.5 * np.sum(np.log(h) + x ** 2 / h)

    p0 = [var0 * 0.05, 0.08, 0.90]
    res = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-8})
    omega, alpha, beta = res.x
    h = np.empty_like(x)
    h[0] = var0
    for t in range(1, len(x)):
        h[t] = omega + alpha * x[t - 1] ** 2 + beta * h[t - 1]
    h_next = omega + alpha * x[-1] ** 2 + beta * h[-1]
    lr_var = omega / (1 - alpha - beta)
    return {"omega": omega, "alpha": alpha, "beta": beta, "persistence": alpha + beta,
            "long_run_vol": float(np.sqrt(lr_var * TD)), "h_next": float(h_next),
            "half_life_days": float(np.log(0.5) / np.log(alpha + beta)),
            "cond_vol": pd.Series(np.sqrt(h * TD), index=r.dropna().index), "converged": bool(res.success)}


def garch_term_vol(g: dict, T_years: float) -> float:
    """Average annualised vol over [0, T] from the GARCH variance forecast."""
    k = g["persistence"]
    lr = g["long_run_vol"] ** 2 / TD
    days = max(1, int(round(T_years * TD)))
    i = np.arange(days)
    h = lr + (k ** i) * (g["h_next"] - lr)
    return float(np.sqrt(h.mean() * TD))
