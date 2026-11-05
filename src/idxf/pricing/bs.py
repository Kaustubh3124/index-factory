"""Black-Scholes-Merton with continuous dividend yield. Closed forms + Greeks.

Greeks are per unit of underlying, vega per 1.00 change in vol (divide by 100 for
"per vol point"), theta per year (divide by 365 for per calendar day).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

N, n = norm.cdf, norm.pdf


def _d1d2(S, K, T, r, q, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vs = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vs
    return d1, d1 - vs


def price(S, K, T, r, q, sigma, kind="call"):
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if kind == "call":
        return S * df_q * N(d1) - K * df_r * N(d2)
    return K * df_r * N(-d2) - S * df_q * N(-d1)


def digital(S, K, T, r, q, sigma, kind="call"):
    """Cash-or-nothing paying 1."""
    _, d2 = _d1d2(S, K, T, r, q, sigma)
    df_r = np.exp(-r * T)
    return df_r * (N(d2) if kind == "call" else N(-d2))


def greeks(S, K, T, r, q, sigma, kind="call") -> dict:
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    sqT = np.sqrt(T)
    gamma = df_q * n(d1) / (S * sigma * sqT)
    vega = S * df_q * n(d1) * sqT
    if kind == "call":
        delta = df_q * N(d1)
        theta = (-S * df_q * n(d1) * sigma / (2 * sqT) - r * K * df_r * N(d2) + q * S * df_q * N(d1))
        rho = K * T * df_r * N(d2)
    else:
        delta = -df_q * N(-d1)
        theta = (-S * df_q * n(d1) * sigma / (2 * sqT) + r * K * df_r * N(-d2) - q * S * df_q * N(-d1))
        rho = -K * T * df_r * N(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(target, S, K, T, r, q, kind="call", lo=1e-4, hi=3.0):
    f = lambda s: price(S, K, T, r, q, s, kind) - target
    return brentq(f, lo, hi, xtol=1e-10)
