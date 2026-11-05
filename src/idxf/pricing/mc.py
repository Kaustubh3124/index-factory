"""Monte Carlo under GBM. Antithetic variates, fixed seed so Greeks by bump-and-reval
use common random numbers (otherwise the noise swamps the bump)."""
from __future__ import annotations

import numpy as np


def gbm_paths(S0, r, q, sigma, times, n_paths, seed=7, antithetic=True):
    """Simulate S at the given times (years, increasing). Returns array (n_paths, len(times))."""
    times = np.asarray(times, dtype=float)
    dt = np.diff(np.concatenate([[0.0], times]))
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, len(times)))
    if antithetic:
        z = np.vstack([z, -z])
    inc = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    return S0 * np.exp(np.cumsum(inc, axis=1))


def mc_price(payoff_fn, S0, r, q, sigma, times, T, n_paths=200_000, seed=7):
    """payoff_fn(paths) -> payoff at T per path. Returns (price, std_error)."""
    paths = gbm_paths(S0, r, q, sigma, times, n_paths, seed)
    pv = np.exp(-r * T) * payoff_fn(paths)
    m = len(pv) // 2
    # antithetic pairs are not independent - average them before taking the std error
    pairs = 0.5 * (pv[:m] + pv[m:2 * m])
    return float(pv.mean()), float(pairs.std(ddof=1) / np.sqrt(m))
