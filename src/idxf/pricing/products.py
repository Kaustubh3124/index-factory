"""Products written on the index.

1. Principal-protected note (PPN), the way Indian market-linked debentures are
   usually built: zero-coupon bond + participation in an at-the-money call, with an
   optional cap and an "Asian tail" (final level = average of the last few monthly
   fixings). The averaging lowers the option cost, which buys more participation.

2. Range accrual: daily coupon accrues for every fixing the index closes inside
   [L, U]. Priced as a strip of digital call spreads in closed form and checked by MC.

Notional is 100 everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import bs
from .mc import gbm_paths

TD = 252


@dataclass
class PPN:
    T: float = 3.0
    protection: float = 1.00          # fraction of notional returned at maturity
    participation: float = 1.0
    cap: float | None = None          # max index return paid, e.g. 0.60
    avg_months: int = 0               # 0 = point-to-point, 6 = average of last 6 monthly closes
    fees: float = 0.015

    def fixing_times(self):
        if self.avg_months <= 0:
            return np.array([self.T])
        return np.array([self.T - (self.avg_months - 1 - i) / 12 for i in range(self.avg_months)])

    def payoff(self, paths, S0):
        final = paths[:, -self.fixing_times().size:].mean(axis=1) if self.avg_months else paths[:, -1]
        perf = final / S0 - 1.0
        up = np.maximum(perf, 0.0)
        if self.cap is not None:
            up = np.minimum(up, self.cap)
        return 100.0 * (self.protection + self.participation * up)


def ppn_closed_form(note: PPN, S0, r, q, sigma) -> dict:
    """Point-to-point only (avg_months == 0)."""
    if note.avg_months:
        raise ValueError("closed form only for point-to-point notes")
    zcb = 100 * note.protection * np.exp(-r * note.T)
    call = bs.price(S0, S0, note.T, r, q, sigma) / S0 * 100
    if note.cap is not None:
        call -= bs.price(S0, S0 * (1 + note.cap), note.T, r, q, sigma) / S0 * 100
    return {"zcb": float(zcb), "option_per_100pct": float(call),
            "value": float(zcb + note.participation * call)}


def ppn_mc(note: PPN, S0, r, q, sigma, n_paths=200_000, seed=7) -> dict:
    times = _grid(note)
    paths = gbm_paths(S0, r, q, sigma, times, n_paths, seed)
    k = note.fixing_times().size
    fix = paths[:, -k:]
    pv = np.exp(-r * note.T) * note.payoff(fix, S0)
    m = len(pv) // 2
    se = 0.5 * (pv[:m] + pv[m:2 * m]).std(ddof=1) / np.sqrt(m)
    return {"value": float(pv.mean()), "std_err": float(se)}


def solve_participation(note: PPN, S0, r, q, sigma, n_paths=200_000, seed=7) -> PPN:
    """Participation that makes the note worth (1 - fees) * 100 to the investor.

    Value is linear in participation, so two MC runs with the same seed pin it down exactly.
    """
    target = 100 * (1 - note.fees)
    v0 = ppn_mc(replace(note, participation=0.0), S0, r, q, sigma, n_paths, seed)["value"]
    v1 = ppn_mc(replace(note, participation=1.0), S0, r, q, sigma, n_paths, seed)["value"]
    if v1 - v0 <= 0:
        raise ValueError("option leg has no value")
    p = (target - v0) / (v1 - v0)
    if p <= 0:
        raise ValueError("zero-coupon bond alone costs more than the budget - protection too high for these rates")
    return replace(note, participation=float(p))


def ppn_greeks(note: PPN, S0, r, q, sigma, n_paths=200_000, seed=7) -> dict:
    """Bump-and-reval with common random numbers. Delta/gamma per 1% spot move,
    vega per vol point, rho per 1bp, theta per calendar day."""
    f = lambda **kw: ppn_mc(kw.get("note", note), kw.get("S", S0), kw.get("r", r), q,
                            kw.get("sigma", sigma), n_paths, seed)["value"]
    # spot bump: keep strike fixed at the original S0 -> bump paths, not S0 in the payoff
    def val_spot(S):
        times = _grid(note)
        paths = gbm_paths(S, r, q, sigma, times, n_paths, seed)
        fix = paths[:, -note.fixing_times().size:]
        return float((np.exp(-r * note.T) * note.payoff(fix, S0)).mean())

    h = 0.01 * S0
    up, mid, dn = val_spot(S0 + h), val_spot(S0), val_spot(S0 - h)
    vega = (f(sigma=sigma + 0.01) - f(sigma=sigma - 0.01)) / 2
    rho = (f(r=r + 0.0001) - f(r=r - 0.0001)) / 2
    shorter = replace(note, T=note.T - 1 / 365)
    theta = f(note=shorter) - mid
    return {"value": mid, "delta_1pct": (up - dn) / 2, "gamma_1pct": up - 2 * mid + dn,
            "vega_1vol": vega, "rho_1bp": rho, "theta_1d": theta}


def _grid(note: PPN):
    # GBM can be sampled exactly at any set of dates, and the payoff only looks at
    # the fixings, so simulating just the fixing dates is exact (no discretisation error)
    return note.fixing_times()


# ---------------------------------------------------------------- range accrual

@dataclass
class RangeAccrual:
    T: float = 1.0
    lower: float = 0.90               # as a fraction of initial level
    upper: float = 1.10
    coupon: float = 0.10              # annual coupon if every fixing is in range
    protection: float = 1.0


def range_accrual_closed_form(ra: RangeAccrual, S0, r, q, sigma) -> dict:
    """Coupon leg = sum over daily fixings t_i of (c/252) * DF(T) * P(L <= S_ti <= U).
    Coupons are paid at maturity, so every fixing is discounted from T, and the
    in-range probability uses the forward measure to t_i (digital spread / DF(t_i))."""
    ts = np.arange(1, int(round(ra.T * TD)) + 1) / TD
    L, U = ra.lower * S0, ra.upper * S0
    p_in = (bs.digital(S0, L, ts, r, q, sigma) - bs.digital(S0, U, ts, r, q, sigma)) / np.exp(-r * ts)
    coupon_leg = 100 * ra.coupon / TD * np.exp(-r * ra.T) * p_in.sum()
    zcb = 100 * ra.protection * np.exp(-r * ra.T)
    return {"coupon_leg": float(coupon_leg), "zcb": float(zcb), "value": float(zcb + coupon_leg),
            "expected_days_in_range": float(p_in.sum())}


def range_accrual_mc(ra: RangeAccrual, S0, r, q, sigma, n_paths=100_000, seed=7) -> dict:
    ts = np.arange(1, int(round(ra.T * TD)) + 1) / TD
    paths = gbm_paths(S0, r, q, sigma, ts, n_paths, seed)
    inside = ((paths >= ra.lower * S0) & (paths <= ra.upper * S0)).sum(axis=1)
    pv = np.exp(-r * ra.T) * 100 * (ra.protection + ra.coupon / TD * inside)
    m = len(pv) // 2
    se = 0.5 * (pv[:m] + pv[m:2 * m]).std(ddof=1) / np.sqrt(m)
    return {"value": float(pv.mean()), "std_err": float(se), "avg_days_in_range": float(inside.mean())}


def solve_range_coupon(ra: RangeAccrual, S0, r, q, sigma, fees=0.01) -> RangeAccrual:
    unit = range_accrual_closed_form(replace(ra, coupon=1.0), S0, r, q, sigma)
    target = 100 * (1 - fees) - unit["zcb"]
    return replace(ra, coupon=float(target / unit["coupon_leg"]))
