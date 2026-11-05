import numpy as np
import pytest

from idxf.pricing import bs
from idxf.pricing.products import (PPN, RangeAccrual, ppn_closed_form, ppn_greeks, ppn_mc,
                                   range_accrual_closed_form, range_accrual_mc, solve_participation)

S, K, T, r, q, vol = 1000.0, 1050.0, 1.5, 0.0675, 0.012, 0.18


def test_put_call_parity():
    c = bs.price(S, K, T, r, q, vol, "call")
    p = bs.price(S, K, T, r, q, vol, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_vs_finite_difference(kind):
    g = bs.greeks(S, K, T, r, q, vol, kind)
    h = 1e-3
    f = lambda **kw: bs.price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), q, kw.get("v", vol), kind)
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((f(S=S + 1) - 2 * f() + f(S=S - 1)), rel=1e-4)
    assert g["vega"] == pytest.approx((f(v=vol + h) - f(v=vol - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((f(r=r + 1e-5) - f(r=r - 1e-5)) / 2e-5, rel=1e-5)
    assert g["theta"] == pytest.approx(-(f(T=T + 1e-5) - f(T=T - 1e-5)) / 2e-5, rel=1e-5)


def test_implied_vol_roundtrip():
    c = bs.price(S, K, T, r, q, 0.2345)
    assert bs.implied_vol(c, S, K, T, r, q) == pytest.approx(0.2345, abs=1e-8)


def test_digital_is_minus_dcall_dk():
    h = 0.01
    d = (bs.price(S, K - h, T, r, q, vol) - bs.price(S, K + h, T, r, q, vol)) / (2 * h)
    assert bs.digital(S, K, T, r, q, vol) == pytest.approx(d, rel=1e-6)


@pytest.mark.parametrize("cap", [None, 0.5])
def test_ppn_mc_matches_closed_form(cap):
    note = PPN(T=3.0, participation=0.8, cap=cap)
    cf = ppn_closed_form(note, S, r, q, vol)["value"]
    mc = ppn_mc(note, S, r, q, vol, n_paths=400_000)
    assert abs(mc["value"] - cf) < 4 * mc["std_err"] + 1e-3


def test_averaging_cheapens_option():
    p2p = solve_participation(PPN(T=3.0), S, r, q, vol)
    avg = solve_participation(PPN(T=3.0, avg_months=6), S, r, q, vol)
    assert avg.participation > p2p.participation


def test_solved_note_prices_at_budget():
    note = solve_participation(PPN(T=3.0, fees=0.015), S, r, q, vol)
    assert ppn_mc(note, S, r, q, vol)["value"] == pytest.approx(98.5, abs=1e-6)


def test_note_delta_close_to_bs():
    note = PPN(T=3.0, participation=1.0)
    g = ppn_greeks(note, S, r, q, vol)
    bs_delta = bs.greeks(S, S, 3.0, r, q, vol)["delta"] * 100 / S * 0.01 * S
    assert g["delta_1pct"] == pytest.approx(bs_delta, rel=0.02)


def test_range_accrual_mc_vs_strip():
    ra = RangeAccrual(T=1.0, lower=0.9, upper=1.1, coupon=0.12)
    cf = range_accrual_closed_form(ra, S, r, q, vol)
    mc = range_accrual_mc(ra, S, r, q, vol, n_paths=100_000)
    assert abs(mc["value"] - cf["value"]) < 4 * mc["std_err"] + 1e-3
