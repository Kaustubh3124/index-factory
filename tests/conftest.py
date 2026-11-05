"""Synthetic market used by the tests. Small enough to run in a couple of seconds,
with the awkward cases planted on purpose: a 1:5 split, a bonus, a special dividend
and a stock that stops trading."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idxf.config import load_config
from idxf.data import Market

ROOT = Path(__file__).resolve().parents[1]

SPLIT = ("S007", "2020-03-16", 5.0)
BONUS = ("S011", "2020-09-14", 2.0)
SPECIAL_DIV = ("S003", "2020-11-16", 40.0)
DEAD = ("S015", "2020-08-03")   # stops trading from this date


@pytest.fixture(scope="session")
def cfg():
    c = load_config(ROOT / "config" / "imlv30.yaml")
    c.base_date = pd.Timestamp("2019-12-31")
    c.universe.min_median_value_cr = 1.0
    c.universe.top_n_by_liquidity = 40
    c.universe.min_history_days = 260
    c.selection.n_constituents = 15
    c.selection.buffer_in = 10
    c.selection.buffer_out = 20
    c.weighting.stock_cap = 0.12
    c.weighting.sector_cap = 0.40
    return c


@pytest.fixture(scope="session")
def market():
    rng = np.random.default_rng(42)
    cal = pd.bdate_range("2018-06-01", "2021-06-30")
    n = 50
    tickers = [f"S{i:03d}" for i in range(n)]
    vols = rng.uniform(0.18, 0.45, n)
    drift = rng.normal(0.10, 0.08, n)
    # make the stocks carrying the planted events attractive (strong, low vol) so
    # they end up in the index and the CA code paths actually run
    for i in (3, 7, 11, 15):
        vols[i], drift[i] = 0.12, 0.40
    mkt_f = rng.standard_normal(len(cal)) * 0.011
    eps = rng.standard_normal((len(cal), n)) * (vols / np.sqrt(252))
    r = 0.8 * mkt_f[:, None] + 0.6 * eps + (drift / 252)
    px = 500 * np.exp(np.cumsum(r, axis=0))
    close = pd.DataFrame(px, index=cal, columns=tickers)

    # plant corporate actions in the raw prices
    t, d, k = SPLIT
    close.loc[d:, t] /= k
    t, d, k = BONUS
    close.loc[d:, t] /= k
    t, d, amt = SPECIAL_DIV
    close.loc[d:, t] -= amt

    traded = pd.DataFrame(True, index=cal, columns=tickers)
    t, d = DEAD
    traded.loc[d:, t] = False
    close = close.where(traded).ffill()

    volume = pd.DataFrame(rng.uniform(2e5, 2e6, (len(cal), n)), index=cal, columns=tickers).where(traded, 0.0)
    value = (close * volume).where(traded, 0.0)
    bench = pd.Series(10000 * np.exp(np.cumsum(mkt_f)), index=cal)
    sectors = pd.Series([["Financials", "IT", "Energy", "Materials", "Industrials", "Health Care"][i % 6]
                         for i in range(n)], index=tickers)
    opn = close * (1 + rng.normal(0, 0.002, close.shape))
    return Market(close=close, last=close, open=opn, volume=volume, value=value, traded=traded,
                  calendar=cal, benchmark=bench, sectors=sectors)


@pytest.fixture(scope="session")
def ca_table():
    return pd.DataFrame([
        {"ex_date": pd.Timestamp(SPLIT[1]), "ticker": SPLIT[0], "action": "split", "factor": SPLIT[2], "amount": 0.0,
         "source": "test", "note": ""},
        {"ex_date": pd.Timestamp(BONUS[1]), "ticker": BONUS[0], "action": "bonus", "factor": BONUS[2], "amount": 0.0,
         "source": "test", "note": ""},
        {"ex_date": pd.Timestamp(SPECIAL_DIV[1]), "ticker": SPECIAL_DIV[0], "action": "special_dividend",
         "factor": 1.0, "amount": SPECIAL_DIV[2], "source": "test", "note": ""},
    ])
