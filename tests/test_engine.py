import numpy as np
import pandas as pd
import pytest

from idxf import corpactions as cax
from idxf.engine import calculate, prepare
from idxf.recon import reconcile
from idxf.shadow import calculate_shadow

from conftest import BONUS, DEAD, SPECIAL_DIV, SPLIT


@pytest.fixture(scope="module")
def run(market, cfg, ca_table):
    return calculate(market, cfg, ca_table)


def test_base_value(run, cfg):
    assert run.level.iloc[0] == pytest.approx(cfg.base_value)


def test_production_matches_shadow(run, market, cfg, ca_table):
    factors, divs, rets, _ = prepare(market, ca_table)
    tw = {r.review.effective: r.weights for r in run.reviews}
    sh = calculate_shadow(market, cfg, rets, tw, factors, divs)
    rec = reconcile(run.level, sh)
    assert rec["passed"], rec


def test_split_does_not_move_index(run, market, cfg):
    """Rerun without telling the engine about the split: the index must jump on the
    ex-date if the stock is a constituent. With the CA file it must not."""
    t, d, k = SPLIT
    d = pd.Timestamp(d)
    held = run.weights.loc[market.offset(d, -1)].get(t, 0.0) > 0
    if not held:
        pytest.skip("split stock not a constituent on the ex-date in this seed")
    w = run.weights.loc[market.offset(d, -1), t]
    true_ret = market.close.loc[d, t] * k / market.close.loc[market.offset(d, -1), t] - 1
    no_ca = calculate(market, cfg, pd.DataFrame(columns=cax.CA_COLUMNS))
    jump = no_ca.level.pct_change().loc[d] - run.level.pct_change().loc[d]
    assert jump == pytest.approx(w * ((1 + true_ret) / k - (1 + true_ret)), rel=1e-6)


def test_split_scales_shares(run):
    t, d, k = SPLIT
    ev = [e for e in run.events if e["type"] == "share_adjustment" and e["ticker"] == t]
    for e in ev:
        assert e["shares_after"] == pytest.approx(e["shares_before"] * k)


def test_special_dividend_moves_divisor_only_if_held(run, market):
    t, d, amt = SPECIAL_DIV
    d = pd.Timestamp(d)
    held = run.weights.loc[market.offset(d, -1)].get(t, 0.0) > 0
    div_change = run.divisor.loc[d] / run.divisor.loc[market.offset(d, -1)] - 1
    if held:
        assert div_change < 0
        ev = [e for e in run.events if e["type"] == "special_dividend"]
        assert len(ev) == 1
    else:
        assert abs(div_change) < 1e-12


def test_dead_stock_removed(run, market, cfg):
    t, d = DEAD
    after = market.offset(pd.Timestamp(d), cfg.data_quality.stale_days + 1)
    assert run.weights.loc[after:].get(t, pd.Series(0.0)).max() == 0.0


def test_weights_sum_to_one(run):
    s = run.weights.sum(axis=1)
    assert np.allclose(s, 1.0)


def test_rebalance_keeps_level(run):
    # level on an effective date is computed before the rebalance; the next day's
    # return must equal the new weights times stock returns - checked by recon,
    # here we just make sure no review created a jump in the divisor
    d = run.divisor
    jumps = (d / d.shift(1) - 1).abs().dropna()
    rebal_days = list(run.rebalance_trades)
    assert jumps.loc[jumps.index.isin(rebal_days)].max() < 1e-12


def test_detector_finds_planted_actions(market):
    found = cax.detect_candidates(market.open, market.last, market.close, market.volume,
                                  market.traded, market.benchmark)
    got = {(r.ticker, r.ex_date.strftime("%Y-%m-%d"), r.factor) for r in found.itertuples()
           if r.suggested == "split_or_bonus"}
    assert (SPLIT[0], SPLIT[1], SPLIT[2]) in got
    assert (BONUS[0], BONUS[1], BONUS[2]) in got


def test_shadow_without_div_handling_breaks(run, market, cfg, ca_table):
    """Sanity check on the recon itself: the naive shadow (no ex-weight step) must fail,
    otherwise the recon would not be able to catch this class of bug."""
    _, _, rets, _ = prepare(market, ca_table)
    tw = {r.review.effective: r.weights for r in run.reviews}
    rec = reconcile(run.level, calculate_shadow(market, cfg, rets, tw))
    assert not rec["passed"]
    assert rec["first_break"] == SPECIAL_DIV[1]


def test_adjusted_prices_continuous(market, ca_table):
    f = cax.share_factors(ca_table, market.calendar, market.tickers)
    d = cax.special_dividends(ca_table, market.calendar, market.tickers)
    adj = cax.adjusted_prices(market.close, cax.price_adjustment(market.close, f, d))
    t, d, k = SPLIT
    i = market.calendar.get_loc(pd.Timestamp(d))
    r_adj = adj[t].iloc[i] / adj[t].iloc[i - 1] - 1
    assert abs(r_adj) < 0.15
