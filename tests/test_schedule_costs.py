import numpy as np
import pandas as pd
import pytest

from idxf.costs import capacity, net_of_cost, rebalance_costs
from idxf.engine import calculate, prepare
from idxf.monitor import proforma
from idxf.schedule import review_dates


def test_review_dates_quarter_ends(market, cfg):
    revs = review_dates(market.calendar, cfg)
    assert revs[0].initial and revs[0].effective == cfg.base_date
    for r in revs[1:]:
        assert r.effective.month in (3, 6, 9, 12)
        # effective date is the last trading day of its month
        nxt = market.calendar[market.calendar.get_loc(r.effective) + 1]
        assert nxt.month != r.effective.month
        i = market.calendar.get_loc(r.effective)
        assert market.calendar.get_loc(r.determination) == i - cfg.schedule.determination_lag
        assert r.determination < r.announcement < r.effective


def test_bad_base_date_raises(market, cfg):
    import copy
    c = copy.deepcopy(cfg)
    c.base_date = pd.Timestamp("2019-12-17")
    with pytest.raises(ValueError):
        review_dates(market.calendar, c)


@pytest.fixture(scope="module")
def run_and_rets(market, cfg, ca_table):
    _, _, rets, adj = prepare(market, ca_table)
    return calculate(market, cfg, ca_table), rets, adj


def test_costs_positive_and_net_below_gross(run_and_rets, market, cfg):
    run, rets, _ = run_and_rets
    c = rebalance_costs(run, market, rets, cfg)
    assert (c["cost_bps"] > 0).all()
    assert ((c["one_way_turnover"] >= 0) & (c["one_way_turnover"] <= 1)).all()
    net = net_of_cost(run.level, c)
    assert net.iloc[-1] < run.level.iloc[-1]
    # cost only bites on review days
    gap = (net / run.level)
    changes = gap[gap.diff().abs() > 1e-12].index
    assert set(changes) <= set(c.index)


def test_capacity_scales_with_participation(run_and_rets, market, cfg):
    run, _, _ = run_and_rets
    c10 = capacity(run, market, cfg, 0.10)
    c20 = capacity(run, market, cfg, 0.20)
    assert np.allclose(c20, 2 * c10)


def test_proforma_on_effective_date_matches_review(run_and_rets, market, cfg):
    """Running the monitor on a determination date must reproduce that review's selection."""
    run, rets, adj = run_and_rets
    rv = run.reviews[2]
    prev_eff = run.reviews[1].review.effective
    w_before = run.weights.loc[rv.review.determination]
    pf = proforma(market, cfg, rets, adj, rv.review.determination, w_before)
    chosen = set(pf["trades"].index[pf["trades"]["w_proforma"] > 0])
    assert chosen == set(rv.constituents)
    assert prev_eff < rv.review.determination
