import numpy as np
import pandas as pd
import pytest

from idxf.selection import select
from idxf.weighting import cap_sectors, cap_weights, check_constraints, target_weights


def test_cap_weights_respects_cap_and_sums_to_one():
    w = pd.Series([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05], index=list("abcdefgh"))
    out = cap_weights(w, cap=0.15)
    assert out.sum() == pytest.approx(1.0)
    assert out.max() <= 0.15 + 1e-12
    # names that were below the cap keep their relative sizes
    assert out["g"] / out["h"] == pytest.approx(1.0)


def test_cap_infeasible():
    with pytest.raises(ValueError):
        cap_weights(pd.Series([1.0] * 5), cap=0.1)


def test_sector_cap():
    idx = [f"x{i}" for i in range(20)]
    w = pd.Series(np.linspace(2, 1, 20), index=idx)
    w /= w.sum()
    sec = pd.Series(["A"] * 10 + ["B"] * 5 + ["C"] * 5, index=idx)
    out = cap_sectors(cap_weights(w, 0.10), sec, cap=0.35, stock_cap=0.10)
    sw = out.groupby(sec).sum()
    assert sw.max() <= 0.35 + 1e-9
    assert out.max() <= 0.10 + 1e-9
    assert out.sum() == pytest.approx(1.0)


def test_buffer_rule(cfg):
    ranks = pd.DataFrame({"rank": range(1, 31)}, index=[f"t{i}" for i in range(1, 31)])
    c = cfg
    old = (c.selection.n_constituents, c.selection.buffer_in, c.selection.buffer_out)
    c.selection.n_constituents, c.selection.buffer_in, c.selection.buffer_out = 5, 3, 8
    try:
        # incumbent at rank 7 survives, new name at rank 4 does not get in
        chosen = select(ranks, {"t7", "t2"}, c)
        assert chosen[:3] == ["t1", "t2", "t3"]
        assert "t7" in chosen
        assert len(chosen) == 5
        assert "t4" in chosen and "t5" not in chosen
    finally:
        c.selection.n_constituents, c.selection.buffer_in, c.selection.buffer_out = old


def test_target_weights_constraints(cfg):
    rng = np.random.default_rng(1)
    idx = [f"s{i}" for i in range(30)]
    vol = pd.Series(rng.uniform(0.12, 0.5, 30), index=idx)
    sec = pd.Series(rng.choice(["F", "IT", "E", "M", "H"], 30), index=idx)
    w = target_weights(vol, sec, cfg)
    chk = check_constraints(w, sec, cfg)
    assert chk["stock_cap_ok"] and chk["sector_cap_ok"]
    assert chk["sum"] == pytest.approx(1.0)
