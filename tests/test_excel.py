"""Recalculate the generated workbook with LibreOffice and compare its formulas with Python.

openpyxl only writes formulas; nothing evaluates them. Two real bugs were found this way
(NORM.S.DIST written without the _xlfn. prefix -> #NAME?, and -B13^2 being (+B13)^2 in
Excel), so this test stays even though it is slow. Skipped if soffice is not installed.
"""
import shutil
import subprocess

import openpyxl
import pytest

from idxf import analytics as an
from idxf.costs import rebalance_costs
from idxf.engine import calculate, prepare
from idxf.pricing import bs
from idxf.recon import reconcile
from idxf.report.excel import build_workbook
from idxf.shadow import calculate_shadow

pytestmark = pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice not installed")


def test_workbook_formulas_match_python(tmp_path, market, cfg, ca_table):
    factors, divs, rets, _ = prepare(market, ca_table)
    run = calculate(market, cfg, ca_table)
    rec = reconcile(run.level, calculate_shadow(market, cfg, rets, {r.review.effective: r.weights for r in run.reviews},
                                                factors, divs))
    costs = rebalance_costs(run, market, rets, cfg)
    stats, rel = an.perf_stats(run.level), an.relative_stats(run.level, market.benchmark)
    S0, T, r, q, sig = float(run.level.iloc[-1]), 3.0, 0.0675, 0.012, 0.18
    zcb = 100 * __import__("math").exp(-r * T)
    part = (100 * (1 - 0.015) - zcb) / (bs.price(S0, S0, T, r, q, sig) / S0 * 100)
    src = tmp_path / "wb.xlsx"
    build_workbook(src, cfg, run, market, stats, rel, costs, rec,
                   {"S0": S0, "T": T, "r": r, "q": q, "sigma": sig, "protection": 1.0, "fees": 0.015,
                    "participation_cf": part})

    out = tmp_path / "calc"
    out.mkdir()
    subprocess.run(["soffice", "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(out), str(src)],
                   check=True, capture_output=True, timeout=180)
    wb = openpyxl.load_workbook(out / "wb.xlsx", data_only=True)

    errors = [c.coordinate for ws in wb for row in ws.iter_rows() for c in row
              if isinstance(c.value, str) and c.value.startswith("#")]
    assert not errors

    np_ = wb["Note_Pricing"]
    v = {np_.cell(i, 1).value: np_.cell(i, 2).value for i in range(3, 25) if np_.cell(i, 1).value}
    assert v["Participation"] == pytest.approx(part, rel=1e-10)
    g = bs.greeks(S0, S0, T, r, q, sig)
    assert v["Call delta"] == pytest.approx(g["delta"], rel=1e-10)
    assert v["Note vega per vol pt (per 100)"] == pytest.approx(part * g["vega"] / S0, rel=1e-10)

    ws = wb["Constituents"]
    labels = {ws.cell(i, 4).value: ws.cell(i, 5).value for i in range(1, ws.max_row + 1)}
    assert labels["Index level"] == pytest.approx(run.level.iloc[-1], rel=1e-12)
