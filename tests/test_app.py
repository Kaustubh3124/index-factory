"""Smoke test for the dashboard: every page renders without an exception, the rule lab runs
a custom back-test, and the note pricer works in closed form and Monte Carlo mode.
Uses the real data in data/panel.parquet, so it takes a while (~40 s). Skipped if
streamlit is not installed."""
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


@pytest.fixture(scope="module")
def app():
    at = st_testing.AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


def goto(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    return at


def test_overview(app):
    at = goto(app, "Overview")
    assert "Annual return" in [m.label for m in at.metric]


def test_holdings(app):
    goto(app, "Holdings and reviews")


def test_rule_lab_custom_run(app):
    at = goto(app, "Rule lab")
    at.slider[0].set_value(20)                      # 20 stocks instead of 30
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1


def test_note_pricer_closed_form_and_mc(app):
    at = goto(app, "Note pricer")
    labels = [m.label for m in at.metric]
    assert "Participation" in labels
    part_cf = next(m.value for m in at.metric if m.label == "Participation")
    at.selectbox[1].set_value(6).run()              # average of the last 6 monthly closes
    assert not at.exception, at.exception
    part_avg = next(m.value for m in at.metric if m.label == "Participation")
    assert float(part_avg.rstrip("%")) > float(part_cf.rstrip("%"))


def test_data_quality(app):
    goto(app, "Data quality")
