"""Reconcile the production calc against the shadow calc."""
from __future__ import annotations

import pandas as pd


def reconcile(prod: pd.Series, shadow: pd.Series, tol: float = 1e-8) -> dict:
    a, b = prod.align(shadow, join="outer")
    rel = (a / b - 1.0).abs()
    breaks = rel[rel > tol]
    return {
        "days": int(len(rel)),
        "max_rel_diff": float(rel.max()),
        "mean_rel_diff": float(rel.mean()),
        "n_breaks": int(len(breaks)),
        "first_break": None if breaks.empty else str(breaks.index[0].date()),
        "breaks": breaks,
        "passed": bool(breaks.empty),
    }
