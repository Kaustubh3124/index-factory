"""Rebalance calendar.

    determination date  (D)  : data cut-off, scores and selection are computed on this close
    announcement date   (A)  : new composition published (A = E - announcement_lag)
    effective date      (E)  : last trading day of the review month; new shares are
                               fixed on E's close and the index trades with them from E+1
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Review:
    determination: pd.Timestamp
    announcement: pd.Timestamp
    effective: pd.Timestamp
    initial: bool = False


def review_dates(calendar: pd.DatetimeIndex, cfg) -> list[Review]:
    sch = cfg.schedule
    cal = pd.Series(calendar, index=calendar)
    month_ends = cal.groupby([calendar.year, calendar.month]).max()

    out = []
    for eff in month_ends:
        if eff < cfg.base_date:
            continue
        if eff.month not in sch.months and eff != cfg.base_date:
            continue
        i = calendar.get_loc(eff)
        det = calendar[i - sch.determination_lag]
        ann = calendar[i - sch.announcement_lag]
        out.append(Review(det, ann, eff, initial=(eff == cfg.base_date)))

    if not out or not out[0].initial:
        raise ValueError(f"base date {cfg.base_date.date()} is not a month-end trading day in the calendar")
    # drop a review whose effective date is the very last day of data - nothing to trade after it
    if out[-1].effective == calendar[-1]:
        out = out[:-1]
    return out
