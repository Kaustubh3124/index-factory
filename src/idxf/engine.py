"""Index calculation.

Two independent calculators live in this package:

  engine.py   "production" calc. Holds index shares per constituent and a divisor,
              applies corporate actions as share/divisor changes on the ex-date,
              level = sum(shares * price) / divisor.
  shadow.py   return-chaining calc. Uses CA-adjusted returns and drifting weights,
              never touches shares or the divisor.

They are written separately on purpose and reconciled every run (recon.py). If they
disagree by more than 1e-8 in relative terms, something is wrong in one of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import corpactions as cax
from .data import Market, stale_run
from .schedule import Review, review_dates
from .selection import factor_scores, select
from .universe import eligible_universe
from .weighting import check_constraints, target_weights


@dataclass
class ReviewResult:
    review: Review
    universe: pd.DataFrame
    scores: pd.DataFrame
    constituents: list[str]
    weights: pd.Series
    adds: list[str]
    deletes: list[str]
    checks: dict


@dataclass
class IndexRun:
    level: pd.Series
    divisor: pd.Series
    weights: pd.DataFrame               # end-of-day weights (after any rebalance that day)
    reviews: list[ReviewResult]
    rebalance_trades: dict = field(default_factory=dict)   # effective date -> DataFrame
    events: list[dict] = field(default_factory=list)
    shares: dict = field(default_factory=dict)             # effective date -> shares after rebalance


def prepare(mkt: Market, ca: pd.DataFrame):
    factors = cax.share_factors(ca, mkt.calendar, mkt.tickers)
    divs = cax.special_dividends(ca, mkt.calendar, mkt.tickers)
    adj = cax.price_adjustment(mkt.close, factors, divs)
    rets = cax.adjusted_returns(mkt.close, adj)
    adj_close = cax.adjusted_prices(mkt.close, adj)
    return factors, divs, rets, adj_close


def run_reviews(mkt: Market, cfg, rets, adj_close) -> list[ReviewResult]:
    out: list[ReviewResult] = []
    incumbents: set[str] = set()
    for rv in review_dates(mkt.calendar, cfg):
        uni = eligible_universe(mkt, rv.determination, cfg)
        names = uni.index[uni["eligible"]]
        sc = factor_scores(adj_close, rets, rv.determination, names, cfg)
        cons = select(sc, incumbents, cfg)

        wvol = rets.loc[:rv.determination, cons].iloc[-cfg.weighting.vol_window:].std() * np.sqrt(252)
        w = target_weights(wvol, mkt.sectors, cfg)
        res = ReviewResult(
            review=rv, universe=uni, scores=sc, constituents=cons, weights=w,
            adds=sorted(set(cons) - incumbents), deletes=sorted(incumbents - set(cons)),
            checks=check_constraints(w, mkt.sectors, cfg),
        )
        out.append(res)
        incumbents = set(cons)
    return out


def calculate(mkt: Market, cfg, ca: pd.DataFrame, reviews: list[ReviewResult] | None = None) -> IndexRun:
    factors, divs, rets, adj_close = prepare(mkt, ca)
    if reviews is None:
        reviews = run_reviews(mkt, cfg, rets, adj_close)
    by_eff = {r.review.effective: r for r in reviews}

    cal = mkt.calendar[mkt.calendar >= cfg.base_date]
    px = mkt.close.loc[cal]
    stale = stale_run(mkt.traded).loc[cal]
    fac = factors.loc[cal]
    dv = divs.loc[cal]
    dq = cfg.data_quality

    tickers = list(mkt.tickers)
    col = {t: i for i, t in enumerate(tickers)}
    shares = np.zeros(len(tickers))
    divisor = 1.0

    levels, divisors, wts = [], [], []
    prev_p = None   # previous day's closes, needed for the special-dividend adjustment
    run = IndexRun(level=None, divisor=None, weights=None, reviews=reviews)

    for k, d in enumerate(cal):
        p = px.loc[d].to_numpy(dtype=float)
        held = shares > 0

        if k > 0:
            # 1) share-based CAs on the ex-date: shares * factor, no divisor change
            f = fac.loc[d].to_numpy()
            hit = held & (f != 1.0)
            for j in np.flatnonzero(hit):
                run.events.append({"date": d, "type": "share_adjustment", "ticker": tickers[j],
                                   "factor": float(f[j]), "shares_before": float(shares[j]),
                                   "shares_after": float(shares[j] * f[j])})
            shares_cum = shares
            shares = shares * f

            # 2) special dividends: the price drops on the ex-date but the index should
            #    not, so the divisor comes down by the cash paid out.
            #    D_new = D_old * (MV_cum - cash) / MV_cum
            cash = float(np.nansum(shares * dv.loc[d].to_numpy()))
            if cash > 0:
                mv_cum = float(np.nansum(shares_cum * prev_p))
                new_div = divisor * (mv_cum - cash) / mv_cum
                run.events.append({"date": d, "type": "special_dividend", "cash_per_index": cash / divisor,
                                   "divisor_before": divisor, "divisor_after": new_div})
                divisor = new_div

        mv = float(np.nansum(shares * p))
        level = cfg.base_value if k == 0 else mv / divisor

        # 3) a constituent that has stopped trading is removed at its last price;
        #    proceeds go to the remaining names pro rata, so MV and divisor are unchanged
        if k > 0:
            dead = held & (stale.loc[d].to_numpy() >= dq.stale_days)
            if dead.any():
                proceeds = float(np.sum(shares[dead] * p[dead]))
                alive = held & ~dead
                for j in np.flatnonzero(dead):
                    run.events.append({"date": d, "type": "removal_stale", "ticker": tickers[j],
                                       "price": float(p[j]), "days_without_trade": int(stale.loc[d].iloc[j])})
                shares[dead] = 0.0
                mv_alive = float(np.sum(shares[alive] * p[alive]))
                shares[alive] *= (mv_alive + proceeds) / mv_alive

            # data-quality flags on constituents (informational)
            r = rets.loc[d].to_numpy()
            flag = held & (np.abs(np.nan_to_num(r)) > dq.max_unexplained_move)
            for j in np.flatnonzero(flag):
                run.events.append({"date": d, "type": "dq_large_move", "ticker": tickers[j],
                                   "return": float(r[j])})

        # 4) rebalance at the close of the effective date
        if d in by_eff:
            res = by_eff[d]
            old_w = _weights(shares, p)
            base_mv = level * divisor if k > 0 else cfg.base_value * divisor
            new_shares = np.zeros_like(shares)
            for t, w in res.weights.items():
                new_shares[col[t]] = w * base_mv / p[col[t]]
            new_w = _weights(new_shares, p)
            run.rebalance_trades[d] = pd.DataFrame({
                "w_before": pd.Series(old_w, index=tickers),
                "w_after": pd.Series(new_w, index=tickers),
                "shares_before": pd.Series(shares, index=tickers),
                "shares_after": pd.Series(new_shares, index=tickers),
                "price": pd.Series(p, index=tickers),
            }).query("w_before > 0 or w_after > 0")
            shares = new_shares
            run.shares[d] = pd.Series(shares, index=tickers)[shares > 0]
            mv = float(np.nansum(shares * p))
            # composition change keeps MV, so the divisor only needs a re-sync for rounding
            divisor = mv / level

        levels.append(level)
        divisors.append(divisor)
        wts.append(_weights(shares, p))
        prev_p = p

    run.level = pd.Series(levels, index=cal, name=cfg.ticker)
    run.divisor = pd.Series(divisors, index=cal, name="divisor")
    run.weights = pd.DataFrame(wts, index=cal, columns=tickers)
    run.weights = run.weights.loc[:, (run.weights > 0).any()]
    return run


def _weights(shares, p):
    mv = shares * np.nan_to_num(p)
    tot = mv.sum()
    return mv / tot if tot > 0 else mv
