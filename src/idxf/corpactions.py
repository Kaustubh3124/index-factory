"""Corporate actions: detection from raw prices, the curated CA file, and adjustment factors.

Convention used everywhere in this repo: `factor` is the number of new shares a
holder has for each old share after the event.
    2-for-1 split / 1:1 bonus  -> factor 2.0
    1:5 split (FV 10 -> FV 2)  -> factor 5.0
    demerger, value reinvested -> factor = P(cum) / P(ex, theoretical)
A price series is continuous across the event if you multiply post-event prices
by the factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# multipliers we actually see on NSE (splits and bonus issues)
STANDARD_FACTORS = np.array([1.2, 1.25, 4 / 3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0])

CA_COLUMNS = ["ex_date", "ticker", "action", "factor", "amount", "source", "note"]

# actions that change index shares (and nothing else). "data_fix" is a level shift in
# the source data with no economic event behind it - see data/README.md
SHARE_ACTIONS = ("split", "bonus", "consolidation", "demerger", "merger", "data_fix")


def detect_spikes(close: pd.DataFrame, traded: pd.DataFrame, benchmark: pd.Series,
                  threshold: float = 0.20, revert_tol: float = 0.05) -> pd.DataFrame:
    """Single bad closing prices: a big move on day t that is fully undone on the next
    traded day (market-relative). These go to price_overrides.csv, not the CA file."""
    c = close.where(traded)
    b = benchmark.reindex(close.index).ffill()
    rows = []
    for t in close.columns:
        s = c[t].dropna()
        if len(s) < 3:
            continue
        bm = b.reindex(s.index)
        rel = (s / s.shift(1)) / (bm / bm.shift(1))
        nxt = rel.shift(-1)
        hit = (np.abs(np.log(rel)) > threshold) & (np.abs(np.log(rel * nxt)) < revert_tol)
        for d in s.index[hit.fillna(False).to_numpy()]:
            rows.append({"date": d, "ticker": t, "close": s.loc[d],
                         "rel_move": round(float(rel.loc[d]) - 1, 4), "next_rel_move": round(float(nxt.loc[d]) - 1, 4)})
    return pd.DataFrame(rows, columns=["date", "ticker", "close", "rel_move", "next_rel_move"])


def detect_candidates(open_: pd.DataFrame, last: pd.DataFrame, close: pd.DataFrame,
                      volume: pd.DataFrame, traded: pd.DataFrame, benchmark: pd.Series,
                      benchmark_open: pd.Series | None = None, threshold: float = 0.18,
                      close_threshold: float = 0.30) -> pd.DataFrame:
    """Flag overnight gaps that look like a corporate action rather than a market move,
    and collect the evidence needed to decide what each one is.

    Gap: ex-day open vs previous last trade, divided by the NIFTY 50's own overnight
    gap so a market-wide gap does not trigger anything.

    Evidence columns
        implied        share factor implied by the gap (1 / relative gap)
        fit_err        |log(implied / nearest standard ratio)|
        persist        5-day median close after vs before, market-relative. A real
                       split/bonus/demerger gives persist ~ 1/implied; a bad print
                       that reverts gives persist ~ 1
        vol_ratio      20-day median volume after vs before. Splits and bonuses
                       multiply share volume by roughly the factor; demergers do not
        gap_days       trading days without a bar just before the flagged day; a big
                       number means a hole in the data, not a corporate action

    Two triggers: the open gap (threshold) and the close-to-close move (close_threshold).

    Classification is only a suggestion - every row still gets looked at by hand
    before it goes into corporate_actions.csv.
    """
    prev = last.shift(1)
    gap = open_ / prev
    # market's own overnight gap: index open vs previous index close. Using close-to-close
    # here was a bug - on 13 Mar 2020 the NIFTY opened -10% and closed +4%, which made
    # dozens of stocks look like they had a corporate action
    b_open = benchmark_open if benchmark_open is not None else benchmark
    mkt_gap = (b_open / benchmark.shift(1)).reindex(gap.index)
    rel = gap.div(mkt_gap, axis=0)
    # second trigger on close-to-close (market relative): the 2026 source files have
    # unreliable opens for some tickers, and a CA must never depend on one field only
    cl = close.where(traded)
    rel_c = (cl / cl.ffill().shift(1)).div(benchmark / benchmark.shift(1), axis=0)
    # (higher bar than the open trigger: NSE price bands stop most stocks at 20% a day,
    # so a 30%+ close-to-close move with no CA behind it is rare)
    trig = ((rel < 1 - threshold) | (rel > 1 + threshold)
            | (rel_c < 1 - close_threshold) | (rel_c > 1 + close_threshold))
    hits = rel.where(trig).stack().dropna()
    extra = rel_c.where(trig & rel.isna()).stack().dropna()
    hits = pd.concat([hits, extra])
    hits = hits[~hits.index.duplicated()]

    b = benchmark.reindex(close.index).ffill()
    rows = []
    for (d, t), r in hits.items():
        i = close.index.get_loc(d)
        c = close[t].where(traded[t])
        before, after = c.iloc[max(0, i - 5):i].median(), c.iloc[i:i + 5].median()
        mb, ma = b.iloc[max(0, i - 5):i].median(), b.iloc[i:i + 5].median()
        persist = (after / before) / (ma / mb)
        v = volume[t].where(traded[t])
        vol_ratio = v.iloc[i:i + 20].median() / v.iloc[max(0, i - 20):i].median()
        gap_days = int((~traded[t].iloc[max(0, i - 260):i]).iloc[::-1].cummin().sum())
        # market-relative close-to-close move on the flagged day: if the open was a bad
        # tick but the close is fine, this is small and there is nothing to do
        j = c.iloc[:i].last_valid_index()
        close_move = (c.iloc[i] / c.loc[j]) / (b.iloc[i] / b.loc[j]) if j is not None else np.nan

        implied_open = 1.0 / r
        implied_close = 1.0 / close_move if np.isfinite(close_move) and close_move > 0 else np.nan
        # the open (pre-open price discovery) is the right number for a demerger and is
        # not polluted by the day's own move. Only fall back to the close when the open
        # is clearly inconsistent with where the stock actually closed
        implied = implied_open
        if np.isfinite(implied_close) and abs(np.log(implied_open / implied_close)) > 0.25:
            implied = implied_close
        k = STANDARD_FACTORS[np.argmin(np.abs(np.log(STANDARD_FACTORS) - np.log(implied)))]
        err = abs(np.log(implied / k))
        lasting = abs(np.log(persist * implied)) < 0.25
        reverted = abs(np.log(persist)) < 0.15
        if gap_days >= 3:
            kind = "data_gap"
        elif abs(np.log(close_move)) < 0.12:
            kind = "open_tick"
        elif reverted:
            kind = "bad_print"
        elif lasting and err < 0.06:
            kind = "split_or_bonus"
        else:
            kind = "review"
        rows.append({"ex_date": d, "ticker": t, "suggested": kind,
                     "factor": round(float(k) if kind == "split_or_bonus" else implied, 6),
                     "implied": round(implied, 4), "implied_open": round(implied_open, 4),
                     "implied_close": round(float(implied_close), 4),
                     "fit_err": round(err, 4), "persist": round(persist, 4),
                     "vol_ratio": round(vol_ratio, 2), "gap_days": gap_days,
                     "close_move": round(float(close_move), 4)})
    cols = ["ex_date", "ticker", "suggested", "factor", "implied", "implied_open", "implied_close",
            "fit_err", "persist", "vol_ratio", "gap_days", "close_move"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["ex_date", "ticker"]).reset_index(drop=True)


def boundary_scaling(close: pd.DataFrame, open_: pd.DataFrame, traded: pd.DataFrame, benchmark: pd.Series,
                     boundary: pd.Timestamp, window: int = 10, tol: float = 0.03) -> pd.DataFrame:
    """Check a join between two vendor files for back-adjustment mismatches.

    The 2026 source file is partly back-adjusted for 2026 corporate actions: closes are
    divided by some factor k1 but the open field is left on the raw basis until the real
    ex-date. That leaves a fingerprint - open / previous close sits at ~k1 every day
    instead of ~1 - which separates a vendor rescaling from a genuine move on the
    boundary day (e.g. ITC on 1 Jan 2026 fell ~10% for real; the next day's open was
    on the same basis as the close).
    """
    i = close.index.get_loc(boundary)
    win = close.index[i + 1:i + 1 + window]
    k1 = (open_.loc[win] / close.shift(1).loc[win]).median()
    mkt = benchmark.iloc[i] / benchmark.iloc[i - 1]
    move = close.iloc[i] / close.iloc[i - 1] / mkt
    ok = traded.iloc[i] & traded.iloc[i - 1]
    out = pd.DataFrame({"open_basis_k1": k1, "boundary_close_move": move})[ok]
    out["vendor_scaled"] = np.abs(np.log(out["open_basis_k1"])) > tol
    out["real_move"] = ~out["vendor_scaled"] & (np.abs(np.log(out["boundary_close_move"])) > 0.05)
    return out[out["vendor_scaled"] | out["real_move"]].sort_values("open_basis_k1", ascending=False)


def load_ca(path) -> pd.DataFrame:
    ca = pd.read_csv(path, comment="#", parse_dates=["ex_date"])
    for c in CA_COLUMNS:
        if c not in ca.columns:
            ca[c] = np.nan
    ca["factor"] = ca["factor"].fillna(1.0).astype(float)
    ca["amount"] = ca["amount"].fillna(0.0).astype(float)
    return ca[CA_COLUMNS].sort_values(["ex_date", "ticker"]).reset_index(drop=True)


def share_factors(ca: pd.DataFrame, calendar: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Matrix of share multipliers (1.0 except on ex-dates)."""
    m = pd.DataFrame(1.0, index=calendar, columns=pd.Index(tickers))
    for r in ca.itertuples():
        if r.action in SHARE_ACTIONS and r.ticker in m.columns:
            if r.ex_date not in m.index:
                # CA on a non-trading day in our calendar -> next trading day
                pos = m.index.searchsorted(r.ex_date)
                if pos >= len(m.index):
                    continue
                d = m.index[pos]
            else:
                d = r.ex_date
            m.loc[d, r.ticker] *= r.factor
    return m


def special_dividends(ca: pd.DataFrame, calendar, tickers) -> pd.DataFrame:
    d = pd.DataFrame(0.0, index=calendar, columns=pd.Index(tickers))
    for r in ca.itertuples():
        if r.action == "special_dividend" and r.ticker in d.columns and r.ex_date in d.index:
            d.loc[r.ex_date, r.ticker] += r.amount
    return d


def price_adjustment(close: pd.DataFrame, factors: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative adjustment A such that  r_t = P_t * A_t / P_{t-1} - 1  is a clean return.

    Theoretical ex-price (per new share) = P_{t-1} / f - D, so
        A = P_{t-1} / (P_{t-1} / f - D)
    which collapses to A = f for a plain split/bonus and to P/(P-D) for a special dividend.
    """
    f = factors.reindex_like(close).fillna(1.0)
    d = divs.reindex_like(close).fillna(0.0)
    prev = close.shift(1)
    a = prev / (prev / f - d)
    return a.where(d != 0, f)


def adjusted_returns(close: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Daily price returns that are clean across corporate actions (adj from price_adjustment)."""
    return close * adj.reindex_like(close).fillna(1.0) / close.shift(1) - 1.0


def adjusted_prices(close: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Back-adjusted prices (latest prices unchanged, history divided down)."""
    f = factors.reindex_like(close).fillna(1.0)
    # cumulative factor from each day to the end, excluding that day's own event
    cum_after = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return close / cum_after
