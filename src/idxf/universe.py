"""Eligible universe on a determination date (methodology section 3)."""
from __future__ import annotations

import pandas as pd

from .data import Market


def eligible_universe(mkt: Market, d: pd.Timestamp, cfg) -> pd.DataFrame:
    u = cfg.universe
    hist_win = mkt.window(d, u.min_history_days)
    liq_win = mkt.window(d, u.liquidity_window)

    first_seen = mkt.traded.idxmax()          # first day each ticker traded
    has_history = (first_seen <= hist_win[0]) & mkt.traded.loc[:d].any()

    traded_ratio = mkt.traded.loc[liq_win].mean()
    med_value_cr = mkt.value.loc[liq_win].where(mkt.traded.loc[liq_win]).median() / 1e7
    price = mkt.close.loc[d]

    df = pd.DataFrame({
        "has_history": has_history,
        "traded_ratio": traded_ratio,
        "med_value_cr": med_value_cr,
        "price": price,
        "traded_on_d": mkt.traded.loc[d],
        "sector": mkt.sectors,
    })
    df["excluded"] = df.index.isin(u.exclude)
    df["reason"] = ""
    df.loc[~df["has_history"], "reason"] = "history"
    df.loc[(df["reason"] == "") & (df["traded_ratio"] < u.min_trading_ratio), "reason"] = "trading_ratio"
    df.loc[(df["reason"] == "") & (df["med_value_cr"] < u.min_median_value_cr), "reason"] = "liquidity"
    df.loc[(df["reason"] == "") & (df["price"] < u.min_price), "reason"] = "price"
    df.loc[(df["reason"] == "") & df["excluded"], "reason"] = "manual"
    df.loc[(df["reason"] == "") & ~df["traded_on_d"].astype(bool), "reason"] = "not_trading"

    ok = df["reason"] == ""
    # size proxy: rank on median traded value, keep the top N
    rank = df.loc[ok, "med_value_cr"].rank(ascending=False, method="first")
    df["liq_rank"] = rank.reindex(df.index)
    df.loc[ok & (df["liq_rank"] > u.top_n_by_liquidity), "reason"] = "outside_top_n"
    df["eligible"] = df["reason"] == ""
    return df.sort_values("med_value_cr", ascending=False)
