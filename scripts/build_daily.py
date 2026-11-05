"""Aggregate 1-minute NSE bars into daily bars.

Source: per-ticker parquet files of 1-minute OHLCV (09:15-15:29 IST), raw prices.
Output: one parquet per ticker with
    open, high, low, last      first/max/min/last of the 1-min bars
    close                      VWAP of the 15:00-15:29 bars
    volume, value              shares and approx. traded value (sum of close * volume per bar)
    bars                       number of 1-min bars that day

Why `close` is a 30-minute VWAP: NSE's official closing price is the weighted
average price of trades in the last 30 minutes, not the last print. Using the
last print makes the index noisier and differs from what a provider would use.

usage: python scripts/build_daily.py SRC_DIR OUT_DIR [--delete-src]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate(path: Path) -> pd.DataFrame:
    d = pd.read_parquet(path, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    d = d.dropna(subset=["Close"])
    d["day"] = d["Date"].dt.normalize()
    d["tv"] = d["Close"] * d["Volume"]
    minute = d["Date"].dt.hour * 60 + d["Date"].dt.minute
    last30 = d[(minute >= 15 * 60) & (minute < 15 * 60 + 30)]

    g = d.groupby("day")
    daily = pd.DataFrame({
        "open": g["Open"].first(), "high": g["High"].max(), "low": g["Low"].min(),
        "last": g["Close"].last(), "volume": g["Volume"].sum(), "value": g["tv"].sum(),
        "bars": g.size(),
    })
    g2 = last30.groupby("day")
    vwap = g2["tv"].sum() / g2["Volume"].sum().replace(0, np.nan)
    # index files have no volume -> fall back to the last print
    daily["close"] = vwap.reindex(daily.index).fillna(daily["last"]).round(2)
    daily.index.name = "date"
    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--delete-src", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.src).glob("*.parquet"))
    for i, f in enumerate(files, 1):
        aggregate(f).to_parquet(out / f.name, compression="zstd")
        if a.delete_src:
            os.remove(f)
        if i % 50 == 0:
            print(f"{i}/{len(files)}")


if __name__ == "__main__":
    main()
