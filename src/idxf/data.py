"""Loading the daily panel and turning it into the wide matrices the rest of the code uses.

The raw source is 1-minute bars (see scripts/build_daily.py). Prices are NOT
adjusted for splits/bonuses/demergers, which is what we want: corporate actions
are handled explicitly by the index engine, the same way an index provider has
to handle them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_TICKERS = {"NIFTY50-INDEX", "NIFTYBANK-INDEX", "FINNIFTY-INDEX", "MIDCPNIFTY-INDEX"}


@dataclass
class Market:
    close: pd.DataFrame       # official-style close (VWAP of last 30 min), raw
    last: pd.DataFrame        # last traded price of the day
    open: pd.DataFrame
    volume: pd.DataFrame
    value: pd.DataFrame       # traded value, INR
    traded: pd.DataFrame      # bool, stock actually printed that day
    calendar: pd.DatetimeIndex
    benchmark: pd.Series      # NIFTY 50 closes on the same calendar
    sectors: pd.Series
    benchmark_open: pd.Series | None = None

    @property
    def tickers(self):
        return self.close.columns

    def window(self, end: pd.Timestamp, n: int) -> pd.DatetimeIndex:
        """Last n trading days up to and including `end`."""
        i = self.calendar.get_loc(end)
        return self.calendar[max(0, i - n + 1): i + 1]

    def offset(self, d: pd.Timestamp, k: int) -> pd.Timestamp:
        i = self.calendar.get_loc(d) + k
        i = min(max(i, 0), len(self.calendar) - 1)
        return self.calendar[i]


def build_panel(daily_dirs: list[str | Path], out: str | Path) -> pd.DataFrame:
    """Stack per-ticker daily files (possibly split across several source dirs)."""
    frames = []
    for d in daily_dirs:
        for f in sorted(Path(d).glob("*.parquet")):
            df = pd.read_parquet(f)
            df["ticker"] = f.stem
            frames.append(df.reset_index())
    panel = pd.concat(frames, ignore_index=True)
    panel = (panel.sort_values(["ticker", "date"])
                  .drop_duplicates(["ticker", "date"], keep="last"))
    # only the columns the code uses; zstd keeps the file small enough to commit (~18 MB)
    panel = panel[["date", "ticker", "open", "last", "close", "volume", "value"]]
    panel.to_parquet(out, index=False, compression="zstd", compression_level=9)
    return panel


def apply_symbol_changes(panel: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Splice old -> new symbol: old symbol's rows before `effective`, new symbol's rows from
    `effective` on, all under the new symbol. The index keeps holding the same company."""
    for ch in changes.itertuples():
        eff = pd.Timestamp(ch.effective)
        old = panel[(panel["ticker"] == ch.old) & (panel["date"] < eff)].assign(ticker=ch.new)
        new = panel[(panel["ticker"] == ch.new) & (panel["date"] >= eff)]
        rest = panel[~panel["ticker"].isin([ch.old, ch.new])]
        panel = pd.concat([rest, old, new], ignore_index=True)
    return panel


def load_market(panel_path: str | Path, sectors_path: str | Path | None = None,
                start: str | None = None, end: str | None = None,
                data_dir: str | Path | None = None) -> Market:
    """data_dir, if given, is where symbol_changes.csv and price_overrides.csv live."""
    panel = pd.read_parquet(panel_path)
    overrides = pd.DataFrame(columns=["date", "ticker"])
    if data_dir is not None:
        sc = Path(data_dir) / "symbol_changes.csv"
        if sc.exists():
            panel = apply_symbol_changes(panel, pd.read_csv(sc, comment="#"))
        po = Path(data_dir) / "price_overrides.csv"
        if po.exists():
            overrides = pd.read_csv(po, comment="#", parse_dates=["date"])
        ce = Path(data_dir) / "calendar_exclusions.csv"
        if ce.exists():
            drop = pd.read_csv(ce, comment="#", parse_dates=["date"])["date"]
            panel = panel[~panel["date"].isin(drop)]
    if start:
        panel = panel[panel["date"] >= pd.Timestamp(start)]
    if end:
        panel = panel[panel["date"] <= pd.Timestamp(end)]

    bench_df = panel[panel["ticker"] == "NIFTY50-INDEX"].set_index("date")
    bench_raw = bench_df["last"]
    stocks = panel[~panel["ticker"].isin(INDEX_TICKERS)]

    # Exchange calendar = days the NIFTY 50 printed. Muhurat sessions show up as
    # a short Sunday/holiday session - we keep them, the exchange counts them too.
    calendar = pd.DatetimeIndex(bench_raw.index.unique()).sort_values()

    def wide(col):
        w = stocks.pivot(index="date", columns="ticker", values=col)
        return w.reindex(calendar)

    vol = wide("volume")
    # bad bars from price_overrides.csv are treated as "no trade that day"
    for o in overrides.itertuples():
        if o.ticker in vol.columns and o.date in vol.index:
            vol.loc[o.date, o.ticker] = 0.0
    traded = vol.fillna(0) > 0
    close = wide("close").where(traded).ffill()
    last = wide("last").where(traded).ffill()
    opn = wide("open").where(traded)
    value = wide("value").where(traded, 0.0).fillna(0.0)

    sectors = pd.Series(dtype=object)
    if sectors_path and Path(sectors_path).exists():
        sectors = pd.read_csv(sectors_path).set_index("ticker")["sector"]
    sectors = sectors.reindex(close.columns).fillna("Unclassified")

    return Market(close=close, last=last, open=opn, volume=vol.fillna(0.0),
                  value=value, traded=traded, calendar=calendar,
                  benchmark=bench_raw.reindex(calendar).ffill(), sectors=sectors,
                  benchmark_open=bench_df["open"].reindex(calendar))


def stale_run(traded: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive days each stock has gone without a trade (0 if it traded today)."""
    return _run_lengths((~traded).astype(int))


def _run_lengths(nt: pd.DataFrame) -> pd.DataFrame:
    a = nt.to_numpy()
    out = np.zeros_like(a)
    for j in range(a.shape[1]):
        run = 0
        col = a[:, j]
        for i in range(len(col)):
            run = run + 1 if col[i] else 0
            out[i, j] = run
    return pd.DataFrame(out, index=nt.index, columns=nt.columns)
