"""Command line entry point.

    idxf build-panel   stack per-ticker daily files into data/panel.parquet
    idxf detect-ca     scan raw prices for split/bonus/demerger gaps -> data/ca_candidates.csv
    idxf run           full index run + reports into output/
    idxf monitor       pro-forma review as of a date
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import load_config

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="idxf")
    ap.add_argument("--config", default=str(ROOT / "config" / "imlv30.yaml"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "output"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-panel")
    b.add_argument("dirs", nargs="+", help="folders with one daily parquet per ticker")
    sub.add_parser("detect-ca")
    sub.add_parser("run")
    m = sub.add_parser("monitor")
    m.add_argument("--asof", default=None)
    m.add_argument("--aum", type=float, default=None, help="tracking AUM in INR crore")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    data, out = Path(args.data), Path(args.out)

    if args.cmd == "build-panel":
        from .data import build_panel
        p = build_panel(args.dirs, data / "panel.parquet")
        print(f"panel: {p['ticker'].nunique()} tickers, {len(p):,} rows")

    elif args.cmd == "detect-ca":
        from .corpactions import detect_candidates, detect_spikes
        from .data import load_market
        mkt = load_market(data / "panel.parquet", data_dir=data)
        pd.set_option("display.width", 160)
        spikes = detect_spikes(mkt.close, mkt.traded, mkt.benchmark)
        spikes.to_csv(data / "spike_candidates.csv", index=False)
        print("single-day spikes (candidates for price_overrides.csv):")
        print(spikes.to_string(index=False) if len(spikes) else "  none")
        cand = detect_candidates(mkt.open, mkt.last, mkt.close, mkt.volume, mkt.traded, mkt.benchmark,
                                 mkt.benchmark_open)
        cand.to_csv(data / "ca_candidates.csv", index=False)
        print("\novernight gaps (candidates for corporate_actions.csv):")
        print(cand.to_string(index=False))
        print("\n" + cand["suggested"].value_counts().to_string())

    elif args.cmd == "run":
        from .pipeline import run_all
        s = run_all(cfg, data, out)
        st = s["stats"]
        print(f"\n{cfg.ticker} {s['stats']['start']} -> {s['stats']['end']}: CAGR {st['cagr']:.2%} "
              f"(NIFTY 50 {st['bench_cagr']:.2%}), vol {st['vol']:.2%}, MDD {st['max_drawdown']:.2%}")
        print(f"recon passed: {s['recon']['passed']}  max rel diff {s['recon']['max_rel_diff']:.2e}")
        print(f"outputs in {out}")

    elif args.cmd == "monitor":
        from .data import load_market
        from .engine import calculate, prepare
        from .corpactions import load_ca
        from .monitor import proforma
        mkt = load_market(data / "panel.parquet", data / "sectors.csv", data_dir=data)
        ca = load_ca(data / "corporate_actions.csv")
        _, _, rets, adj = prepare(mkt, ca)
        run = calculate(mkt, cfg, ca)
        asof = pd.Timestamp(args.asof) if args.asof else mkt.calendar[-1]
        asof = mkt.calendar[mkt.calendar <= asof][-1]
        pf = proforma(mkt, cfg, rets, adj, asof, run.weights.loc[asof], args.aum)
        pd.set_option("display.width", 160)
        print(f"pro-forma as of {asof.date()}, one-way turnover {pf['one_way_turnover']:.1%}\n")
        print(pf["trades"].round(3).to_string())
        print("\nnames near the buffer thresholds:\n")
        print(pf["watchlist"][["rank", "score", "incumbent", "in_proforma"]].round(3).to_string())


if __name__ == "__main__":
    main()
