"""Compare the final rulebook against the simpler versions it replaced.

    python scripts/run_ablation.py            -> output/ablation.csv, output/ablation.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from idxf.ablation import run_variants
from idxf.config import load_config
from idxf.corpactions import load_ca
from idxf.data import load_market

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = load_config(ROOT / "config" / "imlv30.yaml")
    mkt = load_market(ROOT / "data" / "panel.parquet", ROOT / "data" / "sectors.csv", data_dir=ROOT / "data")
    ca = load_ca(ROOT / "data" / "corporate_actions.csv")
    table, levels = run_variants(mkt, cfg, ca)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "ablation.csv", float_format="%.4f")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, v in levels.items():
        ax.plot(v.index, v, lw=1.6 if k.startswith("E_") else 0.9, label=k)
    b = mkt.benchmark.reindex(v.index).ffill()
    ax.plot(b.index, b / b.iloc[0] * 1000, color="k", lw=0.8, ls="--", label="NIFTY 50")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Rule variants, same data (log scale)")
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=130)


if __name__ == "__main__":
    main()
