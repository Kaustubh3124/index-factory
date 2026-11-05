"""Charts used in README.md (written to docs/img). Run after `idxf run` and the ablation."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
IMG = ROOT / "docs" / "img"
NAVY, GREY, RED, GREEN = "#1f3b5a", "#8a94a3", "#b5473a", "#3d7a5a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def perf():
    lv = pd.read_csv(OUT / "levels.csv", index_col=0, parse_dates=True)
    b = lv["NIFTY50"] / lv["NIFTY50"].iloc[0] * 1000
    p = lv["parent_ew"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lv.index, lv["IMLV30"], color=NAVY, lw=1.4, label="IMLV30")
    ax.plot(lv.index, lv["IMLV30_net"], color=NAVY, lw=0.8, ls="--", label="IMLV30 net of costs")
    ax.plot(lv.index, p, color=GREEN, lw=0.9, label="Eligible universe, equal weight")
    ax.plot(lv.index, b, color=GREY, lw=1.1, label="NIFTY 50")
    ax.set_title("IMLV30 vs NIFTY 50 and its own parent universe (base 1000 = 28 Jun 2019)", loc="left", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "performance.png", dpi=130)


def greeks():
    g = pd.read_csv(OUT / "note_greeks_profile.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for tau, c in zip((3.0, 2.0, 1.0, 0.25), (NAVY, "#4b6b8f", "#8aa1bb", RED)):
        s = g[g.tau == tau]
        axes[0].plot(s.spot_pct * 100, s.value, color=c, label=f"{tau:g}y left")
        axes[1].plot(s.spot_pct * 100, s.delta_1pct, color=c)
        axes[2].plot(s.spot_pct * 100, s.vega_1vol, color=c)
    for ax, t in zip(axes, ["Note value (per 100)", "Delta per 1% index move", "Vega per vol point"]):
        ax.set_title(t, loc="left", fontsize=9)
        ax.set_xlabel("Index level, % of initial")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "note_greeks.png", dpi=130)


def sensitivity():
    s = pd.read_csv(OUT / "participation_sensitivity.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.imshow(s.values * 100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(s.columns)), [f"{float(c) * 100:.2f}%" for c in s.columns])
    ax.set_yticks(range(len(s.index)), [f"{float(v) * 100:.1f}%" for v in s.index])
    ax.set_xlabel("INR rate")
    ax.set_ylabel("Volatility")
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            v = s.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", color="white" if v > 90 else "black", fontsize=8)
    ax.set_title("3y PPN participation vs rates and vol", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "participation_sensitivity.png", dpi=130)


def garch():
    v = pd.read_csv(OUT / "garch_conditional_vol.csv", index_col=0, parse_dates=True).iloc[:, 0]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(v.index, v * 100, color=NAVY, lw=0.9)
    ax.set_ylabel("annualised vol, %")
    ax.set_title("GARCH(1,1) conditional volatility of IMLV30", loc="left", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(IMG / "garch_vol.png", dpi=130)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    perf()
    greeks()
    sensitivity()
    garch()
    if (OUT / "ablation.png").exists():
        (IMG / "ablation.png").write_bytes((OUT / "ablation.png").read_bytes())
    print("charts written to", IMG)
