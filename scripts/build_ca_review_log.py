"""Close the loop on the corporate-action scan.

1. Every row in data/ca_candidates.csv gets a decision in data/ca_review_log.csv:
   either it is in corporate_actions.csv / price_overrides.csv, or there is a reason
   why nothing was done.
2. Residual check: after applying everything, list adjusted daily moves above 25% for
   stocks that were in the eligible universe at the time. Anything here that is not a
   known real event means the CA file is still missing something.

    python scripts/build_ca_review_log.py
"""
from pathlib import Path

import pandas as pd

from idxf.config import load_config
from idxf.corpactions import load_ca
from idxf.data import load_market
from idxf.engine import prepare, run_reviews

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# candidates looked at by hand and left alone, with the reason
NO_ACTION = {
    ("CENTRALBK", "2018-09-24"): "real move in the Sep-2018 IL&FS sell-off; volume x13, not a share event",
    ("JWL", "2018-11-29"): "real move in an illiquid stock",
    ("LINDEINDIA", "2019-01-25"): "real move; volume fell afterwards, so not a bonus",
    ("YESBANK", "2019-02-14"): "real move",
    ("CGPOWER", "2019-08-20"): "real: accounting irregularities disclosed, lower circuits",
    ("CGPOWER", "2019-08-21"): "real: second lower circuit",
    ("TATACOMM", "2019-09-17"): "unverified -35% day followed by 5% upper circuits; never in the eligible universe around it",
    ("SWSOLAR", "2019-11-18"): "real post-IPO sell-off",
    ("IDEA", "2019-11-19"): "real move (AGR dues)",
    ("SUZLON", "2020-01-03"): "real move",
    ("IDEA", "2020-02-19"): "real move (AGR dues)",
    ("YESBANK", "2020-03-06"): "real: RBI moratorium",
    ("YESBANK", "2020-03-16"): "real: rebound after the reconstruction scheme",
    ("YESBANK", "2020-03-17"): "real: rebound after the reconstruction scheme",
    ("IDEA", "2020-03-18"): "real move",
    ("BLS", "2020-03-13"): "real move on the 13-Mar-2020 circuit-breaker day",
    ("BANDHANBNK", "2020-03-26"): "real rebound",
    ("INDUSINDBK", "2020-03-26"): "real rebound",
    ("HUDCO", "2020-06-29"): "real move",
    ("YESBANK", "2020-07-23"): "real: FPO priced at INR 12 - dilution, not a share split",
    ("TTML", "2020-10-21"): "real move",
    ("PNBHOUSING", "2021-06-01"): "real move (Carlyle deal)",
    ("ZEEL", "2021-09-14"): "real: Invesco / Sony news",
    ("ZEEL", "2021-09-22"): "real: Sony merger announcement",
    ("GPIL", "2022-05-23"): "real: steel and iron ore export duty sell-off",
    ("BEML", "2022-09-08"): "unclear (volume fell to 0.12x); never held",
    ("ZEEL", "2024-01-23"): "real: Sony merger called off",
    ("PAYTM", "2024-02-01"): "real: RBI action on Paytm Payments Bank",
    ("PAYTM", "2024-02-02"): "real: second lower circuit",
    ("IIFL", "2024-03-05"): "real: RBI restriction on gold loans",
    ("IIFL", "2024-03-06"): "real: second lower circuit",
    ("HONASA", "2024-11-18"): "real: results",
    ("ADANIENSOL", "2024-11-21"): "real: US indictment news",
    ("WHIRLPOOL", "2025-01-30"): "real: parent stake sale",
    ("SARDAEN", "2025-08-04"): "real move",
    ("TARIL", "2025-11-10"): "unclear; never in the eligible universe",
    ("TATACHEM", "2026-09-15"): "real move",
    ("IDBI", "2020-06-01"): "real move",
    ("ADANIGREEN", "2024-11-21"): "real: US indictment news (same day as ADANIENSOL)",
}

# big adjusted moves that the residual check finds and that are real, documented events
RESIDUAL_KNOWN = {
    ("YESBANK", "2018-09-21"): "RBI cuts CEO tenure", ("ADANIPOWER", "2018-10-10"): "real rally",
    ("ZEEL", "2019-01-25"): "promoter pledge sell-off", ("CGPOWER", "2019-02-13"): "real move",
    ("YESBANK", "2019-04-30"): "Q4 loss", ("ADANIENT", "2019-05-20"): "exit-poll rally",
    ("YESBANK", "2019-10-03"): "rebound", ("YESBANK", "2020-03-05"): "pre-moratorium swing",
    ("YESBANK", "2020-03-09"): "reconstruction news", ("YESBANK", "2020-03-11"): "reconstruction news",
    ("ADANIPOWER", "2020-03-12"): "COVID sell-off", ("ASHOKLEY", "2020-03-19"): "COVID sell-off",
    ("AXISBANK", "2020-03-23"): "COVID crash day (NIFTY -13%)", ("BAJAJFINSV", "2020-03-23"): "COVID crash day",
    ("CHOLAFIN", "2020-03-23"): "COVID crash day", ("M&MFIN", "2020-03-23"): "COVID crash day",
    ("MFSL", "2020-03-23"): "COVID crash day", ("ICICIPRULI", "2020-03-25"): "relief rally",
    ("JINDALSTEL", "2020-04-07"): "relief rally", ("GLENMARK", "2020-06-22"): "FabiFlu approval",
    ("ADANIENT", "2023-02-01"): "Hindenburg report", ("ADANIENT", "2023-02-02"): "Hindenburg report",
    ("OFSS", "2024-01-18"): "results", ("RECLTD", "2024-06-04"): "election results day",
    ("INDUSINDBK", "2025-03-11"): "derivatives accounting disclosure",
}
AUTO_REASON = {
    "open_tick": "open-only bad tick; close not affected, nothing to do",
    "bad_print": "print that reverts; the bad bar is in price_overrides.csv or only the open was wrong",
    "data_gap": "hole in the source data; handled by the trading-ratio screen and the stale-stock rule",
}


def main():
    cand = pd.read_csv(DATA / "ca_candidates.csv", parse_dates=["ex_date"])
    ca = load_ca(DATA / "corporate_actions.csv")
    key_ca = {(r.ticker, r.ex_date.strftime("%Y-%m-%d")): f"{r.action} {r.factor:g}" for r in ca.itertuples()}
    rows = []
    for r in cand.itertuples():
        k = (r.ticker, r.ex_date.strftime("%Y-%m-%d"))
        if k in key_ca:
            dec, why = "applied", key_ca[k]
        elif k in NO_ACTION:
            dec, why = "no_action", NO_ACTION[k]
        elif r.suggested in AUTO_REASON:
            dec, why = "no_action", AUTO_REASON[r.suggested]
        else:
            dec, why = "OPEN", "needs a decision"
        rows.append({"ex_date": k[1], "ticker": r.ticker, "suggested": r.suggested, "decision": dec, "reason": why})
    log = pd.DataFrame(rows)
    # CA rows that did not come from the scanner (boundary check, manual)
    extra = [{"ex_date": d, "ticker": t, "suggested": "-", "decision": "applied", "reason": f"{a} (not from the gap scan)"}
             for (t, d), a in key_ca.items() if not ((log.ticker == t) & (log.ex_date == d)).any()]
    log = pd.concat([log, pd.DataFrame(extra)]).sort_values(["ex_date", "ticker"])
    log.to_csv(DATA / "ca_review_log.csv", index=False)
    print(log["decision"].value_counts().to_string())
    still_open = log[log.decision == "OPEN"]
    if len(still_open):
        print("\nstill open:\n" + still_open.to_string(index=False))

    # ---- residual check
    cfg = load_config(ROOT / "config" / "imlv30.yaml")
    mkt = load_market(DATA / "panel.parquet", DATA / "sectors.csv", data_dir=DATA)
    _, _, rets, adj = prepare(mkt, ca)
    reviews = run_reviews(mkt, cfg, rets, adj)
    eligible = pd.DataFrame(False, index=mkt.calendar, columns=mkt.tickers)
    for rv, nxt in zip(reviews, reviews[1:] + [None]):
        end = nxt.review.effective if nxt else mkt.calendar[-1]
        names = rv.universe.index[rv.universe["eligible"]]
        eligible.loc[rv.review.determination - pd.Timedelta(days=400):end, names] = True
    big = rets.where(eligible).stack()
    big = big[big.abs() > 0.25]
    known = {(t, d) for (t, d) in NO_ACTION} | set(RESIDUAL_KNOWN)
    flagged = [(d.strftime("%Y-%m-%d"), t, round(v, 4), (t, d.strftime("%Y-%m-%d")) in known)
               for (d, t), v in big.items()]
    unexplained = [f for f in flagged if not f[3]]
    print(f"\nresidual |adjusted move| > 25% in eligible names: {len(flagged)}, unexplained: {len(unexplained)}")
    for f in unexplained:
        print("  ", f[0], f[1], f[2])


if __name__ == "__main__":
    main()
