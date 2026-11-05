# Data

`panel.parquet` (daily bars for 578 stocks and the NIFTY 50, 2018 to Oct 2026, about 18 MB) is committed, so `idxf run`, the tests and the dashboard work straight after cloning. The 1-minute source data (several GB) is not committed. To rebuild the panel from scratch:

1. Get the 1-minute bars. I used the public datasets at
   github.com/ganeshbiyer/Nse_Historical_Data (2018-2025) and
   github.com/ganeshbiyer/Nse_Historical_Data_2026 (2026). Both are several GB; clone with
   `--filter=blob:none --no-checkout` and check files out in batches if disk is tight.
2. Aggregate to daily bars, one file per ticker:

       python scripts/build_daily.py <repo_2018_2025> data/daily/2018_2025 --delete-src
       python scripts/build_daily.py <repo_2026> data/daily/2026 --delete-src

3. Stack into one panel (the later folder wins on overlapping dates):

       idxf build-panel data/daily/2018_2025 data/daily/2026

4. Re-run the scan and the review log if anything in the source changed:

       idxf detect-ca
       python scripts/build_ca_review_log.py

## Files

| File | What it is |
|---|---|
| `corporate_actions.csv` | Everything the engine applies: splits, bonuses, demergers, one merger, one special dividend and `data_fix` rows. `factor` = new shares per old share. Every row has a note. |
| `ca_candidates.csv` | Raw output of the gap scanner with the evidence columns (implied factor from the open and from the close, 5-day persistence, volume ratio, data-gap length). |
| `spike_candidates.csv` | Single-day closes that are fully reversed the next day. |
| `ca_review_log.csv` | One line per scanner candidate with the decision taken and why. |
| `price_overrides.csv` | Bars treated as "no trade that day". |
| `calendar_exclusions.csv` | Dates in the source that were not real sessions. |
| `symbol_changes.csv` | Ticker renames spliced into one series. |
| `sectors.csv` | GICS-like level-1 sector for every ticker, mapped by hand. |

## What the cleaning found

The source is a good free dataset, but it is not an index provider's feed, and most of the work in this project ended up here.

**Raw prices, not adjusted.** Mostly. Splits and bonuses show up as overnight gaps, which is what we want, because the engine applies them itself. But a few tickers are partly adjusted:

- Persistent Systems flips between a half-scaled and an unscaled series four times between 2019 and 2023.
- Elgi Equipments doubles in level overnight in September 2020 with nothing behind it.
- Tata Steel's 2022 1:10 split is already back-adjusted in the file and never shows up.

All of these go in as `data_fix` entries (the same mechanics as a split) so the index market value stays continuous.

**The 2026 file is partly back-adjusted for 2026 events.** For 14 stocks the closes in the 2026 file are pre-divided by some factor k1, while the `open` field stays on the raw basis until the real ex-date. Kotak Bank is the cleanest example: ×3.0 at the file join on 1 Jan 2026 and another ×1.667 on the real ex-date of 12 Jan, which together make its 1:5 split. The fingerprint is that open / previous close sits at k1 every day for weeks. `corpactions.boundary_scaling` uses this to separate vendor rescaling from real moves on the join date. ITC (−10%) and Godfrey Phillips (−17%) on 1 Jan 2026 were real; their next-day opens were normal.

**A mock trading session.** Saturday 8 Nov 2025 has bars for about 550 tickers and shows the NIFTY 50 at −10%, with around 100 stocks gapping 20% and reverting on Monday. NSE runs mock sessions on some Saturdays and this one leaked into the data. Removed from the calendar, along with a one-bar "session" on 16 May 2026. The real weekend sessions are kept: the Budget days, the 2024 DR-site live sessions and the Diwali Muhurat sessions.

**Bad bars.** Six one-day bars at roughly 2x or 0.5x the surrounding prices (Cochin Shipyard, HAL and Sonata Software on 21 Sep 2023 all came from the same bad file day). They are treated as no-trade days.

**Ticker histories that are not what the name says.** The IDFCFIRSTB file before 10 Oct 2024 is IDFC Ltd, the holding company that merged into the bank at 155:100. It is handled as a security conversion. TATAMOTORS becomes TMPV after the October 2025 demerger, and LTIM appears under both LTIM and LTM in 2026.

**Holes.** Several stocks have gaps of weeks or months (Patanjali, ICICI Securities, Techno Electric, Anant Raj, Welspun Living). The trading-ratio screen keeps them out of the universe, and the stale-stock rule would remove a constituent. No constituent was ever affected.

**Universe drift at the join.** 49 tickers in the 2018-2025 file are missing from the 2026 file because the 2026 set is the 2026 NIFTY 500 list. For the index this looks like a stock that stops trading, which the 5-day stale rule handles without look-ahead. No constituent was affected.

## Decision rules used on the scan

1. Gap matches a standard ratio, persists for at least 5 days and fits a known event: apply the announced ratio.
2. Demerger: factor from the pre-open price on the ex-date (methodology 9.2), not from the close. Siemens on 7 Apr 2025 is the example where it matters. The close-based number (1.70) includes a 12% fall on the tariff sell-off day; the open-based one is 1.94.
3. Gap that reverts, open-only bad tick, or data hole: no action, reason logged.
4. Genuine market move (Yes Bank 2020, Adani after Hindenburg, Paytm, IIFL, the 23 Mar 2020 crash): no action, reason logged.
5. Materiality: any candidate still uncertain after 1-4 was checked against index membership. The four that stayed uncertain (Tata Comm 2019, BEML 2022, TARIL 2025, Central Bank 2018) were never held, so the decision does not affect the index.

Final state: 1,154 scanner candidates plus 12 entries from the boundary check and manual research. 89 applied, 1,077 no action with a reason, 0 open. The residual check in `build_ca_review_log.py` finds 34 adjusted daily moves above 25% in eligible names, all of them matched to real events.
