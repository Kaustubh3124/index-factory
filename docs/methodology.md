# India Momentum Low-Vol 30 Index (IMLV30): Index Methodology

Version 1.0. All parameters live in `config/imlv30.yaml`; section references below are used in the code comments.

---

## 1. Index objective

IMLV30 tracks 30 liquid Indian equities with strong *risk-adjusted* price momentum and low volatility. The two signals are blended because each one on its own has an obvious failure mode. Pure momentum indices crash hard at turning points (March 2020 is the textbook case) and pure low-vol indices drift into a staples-and-utilities basket that lags in strong markets. The blend tries to keep most of the momentum premium while cutting the drawdowns.

IMLV30 is a price return index in INR. Regular cash dividends are not reinvested.

## 2. Definitions

| Term | Meaning |
|---|---|
| Trading day | A day on which the NIFTY 50 is calculated, after removing non-sessions (section 10) |
| Close | Official-style closing price: volume-weighted average of trades from 15:00 to 15:29:59 IST |
| Determination date (D) | Data cut-off for a review. 10 trading days before E |
| Announcement date (A) | Review results are published. 5 trading days before E |
| Effective date (E) | Last trading day of March, June, September and December. Index shares are set on E's close and the new composition trades from E+1 |
| Index shares | Notional number of shares of each constituent held by the index |
| Median traded value | Median of daily traded value (INR) over the liquidity window, counting only days the stock traded |

## 3. Eligible universe

### 3.1 Starting universe

All NSE equities in the data set (see section 10 for what that means). NIFTY index series are excluded.

### 3.2 Screens, applied on D in this order

1. **History**: at least 260 trading days of price history.
2. **Trading frequency**: traded on at least 95% of trading days in the last 126.
3. **Liquidity**: median daily traded value over the last 126 trading days of at least INR 25 crore.
4. **Price**: close on D of at least INR 20.
5. **Manual exclusions**: none at present.
6. **Traded on D**: must have a trade on the determination date.

### 3.3 Size proxy

The usual size measure is free-float market cap, but the data has no share counts or free-float factors. Median traded value is used instead. Securities that pass 3.2 are ranked on median traded value and the top 150 form the **Eligible Universe**, which in practice is roughly the NIFTY 100 plus the more liquid mid caps.

There is no buffer on this cut-off. Section 12 explains why that is the weakest rule in the book.

## 4. Factor scores

All returns use corporate-action-adjusted prices (section 9). For each stock in the Eligible Universe on D:

- 12-month momentum: `M12 = P(D-21) / P(D-252) - 1`
- 6-month momentum: `M6 = P(D-21) / P(D-126) - 1`

  The latest month (21 trading days) is skipped because one-month returns tend to reverse, which only adds noise to a momentum signal.
- Volatility: `σ = stdev(daily returns over 252 days) × √252`
- Risk-adjusted momentum: `RAM = ½·z(M12/σ) + ½·z(M6/σ)`
- Momentum z-score: `Z_MOM = z(RAM)`
- Low-volatility z-score: `Z_LV = z(−σ)`
- Composite score: `S = 0.5·Z_MOM + 0.5·Z_LV`

`z(·)` is the cross-sectional z-score inside the Eligible Universe (population standard deviation), winsorised at ±3. Stocks are ranked on S, highest first; ties go to the stock with the higher median traded value.

## 5. Constituent selection

Target: **30** constituents.

1. Every stock ranked 1 to 20 is selected.
2. Existing constituents ranked 21 to 40 are kept, in rank order, until 30 are selected.
3. Any remaining places go to the highest-ranked stocks not yet selected.

The buffer keeps turnover down. A constituent drifting from rank 28 to rank 33 is not sold, and a non-constituent at rank 25 has to reach the top 20 before it is bought, unless rule 3 needs it to fill a slot.

## 6. Weighting

1. **Inverse volatility**: `w_i ∝ 1/σ_i`, with σ over the last 126 trading days up to D.
2. **Stock cap and floor**: no stock above 8% or below 1%. Weights above the cap are set to the cap and the excess goes to the uncapped stocks in proportion to their weights. Repeat until nothing breaches.
3. **Sector cap**: no sector above 30%. A sector above the cap is scaled down to it and the excess goes to the other sectors pro rata; the stock cap is then re-applied inside the sectors that are not capped. Alternate until both constraints hold.

If the sector step pushes a stock below the 1% floor, the floor is not re-imposed. The review output reports `min_stock` so this is visible rather than silently forced.

Target weights are computed on D and implemented at the close of E, so weights at E equal the targets exactly. Between reviews they drift with prices.

## 7. Review schedule

| Event | Timing |
|---|---|
| Determination | E − 10 trading days |
| Announcement | E − 5 trading days |
| Effective | Close of the last trading day of Mar / Jun / Sep / Dec |

The base date (28 June 2019) is treated as the initial review.

## 8. Index calculation

```
        Σ_i  N_i,t × P_i,t
I_t  =  ──────────────────
               D_t
```

- `N_i,t`: index shares of stock i
- `P_i,t`: close of stock i on day t (last available close if it did not trade)
- `D_t`: divisor

**Base**: I = 1000 at the close of the base date, with D = 1.

**Rebalancing**: at the close of E, new index shares are set so each stock gets its target weight and the index market value does not change:

```
N_i,new = w_i × (I_E × D_E) / P_i,E
```

Because market value is held constant, a review never changes the divisor. Providers that hold real free-float share counts change the divisor at every review instead; the two set-ups give the same index levels.

**When the divisor does change**: only when something moves the index market value without being an investment return. In this methodology that is a special dividend (9.3). Splits and bonuses never touch the divisor; they go through index shares. Over the back-test no special dividend hit a constituent, so the published divisor stayed at 1.0. The divisor path is covered by the test suite.

## 9. Corporate actions and other events

General principle: only market moves should move the index. A corporate action must not.

### 9.1 Splits, consolidations and bonus issues

On the ex-date, index shares are multiplied by the share factor f (new shares per old share: 2 for a 1:1 bonus, 5 for a split from FV 10 to FV 2). The divisor does not change.

### 9.2 Demergers (spin-offs)

The demerged company is not added to the index. The index is treated as selling the demerged shares at their theoretical value and putting the proceeds back into the parent, which works out as a share adjustment on the parent with

```
f = P_cum / P_ex
```

where `P_cum` is the parent's close before the ex-date and `P_ex` is the parent's price discovered in the special pre-open session on the ex-date. This is a simplification; some providers keep the spin-off in the index for a few days instead.

### 9.3 Special dividends

The index does not include dividends, but a special dividend is a return of capital, not a market move. On the ex-date:

```
D_new = D_old × (MV_cum − C) / MV_cum
```

where `MV_cum` is the index market value at the previous close and `C` is the cash paid on the index shares. Regular dividends are not adjusted.

### 9.4 Suspensions and delistings

A constituent with no trades for 5 consecutive trading days is removed at the close of the fifth day at its last traded price. The proceeds go to the remaining constituents in proportion to their weights, so the divisor does not change. The slot stays empty until the next review.

### 9.5 Rights issues, mergers, buy-backs

Rights issues are not modelled. One with a meaningful discount would show up in the gap scan, and none was identified among the 1,154 scanner candidates reviewed (`data/ca_review_log.csv`). If one had been, it would be handled as a price adjustment using the theoretical ex-rights price.

Mergers where the index holds the target are treated as a conversion into the acquirer at the exchange ratio. The only case in the data is IDFC Ltd into IDFC First Bank (155:100), see `data/corporate_actions.csv`.

### 9.6 Data quality checks

Every day the engine flags constituents with an adjusted daily return above 35% in absolute value. Flags go to `output/events.csv` for review; they do not change the calculation. There were none over the back-test once the data cleaning in section 10 was done.

## 10. Data

- **Source**: public 1-minute bars for the NIFTY 500 constituents and NIFTY 50, 2018 to October 2026, aggregated to daily bars with `scripts/build_daily.py`. Prices are raw, so corporate actions have to be applied explicitly.
- **Survivorship bias**: the universe comes from the NIFTY 500 list at the time the files were built. Companies that dropped out or were delisted earlier are missing, which flatters any back-test. The effect is smaller for momentum than for value, since the losers that disappeared would mostly not have been selected anyway, but it is there.
- **No free-float data**: size is proxied by median traded value (3.3).
- **No dividends**: hence a price return index only.
- **Cleaning** (details and decisions in `data/README.md`):
  - two non-sessions removed from the calendar, including a Saturday mock-trading session in November 2025 that showed the NIFTY 50 down 10%
  - six bad bars treated as no-trade days
  - two symbol changes spliced (TATAMOTORS to TMPV, LTIM to LTM)
  - 89 corporate-action and data-fix entries, each checked against price, volume and open/close evidence
  - the 2018-2025 and 2026 source files disagree on back-adjustment for 14 stocks; the join is handled by `corpactions.boundary_scaling`
- **Back-test**: all index levels are hypothetical. The rules and parameters were written into the config before the first back-test run and are round numbers taken from published factor index methodologies, not tuned on this data. Section 11 shows the alternatives that were tested afterwards and were not adopted.

## 11. Design choices tested

Each rule had to justify itself against the version without it. Same data, same dates, 28 Jun 2019 to 16 Oct 2026 (`scripts/run_ablation.py`):

| Variant | CAGR | Net of costs | Vol | Return / vol | Max DD | Avg one-way turnover | Max sector weight |
|---|---|---|---|---|---|---|---|
| A. Momentum only, equal weight, no buffer | 16.9% | 16.0% | 22.3% | 0.75 | −38.6% | 56% | 50% |
| B. 50/50 blend, equal weight, no buffer | 16.3% | 15.6% | 18.3% | 0.89 | −34.9% | 49% | 47% |
| C. Blend, inverse-vol weights, no buffer | 14.9% | 14.3% | 17.6% | 0.85 | −34.0% | 48% | 44% |
| D. C plus the 20/40 buffer | 14.8% | 14.3% | 17.2% | 0.86 | −32.9% | 39% | 48% |
| **E. D plus the 30% sector cap (IMLV30)** | **14.7%** | **14.2%** | **17.2%** | **0.86** | **−32.9%** | **39%** | **30%** |
| F. Low-vol only | 11.1% | 11.0% | 14.6% | 0.76 | −30.3% | 14% | 30% |

What this says:

- **Blending in low vol** (A to B) cut volatility from 22% to 18% and improved return per unit of risk from 0.75 to 0.89, for 0.6% of CAGR.
- **The buffer** (C to D) cut turnover from 48% to 39% a review and cost from 13.8 to 10.9 bps, with no loss of return.
- **The sector cap** (D to E) brought the largest sector from 48% to 30% for 0.1% of CAGR.
- **Inverse-vol weighting did not pay in this sample** (B to C). It lowered vol a little but cost 1.4% of CAGR and the best risk-adjusted variant is B. It stays in the rulebook because switching after looking at one 7-year back-test would be fitting to that back-test. It is the first rule to revisit with more data, especially since inverse vol also tilts the index towards less liquid defensive names (section 12).

## 12. Known issues

1. **No buffer on the universe size cut (3.3).** Stocks that sit near liquidity rank 150 can be forced out and back in for reasons unrelated to the factors. Marico was a top holding, slipped to liquidity rank 152 at the June 2026 review, was deleted, and came back in September at rank 148. Nykaa (rank 180) did the same, and Jindal Steel went out in September at rank 168. The fix is the same 20/40-style buffer applied to the size cut, for example incumbents stay eligible down to rank 175.
2. **Capacity is set by the defensive names.** Inverse-vol weights favour low-volatility stocks, which in India are often less traded (Colgate, Shree Cement, Torrent Power and Coromandel were the hardest trades in 2019-2020). At INR 2,000 crore tracking AUM the hardest trade needed up to about 10 days at 20% participation in 2020; median about 4 days.
3. **Survivorship bias** (section 10).
4. **Price return only.** A total return version needs a dividend file.
