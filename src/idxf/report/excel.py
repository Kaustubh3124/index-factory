"""Excel workbook with live formulas.

The point of this file is that someone on a desk can open the workbook and
re-derive the index level and the note price without trusting the Python:
  Constituents  : MV = shares * price, weight = MV / total, level = total MV / divisor
  Note_Pricing  : Black-Scholes call (NORMSDIST), zero-coupon bond, participation solve
"""
from __future__ import annotations

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HDR = Font(bold=True, color="FFFFFF")
HDR_FILL = PatternFill("solid", fgColor="1F3B5A")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN = Border(bottom=Side(style="thin", color="D0D4DA"))


def _header(ws, row, labels):
    for j, h in enumerate(labels, 1):
        c = ws.cell(row=row, column=j, value=h)
        c.font, c.fill = HDR, HDR_FILL
        c.alignment = Alignment(horizontal="center")


def _widths(ws, widths):
    for j, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(j)].width = w


def build_workbook(path, cfg, run, mkt, stats, rel, costs, recon, note_inputs: dict):
    wb = Workbook()

    # ---- Constituents (live)
    ws = wb.active
    ws.title = "Constituents"
    d = run.level.index[-1]
    # current shares = shares implied by today's weights (includes any CA since the last review)
    cur_w = run.weights.loc[d]
    held = cur_w[cur_w > 0].index
    px = mkt.close.loc[d, held]
    mv_target = run.level.iloc[-1] * run.divisor.iloc[-1]
    cur_shares = cur_w[held] * mv_target / px

    ws["A1"] = f"{cfg.name} ({cfg.ticker}) - holdings as of {d.date()}"
    ws["A1"].font = Font(bold=True, size=12)
    ws["A2"] = "Yellow cells are inputs. Everything else is a formula."
    _header(ws, 4, ["Ticker", "Sector", "Index shares", "Close (INR)", "Market value", "Weight"])
    r0 = 5
    for i, t in enumerate(cur_w[held].sort_values(ascending=False).index):
        r = r0 + i
        ws.cell(r, 1, t)
        ws.cell(r, 2, mkt.sectors.get(t, ""))
        ws.cell(r, 3, float(cur_shares[t])).number_format = "0.000000"
        c = ws.cell(r, 4, float(px[t]))
        c.fill = INPUT_FILL
        c.number_format = "#,##0.00"
        ws.cell(r, 5, f"=C{r}*D{r}").number_format = "#,##0.00"
        ws.cell(r, 6, f"=E{r}/$E${r0 + len(held) + 1}").number_format = "0.00%"
    rt = r0 + len(held) + 1
    ws.cell(rt, 4, "Total MV").font = Font(bold=True)
    ws.cell(rt, 5, f"=SUM(E{r0}:E{rt - 2})").number_format = "#,##0.00"
    ws.cell(rt, 6, f"=SUM(F{r0}:F{rt - 2})").number_format = "0.00%"
    ws.cell(rt + 1, 4, "Divisor")
    c = ws.cell(rt + 1, 5, float(run.divisor.iloc[-1]))
    c.number_format = "0.0000000000"
    ws.cell(rt + 2, 4, "Index level").font = Font(bold=True)
    ws.cell(rt + 2, 5, f"=E{rt}/E{rt + 1}").number_format = "#,##0.00"
    ws.cell(rt + 3, 4, "Python level")
    ws.cell(rt + 3, 5, float(run.level.iloc[-1])).number_format = "#,##0.00"
    ws.cell(rt + 4, 4, "Check (should be 0)")
    ws.cell(rt + 4, 5, f"=E{rt + 2}-E{rt + 3}").number_format = "0.000000"
    _widths(ws, [14, 22, 16, 14, 18, 10])
    ws.freeze_panes = "A5"

    # ---- History
    ws = wb.create_sheet("History")
    _header(ws, 1, ["Date", cfg.ticker, "Divisor", "NIFTY 50", "Index daily ret", "NIFTY daily ret"])
    b = mkt.benchmark.reindex(run.level.index)
    for i, (dt, lv) in enumerate(run.level.items()):
        r = i + 2
        ws.cell(r, 1, dt.to_pydatetime()).number_format = "yyyy-mm-dd"
        ws.cell(r, 2, float(lv)).number_format = "0.00"
        ws.cell(r, 3, float(run.divisor.loc[dt])).number_format = "0.0000000000"
        ws.cell(r, 4, float(b.loc[dt])).number_format = "0.00"
        if i > 0:
            ws.cell(r, 5, f"=B{r}/B{r - 1}-1").number_format = "0.00%"
            ws.cell(r, 6, f"=D{r}/D{r - 1}-1").number_format = "0.00%"
    _widths(ws, [12, 12, 16, 12, 14, 14])
    ws.freeze_panes = "A2"

    # ---- Reviews
    ws = wb.create_sheet("Reviews")
    _header(ws, 1, ["Effective", "Determination", "Adds", "Deletes", "One-way turnover", "Cost (bps)",
                    "Max days to trade", "Stock cap ok", "Sector cap ok", "Effective N"])
    for i, rv in enumerate(run.reviews):
        r = i + 2
        e = rv.review.effective
        ws.cell(r, 1, e.to_pydatetime()).number_format = "yyyy-mm-dd"
        ws.cell(r, 2, rv.review.determination.to_pydatetime()).number_format = "yyyy-mm-dd"
        ws.cell(r, 3, ", ".join(rv.adds) if not rv.review.initial else "(initial build)")
        ws.cell(r, 4, ", ".join(rv.deletes))
        if e in costs.index:
            ws.cell(r, 5, float(costs.loc[e, "one_way_turnover"])).number_format = "0.0%"
            ws.cell(r, 6, float(costs.loc[e, "cost_bps"])).number_format = "0.0"
            ws.cell(r, 7, float(costs.loc[e, "max_days_to_trade"])).number_format = "0.00"
        ws.cell(r, 8, rv.checks["stock_cap_ok"])
        ws.cell(r, 9, rv.checks["sector_cap_ok"])
        ws.cell(r, 10, round(rv.checks["effective_n"], 1))
    _widths(ws, [12, 13, 60, 60, 15, 10, 15, 12, 12, 11])

    # ---- Note pricing (live BS)
    ws = wb.create_sheet("Note_Pricing")
    ni = note_inputs
    ws["A1"] = "Principal-protected note on the index - point-to-point, closed form"
    ws["A1"].font = Font(bold=True, size=12)
    rows = [("Spot (index level)", ni["S0"], "0.00"), ("Strike (% of spot)", 1.0, "0%"),
            ("Tenor (years)", ni["T"], "0.00"), ("Rate (cont.)", ni["r"], "0.00%"),
            ("Dividend yield", ni["q"], "0.00%"), ("Volatility", ni["sigma"], "0.00%"),
            ("Protection", ni["protection"], "0%"), ("Fees (upfront)", ni["fees"], "0.00%")]
    for i, (lab, v, fmt) in enumerate(rows):
        ws.cell(3 + i, 1, lab)
        c = ws.cell(3 + i, 2, float(v))
        c.number_format, c.fill = fmt, INPUT_FILL
    # B3 S, B4 k, B5 T, B6 r, B7 q, B8 vol, B9 prot, B10 fees
    f = [
        ("K", "=B3*B4", "0.00"),
        ("d1", "=(LN(B3/B12)+(B6-B7+0.5*B8^2)*B5)/(B8*SQRT(B5))", "0.0000"),
        ("d2", "=B13-B8*SQRT(B5)", "0.0000"),
        # NORMSDIST rather than NORM.S.DIST: openpyxl writes the newer name without the
        # _xlfn. prefix and both Excel and LibreOffice then show #NAME?
        ("Call price (index pts)", "=B3*EXP(-B7*B5)*NORMSDIST(B13)-B12*EXP(-B6*B5)*NORMSDIST(B14)", "0.00"),
        ("Call per 100 notional", "=B15/B3*100", "0.00"),
        ("Zero-coupon bond (per 100)", "=100*B9*EXP(-B6*B5)", "0.00"),
        ("Option budget (per 100)", "=100*(1-B10)-B17", "0.00"),
        ("Participation", "=B18/B16", "0.0%"),
        ("Call delta", "=EXP(-B7*B5)*NORMSDIST(B13)", "0.0000"),
        # note the brackets: in Excel unary minus binds tighter than ^, so -B13^2 is (+B13^2)
        ("Note vega per vol pt (per 100)", "=B19*EXP(-B7*B5)*EXP(-(B13^2)/2)/SQRT(2*PI())*SQRT(B5)", "0.000"),
        ("Python participation (check)", float(ni["participation_cf"]), "0.0%"),
    ]
    for i, (lab, form, fmt) in enumerate(f):
        ws.cell(12 + i, 1, lab)
        ws.cell(12 + i, 2, form).number_format = fmt
    _widths(ws, [34, 16])

    # ---- Stats & recon
    ws = wb.create_sheet("Stats")
    _header(ws, 1, ["Metric", "Value"])
    r = 2
    for k, v in {**stats, **rel}.items():
        if isinstance(v, (int, float)):
            ws.cell(r, 1, k)
            ws.cell(r, 2, float(v)).number_format = "0.0000"
            r += 1
    r += 1
    ws.cell(r, 1, "Recon: production vs shadow").font = Font(bold=True)
    for k in ("days", "max_rel_diff", "n_breaks", "passed"):
        r += 1
        ws.cell(r, 1, k)
        ws.cell(r, 2, recon[k])
    _widths(ws, [32, 18])

    wb.save(path)
