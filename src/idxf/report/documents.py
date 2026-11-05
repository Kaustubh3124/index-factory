"""Text deliverables: index review notices (markdown) and an indicative term sheet (docx)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor


def review_notice(cfg, rv, prev_weights: pd.Series | None, sectors: pd.Series) -> str:
    r = rv.review
    lines = [
        f"# {cfg.name} ({cfg.ticker}) - {'Initial composition' if r.initial else 'Quarterly review'}",
        "",
        f"Determination date: {r.determination.date()}  ",
        f"Announcement date: {r.announcement.date()}  ",
        f"Effective after the close of: {r.effective.date()}",
        "",
    ]
    if not r.initial:
        lines += ["## Changes", "", "| Action | Ticker | Sector | Rank on determination date |", "|---|---|---|---|"]
        for t in rv.adds:
            lines.append(f"| Addition | {t} | {sectors.get(t, '')} | {int(rv.scores.loc[t, 'rank'])} |")
        for t in rv.deletes:
            rk = rv.scores["rank"].get(t)
            if pd.notna(rk):
                why = f"{int(rk)}"
            else:
                u = rv.universe.loc[t] if t in rv.universe.index else None
                if u is not None and u["reason"] == "outside_top_n":
                    why = f"not eligible: liquidity rank {int(u['liq_rank'])} (top {len(rv.universe[rv.universe.eligible])} kept)"
                elif u is not None:
                    why = f"not eligible: {u['reason']}"
                else:
                    why = "not eligible: no data"
            lines.append(f"| Deletion | {t} | {sectors.get(t, '')} | {why} |")
        if not rv.adds and not rv.deletes:
            lines.append("| - | No constituent changes | | |")
        lines.append("")
    lines += ["## Target weights at the effective date", "", "| Ticker | Sector | Weight |", "|---|---|---|"]
    for t, w in rv.weights.items():
        lines.append(f"| {t} | {sectors.get(t, '')} | {w * 100:.2f}% |")
    c = rv.checks
    lines += ["", f"Constituents: {c['n']}, effective N: {c['effective_n']:.1f}, "
                  f"max stock weight {c['max_stock'] * 100:.2f}%, max sector weight {c['max_sector'] * 100:.2f}%.", ""]
    return "\n".join(lines)


def write_notices(out_dir: Path, cfg, run, sectors):
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = None
    for rv in run.reviews:
        name = f"{rv.review.effective.date()}_{'initial' if rv.review.initial else 'review'}.md"
        (out_dir / name).write_text(review_notice(cfg, rv, prev, sectors))
        prev = rv.weights


def term_sheet_docx(path, cfg, note, pricing: dict, greeks: dict, S0: float, trade_date: pd.Timestamp):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(9.5)

    h = doc.add_paragraph()
    run = h.add_run("INDICATIVE TERMS - FOR DISCUSSION PURPOSES ONLY")
    run.bold, run.font.size, run.font.color.rgb = True, Pt(8), RGBColor(0xB5, 0x47, 0x3A)
    t = doc.add_paragraph()
    run = t.add_run(f"{note.T:.0f}Y Principal Protected Market-Linked Note on the {cfg.name}")
    run.bold, run.font.size = True, Pt(14)

    avg = (f"Arithmetic average of the Index closing levels on the last {note.avg_months} monthly "
           f"Averaging Dates up to and including the Final Valuation Date") if note.avg_months else \
        "Index closing level on the Final Valuation Date"
    cap_txt = f"{note.cap * 100:.0f}%" if note.cap is not None else "None"
    rows = [
        ("Issuer", "[Issuer] (illustrative)"),
        ("Underlying", f"{cfg.name} (Bloomberg-style ticker: {cfg.ticker}), price return"),
        ("Currency", "INR"),
        ("Denomination", "INR 1,00,000 per Note"),
        ("Trade date / Initial Valuation Date", str(trade_date.date())),
        ("Tenor", f"{note.T:.0f} years"),
        ("Initial Level", f"Index closing level on the Initial Valuation Date (indicatively {S0:,.2f})"),
        ("Final Level", avg),
        ("Index Performance", "max(0, Final Level / Initial Level - 1)"),
        ("Principal Protection", f"{note.protection * 100:.0f}% of Denomination at maturity, subject to Issuer credit"),
        ("Participation Rate", f"{note.participation * 100:.1f}% (indicative, fixed on trade date)"),
        ("Cap on Index Performance", cap_txt),
        ("Redemption Amount", "Denomination x [Principal Protection + Participation Rate x min(Index Performance, Cap)]"),
        ("Coupon", "None"),
        ("Fees", f"{note.fees * 100:.2f}% upfront, included in the participation rate"),
        ("Calculation Agent", "[Issuer]"),
    ]
    tbl = doc.add_table(rows=0, cols=2)
    tbl.style = "Light Grid Accent 1"
    for k, v in rows:
        c = tbl.add_row().cells
        c[0].text, c[1].text = k, v
        c[0].paragraphs[0].runs[0].bold = True

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Pricing summary (per 100 notional, internal)").bold = True
    pr = doc.add_table(rows=0, cols=2)
    pr.style = "Light List Accent 1"
    items = [
        ("Model", "Black-Scholes / GBM Monte Carlo, flat rate and dividend yield"),
        ("Rate / dividend yield / vol", f"{pricing['r'] * 100:.2f}% / {pricing['q'] * 100:.2f}% / {pricing['sigma'] * 100:.2f}%"),
        ("Zero-coupon bond", f"{pricing['zcb']:.2f}"),
        ("Option value at 100% participation", f"{pricing['option_unit']:.2f}"),
        ("Note value to investor", f"{pricing['value']:.2f}  (MC std err {pricing['std_err']:.3f})"),
        ("Delta per 1% index move", f"{greeks['delta_1pct']:.3f}"),
        ("Vega per vol point", f"{greeks['vega_1vol']:.3f}"),
        ("Rho per bp", f"{greeks['rho_1bp']:.4f}"),
        ("Theta per day", f"{greeks['theta_1d']:.4f}"),
    ]
    for k, v in items:
        c = pr.add_row().cells
        c[0].text, c[1].text = k, v

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Key risks").bold = True
    for txt in [
        "Principal protection depends on the Issuer's credit; it is not a guarantee by any third party.",
        "Returns are linked to a price-return index: dividends paid by index constituents are not passed on.",
        "Index levels before the launch date are back-tested and do not represent actual performance.",
        "The Issuer hedges with liquid index futures/options on a proxy index; the hedge carries basis risk "
        "that is priced into the Participation Rate.",
        "Early exit before maturity is at the prevailing secondary price, which can be below Denomination.",
    ]:
        doc.add_paragraph(txt, style="List Bullet")

    f = doc.add_paragraph("This document is a student project illustration and not an offer of securities.")
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    f.runs[0].font.size = Pt(7)
    f.runs[0].italic = True
    doc.save(path)
