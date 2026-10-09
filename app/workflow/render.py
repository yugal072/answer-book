"""Serving formats for a :class:`SolvedPaper`: JSON, SSE frames, PDF.

All three are rendered from the SAME solved-paper dict, so they cannot drift:
the JSON file is the dict, the SSE terminal event summarises it, and the PDF
prints it. Nothing here re-derives a status or a confidence.
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape

FONT_DIR = Path(__file__).parent / "fonts"

STATUS_LABEL = {
    "verified": "VERIFIED",
    "cached": "VERIFIED (reused from knowledge base)",
    "unverified": "UNVERIFIED - TEACHER REVIEW REQUIRED",
    "failed": "FAILED - NO ANSWER PRODUCED",
}
OUTCOME_LABEL = {
    "complete": "Complete - every answer verified",
    "needs_review": "Needs review - some answers are unverified",
    "partial": "Partial - some questions failed",
    "failed": "Failed - no question could be answered",
}


# --------------------------------------------------------------------------- JSON
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def safe_paper_id(paper_id: str) -> str:
    """Paper ids become file names; never let one escape the output directory."""
    cleaned = _SAFE_ID.sub("_", paper_id or "")
    if not cleaned or cleaned.startswith("."):
        raise ValueError(f"invalid paper id: {paper_id!r}")
    return cleaned


def write_json(solved: Dict[str, Any], output_dir: Path) -> Path:
    """Atomically write ``<output_dir>/<paper_id>.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{safe_paper_id(solved['paper_id'])}.json"
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(solved, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


# --------------------------------------------------------------------------- SSE
def format_sse(event_id: int, event: str, data: Dict[str, Any]) -> str:
    """One Server-Sent-Events frame (``id`` / ``event`` / ``data`` + blank line)."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event_id}\nevent: {event}\ndata: {payload}\n\n"


# --------------------------------------------------------------------------- PDF
def _register_fonts() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular, bold = FONT_DIR / "DejaVuSans.ttf", FONT_DIR / "DejaVuSans-Bold.ttf"
    if regular.exists() and bold.exists():
        if "AB-Sans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("AB-Sans", str(regular)))
            pdfmetrics.registerFont(TTFont("AB-Sans-Bold", str(bold)))
        return "AB-Sans", "AB-Sans-Bold"
    return "Helvetica", "Helvetica-Bold"  # last resort; non-Latin glyphs may not render


def _p(text: Any) -> str:
    return escape(str(text if text is not None else "")).replace("\n", "<br/>")


def build_pdf(solved: Dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font, bold = _register_fonts()
    base = ParagraphStyle("base", fontName=font, fontSize=9.5, leading=13)
    h1 = ParagraphStyle("h1", parent=base, fontName=bold, fontSize=17, leading=21, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=base, fontName=bold, fontSize=11, leading=14, spaceBefore=8, spaceAfter=3)
    small = ParagraphStyle("small", parent=base, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"))
    status_colors = {"verified": "#0b6b2f", "cached": "#0b6b2f", "unverified": "#9a5b00", "failed": "#a11"}

    buf = io.BytesIO()
    meta = solved.get("metadata", {})
    title = f"Solution Book - {meta.get('subject') or 'Question Paper'}"
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=title, author="Answer Book")
    s = solved["summary"]
    story: List[Any] = [
        Paragraph(_p(title), h1),
        Paragraph(
            _p(" | ".join(x for x in [meta.get("board"), meta.get("class_name") or meta.get("class"), f"Paper {solved['paper_id']}"] if x)),
            small,
        ),
        Spacer(1, 4),
    ]
    tbl = Table(
        [["Outcome", OUTCOME_LABEL.get(solved["outcome"], solved["outcome"])],
         ["Questions", f"{s['total']}  (verified {s['verified']}, reused {s['cached']}, unverified {s['unverified']}, failed {s['failed']})"],
         ["Total marks", str(solved.get("total_marks") or "-")]],
        colWidths=[28 * mm, 140 * mm],
    )
    tbl.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 9),
                             ("FONTNAME", (0, 0), (0, -1), bold), ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                             ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0"))]))
    story += [tbl, Spacer(1, 6)]
    if solved["outcome"] != "complete":
        story.append(Paragraph("<b>Note:</b> answers marked UNVERIFIED or FAILED were not confirmed by the verification "
                               "pipeline and must be checked by a teacher before use.", base))

    for r in solved["results"]:
        q, sol = r["question"], r["solution"]
        col = status_colors.get(r["status"], "#000000")
        head = Paragraph(
            f"Q{_p(q['number'])}  <font size=8 color='#555555'>[{q['marks']} mark(s) | {_p(q['type'])}"
            f"{' | section ' + _p(q['section']) if q.get('section') else ''}]</font>  "
            f"<font size=8 color='{col}'><b>{_p(STATUS_LABEL.get(r['status'], r['status']))}</b></font>", h2)
        block: List[Any] = [head, Paragraph(_p(q["text"]), base)]
        if q.get("options"):
            block.append(Paragraph("<br/>".join(_p(o) for o in q["options"]), base))
        if sol.get("answer"):
            block.append(Spacer(1, 3))
            block.append(Paragraph(f"<b>Answer:</b> {_p(sol['answer'])}", base))
            if sol.get("steps"):
                block.append(Paragraph("<b>Steps:</b><br/>" + "<br/>".join(f"{i}. {_p(x)}" for i, x in enumerate(sol["steps"], 1)), base))
            if sol.get("mark_split"):
                block.append(Paragraph("<b>Mark split:</b> " + "; ".join(f"{_p(m.get('marks'))} - {_p(m.get('for'))}" for m in sol["mark_split"]), base))
            if sol.get("common_mistakes"):
                block.append(Paragraph("<b>Common mistakes:</b><br/>" + "<br/>".join(
                    f"- {_p(m.get('wrong'))}: {_p(m.get('why'))}" for m in sol["common_mistakes"]), base))
        else:
            block.append(Paragraph("<i>No answer was produced for this question.</i>", base))
        foot = f"verified_by: {sol.get('verified_by')} | confidence: {sol.get('confidence'):.2f} | origin: {r['origin']}"
        if r["status"] in ("unverified", "failed"):
            problems = [d["message"] for d in r.get("diagnostics", []) if d["severity"] == "error"]
            ca = r.get("confidence_assessment") or {}
            problems += ca.get("reasons", [])[:3]
            if problems:
                foot += " | why: " + "; ".join(problems[:4])
        block.append(Paragraph(_p(foot), small))
        story.append(KeepTogether(block[:3]))
        story += block[3:]

    doc.build(story)
    return buf.getvalue()
