"""Live end-to-end tests for the vision ingestion path.

These call the real Groq vision API. They are skipped unless
``GROQ_API_KEY`` is set, so the offline suite stays free::

    GROQ_API_KEY=... pytest tests -m live -q
    pytest tests -m "not live" -q      # everything else

They deliberately use a *real* page of a *real* test paper as the input,
and check the extraction against what the deterministic PDF path gets
from the very same page. That is the only way to know the two paths agree
on real input rather than on a fixture someone tuned until it passed.
"""
import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pymupdf = pytest.importorskip("pymupdf", reason="pymupdf needed to render pages")

from app.ingestion.image_loader import (  # noqa: E402
    extract_paper_from_image_bytes,
    extract_paper_from_images,
)
from app.ingestion.paper_loader import ingest_paper  # noqa: E402

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY is not set",
    ),
]

PAPERS = ROOT / "data" / "papers"


def render_page(pdf_path: Path, page_index: int, dpi: int = 150):
    """Render one page of a paper to a PIL image."""
    from PIL import Image

    with pymupdf.open(str(pdf_path)) as doc:
        pixmap = doc[page_index].get_pixmap(dpi=dpi)
        return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


def as_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_a_clean_scan_matches_the_pdf_extraction():
    """Page 1 of paper 455 holds three MCQs; the vision path must find the
    same three, with the same numbers, marks and option counts."""
    pdf = PAPERS / "question_paper_455.pdf"
    expected = [q for q in ingest_paper(pdf).questions if q.page == 1]

    data = as_png_bytes(render_page(pdf, 0))
    paper = extract_paper_from_image_bytes(data)

    assert [q.number for q in paper.questions] == [q.number for q in expected]
    assert [q.marks for q in paper.questions] == [q.marks for q in expected]
    assert [len(q.options or []) for q in paper.questions] == [
        len(q.options or []) for q in expected
    ]
    assert paper.total_marks == sum(q.marks for q in expected)


def test_a_phone_photo_of_the_same_page_still_extracts():
    """Rotation, blur and a dimmer exposure must not cost a question."""
    pdf = PAPERS / "question_paper_455.pdf"
    expected = [q for q in ingest_paper(pdf).questions if q.page == 1]

    from PIL import ImageEnhance, ImageFilter

    image = render_page(pdf, 0)
    image = ImageEnhance.Brightness(
        image.rotate(2.5, expand=True, fillcolor="white").filter(
            ImageFilter.GaussianBlur(0.8)
        )
    ).enhance(0.96)

    paper = extract_paper_from_image_bytes(as_png_bytes(image))
    assert [q.number for q in paper.questions] == [q.number for q in expected]


def test_a_dense_page_is_tiled_and_never_duplicated():
    """Paper 471's passage page overflows the output budget, so the tiling
    path runs. The result must not contain the same question twice - the
    failure mode that overlapping tiles actually produce."""
    pdf = PAPERS / "question_paper_471.pdf"
    parent = next(q for q in ingest_paper(pdf).questions if q.number == "1")

    data = as_png_bytes(render_page(pdf, 1, dpi=170))
    paper = extract_paper_from_image_bytes(data)

    assert paper.questions, "a dense page must not extract to nothing"
    assert paper.questions[0].number == parent.number
    assert paper.questions[0].marks == parent.marks

    seen = set()
    for q in paper.questions:
        key = "".join(ch for ch in q.text.lower() if ch.isalnum())[:120]
        assert key not in seen, f"question {q.number} was extracted twice"
        seen.add(key)


def test_two_pages_assemble_into_one_paper(tmp_path):
    pdf = PAPERS / "question_paper_455.pdf"
    first = tmp_path / "page1.png"
    second = tmp_path / "page2.png"
    first.write_bytes(as_png_bytes(render_page(pdf, 0)))
    second.write_bytes(as_png_bytes(render_page(pdf, 1)))

    paper = extract_paper_from_images([first, second])
    assert paper.total_questions >= 6
    assert {q.page for q in paper.questions} == {1, 2}
    assert len(paper.fingerprint) == 64
