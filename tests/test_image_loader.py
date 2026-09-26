"""Offline tests for the image OCR ingestion path (no API calls, no key needed).

Run from backend/:  pytest tests/test_image_loader.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import image_loader as il
from app.models.loaders_models import Paper, Question


def _qd(**kw):
    base = {"question_number": "1", "question_text": "What is 2+2?",
            "marks": 2, "question_type": "numerical", "options": [],
            "source_page": 1, "confidence": 0.9, "section": None,
            "has_figure": False, "choice_group": None}
    base.update(kw)
    return base


def test_output_validates_against_canonical_schema():
    q = il.to_canonical(_qd(question_type="mcq", options=["(A) 3", "(B) 4"]))
    assert isinstance(q, Question)
    assert (q.number, q.type, q.marks, q.page) == ("1", "mcq", 2, 1)
    assert q.options == ["(A) 3", "(B) 4"]


def test_type_mapping():
    assert il.to_canonical(_qd(question_type="short_answer")).type == "short"
    assert il.to_canonical(_qd(question_type="descriptive")).type == "long"
    assert il.to_canonical(_qd(question_type="unknown")).type == "short"
    with pytest.raises(Exception):
        Question(number="1", text="x", type="descriptive")  # not canonical


def test_empty_options_become_none_and_float_marks():
    assert il.to_canonical(_qd(options=[])).options is None
    assert il.to_canonical(_qd(marks=5.0)).marks == 5
    assert il.to_canonical(_qd(marks=None)).marks is None


def test_choice_group_from_or_text():
    items = il.parse_and_validate(
        '{"questions": [{"question_number": "5", "question_text": "Do this. OR Do that."}]}')
    assert il.to_canonical(items[0]).choice_group == "5"


def test_malformed_model_output_rejected():
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate("not json {{{")
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate('{"questions": [{"question_text": "no number"}]}')
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate('{"answers": []}')


def test_image_validation(tmp_path):
    with pytest.raises(il.ImageError):
        il.load_image_bytes(tmp_path / "missing.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(il.ImageError):
        il.load_image_bytes(bad)
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"GIF89a....")
    with pytest.raises(il.ImageError):
        il.load_image_bytes(gif)


def test_tile_merge_keeps_boundary_options():
    top = [_qd(question_number="7", question_text="Formula?",
               options=["(A) 1", "(B) a", "(C) -1"])]
    bottom = [_qd(question_number="7", question_text="Formula?",
                  options=["(C) -1", "(D) 0"])]
    merged = il._merge_tile_questions(top, bottom)
    assert len(merged) == 1
    assert merged[0]["options"] == ["(A) 1", "(B) a", "(C) -1", "(D) 0"]


def test_split_row_avoids_text():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    bands = [(60, 80), (140, 160), (240, 260), (320, 340)]
    for y0, y1 in bands:
        d.rectangle([20, y0, 380, y1], fill="black")
    split = il._find_split_row(img, search_radius=120)
    assert 80 <= split <= 320
    assert all(not (y0 <= split <= y1) for y0, y1 in bands)


def test_paper_assembly_convention():
    paper = il.assemble_paper(
        [il.to_canonical(_qd(question_number="1", marks=2)),
         il.to_canonical(_qd(question_number="2", marks=3))],
        [b"bytes"],
    )
    assert isinstance(paper, Paper)
    assert paper.paper_id == f"pap_{paper.fingerprint[:8]}"
    assert paper.status == "ready"
    assert (paper.total_questions, paper.total_marks) == (2, 5)
    assert il.to_langgraph_questions(paper)[0]["number"] == "1"


def test_nested_tiling_recovers_after_tile_truncation(monkeypatch):
    """A tile that overflows is split again (depth 2 max); results merge."""
    from PIL import Image

    calls = []

    def fake_data_url(data_url, source_page, model):
        calls.append(source_page)
        if len(calls) <= 2:
            raise il.TruncationError("overflow at full page and first tile")
        n = len(calls)
        return ([_qd(question_number=str(n),
                     question_text=f"Tile question {n}")],
                {})

    monkeypatch.setattr(il, "_extract_data_url", fake_data_url)
    img = Image.new("RGB", (600, 1600), "white")
    items, _ = il._extract_page(img, 1, "test-model")
    assert len(items) >= 2
    assert len(calls) == 5  # full + tile + 2 sub-tiles + second tile; capped depth


def test_prepare_handles_photo_sized_image():
    from PIL import Image

    img = Image.new("RGB", (800, 1200), "white")
    out = il._prepare(img)
    assert out.size == (800, 1200)
    assert out.mode == "RGB"
