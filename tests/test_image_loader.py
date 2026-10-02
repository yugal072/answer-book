"""Tests for the image (vision) ingestion path.

The vision *transport* is stubbed - that is the only thing these tests
replace. Everything above it runs for real: image validation and decoding,
preparation, the token-budget tiling path, overlap dedupe, boundary
reassembly, canonical conversion, multi-page assembly and structural
validation. No test asserts merely that "something came back": each one
checks the concrete values that reach the canonical model.

These tests need no API key and make no network calls.
"""
import io
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion import image_loader as il  # noqa: E402
from app.models.loaders_models import Paper, Question  # noqa: E402


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _qd(**kw):
    base = {
        "question_number": "1",
        "question_text": "What is 2+2?",
        "marks": 2,
        "question_type": "numerical",
        "options": [],
        "source_page": 1,
        "confidence": 0.9,
        "section": None,
        "has_figure": False,
        "choice_group": None,
    }
    base.update(kw)
    return base


def _response(questions, metadata=None):
    """A model response body plus the metadata dict the loader reads out."""
    return (
        json.dumps(
            {"metadata": metadata or {}, "questions": questions}
        ),
        {},
    )


def _image_bytes(size=(900, 1400), fmt="PNG", color="white"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def _stub_vision(monkeypatch, handler):
    """Replace only the API transport call."""
    monkeypatch.setattr(il, "_vision_call", handler)


# --------------------------------------------------------------------------
# Canonical conversion (pure, no model involved)
# --------------------------------------------------------------------------

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


def test_invalid_marks_values_become_unknown_rather_than_guesses():
    assert il.to_canonical(_qd(marks="three")).marks is None
    assert il.to_canonical(_qd(marks="2.5")).marks is None
    assert il.to_canonical(_qd(marks=-1)).marks == -1  # caught by validation


def test_choice_group_from_or_text():
    items = il.parse_and_validate(
        '{"questions": [{"question_number": "5", "question_text": "Do this. OR Do that."}]}'
    )
    assert il.to_canonical(items[0]).choice_group == "5"


def test_to_canonical_never_invents_a_section_or_figure():
    q = il.to_canonical(_qd(section=None, has_figure=False))
    assert q.section is None
    assert q.has_figure is False
    assert q.chapter is None


# --------------------------------------------------------------------------
# Model response validation
# --------------------------------------------------------------------------

def test_malformed_model_output_rejected():
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate("not json {{{")
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate('{"questions": [{"question_text": "no number"}]}')
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate('{"answers": []}')
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate('{"questions": "not a list"}')
    with pytest.raises(il.ExtractionError):
        il.parse_and_validate('{"questions": ["not an object"]}')


def test_fenced_json_is_accepted():
    items = il.parse_and_validate(
        '```json\n{"questions": [{"question_number": "1", '
        '"question_text": "What is 2+2?"}]}\n```'
    )
    assert items[0]["question_number"] == "1"


def test_unknown_question_type_falls_back_to_short():
    items = il.parse_and_validate(
        '{"questions": [{"question_number": "1", "question_text": "x", '
        '"question_type": "interpretive_dance"}]}'
    )
    assert items[0]["question_type"] == "unknown"
    assert il.to_canonical(items[0]).type == "short"


# --------------------------------------------------------------------------
# Image input validation
# --------------------------------------------------------------------------

def test_image_file_validation(tmp_path):
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
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(il.ImageError, match="empty"):
        il.load_image_bytes(empty)


def test_image_file_extension_is_checked(tmp_path):
    text = tmp_path / "paper.png"
    text.write_bytes(b"hello")
    with pytest.raises(il.ImageError, match="[Ii]nvalid/corrupt"):
        il.load_image_bytes(text)


def test_bytes_entry_point_rejects_bad_input():
    with pytest.raises(il.ImageError, match="Empty"):
        il.extract_paper_from_image_bytes(b"")
    with pytest.raises(il.ImageError):
        il.extract_paper_from_image_bytes(b"\x89PNG\r\n\x1a\ngarbage")
    with pytest.raises(il.ImageError, match="page"):
        il.extract_paper_from_image_bytes(_image_bytes(), source_page=0)


def test_a_gif_bytes_payload_is_rejected_even_with_a_png_extension():
    gif = io.BytesIO()
    Image.new("RGB", (10, 10)).save(gif, format="GIF")
    with pytest.raises(il.ImageError, match="Unsupported image format"):
        il.open_image(gif.getvalue(), "test")


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_supported_formats_are_accepted(fmt):
    image = il.open_image(_image_bytes(fmt=fmt), fmt)
    assert image.mode == "RGB"


# --------------------------------------------------------------------------
# End-to-end with a stubbed transport
# --------------------------------------------------------------------------

def test_valid_image_produces_a_canonical_paper(monkeypatch):
    _stub_vision(monkeypatch, lambda url, model: _response(
        [
            _qd(question_number="1", question_text="What is 2+2?",
                question_type="numerical", marks=2, source_page=1),
            _qd(question_number="2", question_text="Pick one.",
                question_type="mcq", marks=1,
                options=["(A) 3", "(B) 4", "(C) 5", "(D) 6"]),
        ],
        metadata={"subject": "Mathematics Part 1", "class": "Class 9",
                  "board": "CBSE"},
    ))
    paper = il.extract_paper_from_image_bytes(_image_bytes())

    assert isinstance(paper, Paper)
    assert paper.status == "ready"
    assert paper.total_questions == 2
    assert paper.total_marks == 3
    assert (paper.subject, paper.class_name, paper.board) == (
        "Mathematics Part 1", "Class 9", "CBSE",
    )
    assert [q.number for q in paper.questions] == ["1", "2"]
    assert paper.questions[1].options == ["(A) 3", "(B) 4", "(C) 5", "(D) 6"]
    assert all(q.page == 1 for q in paper.questions)


def test_partial_marks_do_not_produce_a_misleading_total(monkeypatch):
    _stub_vision(monkeypatch, lambda url, model: _response(
        [_qd(question_number="1", marks=2, question_text="First question?"),
         _qd(question_number="2", marks=None, question_text="Second question?")]
    ))
    paper = il.extract_paper_from_image_bytes(_image_bytes())
    assert paper.questions[1].marks is None
    assert paper.total_marks is None, (
        "a partial sum must not be reported as the paper's total"
    )


def test_duplicate_numbers_are_reported_not_hidden(monkeypatch):
    _stub_vision(monkeypatch, lambda url, model: _response(
        [_qd(question_number="1", question_text="First?"),
         _qd(question_number="1", question_text="Second?")]
    ))
    collected = []
    paper = il.extract_paper_from_image_bytes(
        _image_bytes(), diagnostics=collected
    )
    assert len(paper.questions) == 2
    assert any(d.code == "duplicate_number" for d in collected)


def test_low_confidence_is_surfaced(monkeypatch):
    _stub_vision(monkeypatch, lambda url, model: _response(
        [_qd(question_number="1", confidence=0.2)]
    ))
    collected = []
    il.extract_paper_from_image_bytes(_image_bytes(), diagnostics=collected)
    assert any(d.code == "low_confidence" for d in collected)


def test_a_response_with_no_questions_fails_loudly(monkeypatch):
    _stub_vision(monkeypatch, lambda url, model: _response([]))
    with pytest.raises(il.ValidationError, match="no_questions"):
        il.extract_paper_from_image_bytes(_image_bytes())


# --------------------------------------------------------------------------
# Truncation and tiling
# --------------------------------------------------------------------------

def _truncating(count):
    """A transport that reports `count` truncated responses, then succeeds.

    TruncationError is exactly what the real transport raises when the
    model stops at the token budget, so the tiling path above it runs for
    real from here on.
    """
    calls = []

    def handler(url, model):
        calls.append(url)
        if len(calls) <= count:
            raise il.TruncationError("Model output hit the token budget.")
        return _response(
            [_qd(question_number=str(len(calls)),
                 question_text=f"Question from call {len(calls)}")]
        )

    return handler, calls


def test_truncation_on_a_small_image_is_never_silent(monkeypatch):
    """Below the tiling threshold there is nowhere left to split, so the
    caller must hear about it instead of receiving partial data."""
    handler, calls = _truncating(99)
    _stub_vision(monkeypatch, handler)
    small = Image.new("RGB", (600, 400), "white")
    with pytest.raises(il.TruncationError):
        il._extract_page(small, 1, "test-model")
    assert len(calls) == 1, "a tile smaller than MIN_TILE_HEIGHT must not loop"


def test_a_dense_page_is_retiled_until_the_output_fits(monkeypatch):
    handler, calls = _truncating(1)          # only the full page overflows
    _stub_vision(monkeypatch, handler)
    dense = Image.new("RGB", (900, 2400), "white")
    items, _meta = il._extract_page(dense, 1, "test-model")

    assert len(calls) == 3, "expected full page + two tiles"
    numbers = [item["question_number"] for item in items]
    assert len(numbers) == len(set(numbers)), "tiling duplicated questions"
    assert len(items) == 2
    assert all(item["source_page"] == 1 for item in items)


def test_a_tile_that_still_overflows_is_split_again(monkeypatch):
    # full page + first tile overflow, then everything fits
    handler, calls = _truncating(2)
    _stub_vision(monkeypatch, handler)
    dense = Image.new("RGB", (900, 2400), "white")
    items, _meta = il._extract_page(dense, 1, "test-model")
    assert len(calls) == 5
    assert len(items) >= 2
    assert len({i["question_number"] for i in items}) == len(items)


def test_overlap_dedup_keeps_boundary_options_once():
    top = [_qd(question_number="7", question_text="Formula?",
               options=["(A) 1", "(B) a", "(C) -1"])]
    bottom = [_qd(question_number="7", question_text="Formula?",
                  options=["(C) -1", "(D) 0"])]
    merged = il._merge_tile_questions(top, bottom)
    assert len(merged) == 1
    assert merged[0]["options"] == ["(A) 1", "(B) a", "(C) -1", "(D) 0"]


def test_boundary_question_reassembles_from_two_tiles():
    top = [_qd(question_number="9", question_text="The chord of length 24 cm",
               options=["(A) x", "(B) y"])]
    bottom = [_qd(question_number="9", question_text="The chord of length 24 cm",
                  options=["(C) z", "(D) w"])]
    merged = il._merge_tile_questions(top, bottom)
    assert len(merged) == 1
    assert merged[0]["options"] == ["(A) x", "(B) y", "(C) z", "(D) w"]
    assert merged[0]["question_text"] == "The chord of length 24 cm"


def test_a_straddling_question_keeps_both_fragments_in_order():
    top = [_qd(question_number="4", question_text="A pattern of square tiles")]
    bottom = [_qd(question_number="4",
                  question_text="starts with 1 tile in Stage 1.")]
    merged = il._merge_tile_questions(top, bottom)
    assert len(merged) == 1
    assert merged[0]["question_text"] == (
        "A pattern of square tiles starts with 1 tile in Stage 1."
    )


def test_a_longer_overlapping_text_wins_over_a_fragment():
    top = [_qd(question_number="4", question_text="Short bit")]
    bottom = [_qd(question_number="4",
                  question_text="Short bit and the rest of the sentence")]
    merged = il._merge_tile_questions(top, bottom)
    assert merged[0]["question_text"] == (
        "Short bit and the rest of the sentence"
    )


def test_split_row_avoids_text():
    from PIL import ImageDraw

    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    bands = [(60, 80), (140, 160), (240, 260), (320, 340)]
    for y0, y1 in bands:
        d.rectangle([20, y0, 380, y1], fill="black")
    split = il._find_split_row(img, search_radius=120)
    assert 80 <= split <= 320
    assert all(not (y0 <= split <= y1) for y0, y1 in bands)


# --------------------------------------------------------------------------
# Multi-page assembly
# --------------------------------------------------------------------------

def test_multi_page_assembly_keeps_page_numbers(tmp_path, monkeypatch):
    seen = []

    def handler(url, model):
        seen.append(url)
        if len(seen) == 1:
            return _response([_qd(question_number="1",
                                  question_text="Page one question?")])
        return _response([_qd(question_number="2",
                              question_text="Page two question?")])

    _stub_vision(monkeypatch, handler)
    first = tmp_path / "p1.png"
    second = tmp_path / "p2.png"
    first.write_bytes(_image_bytes())
    second.write_bytes(_image_bytes())

    paper = il.extract_paper_from_images([first, second])
    assert [q.number for q in paper.questions] == ["1", "2"]
    assert [q.page for q in paper.questions] == [1, 2]
    assert paper.total_questions == 2


def test_metadata_from_any_page_is_kept(tmp_path, monkeypatch):
    calls = []

    def handler(url, model):
        calls.append(url)
        if len(calls) == 1:
            return _response([_qd(question_number="1")], metadata={})
        return _response(
            [_qd(question_number="2")],
            metadata={"subject": "English", "class": "Class 9",
                      "board": "CBSE"},
        )

    _stub_vision(monkeypatch, handler)
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(_image_bytes())
    second.write_bytes(_image_bytes())

    paper = il.extract_paper_from_images([first, second])
    assert paper.subject == "English"
    assert paper.class_name == "Class 9"
    assert paper.board == "CBSE"


def test_a_question_repeated_across_two_images_appears_once(
    tmp_path, monkeypatch
):
    def handler(url, model):
        return _response([_qd(question_number="1",
                              question_text="Read the passage.")])

    _stub_vision(monkeypatch, handler)
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    first.write_bytes(_image_bytes())
    second.write_bytes(_image_bytes())

    paper = il.extract_paper_from_images([first, second])
    assert len(paper.questions) == 1


def test_empty_image_list_is_rejected():
    with pytest.raises(il.ImageError, match="at least one"):
        il.extract_paper_from_images([])


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


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------

def test_prepare_handles_photo_sized_image():
    img = Image.new("RGB", (800, 1200), "white")
    out = il._prepare(img)
    assert out.size == (800, 1200)
    assert out.mode == "RGB"


def test_prepare_caps_oversized_images():
    img = Image.new("RGB", (4000, 3000), "white")
    out = il._prepare(img)
    assert max(out.size) == il.VISION_RENDER_MAX_DIM


def test_prepare_normalises_orientation_and_grayscale():
    exif = Image.new("L", (300, 200), 128)
    out = il._prepare(exif)
    assert out.mode == "RGB"
    assert out.size == (300, 200)


# --------------------------------------------------------------------------
# Transport error paths (the only fully mocked layer)
# --------------------------------------------------------------------------

def _install_fake_groq(monkeypatch, error):
    module = types.ModuleType("groq")

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    error_class = {
        "connection": APIConnectionError,
        "timeout": APITimeoutError,
        "rate_limit": RateLimitError,
    }[error]

    class Groq:  # noqa: N801 - mirrors the real name
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            raise error_class(str(getattr(self, "detail", "fake failure")))

    module.APIConnectionError = APIConnectionError
    module.APITimeoutError = APITimeoutError
    module.RateLimitError = RateLimitError
    module.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", module)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setattr(il.time, "sleep", lambda *_: None)


def _install_groq_returning(monkeypatch, content, finish_reason):
    """A fake Groq client that returns one canned response."""
    module = types.ModuleType("groq")

    class APIConnectionError(Exception):
        pass

    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    class Groq:  # noqa: N801 - mirrors the real name
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        finish_reason=finish_reason,
                    )
                ]
            )

    module.APIConnectionError = APIConnectionError
    module.APITimeoutError = APITimeoutError
    module.RateLimitError = RateLimitError
    module.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", module)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")


def test_a_complete_response_is_returned(monkeypatch):
    body = json.dumps({"metadata": {}, "questions": [_qd()]})
    _install_groq_returning(monkeypatch, body, "stop")
    content, reason = il._vision_call("data:image/png;base64,AAA", "test-model")
    assert json.loads(content)["questions"][0]["question_number"] == "1"
    assert reason == "stop"


def test_a_length_finish_reason_becomes_a_truncation_error(monkeypatch):
    """Partial model output must never be passed on as if it were whole."""
    _install_groq_returning(monkeypatch, '{"questions": [', "length")
    with pytest.raises(il.TruncationError, match="token budget"):
        il._vision_call("data:image/png;base64,AAA", "test-model")


def test_an_unexpected_finish_reason_is_an_error(monkeypatch):
    _install_groq_returning(monkeypatch, '{"questions": []}', "content_filter")
    with pytest.raises(il.ExtractionError, match="unexpected reason"):
        il._vision_call("data:image/png;base64,AAA", "test-model")


def test_an_empty_response_is_an_error(monkeypatch):
    _install_groq_returning(monkeypatch, "   ", "stop")
    with pytest.raises(il.ExtractionError, match="empty response"):
        il._vision_call("data:image/png;base64,AAA", "test-model")


def test_transport_failure_is_reported_not_swallowed(monkeypatch):
    _install_fake_groq(monkeypatch, "connection")
    with pytest.raises(il.ExtractionError, match="transport failure"):
        il._vision_call("data:image/png;base64,AAA", "test-model")


def test_a_daily_quota_error_is_not_retried(monkeypatch):
    """Waiting cannot help a daily token limit, so the caller gets an
    actionable message immediately instead of a slow failure."""
    detail = (
        "Error code: 429 - Rate limit reached for model `qwen/qwen3.8-27b` "
        "on tokens per day (TPD): Limit 200000, Used 197544"
    )
    _install_fake_groq(monkeypatch, "rate_limit")
    original = il._TOKEN_BUDGET_EXHAUSTED

    class FakeGroq(sys.modules["groq"].Groq):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.detail = detail

    monkeypatch.setattr(sys.modules["groq"], "Groq", FakeGroq)
    with pytest.raises(il.ExtractionError, match="daily token"):
        il._vision_call("data:image/png;base64,AAA", "test-model")
    assert original.search(detail)


def test_missing_api_key_is_an_actionable_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(il.ExtractionError, match="GROQ_API_KEY"):
        il._api_key()


def test_the_same_question_under_two_numbers_is_deduped():
    """What overlapping tiles really produce: one question, two numbers."""
    items = [
        _qd(question_number="1(i)", question_text="What does it feel like?",
            options=["(A) x", "(B) y"]),
        _qd(question_number="i", question_text="What does it feel like?",
            options=["(B) y", "(C) z"]),
    ]
    deduped = il._dedupe_by_text(items)
    assert len(deduped) == 1
    assert deduped[0]["question_number"] == "1(i)"
    assert deduped[0]["options"] == ["(A) x", "(B) y", "(C) z"]


def test_distinct_questions_are_not_deduped():
    items = [
        _qd(question_number="1", question_text="What does it feel like?"),
        _qd(question_number="2", question_text="What does the sky look like?"),
    ]
    assert len(il._dedupe_by_text(items)) == 2


def test_the_same_wording_on_two_pages_is_kept():
    """Not a tiling artefact: two pages that really do repeat a question."""
    items = [
        _qd(question_number="1", question_text="Repeat me.", source_page=1),
        _qd(question_number="1", question_text="Repeat me.", source_page=2),
    ]
    assert len(il._dedupe_by_text(items)) == 2
