"""Tests for the structural post-extraction validation layer.

The point of this layer is to catch extraction mistakes that still produce a
perfectly valid ``Paper`` - a duplicated number, a marks total that
disagrees with the paper's own header, a page that does not exist. Each
test feeds a deliberately broken paper and asserts the specific finding.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.errors import ValidationError  # noqa: E402
from app.ingestion.validation import (  # noqa: E402
    raise_if_invalid,
    validate_paper,
)
from app.models.loaders_models import Paper, Question  # noqa: E402


def make_paper(questions, **kwargs):
    defaults = dict(
        paper_id="pap_test",
        fingerprint="a" * 64,
        status="ready",
        questions=questions,
        total_questions=len(questions),
    )
    defaults.update(kwargs)
    return Paper(**defaults)


def codes(report):
    return {d.code for d in report.diagnostics}


def good_question(**kw):
    base = dict(
        number="1",
        text="What is 2 + 2?",
        marks=1,
        type="mcq",
        options=["(A) 3", "(B) 4", "(C) 5", "(D) 6"],
        page=1,
    )
    base.update(kw)
    return Question(**base)


# --------------------------------------------------------------------------
# Healthy input
# --------------------------------------------------------------------------

def test_a_clean_paper_reports_nothing():
    paper = make_paper([good_question(section="A")])
    report = validate_paper(paper, total_pages=1, declared_total_marks=1)
    assert report.ok
    assert report.errors == []
    assert report.warnings == []


# --------------------------------------------------------------------------
# Structural failures
# --------------------------------------------------------------------------

def test_empty_extraction_is_an_error():
    report = validate_paper(make_paper([]))
    assert not report.ok
    assert "no_questions" in codes(report)


def test_duplicate_numbers_are_an_error_on_the_pdf_path():
    paper = make_paper([good_question(), good_question()])
    report = validate_paper(paper, source="pdf")
    assert not report.ok
    assert "duplicate_number" in codes(report)


def test_duplicate_numbers_are_only_a_warning_on_the_image_path():
    """Overlapping photos legitimately show the same question twice."""
    paper = make_paper([good_question(), good_question()])
    report = validate_paper(paper, source="image")
    duplicate = [d for d in report.diagnostics if d.code == "duplicate_number"]
    assert duplicate and duplicate[0].severity == "warning"
    assert "duplicate_number" not in {d.code for d in report.errors}


def test_missing_number_is_an_error():
    paper = make_paper([good_question(number="")])
    assert not validate_paper(paper).ok
    assert "missing_number" in codes(validate_paper(paper))


def test_out_of_order_numbers_are_an_error():
    paper = make_paper(
        [good_question(number="2"), good_question(number="1")]
    )
    assert "impossible_ordering" in codes(validate_paper(paper))


def test_numbering_gap_is_reported():
    paper = make_paper(
        [good_question(number="1"), good_question(number="4")]
    )
    report = validate_paper(paper)
    gap = [d for d in report.diagnostics if d.code == "numbering_gap"]
    assert gap and "2" in gap[0].message


def test_empty_text_is_an_error():
    report = validate_paper(make_paper([good_question(text="   ")]))
    assert not report.ok
    assert "empty_text" in codes(report)


def test_page_outside_the_document_is_an_error():
    report = validate_paper(
        make_paper([good_question(page=9)]), total_pages=3
    )
    assert not report.ok
    assert "page_out_of_range" in codes(report)


def test_invalid_marks_value_is_an_error():
    report = validate_paper(make_paper([good_question(marks=-1)]))
    assert "invalid_marks" in codes(report)


def test_non_canonical_type_is_an_error():
    question = good_question()
    object.__setattr__(question, "type", "essay")
    report = validate_paper(make_paper([question]))
    assert "invalid_type" in codes(report)


def test_mcq_without_options_is_flagged():
    report = validate_paper(
        make_paper([good_question(options=None, type="mcq")])
    )
    assert "mcq_without_options" in codes(report)


def test_empty_option_string_is_an_error():
    report = validate_paper(
        make_paper([good_question(options=["(A) x", "  "])])
    )
    assert "invalid_option" in codes(report)


def test_figure_flag_without_a_reference_is_flagged():
    report = validate_paper(
        make_paper([good_question(text="The answer is four.", has_figure=True)])
    )
    assert "figure_without_reference" in codes(report)


def test_inconsistent_section_is_an_error():
    report = validate_paper(
        make_paper([good_question(section="Z")]), declared_sections={"A", "B"}
    )
    assert "inconsistent_section" in codes(report)


def test_a_section_that_reappears_is_reported():
    report = validate_paper(
        make_paper([
            good_question(number="1", section="A"),
            good_question(number="2", section="B"),
            good_question(number="3", section="A"),
        ])
    )
    assert "section_reappears" in codes(report)


def test_questions_without_a_section_are_only_partially_covered():
    report = validate_paper(
        make_paper([good_question(section="A"), good_question(number="2")])
    )
    assert "partial_sections" in codes(report)


def test_total_questions_mismatch_is_an_error():
    paper = make_paper([good_question()])
    paper.total_questions = 7
    assert "total_questions_mismatch" in codes(validate_paper(paper))


# --------------------------------------------------------------------------
# Cross-checks against what the document itself declared
# --------------------------------------------------------------------------

def test_marks_total_mismatch_against_the_header_is_reported():
    paper = make_paper([good_question(marks=1), good_question(number="2", marks=2)])
    report = validate_paper(paper, declared_total_marks=80)
    assert "marks_total_mismatch" in codes(report)


def test_marks_total_matching_the_header_is_clean():
    paper = make_paper([good_question(marks=1), good_question(number="2", marks=2)])
    report = validate_paper(paper, declared_total_marks=3)
    assert "marks_total_mismatch" not in codes(report)


def test_unknown_marks_are_reported():
    paper = make_paper([good_question(marks=None)])
    assert "unknown_marks" in codes(validate_paper(paper, declared_total_marks=1))


def test_question_count_mismatch_against_the_section_instructions():
    paper = make_paper([good_question()])
    report = validate_paper(paper, declared_question_count=20)
    assert "question_count_mismatch" in codes(report)


def test_low_vision_confidence_is_reported():
    paper = make_paper([good_question()])
    report = validate_paper(paper, source="image", confidences={"1": 0.2})
    assert "low_confidence" in codes(report)


# --------------------------------------------------------------------------
# Raising
# --------------------------------------------------------------------------

def test_raise_if_invalid_carries_the_diagnostics():
    report = validate_paper(make_paper([good_question(), good_question()]))
    with pytest.raises(ValidationError) as excinfo:
        raise_if_invalid(report, context="test run")
    assert "test run" in str(excinfo.value)
    assert excinfo.value.diagnostics
    assert "duplicate_number" in str(excinfo.value)


def test_raise_if_invalid_is_silent_for_a_clean_report():
    raise_if_invalid(validate_paper(make_paper([good_question()])))


def test_identical_wording_under_two_numbers_is_caught():
    """Overlapping vision tiles report one question as "1(i)" and "i";
    a numbering check alone cannot see that."""
    paper = make_paper([
        good_question(number="1(i)", text="What does the silence feel like?"),
        good_question(number="i", text="What does the silence feel like?"),
    ])
    report = validate_paper(paper, source="image")
    assert "duplicate_text" in codes(report)
    assert "duplicate_text" not in {d.code for d in report.errors}
