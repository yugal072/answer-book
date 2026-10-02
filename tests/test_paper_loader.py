"""Tests for PDF ingestion against the real 20-paper corpus.

These are not synthetic fixtures: every assertion runs over the actual
question papers in ``data/papers``. The expectations below were derived
from two independent facts the papers state about themselves - the total
marks in the header and the "Attempt N questions" section instructions -
and are kept here as a regression baseline.

Run from the repository root::

    PYTHONPATH=. pytest tests/test_paper_loader.py -q
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ingestion.errors import DocumentError  # noqa: E402
from app.ingestion.paper_loader import (  # noqa: E402
    extract_header_facts,
    extract_paper_metadata,
    extract_pdf_pages,
    ingest_paper,
    ingest_paper_with_report,
)
from app.models.loaders_models import Paper, Question  # noqa: E402

PAPERS_DIR = ROOT / "data" / "papers"
PAPERS = sorted(PAPERS_DIR.glob("question_paper_*.pdf"))
PAPER_455 = PAPERS_DIR / "question_paper_455.pdf"

#: paper stem -> (question count, total marks, sections)
#: The counts agree with the section instructions of every paper, and the
#: marks agree with the "Total Marks" line of every header.
EXPECTED = {
    "question_paper_455": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_457": (18, 40, ["A", "B", "C", "D", "E"]),
    "question_paper_458": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_459": (18, 40, ["A", "B", "C", "D", "E"]),
    "question_paper_460": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_461": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_462": (18, 40, ["A", "B", "C", "D", "E"]),
    "question_paper_463": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_464": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_465": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_466": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_467": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_468": (18, 40, ["A", "B", "C", "D", "E"]),
    "question_paper_469": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_470": (37, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_471": (17, 40, ["A", "B", "C", "D"]),
    "question_paper_472": (28, 80, ["A", "B", "C", "D", "E"]),
    "question_paper_473": (17, 20, ["A", "B", "C"]),
    "question_paper_474": (17, 40, ["A", "B", "C", "D"]),
    "question_paper_475": (28, 80, ["A", "B", "C", "D", "E"]),
}

pytestmark = pytest.mark.skipif(
    not PAPERS, reason="test paper corpus not present"
)


def ids(papers):
    return [p.stem for p in papers]


@lru_cache(maxsize=None)
def ingest(stem):
    """Ingest once per paper; the corpus checks are read-only."""
    return ingest_paper(PAPERS_DIR / f"{stem}.pdf")


@lru_cache(maxsize=None)
def ingest_reported(stem):
    return ingest_paper_with_report(PAPERS_DIR / f"{stem}.pdf")


@lru_cache(maxsize=None)
def pages_of(stem):
    return extract_pdf_pages(PAPERS_DIR / f"{stem}.pdf")


# --------------------------------------------------------------------------
# Corpus-wide
# --------------------------------------------------------------------------

def test_the_whole_corpus_is_present():
    assert len(PAPERS) == 20
    assert {p.stem for p in PAPERS} == set(EXPECTED)


@pytest.mark.parametrize("paper_path", PAPERS, ids=ids(PAPERS))
def test_question_count_and_marks_match_the_papers_own_declarations(
    paper_path,
):
    expected_n, expected_marks, expected_sections = EXPECTED[paper_path.stem]
    paper, report = ingest_reported(paper_path.stem)

    assert report.errors == [], [str(d) for d in report.errors]
    assert isinstance(paper, Paper)
    assert paper.total_questions == expected_n
    assert len(paper.questions) == expected_n
    assert paper.total_marks == expected_marks
    assert paper.sections == expected_sections
    assert paper.status == "ready"

    # The marks we extracted add up to the marks the header declares, and
    # the question total matches what the sections say they contain.
    facts = extract_header_facts(extract_pdf_pages(paper_path))
    assert facts["total_marks"] == paper.total_marks


@pytest.mark.parametrize("paper_path", PAPERS, ids=ids(PAPERS))
def test_every_question_is_structurally_sound(paper_path):
    paper = ingest(paper_path.stem)

    for q in paper.questions:
        assert Question.model_validate(q.model_dump())
        assert q.text.strip(), f"question {q.number} has no text"
        assert q.number.isdigit()
        assert q.page is not None and q.page >= 1
        assert q.type in {"numerical", "mcq", "short", "long"}
        if q.marks is not None:
            assert 0 < q.marks <= 100
        if q.options is not None:
            assert q.options and all(o.strip() for o in q.options)

    numbers = [int(q.number) for q in paper.questions]
    assert numbers == sorted(numbers), "question numbers are out of order"
    assert len(numbers) == len(set(numbers)), "duplicate question numbers"
    assert numbers == list(range(1, len(numbers) + 1)), (
        "question numbering is not a contiguous 1..N run"
    )
    assert sum(q.marks for q in paper.questions) == paper.total_marks


@pytest.mark.parametrize("paper_path", PAPERS, ids=ids(PAPERS))
def test_metadata_is_extracted_for_every_paper(paper_path):
    paper = ingest(paper_path.stem)
    assert paper.subject, "subject missing"
    assert paper.class_name, "class missing"
    assert paper.board, "board missing"
    assert "Total Marks" not in (paper.subject or "")


@pytest.mark.parametrize("paper_path", PAPERS, ids=ids(PAPERS))
def test_no_source_wording_is_lost(paper_path):
    """Every content line of the document survives into the extraction.

    This is the anti-silent-truncation check: if a heading, a section
    instruction or a wrapped line were mistaken for structure and dropped,
    it would not be found in the extracted text.
    """
    from app.ingestion.text_parser import _classify

    paper = ingest(paper_path.stem)
    produced = " ".join(
        q.text + " " + " ".join(q.options or []) + " " + (q.chapter or "")
        for q in paper.questions
    )
    produced = re.sub(r"\s+", " ", produced)

    started = False
    in_instructions = False
    for lines in pages_of(paper_path.stem):
        for line in lines:
            kind = _classify(line)
            if kind == "instructions_heading":
                started = in_instructions = True
                continue
            if kind in ("section", "instruction", "furniture"):
                in_instructions = False
                continue
            if kind == "chapter":
                in_instructions = False
                assert re.sub(r"\s+", " ", line) in produced, (
                    f"chapter tag lost: {line!r}"
                )
                continue
            if in_instructions and kind == "question":
                continue
            in_instructions = False
            if not started and kind == "question":
                started = True
            if not started or kind == "bare_number":
                continue
            expected = re.sub(r"\s+", " ", line)
            if kind == "question":
                # The number is structural, and marks printed after the
                # stem are captured as Question.marks; neither is wording.
                expected = re.sub(
                    r"^\s*(?:Q\s*\.?\s*)?\d{1,3}\s*[.)]?\s+", "", expected
                )
                expected = re.sub(r"\s\d{1,3}$", "", expected)
                if not expected:
                    continue
            assert expected in produced, f"lost source line: {line!r}"


# --------------------------------------------------------------------------
# Content spot checks against the actual papers
# --------------------------------------------------------------------------

def test_mcq_options_labels_order_and_text():
    paper = ingest_paper(PAPER_455)
    q1 = paper.questions[0]
    assert q1.number == "1"
    assert q1.section == "A"
    assert q1.marks == 1
    assert q1.type == "mcq"
    assert q1.text == (
        "How many parts do the coordinate axes divide the plane into?"
    )
    assert q1.options == ["(A) 2", "(B) 6", "(C) 4", "(D) 8"]
    assert q1.page == 1
    assert q1.has_figure is False


def test_marks_are_per_question_not_per_section():
    """Sections in these papers carry different marks (1, 2, 3 and 5), so a
    single marks value per section would corrupt most of the paper."""
    paper = ingest_paper(PAPER_455)
    by_section = {}
    for q in paper.questions:
        by_section.setdefault(q.section, set()).add(q.marks)
    assert by_section["A"] == {1}
    assert by_section["B"] == {2}
    assert by_section["E"] == {5}
    assert paper.total_marks == 80


def test_internal_choice_keeps_the_alternative_and_sets_choice_group():
    paper = ingest_paper(PAPER_455)
    q21 = next(q for q in paper.questions if q.number == "21")
    assert q21.choice_group == "21"
    assert "OR" in q21.text
    assert "number line" in q21.text  # the alternative is preserved
    assert q21.marks == 2


def test_section_title_is_preserved():
    paper = ingest_paper(PAPERS_DIR / "question_paper_473.pdf")
    titles = {q.section: q.section_title for q in paper.questions}
    assert titles["A"] == "Grammar"
    assert titles["B"] == "Literature (Very Short Answer)"
    assert titles["C"] == "Literature (Short Answer)"


def test_sub_question_options_are_not_stolen_by_the_parent():
    """Paper 471's passage question has options under sub-part (i); they
    must stay in the wording, not become the parent question's options."""
    paper = ingest_paper(PAPERS_DIR / "question_paper_471.pdf")
    q1 = paper.questions[0]
    assert q1.number == "1"
    assert q1.type != "mcq"
    assert q1.marks == 10
    assert "(A) 50 litres" in q1.text
    assert "How Much Water a City Drinks" in q1.text


def test_a_full_passage_question_is_not_truncated():
    """Regression: a wrapped line ending in 'instructions.' was once read
    as a heading, which silently cut the question in half."""
    paper = ingest_paper(PAPERS_DIR / "question_paper_472.pdf")
    q2 = next(q for q in paper.questions if q.number == "2")
    assert "afternoon of suitcases and last instructions." in q2.text
    assert "where love learned to fly without a net." in q2.text
    assert "(vi)" in q2.text
    assert q2.marks == 10


def test_chapter_tags_are_captured_and_kept_out_of_the_text():
    paper = ingest_paper(PAPER_455)
    q1 = paper.questions[0]
    assert q1.chapter == "[Ch 1: Orienting Yourself: The Use of Coordinates]"
    assert "Ch 1" not in q1.text


def test_questions_keep_their_page_across_boundaries():
    paper = ingest_paper(PAPER_455)
    pages = [q.page for q in paper.questions]
    assert pages == sorted(pages)
    assert min(pages) == 1
    assert max(pages) <= 9


def test_a_question_spanning_a_page_break_keeps_both_halves():
    paper = ingest_paper(PAPERS_DIR / "question_paper_471.pdf")
    q1 = next(q for q in paper.questions if q.number == "1")
    # Q1 starts on page 2 and its last sub-question is printed on page 3.
    assert q1.page == 2
    assert "main takeaway" in q1.text


def test_header_metadata_matches_the_image_format_fields():
    paper = ingest_paper(PAPER_455)
    assert (paper.subject, paper.class_name, paper.board) == (
        "Mathematics Part 1", "Class 9", "CBSE",
    )


def test_metadata_helper_contract_is_unchanged():
    pages = [["Subject: Mathematics Part 1  |  Class: Class 9",
              "Total Marks: 80  |  Time Allowed: 3h  |  CBSE"]]
    assert extract_paper_metadata(pages) == {
        "subject": "Mathematics Part 1", "class": "Class 9", "board": "CBSE",
    }
    assert extract_paper_metadata([["Q1. Something?"]]) == {}


def test_board_is_not_invented_from_a_capitalised_word():
    pages = [["Subject: Science  |  Class: Class 9", "CBSE is a board."]]
    facts = extract_header_facts(pages)
    assert facts["subject"] == "Science"
    assert "board" not in facts


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def test_missing_file(tmp_path):
    with pytest.raises(DocumentError, match="not found"):
        ingest_paper(tmp_path / "nope.pdf")


def test_empty_file(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    with pytest.raises(DocumentError, match="empty"):
        ingest_paper(empty)


def test_corrupt_pdf(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 this is not a pdf at all")
    with pytest.raises(DocumentError):
        ingest_paper(broken)


def test_directory_instead_of_file(tmp_path):
    with pytest.raises(DocumentError, match="Not a file"):
        ingest_paper(tmp_path)


def test_pdf_without_a_text_layer(tmp_path):
    """A scan has no text: the loader must say so, not return an empty
    Paper that looks like a successful extraction."""
    blank = tmp_path / "scan.pdf"
    blank.write_bytes(_blank_pdf_bytes())
    with pytest.raises(DocumentError, match="no usable text layer"):
        ingest_paper(blank)


def test_diagnostics_are_collected_for_the_caller(tmp_path):
    collected = []
    ingest_paper(PAPER_455, diagnostics=collected)
    assert isinstance(collected, list)


def _blank_pdf_bytes() -> bytes:
    """A minimal, valid, text-free PDF (one empty page)."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1, xref,
    )
    return bytes(out)


def test_both_helper_spellings_resolve_to_the_same_function():
    """PR #13 renamed these helpers with a leading underscore; the repo's
    own tests import the public names, so both must keep working."""
    from app.ingestion import paper_loader as pl

    assert pl._extract_pdf_pages is pl.extract_pdf_pages
    assert pl._extract_paper_metadata is pl.extract_paper_metadata
    assert pl._parse_questions is pl.parse_questions
