"""Offline tests for PDF ingestion metadata (no API calls, no key needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.paper_loader import (
    extract_paper_metadata,
    extract_pdf_pages,
    ingest_paper,
)

ASSET = Path(__file__).resolve().parent.parent / "data" / "papers" / "question_paper_455.pdf"
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="paper 455 not present")


def test_header_metadata_captured():
    pages = [["Subject: Mathematics Part 1  |  Class: Class 9",
              "Total Marks: 80  |  Time Allowed: 3h  |  CBSE"]]
    assert extract_paper_metadata(pages) == {
        "subject": "Mathematics Part 1", "class": "Class 9", "board": "CBSE"}
    assert extract_paper_metadata([["Q1. Something?"]]) == {}


@needs_asset
def test_ingest_paper_metadata_matches_image_format():
    paper = ingest_paper(ASSET)
    assert (paper.subject, paper.class_name, paper.board) == (
        "Mathematics Part 1", "Class 9", "CBSE")
    assert paper.total_questions == 37
    assert paper.status == "ready"
