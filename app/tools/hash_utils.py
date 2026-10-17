"""Hashing and text normalization utilities for paper and question caching.

Provides deterministic SHA-256 fingerprinting for:
1. Paper-level Recommendation A caching (file bytes hash).
2. Question-level composite signature caching (text + marks + board + class_name).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re


def normalize_text(text: str) -> str:
    """Normalize question text to ensure deterministic hashing across minor variations.

    Transformations:
    - Lowercases all characters.
    - Standardizes curly/smart quotes to standard ASCII quotes.
    - Standardizes typographic dashes/minuses (—, –, −) to standard hyphens (-).
    - Removes leading question labels (e.g., 'Q.1:', 'Question 1 -', '1.').
    - Collapses consecutive whitespace (spaces, tabs, newlines) into a single space.
    - Trims trailing whitespace and non-essential trailing punctuation (periods, colons).
    """
    if not text:
        return ""

    t = str(text).lower()

    # Standardize unicode quotes
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2018", "'").replace("\u2019", "'")

    # Standardize typographic dashes and minus signs
    t = t.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")

    # Strip leading question prefixes like "Q.1:", "Question 1 -", "1.", "1)"
    t = re.sub(
        r"^(?:q(?:uestion)?\s*\.?\s*\d+\s*[:\.\-\)]|\d+\s*[:\.\-\)])\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )

    # Collapse multiple whitespaces and trim
    t = re.sub(r"\s+", " ", t).strip()

    # Strip trailing punctuation (trailing periods or colons that don't affect question meaning)
    t = re.sub(r"[\.\:\s]+$", "", t)

    return t


def generate_paper_fingerprint(source: bytes | str | Path) -> str:
    """Generate a SHA-256 fingerprint of the raw paper file.

    Args:
        source: Raw file bytes, or a Path/str pointing to the paper file on disk.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.
    """
    if isinstance(source, bytes):
        raw_bytes = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot generate fingerprint: file not found at {path}"
            )
        with open(path, "rb") as f:
            raw_bytes = f.read()
    else:
        raise TypeError(
            f"Unsupported source type for fingerprint generation: {type(source)}"
        )

    return hashlib.sha256(raw_bytes).hexdigest()


def generate_question_signature(
    question_text: str,
    marks: int,
    board: str = "CBSE",
    class_name: str = "",
    *,
    options: list[str] | None = None,
    has_figure: bool = False,
    subject: str = "",
) -> str:
    """Generate a deterministic composite signature for question-level caching.

    Formula (v2): SHA-256(version | normalized text | marks | board | class | subject | options | figure flag).
    Optional context fields prevent reusing an MCQ answer when its options differ,
    or reusing an answer when the question explicitly depends on a figure.

    Rationale:
        - Includes marks: Prevents the "marks blindness" bug where a 2-mark brief definition
          is mistakenly reused for a 5-mark in-depth derivation question.
        - Includes board and class_name: Ensures questions mapped to specific syllabi or grades
          maintain pedagogical correctness.
    """
    norm_text = normalize_text(question_text)
    norm_board = (board or "CBSE").strip().upper()
    norm_class = (class_name or "").strip().lower()
    norm_subject = (subject or "").strip().lower()
    norm_options = "\u241f".join(
        normalize_text(option) for option in (options or [])
    )

    composite_key = (
        f"v2|{norm_text}|{marks}|{norm_board}|{norm_class}|{norm_subject}|"
        f"{norm_options}|figure={int(bool(has_figure))}"
    )
    return hashlib.sha256(composite_key.encode("utf-8")).hexdigest()
