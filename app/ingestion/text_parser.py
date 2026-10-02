"""Structure-aware parser for the text layer of a question paper.

Pure, deterministic, no network, no model calls. It takes pages of raw
lines (as produced by :func:`app.ingestion.paper_loader.extract_pdf_pages`)
and returns canonical :class:`~app.models.loaders_models.Question` objects
plus the diagnostics gathered on the way.

The parser is driven by the *observed* structure of real question papers
rather than by generic regex guessing. A typical paper page looks like::

    Q.NO. QUESTIONS MARKS                  <- repeated table header (noise)
    Section B                             <- section heading (optional title)
    Attempt 4 out of 6.                   <- section instruction (declares count)
    12 How did Leela help resolve it? 1   <- question start + trailing marks
    [Ch 11: Twin Melodies]                <- chapter tag (kept, not question text)
    13 The chord of length 24 cm ...      <- next question
    2                                     <- standalone marks line
    OR                                    <- internal-choice separator
    Find the area of the minor sector.    <- OR alternative (same question)

Rules that matter for faithfulness:

* Nothing is invented. A field that is not visible in the text stays
  ``None``/``False``.
* Wrapped lines are joined with single spaces; characters, symbols, units
  and wording are otherwise left untouched.
* Text that merely *looks* like a number ("110 degrees", "5 kg") never
  becomes a question number, and a question is never merged into another
  silently: the two numbering strategies below make that recoverable.
* Repeated page furniture (headers/footers/watermarks) is removed by
  repetition + position statistics, not by hardcoded brand strings.
* Only the MCQ options belonging to the stem are lifted into
  ``Question.options``; options of sub-questions stay in the text where
  they cannot be mistaken for the stem's own options.

Known limitations are documented in ``docs/ingestion.md``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from app.models.loaders_models import Question

# --------------------------------------------------------------------------
# Line-level patterns
# --------------------------------------------------------------------------

# "Q.NO. QUESTIONS MARKS", "Q NO  Question  Marks", "QNo. Question"
_QNO_HEADER = re.compile(r"^\s*Q\s*\.?\s*N\s*O\s*\.?\b", re.IGNORECASE)

# "Page 3/9", "Page 3 of 9", "Page 3 of 9  |  cbse.com"
_PAGE_FOOTER = re.compile(
    r"^\s*Page\s+\d+\s*(?:/|\bof\b)\s*\d+\b", re.IGNORECASE
)

# "Section A", "Section B - Literature", "SECTION C: Grammar", "Part A"
# The label must already be upper-case/roman/digits in the source so that
# ordinary prose such as "Part of the whole" is not read as a heading.
_SECTION = re.compile(
    r"^\s*(?i:Section|Part|Unit)\s*[-–—:]?\s*"
    r"(?P<label>[A-Z]{1,2}|[IVXLC]{1,5}|\d{1,3})"
    r"(?:\s*[—–\-:|]\s*(?P<title>.+?))?\s*$"
)

# "Attempt all 20 questions. Each carries 1 mark(s)." / "Attempt 4 out of 6."
# Deliberately narrow: only the shapes that actually declare how a section is
# to be attempted. A wrapped sentence such as "... last instructions." must
# never be mistaken for an instruction.
_ATTEMPT = re.compile(
    r"^\s*(?:Attempt|Answer)\b(?=.*(?:\ball\s+\d+\s+questions?\b"
    r"|\b\d+\s*out\s*of\s*\d+\b"
    r"|\bcarries?\s+\d+\s*marks?\b"
    r"|\bany\s+\d+\s+questions?\b"
    r"|^\s*Answer\s+the\s+following\b))",
    re.IGNORECASE,
)
# A candidate heading for the block of general instructions at the top of a
# paper ("General Instructions:", "Instructions", "Note:"). The whole line
# must be the heading, and it is only honoured in the preamble: a wrapped
# line that merely ends a sentence with "instructions." is question wording.
_INSTRUCTIONS_HEADING = re.compile(
    r"^\s*(?:(?:general\s+)?instructions?|notes?)\s*:\s*$"
    r"|^\s*(?:general\s+)?instructions?\s*$",
    re.IGNORECASE,
)
# Count a section declares: "Attempt all 20 questions" / "Attempt 4 out of 6"
_DECLARED_COUNT = re.compile(
    r"^\s*(?:Attempt|Answer)\b"
    r"(?:.*?\ball\b\s+(?P<all>\d+)\s*questions?"
    r"|.*?\b(?P<pick>\d+)\s*out\s*of\s+\d+)",
    re.IGNORECASE,
)
# "Each carries 1 mark(s)." / "Each question carries 2 marks."
_CARRIED_MARKS = re.compile(
    r"\bcarries?\s+(?P<marks>\d{1,3})\s*mark", re.IGNORECASE
)

# "[Ch 4: Exploring Algebraic Identities]" - chapter tag printed by the
# paper generator. Retained as Question.chapter, never part of the text.
_CHAPTER_TAG = re.compile(r"^\s*\[Ch\s*\d+\s*:.*\]\s*$", re.IGNORECASE)

# Question start, several numbering styles:
#   "12 Some text"   "12. Some text"   "12) Some text"   "Q12 Some text"
_QUESTION_START = re.compile(
    r"^\s*(?:Q\s*\.?\s*)?(?P<number>\d{1,3})\s*(?P<dot>[.)])?\s+(?P<rest>\S.*)$"
)
# Same with no body at all - a bare number line (marks slot / noise).
_BARE_NUMBER = re.compile(r"^\s*(?:Q\s*\.?\s*)?\d{1,3}\s*$")

# MCQ option line: "(A) text". Uppercase A-H only, so that sub-parts
# written as "(i)", "(ii)" or "(a)" are never read as options.
_OPTION = re.compile(r"^\s*\((?P<label>[A-H])\)\s*(?P<text>.*)$")
# Option markers anywhere on a line (several options on one line).
_OPTION_INLINE = re.compile(r"\((?P<label>[A-H])\)\s*")

# Sub-part markers: "(i)", "(ii)", "(a)", "(A)" in a sub-question body,
# and bullet lists.
_SUBPART = re.compile(r"^\s*\((?P<label>[A-Za-z]{1,4}|[ivx]{1,4})\)\s+\S")
_BULLET = re.compile(r"^\s*[•▪◦●*·]\s+\S")

# Internal-choice separator on a line of its own.
_OR_SEPARATOR = re.compile(r"^\s*OR\s*$", re.IGNORECASE)

# Marks printed after a sentence: "... resolved. 1"
_TRAILING_MARKS = re.compile(r"^(?P<body>.*[.?!…।])\s+(?P<marks>\d{1,3})$")

# A figure is only claimed when the text *references* one:
# "Figure 3", "Fig. 2", "the diagram below", "as shown in the figure".
_FIGURE_REF = re.compile(
    r"\b(?:fig(?:ure)?\.?|diagram|graph|plot|sketch)\s*"
    r"(?:\d+|(?:below|above|here|shown|given|accompanying|adjacent|opposite)\b)"
    r"|\b(?:see|shown in|as shown in|refer to)\s+(?:the\s+|given\s+)?"
    r"(?:fig(?:ure)?\.?|diagram|graph|plot)\b"
    r"|\bbelow\s+(?:is|are)\s+the\s+(?:fig(?:ure)?|diagram|graph)\b",
    re.IGNORECASE,
)

_NUMERICAL_HINTS = re.compile(
    r"\b(calculate|compute|find|evaluate|determine|obtain|derive|"
    r"prove|show that|verify|how many|how much|number of|value of|"
    r"distance|area|perimeter|volume|height|width|length|cost|fare|rate|"
    r"probability|equation|sum of|product of|simplify|factorise|expand)\b",
    re.IGNORECASE,
)
_CASE_STUDY = re.compile(r"\bcase\s+(?:study|based)\b", re.IGNORECASE)

_MAX_HEADING_LEN = 90
_MAX_MARKS = 100
_MAX_OPTION_SCAN = 15

#: A body that opens with a quantity or a unit is a wrapped continuation of
#: the previous line ("110 degrees", "5 kg", "3 marks"), never a question.
_UNIT_LEAD = re.compile(
    r"^\s*(?:\d|[.°%]"
    r"|(?:degrees?|radians?|kg|kgs?|gm|grams?|cm|mm|km|ml|l\b|litres?|liters?|"
    r"marks?|points?|units?|years?|days?|hours?|minutes?|seconds?|months?|"
    r"rupees?|rs\.?|₹|\$|%|times|sq\.?|feet|foot|inch(?:es)?|metres?|meters?)\b)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Data carriers
# --------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Output of :func:`parse_pages`."""

    questions: List[Question] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: How many questions the sections *declare* ("Attempt all 20 questions").
    declared_count: Optional[int] = None
    #: How question boundaries were found: "strict" or "relaxed".
    strategy: str = "strict"
    #: Section labels the document actually printed as headings.
    sections_seen: List[str] = field(default_factory=list)
    #: Numeric lines that looked like question starts but were not promoted.
    rejected: List[str] = field(default_factory=list)


@dataclass
class _Block:
    """One question under construction."""

    number: str
    page: int
    section: Optional[str] = None
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    section_marks: Optional[int] = None
    lines: List[Tuple[int, str]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Page furniture removal
# --------------------------------------------------------------------------

def strip_running_heads(pages: Sequence[Sequence[str]]) -> List[List[str]]:
    """Drop repeated page furniture (headers, footers, watermarks).

    A line is removed only when it is *both*

    * repeated on at least 60% of the pages (so genuine content, which
      normally appears once, is never touched), and
    * sitting in the outer band of the page (first/last few lines), where
      running heads live, and
    * not a bare number (a marks slot must survive) and not a question
      start.

    The two explicit furniture patterns (page numbers, ``Q.NO.`` table
    headers) are applied first regardless of repetition.
    """
    page_count = len(pages)
    cleaned: List[List[str]] = [
        [t for t in (_normalise(l) for l in lines) if t] for lines in pages
    ]
    if page_count < 2:
        return cleaned

    band = 3
    counts: dict = {}
    for lines in cleaned:
        for text in set(lines):
            counts[text] = counts.get(text, 0) + 1

    threshold = max(2, -(-page_count * 6 // 10))  # ceil(60%)
    repeated = {t for t, c in counts.items() if c >= threshold}

    def is_furniture(lines: Sequence[str], index: int, text: str) -> bool:
        if _QNO_HEADER.match(text) or _PAGE_FOOTER.match(text):
            return True
        if text not in repeated or len(text) > 60:
            return False
        if _BARE_NUMBER.match(text) or _QUESTION_START.match(text):
            return False
        # A running head is a label, not a sentence: real question wording
        # ends in punctuation, so a repeated line that does not is
        # furniture. This keeps repeated *content* out of the crossfire.
        if re.search(r"[.?!…।]\s*$", text):
            return False
        return index < band or index >= len(lines) - band

    result: List[List[str]] = []
    for lines in cleaned:
        result.append(
            [
                text
                for i, text in enumerate(lines)
                if not is_furniture(lines, i, text)
            ]
        )
    return result


# --------------------------------------------------------------------------
# Line classification
# --------------------------------------------------------------------------

def _normalise(line: str) -> str:
    """Light, non-destructive cleanup: NBSP -> space, strip the edges."""
    return line.replace(" ", " ").strip()


def _classify(line: str) -> str:
    text = _normalise(line)
    if not text:
        return "blank"
    if _QNO_HEADER.match(text) or _PAGE_FOOTER.match(text):
        return "furniture"
    if _CHAPTER_TAG.match(text):
        return "chapter"
    if len(text) <= _MAX_HEADING_LEN and _SECTION.match(text):
        return "section"
    if _ATTEMPT.match(text):
        return "instruction"
    if _OR_SEPARATOR.match(text):
        return "or"
    if _BARE_NUMBER.match(text):
        return "bare_number"
    if _INSTRUCTIONS_HEADING.match(text):
        return "instructions_heading"
    if _QUESTION_START.match(text):
        return "question"
    return "content"


def _looks_like_terminated_question(line: str) -> bool:
    """Evidence that the previous question block is already complete.

    Used only by the relaxed numbering strategy, so that a wrapped line
    such as ``5 kg of rice`` is never promoted to a question.
    """
    text = _normalise(line)
    if _CHAPTER_TAG.match(text) or _BARE_NUMBER.match(text):
        return True
    if _OPTION.match(text) or _OPTION_INLINE.search(text):
        return True
    return bool(re.search(r"[.?!…।:]\s*$", text))


# --------------------------------------------------------------------------
# Numbering strategies
# --------------------------------------------------------------------------

def _is_question_start(
    line: str,
    expected: int,
    last: Optional[int],
    seen: set,
    relaxed: bool,
) -> Tuple[bool, int]:
    """Decide whether ``line`` opens a new question: (accept, number)."""
    match = _QUESTION_START.match(_normalise(line))
    if not match:
        return False, 0
    number = int(match.group("number"))
    if number == expected:
        return True, number
    if not relaxed:
        return False, number
    # Relaxed: numbering may restart, skip, or begin at an arbitrary value.
    if last is None:
        # Nothing accepted yet, so this is the paper's first question
        # whatever it is numbered. The caller only enters relaxed mode for
        # candidates that survive _promotable_candidate, and the "starts
        # above one" warning is reported by validation.
        return True, number
    # Otherwise only monotonic, never-seen numbers qualify, and the caller
    # additionally requires evidence that the previous block ended.
    if number > last and number not in seen:
        return True, number
    return False, number


def _blocks_from_pages(
    pages: Sequence[Sequence[str]], relaxed: bool
) -> Tuple[List[_Block], List[str], int, List[str]]:
    """Split pages into question blocks.

    Returns ``(blocks, rejected, declared_count, sections_seen)``.
    """
    blocks: List[_Block] = []
    rejected: List[str] = []
    seen: set = set()
    sections_seen: List[str] = []

    current: Optional[_Block] = None
    section: Optional[str] = None
    section_title: Optional[str] = None
    section_marks: Optional[int] = None
    declared = 0
    expected = 1
    last: Optional[int] = None
    in_instructions = False

    def close() -> None:
        nonlocal current
        if current is not None:
            blocks.append(current)
            current = None

    for page_number, lines in enumerate(pages, start=1):
        for raw in lines:
            line = _normalise(raw)
            if not line:
                continue
            kind = _classify(line)

            if kind == "section":
                match = _SECTION.match(line)
                close()
                in_instructions = False
                section = match.group("label").upper()
                if section not in sections_seen:
                    sections_seen.append(section)
                section_title = (match.group("title") or "").strip() or None
                section_marks = None
                continue

            if kind == "instructions_heading":
                # Only a heading before the first question; after that the
                # word "instructions" is ordinary question wording.
                if current is None:
                    in_instructions = True
                else:
                    current.lines.append((page_number, line))
                continue

            if kind == "instruction":
                in_instructions = False
                count = _DECLARED_COUNT.match(line)
                if count:
                    declared += int(count.group("all") or count.group("pick"))
                carried = _CARRIED_MARKS.search(line)
                if carried:
                    section_marks = int(carried.group("marks"))
                continue

            if kind == "furniture":
                continue

            if in_instructions:
                # Leave the instruction block on the first sign of real
                # question content: a section attempt line, a table header,
                # or a question number without the "1." list punctuation.
                if kind == "question":
                    match = _QUESTION_START.match(line)
                    if match.group("dot"):
                        continue
                    in_instructions = False
                else:
                    continue

            if kind == "chapter":
                # Belongs to the question it follows; never question text.
                # An OR alternative can carry its own tag, so all of them are
                # kept rather than only the first.
                if current is not None:
                    if current.chapter is None:
                        current.chapter = line.strip()
                    elif line.strip() not in current.chapter:
                        current.chapter = f"{current.chapter} {line.strip()}"
                continue

            accept, number = _is_question_start(
                line, expected, last, seen, relaxed
            )
            if accept and _boundary_ok(current, relaxed):
                match = _QUESTION_START.match(line)
                close()
                current = _Block(
                    number=str(number),
                    page=page_number,
                    section=section,
                    section_title=section_title,
                    section_marks=section_marks,
                )
                current.lines.append((page_number, match.group("rest").strip()))
                expected = number + 1
                last = number
                seen.add(number)
                continue

            if _QUESTION_START.match(line):
                rejected.append(line[:120])

            if current is None:
                # Preamble (title block, general instructions) before the
                # first question: not question content, and never invented.
                continue

            current.lines.append((page_number, line))

    close()
    return blocks, rejected, declared, sections_seen


def _boundary_ok(current: Optional[_Block], relaxed: bool) -> bool:
    """Guard used by the relaxed strategy only (see module docstring)."""
    if not relaxed or current is None or not current.lines:
        return True
    return _looks_like_terminated_question(current.lines[-1][1])


# --------------------------------------------------------------------------
# Block -> Question
# --------------------------------------------------------------------------

def _option_groups(text: str) -> List[str]:
    """Parse one line into option strings, or [] if it is not an option line.

    Several options printed on one line are all kept, in order, with their
    labels: "(A) 25x^2  (B) 5x^2" -> ["(A) 25x^2", "(B) 5x^2"].
    """
    stripped = _normalise(text)
    if not stripped.startswith("("):
        return []
    groups = list(_OPTION_INLINE.finditer(stripped))
    if not groups:
        return []
    out: List[str] = []
    for index, group in enumerate(groups):
        end = (
            groups[index + 1].start()
            if index + 1 < len(groups)
            else len(stripped)
        )
        body = stripped[group.end():end].strip()
        out.append(f"({group.group('label')}) {body}".rstrip())
    return out


def _parse_option_line(text: str) -> Optional[List[str]]:
    """An option line that *starts* a run: MCQ options always begin at (A).

    Requiring the "A" label is what keeps a sub-part written as ``(a) ...``
    (or even ``(A) ...``) from being lifted out of the question's wording.
    """
    groups = _option_groups(text)
    if not groups or not groups[0].startswith("(A)"):
        return None
    return groups


def _option_label(option: str) -> str:
    match = re.match(r"^\(([A-H])\)", option)
    return match.group(1) if match else ""


def _is_block_boundary_line(text: str) -> bool:
    t = _normalise(text)
    return bool(
        _OR_SEPARATOR.match(t)
        or _BARE_NUMBER.match(t)
        or _CHAPTER_TAG.match(t)
        or _QUESTION_START.match(t)
    )


def _split_options(
    lines: Sequence[Tuple[int, str]]
) -> Tuple[List[str], List[Tuple[int, str]]]:
    """Split a block into (option strings, remaining lines).

    Only the option run attached to the stem is lifted out. The run must
    start before the first sub-part marker and before any boundary line, and
    a non-option line only extends it when the previous option was clearly
    cut mid-sentence (a wrapped option), never after a finished one.
    """
    options: List[str] = []
    run_start = 0
    index = 0
    limit = min(len(lines), _MAX_OPTION_SCAN)

    while index < limit and not options:
        _page, text = lines[index]
        # Checked before the sub-part guard: an option line is also a
        # "(A) ..." line and must not be mistaken for a sub-part.
        parsed = _parse_option_line(text)
        if parsed:
            options = parsed
            run_start = index
        elif _is_block_boundary_line(text) or _SUBPART.match(text):
            break
        index += 1

    if not options:
        return [], list(lines)

    while index < limit:
        _page, text = lines[index]
        groups = _option_groups(text)
        if groups:
            # Only continue the run with the next expected label.
            last_label = _option_label(options[-1])
            expected_label = chr(ord(last_label) + 1) if last_label else ""
            if expected_label and groups[0].startswith(f"({expected_label}"):
                options.extend(groups)
                index += 1
                continue
            break
        if _is_block_boundary_line(text) or _SUBPART.match(text):
            break
        # Wrapped tail of the previous option: only when it was cut off
        # mid-sentence, otherwise this is new content, not a continuation.
        if re.search(r"[.?!…।:]\s*$", options[-1]):
            break
        options[-1] = f"{options[-1]} {_normalise(text)}".strip()
        index += 1

    # Everything before the run is stem wording, everything after it is the
    # rest of the block. Both are kept - nothing is dropped here.
    return options, list(lines[:run_start]) + list(lines[index:])


def _extract_marks(
    stem: str,
    body_lines: List[Tuple[int, str]],
    section_marks: Optional[int],
) -> Tuple[Optional[int], str, List[Tuple[int, str]], str]:
    """Find this question's own marks: (marks, stem, body, source)."""
    marks: Optional[int] = None
    source = "none"
    kept: List[Tuple[int, str]] = []

    # 1. A standalone integer line inside the block is the marks slot.
    for page, line in body_lines:
        if marks is None and _BARE_NUMBER.match(line):
            value = int(line)
            if 0 < value <= _MAX_MARKS:
                marks = value
                source = "standalone_line"
                continue
        kept.append((page, line))

    new_stem = stem
    # 2. Marks printed after the stem or after the last line of the block,
    #    e.g. "How did Leela help? 1". Only accepted directly after sentence
    #    punctuation, so that a question ending "the value is 5." is never
    #    read as 5 marks.
    if marks is None:
        targets = [0] + ([len(kept) - 1] if kept else [])
        for pos in targets:
            target = stem if pos == 0 else kept[pos][1]
            match = _TRAILING_MARKS.match(target)
            if not match:
                continue
            value = int(match.group("marks"))
            if not 0 < value <= _MAX_MARKS:
                continue
            marks = value
            source = "trailing"
            body = match.group("body").rstrip()
            if pos == 0:
                new_stem = body
            else:
                kept[pos] = (kept[pos][0], body)
            break

    # 3. Only when the question states no marks of its own, fall back to the
    #    section's explicit "Each carries N mark(s)" instruction. This is a
    #    structure the paper itself declares, not an inference from length.
    if marks is None and section_marks is not None:
        marks = section_marks
        source = "section_instruction"

    return marks, new_stem, kept, source


def _join(lines: Iterable[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(l for l in lines if l)).strip()


def _classify_type(text: str, options: List[str], marks: Optional[int]) -> str:
    if options:
        return "mcq"
    if _CASE_STUDY.search(text):
        return "long"
    if marks is not None and marks >= 5:
        return "long"
    if _NUMERICAL_HINTS.search(text):
        return "numerical"
    return "short"


def _block_to_question(block: _Block) -> Question:
    stem = block.lines[0][1] if block.lines else ""
    # The chapter tag is captured separately (Question.chapter) and must
    # never appear as question wording.
    body = [(p, t) for p, t in block.lines[1:] if not _CHAPTER_TAG.match(t)]

    marks, stem, body, _source = _extract_marks(
        stem, body, block.section_marks
    )

    options, remaining = _split_options(body)
    has_or = any(_OR_SEPARATOR.match(t) for _, t in body)
    text = _join([stem] + [t for _, t in remaining])

    return Question(
        number=block.number,
        section=block.section,
        text=text,
        marks=marks,
        type=_classify_type(text, options, marks),
        options=options or None,
        has_figure=bool(_FIGURE_REF.search(text)),
        page=block.page,
        choice_group=block.number if has_or else None,
        section_title=block.section_title,
        chapter=block.chapter,
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def _promotable_candidate(candidate: str) -> bool:
    """Is a rejected numeric line worth retrying as a question start?

    Deliberately strict: a wrapped continuation such as ``110 degrees. The
    park designer ...`` or ``5 kg of rice ...`` starts with a number and
    must never be promoted, because promoting it would split one question
    into two.
    """
    if _BARE_NUMBER.match(candidate):
        return False
    match = _QUESTION_START.match(_normalise(candidate))
    if not match:
        return False
    body = match.group("rest")
    # Needs real prose, not a measurement or a stray figure.
    if not re.search(r"[A-Za-z]{3,}", body):
        return False
    if _UNIT_LEAD.match(body):
        return False
    if len(body) < 25 and not re.search(r"[A-Za-z]{3,}\s+\S+\s+\S", body):
        return False
    return True


def parse_pages(pages: Sequence[Sequence[str]]) -> ParseResult:
    """Parse pages of lines into canonical questions.

    The strict (sequential) numbering strategy runs first. If it rejected
    numeric lines that genuinely look like question starts, the relaxed
    strategy is attempted and kept **only** when it yields strictly more
    questions with unique numbers - so a document never ends up with fewer
    questions than the parser could justify, and never gains speculative
    ones.
    """
    cleaned = strip_running_heads(pages)

    blocks, rejected, declared, sections = _blocks_from_pages(
        cleaned, relaxed=False
    )
    result = ParseResult(
        questions=[_block_to_question(b) for b in blocks],
        declared_count=declared or None,
        sections_seen=sections,
        strategy="strict",
        rejected=rejected,
    )

    if any(_promotable_candidate(line) for line in rejected):
        r_blocks, _, _, _ = _blocks_from_pages(cleaned, relaxed=True)
        numbers = [b.number for b in r_blocks]
        if len(r_blocks) > len(blocks) and len(numbers) == len(set(numbers)):
            result.questions = [_block_to_question(b) for b in r_blocks]
            result.strategy = "relaxed"
            result.notes.append(
                f"numbering strategy: relaxed "
                f"({len(blocks)} -> {len(r_blocks)} questions)"
            )
    return result


def parse_lines_to_questions(pages: Sequence[Sequence[str]]) -> List[Question]:
    """Convenience wrapper used by :mod:`app.ingestion.paper_loader`."""
    return parse_pages(pages).questions
