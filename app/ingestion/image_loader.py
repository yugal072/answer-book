"""Image OCR ingestion: question-paper image -> canonical Paper.

IMAGE -> Qwen vision extraction (Groq) -> canonical Question/Paper
(``app.models.loaders_models``), the same representation PDF ingestion
produces, so downstream code never needs to know the input origin.

Design notes
------------
- One vision call per image. Pages that overflow the model output budget
  are retried once as two overlapping tiles whose extractions are merged
  with overlap-dedup and boundary reassembly (no silent truncation).
- Values are never invented: invisible fields use the schema defaults
  (None / False). Unreadable content is reported via low confidence in
  logs, never hallucinated.
- Secrets come from the ``GROQ_API_KEY`` environment variable only.

Public surface::

    extract_paper_from_image_file(path, page_number=1) -> Paper
    extract_paper_from_images(paths)                   -> Paper (multi-page)
    extract_paper_from_image_bytes(data, ...)          -> Paper
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import statistics
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageOps

from app.ingestion.errors import (
    DocumentError,
    ExtractionError,
    IngestionError,
    TruncationError,
    ValidationError,
)
from app.ingestion.validation import raise_if_invalid, validate_paper
from app.models.loaders_models import Paper, Question

log = logging.getLogger(__name__)

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "qwen/qwen3.8-27b")
# Verified 2026-09-19: the Groq on_demand tier (OTPM 1000) rejects single
# requests above ~1000 output tokens, hence the cap plus tile-splitting.
VISION_MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "1000"))
VISION_RENDER_MAX_DIM = int(os.environ.get("VISION_RENDER_MAX_DIM", "2048"))
GROQ_TIMEOUT_S = float(os.environ.get("GROQ_TIMEOUT_S", "60"))

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
#: PIL format names accepted by the bytes entry point. A file whose extension
#: lies (a PNG named .jpg is fine; a GIF is not) is rejected with a clear
#: message instead of being sent to the model.
_PIL_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024

TILE_OVERLAP = 0.25
MIN_TILE_HEIGHT = 800

# Internal vision vocabulary -> canonical type (full wording always stays
# in Question.text; OR alternatives set choice_group, as in PDF ingestion).
TYPE_MAP = {
    "mcq": "mcq",
    "numerical": "numerical",
    "short_answer": "short",
    "descriptive": "long",
    "coding": "long",
    "fill_in_the_blank": "short",
    "true_false": "short",
    "unknown": "short",
}

SYSTEM_PROMPT = """You are extracting questions from a question-paper image. Read the image carefully.

Rules:
1. You are extracting questions, NOT solving them. Never include answers or solutions.
2. Preserve the original wording as closely as possible. Do not paraphrase unnecessarily.
3. Extract EVERY visible question. Do not truncate, even if the page has many questions.
4. Preserve numbering and subquestion structure (e.g. 1, 2, 1(a), 1(b), 2(i), 2(ii)).
5. Preserve mathematical expressions, symbols, units and equations as accurately as possible.
6. Preserve MCQ options exactly as visible, in order, with their labels.
7. Never invent: missing question text, marks, question numbers, MCQ options. Never merge unrelated questions.
8. Use null for any field you cannot determine (marks, options, section, choice_group).
9. If text is genuinely unreadable, keep what you can read and report it via low confidence. Never hallucinate.
10. Handle English, Hindi, Bengali, and mixed-language papers; preserve the original script.
11. Classify question_type as one of: mcq, short_answer, descriptive, numerical, coding, fill_in_the_blank, true_false, unknown. Use unknown when uncertain.
12. If a section header (e.g. "Section A") is visible for the question, record it in section; otherwise use null.
13. Set has_figure to true only when a diagram, figure, graph or plot belongs to the question; otherwise false.
14. If the question offers an internal choice (an "OR" alternative), set choice_group to the question number; otherwise use null.
15. Extract paper metadata when visible: subject, class, and board.
16. Use null when any metadata cannot be determined.
17. Phone photos may be skewed, rotated, shadowed or slightly blurred. Read carefully anyway; use null and low confidence for what is truly unreadable instead of guessing.
18. Return ONLY the requested JSON object, no markdown fences, no commentary.

Required JSON shape:
{
  "metadata": {
    "subject": null,
    "class": null,
    "board": null
  },
  "questions": [
    {
      "question_number": "1",
      "question_text": "...",
      "marks": 5,
      "question_type": "descriptive",
      "options": [],
      "source_page": 1,
      "confidence": 0.96,
      "section": null,
      "has_figure": false,
      "choice_group": null
    }
  ]
}
"""

USER_PROMPT = (
    "Extract all visible questions from this question-paper image into the required JSON shape. "
    "Return only the JSON object."
)

VALID_QTYPES = set(TYPE_MAP)


class ImageError(DocumentError, ValueError):
    """Unusable image input (missing, empty, unsupported, corrupt, huge).

    Subclasses ``ValueError`` as well, so any caller that used to catch
    ``ValueError`` around this module keeps working.
    """


# ``ExtractionError`` and ``TruncationError`` are imported from
# app.ingestion.errors and re-exported here, so the historical
# ``image_loader.ExtractionError`` name keeps working and is now part of the
# shared ingestion hierarchy (``IngestionError``).
__all__ = [
    "ImageError",
    "ExtractionError",
    "TruncationError",
    "IngestionError",
    "ValidationError",
    "load_image_bytes",
    "parse_and_validate",
    "to_canonical",
    "assemble_paper",
    "extract_paper_from_image_file",
    "extract_paper_from_image_bytes",
    "extract_paper_from_images",
    "to_langgraph_questions",
]


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
    if not key:
        raise ExtractionError(
            "GROQ_API_KEY is not set. Export it in the environment; "
            "keys are never hardcoded."
        )
    return key


def load_image_bytes(path: str | Path) -> bytes:
    """Load + validate an image file. Raises ImageError with actionable messages."""
    p = Path(path)
    if not p.exists():
        raise ImageError(f"Image not found: {path}")
    if not p.is_file():
        raise ImageError(f"Not a file: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageError(
            f"Unsupported image type '{p.suffix or '(none)'}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    raw = p.read_bytes()
    if not raw:
        raise ImageError(f"Image is empty: {path}")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(f"Image too large ({len(raw) / 1e6:.1f} MB): {path}")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
    except Exception as e:
        raise ImageError(f"Invalid/corrupt image {path}: {e}") from e
    return raw


def open_image(data: bytes, origin: str = "image bytes") -> Image.Image:
    """Decode image bytes into RGB, with actionable errors.

    Used by the bytes entry point, which previously let PIL's own
    exceptions escape as if they were extraction failures.
    """
    if not data:
        raise ImageError(f"Empty image ({origin})")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(
            f"Image too large ({len(data) / 1e6:.1f} MB, limit "
            f"{MAX_IMAGE_BYTES / 1e6:.0f} MB): {origin}"
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as e:
        raise ImageError(f"Invalid/corrupt image ({origin}): {e}") from e
    if image.format and image.format.upper() not in _PIL_FORMATS:
        raise ImageError(
            f"Unsupported image format {image.format} ({origin}). "
            f"Supported: PNG, JPEG, WEBP."
        )
    return image.convert("RGB")


def _prepare(image: Image.Image) -> Image.Image:
    """Lightweight normalization for scans and phone photos.

    Orientation, downscale-if-huge, gentle contrast, mild sharpening
    (helps slightly blurred phone photos; harmless on clean scans).
    Never upscales, never binarizes.
    """
    from PIL import ImageFilter

    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size
    longest = max(w, h)
    if longest > VISION_RENDER_MAX_DIM:
        scale = VISION_RENDER_MAX_DIM / longest
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.15)
    image = ImageOps.autocontrast(image, cutoff=0.5)
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=70, threshold=3))


def _pil_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    _prepare(image).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rsplit("```", 1)[0].strip():
            t = t.rsplit("```", 1)[0]
    return t.strip()


def _parse_item(item: dict, i: int, default_page: int) -> dict:
    """Validate + normalize one raw model item. Raises ExtractionError."""
    if not isinstance(item, dict):
        raise ExtractionError(f"questions[{i}] is not an object")
    qd = dict(item)
    qd.setdefault("question_text", "")
    qd.setdefault("marks", None)
    qd.setdefault("options", [])
    qd.setdefault("source_page", default_page)
    qd.setdefault("confidence", 0.0)
    qd.setdefault("section", None)
    qd.setdefault("has_figure", False)
    qd.setdefault("choice_group", None)
    if "question_number" not in qd or qd["question_number"] in (None, ""):
        raise ExtractionError(f"questions[{i}] missing required 'question_number'")
    qd["question_number"] = str(qd["question_number"])
    if qd.get("question_type") not in VALID_QTYPES:
        qd["question_type"] = "unknown"
    if qd.get("options") is None:
        qd["options"] = []
    if qd.get("section") is not None:
        qd["section"] = str(qd["section"]).strip().upper().replace("SECTION ", "") or None
    qd["has_figure"] = bool(qd.get("has_figure", False))
    if qd.get("choice_group") is not None:
        qd["choice_group"] = str(qd["choice_group"])
    return qd


def parse_and_validate(
    raw_text: str, default_page: int = 1
) -> list[dict]:
    """Parse model JSON into normalized dicts. Raises ExtractionError."""
    import re as _re

    cleaned = _strip_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model did not return valid JSON: {e}") from e

    if not isinstance(data, dict) or "questions" not in data:
        raise ExtractionError(
            "Model response missing required top-level 'questions' key"
        )

    if not isinstance(data["questions"], list):
        raise ExtractionError("'questions' must be a list")

    out = []

    for i, item in enumerate(data["questions"]):
        qd = _parse_item(item, i, default_page)

        if qd["choice_group"] is None and _re.search(
            r"\bOR\b", qd.get("question_text", "")
        ):
            qd["choice_group"] = qd["question_number"]

        out.append(qd)

    return out


def to_canonical(qd: dict) -> Question:
    """Normalized vision dict -> canonical Question (never fabricates)."""
    options = list(qd["options"]) if qd["options"] else None
    marks: Optional[int] = None
    if qd.get("marks") is not None:
        try:
            f = float(qd["marks"])
            marks = int(f) if f.is_integer() else None
        except (TypeError, ValueError):
            marks = None
    return Question(
        number=qd["question_number"],
        section=qd.get("section"),
        text=qd.get("question_text", ""),
        marks=marks,
        type=TYPE_MAP.get(qd.get("question_type", "unknown"), "short"),
        options=options,
        has_figure=bool(qd.get("has_figure", False)),
        page=qd.get("source_page"),
        choice_group=qd.get("choice_group"),
    )


_TOKEN_BUDGET_EXHAUSTED = re.compile(
    r"tokens per day|\bTPD\b|requests per day|\bRPD\b|monthly", re.IGNORECASE
)


def _vision_call(data_url: str, model: str) -> tuple[str, str]:
    """One Groq vision call -> (content, finish_reason), with transport retries.

    Retries transport errors and short-window rate limits. A *daily* token
    or request limit is not retried: waiting cannot help, and the caller
    gets an actionable message instead of a slow, confusing failure.
    """
    from groq import APIConnectionError, APITimeoutError, Groq, RateLimitError

    client = Groq(api_key=_api_key(), timeout=GROQ_TIMEOUT_S)
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=VISION_MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ],
            )
            break
        except RateLimitError as e:
            detail = str(e)
            if _TOKEN_BUDGET_EXHAUSTED.search(detail):
                raise ExtractionError(
                    f"Groq daily token/request budget exhausted for {model}: "
                    f"{detail}. No retry is attempted because waiting will not "
                    f"help; retry after the quota resets or raise the limit."
                ) from e
            last_err = e
            log.warning("Groq rate limit (attempt %d); waiting 10s", attempt)
            time.sleep(10 * attempt)
        except (APITimeoutError, APIConnectionError) as e:
            last_err = e
            wait = 2 * attempt
            log.warning(
                "Groq transient error (attempt %d): %s; waiting %ss",
                attempt, type(e).__name__, wait,
            )
            time.sleep(wait)
    else:
        raise ExtractionError(f"Groq API transport failure after retry: {last_err}")

    choice = resp.choices[0]
    content = choice.message.content or ""
    finish = choice.finish_reason

    if finish == "length":
        # Never return partial output: the caller re-reads the page as tiles.
        raise TruncationError(
            "Model output hit the token budget "
            f"({VISION_MAX_TOKENS} tokens)."
        )
    if finish not in (None, "stop", "tool_calls"):
        raise ExtractionError(
            f"Model stopped for an unexpected reason: {finish}"
        )
    if not content.strip():
        raise ExtractionError(
            "Model returned an empty response for this page; the image may be "
            "blank, rotated beyond recognition, or unreadable."
        )
    return content, finish


def _extract_data_url(
    data_url: str, source_page: int, model: str
) -> tuple[list[dict], dict]:

    content, _ = _vision_call(data_url, model)

    cleaned = _strip_fences(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Model did not return valid JSON: {e}") from e

    if not isinstance(data, dict) or "questions" not in data:
        raise ExtractionError(
            "Model response missing required top-level 'questions' key"
        )

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    items = parse_and_validate(content, default_page=source_page)

    # The page a question came from is known for certain here: it is the
    # image that was just read. A model-reported page number is never
    # trusted over that, because a wrong page silently corrupts every
    # downstream page reference.
    for item in items:
        item["source_page"] = source_page

    return items, metadata

def _find_split_row(image: Image.Image, search_radius: int = 120) -> int:
    """Row near the vertical middle cutting through whitespace, not text."""
    gray = image.convert("L")
    w, h = gray.size
    px = gray.load()
    darkness = [sum(255 - px[x, y] for x in range(0, w, 4)) for y in range(h)]
    mid = h // 2
    lo, hi = max(0, mid - search_radius), min(h - 1, mid + search_radius)
    band = darkness[lo:hi + 1]
    if not band:
        return mid
    threshold = max(statistics.mean(band) * 0.25, 1.0)
    best, best_key = mid, None
    for i, d in enumerate(band):
        y = lo + i
        key = (d > threshold, abs(y - mid), -y)
        if best_key is None or key < best_key:
            best, best_key = y, key
    return best


def _merge_options(base: list, extra: list) -> list:
    """Union two option lists, order-preserving, de-duplicated by label.

    Two tiles of the same page overlap, and the model may transcribe the
    same option slightly differently in each ("(C) -1" vs "(C) −1"). The
    first spelling wins - nothing is rewritten - but the option is kept
    once, so an MCQ never grows eight options out of four.
    """
    merged = list(base)
    by_label = {}
    for option in merged:
        label = _option_label(option)
        if label:
            by_label.setdefault(label, option)
    for option in extra:
        label = _option_label(option)
        if label and label in by_label:
            continue
        merged.append(option)
        if label:
            by_label[label] = option
    return merged


def _option_label(option: str) -> str:
    match = re.match(r"^\s*\(([A-Za-z])\)", option or "")
    return match.group(1).upper() if match else ""


def _dedupe_key(item: dict) -> tuple:
    """Identity of a vision question for cross-tile/cross-page dedupe."""
    return (
        str(item.get("question_number", "")).strip(),
        re.sub(r"\s+", " ", str(item.get("question_text", ""))).strip()[:200],
    )


def _confidences(items: list) -> dict:
    """Question number -> vision confidence, for low-confidence reporting."""
    return {
        str(item.get("question_number", "")).strip(): item["confidence"]
        for item in items
        if isinstance(item.get("confidence"), (int, float))
    }


def _merge_tile_questions(top: list[dict], bottom: list[dict]) -> list[dict]:
    """Merge two same-page tile extractions without losing boundary content.

    Identical repeats dedupe; straddling questions reassemble in reading
    order; options union by label so an option seen by either tile always
    survives but is never duplicated. Text concatenates only for genuinely
    disjoint fragments.
    """
    merged: list[dict] = []
    seen: set = set()
    by_number: dict = {}
    for q in list(top) + list(bottom):
        key = (
            str(q.get("question_number", "")).strip(),
            str(q.get("question_text", "")).strip()[:200],
            tuple(q.get("options") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        number = str(q.get("question_number", "")).strip()
        prev = by_number.get(number)
        if prev is not None:
            prev["options"] = _merge_options(
                prev.get("options") or [], q.get("options") or []
            )
            if prev.get("question_text") != q.get("question_text"):
                a, b = prev["question_text"], q["question_text"]
                prev["question_text"] = max(a, b, key=len) if (a in b or b in a) \
                    else f"{a} {b}".strip()
            if prev.get("marks") is None:
                prev["marks"] = q.get("marks")
            prev["confidence"] = min(
                prev.get("confidence", 0.0), q.get("confidence", 0.0)
            )
            if not prev.get("section") and q.get("section"):
                prev["section"] = q["section"]
            prev["has_figure"] = bool(
                prev.get("has_figure") or q.get("has_figure")
            )
            continue
        by_number[number] = q
        merged.append(q)
    return merged


def _text_key(item: dict) -> tuple:
    """Normalised identity of a question's wording, for dedupe.

    Overlapping tiles routinely report the same question twice with
    different numbering ("1(i)" in one tile, "i" in the next), so
    identity cannot rest on the number alone. The page is part of the
    key: a question repeated on two *different* pages is not a tiling
    artefact and is left alone.
    """
    text = re.sub(
        r"\s+", " ", str(item.get("question_text", ""))
    ).strip().lower()
    return (item.get("source_page"), re.sub(r"[^a-z0-9]+", "", text)[:160])


def _dedupe_by_text(items: list[dict]) -> list[dict]:
    """Drop a question that repeats an earlier one verbatim.

    The first occurrence wins, so the numbering that reads in document
    order is kept; options and marks from the repeat are merged in rather
    than thrown away.
    """
    kept: list[dict] = []
    by_key: dict = {}
    for item in items:
        key = _text_key(item)
        if not key:
            kept.append(item)
            continue
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = item
            kept.append(item)
            continue
        previous["options"] = _merge_options(
            previous.get("options") or [], item.get("options") or []
        )
        if previous.get("marks") is None:
            previous["marks"] = item.get("marks")
        previous["confidence"] = min(
            previous.get("confidence", 0.0), item.get("confidence", 0.0)
        )
    return kept


def _extract_page(
    image: Image.Image, source_page: int, model: str
) -> tuple[list[dict], dict]:
    """Extract one page; tile-split with overlap only on token-budget overflow.

    A tile that itself overflows is split once more (depth 2 max); merges
    reuse the lossless overlap logic, so boundary options always survive.
    """
    try:
        items, metadata = _extract_data_url(
            _pil_to_data_url(image), source_page, model
        )
        return _dedupe_by_text(items), metadata
    except TruncationError:
        if image.size[1] < MIN_TILE_HEIGHT:
            raise
        log.info("Page %d overflowed the token budget; retrying as overlapping tiles.",
                 source_page)
        w, h = image.size
        overlap = int(h * TILE_OVERLAP)
        split = _find_split_row(image)
        tiles = [image.crop((0, 0, w, split + overlap)),
                 image.crop((0, split - overlap, w, h))]
        items, metadata = _extract_tile_list(
            tiles, source_page, model, _depth=0
        )
        return _dedupe_by_text(items), metadata


def _extract_tile_list(tiles, source_page: int, model: str, _depth: int):
    merged_qs: list[dict] = []
    tile_metas: list[dict] = []
    for tile in tiles:
        try:
            tile_result, tile_meta = _extract_data_url(
                _pil_to_data_url(tile), source_page, model
            )
        except TruncationError:
            if _depth >= 1 or tile.size[1] < MIN_TILE_HEIGHT:
                raise
            log.info("Tile overflowed; splitting again (depth %d).", _depth + 1)
            w, h = tile.size
            overlap = int(h * TILE_OVERLAP)
            split = _find_split_row(tile)
            sub = [tile.crop((0, 0, w, split + overlap)),
                   tile.crop((0, split - overlap, w, h))]
            tile_result, tile_meta = _extract_tile_list(sub, source_page, model, _depth + 1)
        tile_metas.append(tile_meta)
        merged_qs = _merge_tile_questions(merged_qs, tile_result)
    merged_meta: dict = {}
    for meta in tile_metas:
        for k, v in (meta or {}).items():
            if v and k not in merged_meta:
                merged_meta[k] = v
    return merged_qs, merged_meta

def assemble_paper(
    questions: list[Question],
    source_bytes: list[bytes],
    metadata: dict | None = None,
    status: str = "ready",
) -> Paper:
    """Assemble a canonical Paper (same paper_id/fingerprint convention as PDF ingestion).

    ``total_marks`` is the sum only when *every* question has a known marks
    value. A partial sum would silently understate the paper, so it is
    reported as unknown instead - the same rule the PDF path uses.
    """
    h = hashlib.sha256()
    for b in source_bytes:
        h.update(b)
    fingerprint = h.hexdigest() if source_bytes else hashlib.sha256(b"").hexdigest()
    sections = list(dict.fromkeys(q.section for q in questions if q.section))
    known = [q.marks for q in questions if q.marks is not None]
    total_marks = sum(known) if questions and len(known) == len(questions) else None
    return Paper(
        paper_id=f"pap_{fingerprint[:8]}",
        fingerprint=fingerprint,
        status=status,
        subject=metadata.get("subject") if metadata else None,
        class_name=metadata.get("class") if metadata else None,
        board=metadata.get("board") if metadata else None,
        questions=questions,
        total_questions=len(questions),
        total_marks=total_marks,
        sections=sections,
    )


def _merge_metadata(target: dict, extra: dict | None) -> dict:
    """First non-empty value per key wins; metadata is never overwritten
    with a blank and never invented."""
    for key, value in (extra or {}).items():
        if value and not target.get(key):
            target[key] = value
    return target


def extract_paper_from_image_bytes(
    data: bytes,
    source_page: int = 1,
    model: str | None = None,
    diagnostics: Optional[list] = None,
) -> Paper:
    """Image bytes -> canonical Paper."""
    if not isinstance(source_page, int) or source_page < 1:
        raise ImageError(f"Invalid page number: {source_page!r}")
    image = open_image(data, "image bytes")
    items, metadata = _extract_page(
        image, source_page, model or EXTRACTION_MODEL
    )
    paper = assemble_paper(
        [to_canonical(q) for q in items], [data], metadata
    )
    report = validate_paper(
        paper, source="image", total_pages=source_page,
        confidences=_confidences(items),
    )
    if diagnostics is not None:
        diagnostics.extend(report.diagnostics)
    raise_if_invalid(report, context="image ingestion")
    return paper


def extract_paper_from_image_file(
    path: str | Path,
    page_number: int = 1,
    model: str | None = None,
    diagnostics: Optional[list] = None,
) -> Paper:
    """Image file (PNG/JPEG/WebP) -> canonical Paper."""
    raw = load_image_bytes(path)
    image = open_image(raw, str(path))
    items, metadata = _extract_page(
        image, page_number, model or EXTRACTION_MODEL
    )
    paper = assemble_paper(
        [to_canonical(q) for q in items], [raw], metadata
    )
    report = validate_paper(
        paper, source="image", total_pages=page_number,
        confidences=_confidences(items),
    )
    if diagnostics is not None:
        diagnostics.extend(report.diagnostics)
    raise_if_invalid(report, context=f"image ingestion of {Path(path).name}")
    return paper


def extract_paper_from_images(
    paths: list[str | Path],
    model: str | None = None,
    diagnostics: Optional[list] = None,
) -> Paper:
    """Multiple image pages -> one canonical Paper, page numbers preserved.

    Metadata is merged across pages (first non-empty value per key), so a
    subject printed only on the first photo is not lost when page 1 of the
    model response happened to omit it. A question repeated across two
    overlapping photos is kept once.
    """
    if not paths:
        raise ImageError("No images given: pass at least one image path.")

    model = model or EXTRACTION_MODEL
    merged: list[dict] = []
    seen: set = set()
    sources: list[bytes] = []
    metadata: dict = {}
    for idx, p in enumerate(paths, start=1):
        raw = load_image_bytes(p)
        sources.append(raw)
        image = open_image(raw, str(p))
        items, page_metadata = _extract_page(image, idx, model)
        _merge_metadata(metadata, page_metadata)
        if not items:
            log.warning("Page %d (%s) produced no questions", idx, Path(p).name)
        for q in items:
            q["source_page"] = idx
            key = _dedupe_key(q)
            if key in seen:
                continue
            seen.add(key)
            merged.append(q)

    paper = assemble_paper(
        [to_canonical(q) for q in _dedupe_by_text(merged)], sources, metadata
    )
    report = validate_paper(
        paper, source="image", total_pages=len(paths),
        confidences=_confidences(merged),
    )
    if diagnostics is not None:
        diagnostics.extend(report.diagnostics)
    raise_if_invalid(report, context="multi-page image ingestion")
    return paper

def to_langgraph_questions(paper: Paper) -> list[dict]:
    """Canonical questions as plain dicts for the solve graph / answer stage."""
    return [q.model_dump() for q in paper.questions]
