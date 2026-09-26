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
import statistics
import time
from pathlib import Path
from typing import Optional

from PIL import Image, ImageEnhance, ImageOps

from app.models.loaders_models import Paper, Question

log = logging.getLogger(__name__)

EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "qwen/qwen3.8-27b")
# Verified 2026-09-19: the Groq on_demand tier (OTPM 1000) rejects single
# requests above ~1000 output tokens, hence the cap plus tile-splitting.
VISION_MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "1000"))
VISION_RENDER_MAX_DIM = int(os.environ.get("VISION_RENDER_MAX_DIM", "2048"))
GROQ_TIMEOUT_S = float(os.environ.get("GROQ_TIMEOUT_S", "60"))

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
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


class ImageError(ValueError):
    pass


class ExtractionError(RuntimeError):
    pass


class TruncationError(ExtractionError):
    """Model output hit the token budget; partial data is never returned silently."""


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
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageError(
            f"Unsupported image type '{p.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
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


def _vision_call(data_url: str, model: str) -> tuple[str, str]:
    """One Groq vision call -> (content, finish_reason), with transport retries."""
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
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_err = e
            wait = 10 * attempt if isinstance(e, RateLimitError) else 2 * attempt
            log.warning("Groq transient error (attempt %d): %s; waiting %ss",
                        attempt, type(e).__name__, wait)
            time.sleep(wait)
    else:
        raise ExtractionError(f"Groq API transport failure after retry: {last_err}")
    content = resp.choices[0].message.content or ""
    if not content.strip():
        raise ExtractionError("Model returned an empty response")
    if resp.choices[0].finish_reason == "length":
        raise TruncationError("Model output hit the token budget.")
    return content, resp.choices[0].finish_reason


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


def _merge_tile_questions(top: list[dict], bottom: list[dict]) -> list[dict]:
    """Merge two same-page tile extractions without losing boundary content.

    Identical repeats dedupe; straddling questions reassemble in reading
    order; options union order-preservingly so an option seen by either tile
    always survives. Text concatenates only for genuinely disjoint fragments.
    """
    merged: list[dict] = []
    seen: set[tuple] = set()
    by_number: dict[str, dict] = {}
    for q in list(top) + list(bottom):
        key = (q["question_number"].strip(), q["question_text"].strip()[:200],
               tuple(q["options"]))
        if key in seen:
            continue
        seen.add(key)
        prev = by_number.get(q["question_number"].strip())
        if prev is not None:
            for o in q["options"]:
                if o not in prev["options"]:
                    prev["options"].append(o)
            if prev["question_text"] != q["question_text"]:
                a, b = prev["question_text"], q["question_text"]
                prev["question_text"] = max(a, b, key=len) if (a in b or b in a) \
                    else f"{a} {b}".strip()
            if prev["marks"] is None:
                prev["marks"] = q["marks"]
            prev["confidence"] = min(prev["confidence"], q["confidence"])
            if not prev["section"] and q["section"]:
                prev["section"] = q["section"]
            prev["has_figure"] = prev["has_figure"] or q["has_figure"]
            continue
        by_number[q["question_number"].strip()] = q
        merged.append(q)
    return merged


def _extract_page(
    image: Image.Image, source_page: int, model: str
) -> tuple[list[dict], dict]:
    """Extract one page; tile-split with overlap only on token-budget overflow.

    A tile that itself overflows is split once more (depth 2 max); merges
    reuse the lossless overlap logic, so boundary options always survive.
    """
    try:
        return _extract_data_url(_pil_to_data_url(image), source_page, model)
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
        return _extract_tile_list(tiles, source_page, model, _depth=0)


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
    """Assemble a canonical Paper (same paper_id/fingerprint convention as PDF ingestion)."""
    h = hashlib.sha256()
    for b in source_bytes:
        h.update(b)
    fingerprint = h.hexdigest() if source_bytes else hashlib.sha256(b"").hexdigest()
    sections = list(dict.fromkeys(q.section for q in questions if q.section))
    return Paper(
        paper_id=f"pap_{fingerprint[:8]}",
        fingerprint=fingerprint,
        status=status,
        subject=metadata.get("subject") if metadata else None,
        class_name=metadata.get("class") if metadata else None,
        board=metadata.get("board") if metadata else None,
        questions=questions,
        total_questions=len(questions),
        total_marks=sum(q.marks or 0 for q in questions) or None,
        sections=sections,
    )


def extract_paper_from_image_bytes(data: bytes, source_page: int = 1,
                                   model: str | None = None) -> Paper:
    """Image bytes -> canonical Paper."""
    image = Image.open(io.BytesIO(data)).convert("RGB")
    items, metadata = _extract_page(
        image, source_page, model or EXTRACTION_MODEL
    )
    return assemble_paper(
        [to_canonical(q) for q in items], [data], metadata
    )

def extract_paper_from_image_file(path: str | Path, page_number: int = 1,
                                  model: str | None = None) -> Paper:
    """Image file (PNG/JPEG/WebP) -> canonical Paper."""
    raw = load_image_bytes(path)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    items, metadata = _extract_page(
        image, page_number, model or EXTRACTION_MODEL
    )
    for q in items:
        q["source_page"] = page_number
    return assemble_paper(
        [to_canonical(q) for q in items], [raw], metadata
    )

def extract_paper_from_images(paths: list[str | Path],
                              model: str | None = None) -> Paper:
    """Multiple image pages -> one canonical Paper, page numbers preserved."""
    model = model or EXTRACTION_MODEL
    merged: list[dict] = []
    seen: set[tuple] = set()
    sources: list[bytes] = []
    metadata: dict = {}
    for idx, p in enumerate(paths, start=1):
        raw = load_image_bytes(p)
        sources.append(raw)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        items, page_metadata = _extract_page(image, idx, model)
        if not metadata:
            metadata = page_metadata
        for q in items:
            q["source_page"] = idx
            key = (q["question_number"].strip(), q["question_text"].strip()[:200])
            if key in seen:
                continue
            seen.add(key)
            merged.append(q)
    return assemble_paper(
        [to_canonical(q) for q in merged], sources, metadata
    )

def to_langgraph_questions(paper: Paper) -> list[dict]:
    """Canonical questions as plain dicts for the solve graph / answer stage."""
    return [q.model_dump() for q in paper.questions]
