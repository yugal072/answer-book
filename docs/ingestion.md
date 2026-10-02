# Ingestion

How a question paper becomes a canonical `Paper`. This document describes
what the code actually does, including where it refuses to guess.

* `app/ingestion/text_parser.py` - the deterministic PDF text parser
* `app/ingestion/paper_loader.py` - PDF I/O, header facts, validate-or-raise
* `app/ingestion/image_loader.py` - the vision path (Groq)
* `app/ingestion/validation.py` - structural post-extraction validation
* `app/ingestion/errors.py` - the exception hierarchy
* `app/models/loaders_models.py` - the canonical `Paper` / `Question`

---

## 1. The two paths, one output

Both paths produce the same canonical model, so downstream code never
needs to know where a paper came from:

```
PDF  -> pypdf text layer -> text_parser -> validate -> Paper
PNG/JPEG/WEBP -> Groq vision -> merge/dedupe -> validate -> Paper
```

The PDF path is fully local, deterministic and fast. The vision path is
used only for images and scans, where there is no text layer to parse.

## 2. PDF path

### 2.1 Reading

`extract_pdf_pages()` returns the text layer as pages of lines, with
running furniture removed, and fails loudly on unusable input:

| Input | Result |
| --- | --- |
| missing file / directory | `DocumentError` |
| empty file / corrupt PDF | `DocumentError` |
| PDF with no usable text layer (a scan) | `DocumentError` naming the image path as the alternative |

A scan never returns an empty `Paper` dressed up as a success.

### 2.2 Removing page furniture

Repeated headers, footers, table headers and watermarks are dropped. A
line is only treated as furniture when it is **all** of:

* repeated on at least 60% of the pages (real content appears once),
* short (≤ 60 characters),
* in the outer band of the page (first/last 3 lines),
* not a bare number (a marks slot must survive) and not a question start,
* not a complete sentence (running heads are labels, not wording).

The two shapes that are always furniture - `Page 3/9` and `Q.NO. QUESTIONS
MARKS` - are removed regardless of repetition. No brand or site name is
hardcoded anywhere.

### 2.3 Finding question boundaries

Two strategies, in order:

1. **Strict (sequential).** A line opens a question when its leading
   number is the next expected one. This is the right rule for the vast
   majority of papers and the one the corpus uses.
2. **Relaxed (monotonic).** If the strict pass rejected numeric lines that
   genuinely look like question starts, the whole document is re-parsed
   accepting any strictly increasing, never-seen number, provided the
   previous block looks finished (ends in punctuation, a chapter tag or a
   marks line). The relaxed result is kept **only** if it yields strictly
   more questions with unique numbers.

The relaxed pass exists so that a paper which starts at "7", skips a
number, or restarts its numbering is not silently mangled. It is guarded
against the classic false positive: a wrapped line that begins with a
measurement (`110 degrees.`, `5 kg`, `75 percent of ...`) is never
promoted to a question, because the body must read as prose and must not
open with a quantity or a unit.

The preamble - title block, subject line, "General Instructions" and its
numbered list - is not question content. A heading-only match is required
(`Instructions`, `General Instructions:`, `Note:`) and it is only honoured
before the first question, so a wrapped sentence that happens to end in
the word "instructions." stays in the question it belongs to.

### 2.4 Sections

Headings are recognised as `Section B`, `Section B - Literature (Very
Short Answer)`, `SECTION C: Grammar`, `Part A` and similar. The label must
already be upper-case/roman/digits in the source, so prose such as "Part
of the whole equals ..." is not turned into a section.

* `Question.section` keeps the short label (`"B"`) - the value downstream
  code and `Paper.sections` use.
* `Question.section_title` keeps the full heading
  (`"Literature (Very Short Answer)"`).
* A section is never lost across a page break.

### 2.5 Marks

Marks are read per question, in this order:

1. a standalone integer line inside the question block (`2`),
2. a number printed after sentence punctuation on the stem or on the last
   line of the block (`... resolve the conflict? 1`),
3. only if the question states none of its own: the section's explicit
   "Each carries N mark(s)" instruction.

Consequences that matter:

* A question ending in a number is **not** read as marks: "A train travels
  5." keeps `marks = None`, because the marks patterns only fire directly
  after sentence punctuation.
* A section's uniform marks are never assumed. Marks are only inherited
  from the section when the paper itself declares them.
* If marks cannot be determined, they stay `None`.
* `Paper.total_marks` is the sum only when **every** question has known
  marks. A partial sum would understate the paper, so it is reported as
  unknown instead.

### 2.6 MCQ options

Options are lifted out of the question's wording only when they form a
run attached to the stem:

* a run starts at label `(A)` - an MCQ always begins at A, which is what
  keeps a sub-part written as `(a) ...` from being mistaken for options;
* labels and order are preserved, several options on one line included;
* a following line extends the previous option only when that option was
  clearly cut mid-sentence (a wrapped option);
* options belonging to **sub-questions** stay in the text, where they
  cannot be mistaken for the stem's own options. A passage question with
  options under `(i)` therefore keeps `options = None` and reads as a
  long question, not an MCQ with twelve options.

### 2.7 Internal choice, sub-parts, chapters, figures

* A standalone `OR` line marks an internal alternative: the alternative
  text is kept in `Question.text` and `choice_group` is set to the
  question number. A lowercase "or" in prose does not.
* Sub-parts `(i)`, `(ii)`, `(a)` and bullets stay inside their parent
  question's text.
* Chapter tags (`[Ch 4: Exploring Algebraic Identities]`) are captured in
  `Question.chapter` - which the solve graph already reads - and never
  appear in `Question.text`. When an OR alternative carries its own tag,
  all tags are kept.
* `has_figure` is set only when the wording actually references a figure
  ("Figure 3", "the diagram below", "as shown in the figure"). The mere
  word "figure" in unrelated prose does not set it.

### 2.8 Header facts

`extract_paper_metadata()` returns the canonical three keys `subject`,
`class`, `board` - the same fields the image path fills, so both paths
produce equivalent metadata. `extract_header_facts()` additionally returns
`total_marks` and `time_allowed` when the header states them; the declared
total is used to cross-check the extracted marks.

The board is only read from the header line that also carries the other
header fields, so a capitalised word in question text can never become a
board name. Nothing is invented: a paper with no subject line yields
`subject = None`.

## 3. Image (vision) path

`extract_paper_from_image_file` / `_bytes` / `_images` accept PNG, JPEG
and WEBP. Input is validated before anything is sent to the model
(missing, empty, oversized, corrupt, lying extension, invalid page
number), and failures raise `ImageError`.

Per page:

1. `_prepare()` normalises the image - EXIF orientation, downscale to
   `VISION_RENDER_MAX_DIM`, gentle contrast, autocontrast, mild
   unsharpening for phone photos. It never upscales or binarises.
2. One vision call extracts questions, marks, options, section, type,
   figure flag, choice group and header metadata.
3. If the model stops at the token budget, the page is re-read as two
   overlapping tiles split at a whitespace row; a tile that still
   overflows is split again (depth capped at 2). Beyond that the error is
   raised - **truncated model output is never returned**.
4. Tiles are merged: identical repeats are dropped, a question straddling
   the boundary is reassembled in reading order, and options are unioned
   by label so a boundary option always survives exactly once.
5. A second dedupe pass removes a question that repeats verbatim under a
   different number (`1(i)` and `i`) - the failure mode overlapping tiles
   really do produce. It is scoped to a single page, so a question
   repeated on two different pages is kept.
6. The page reported for each question is the page that was actually
   read, never a page number the model claimed.

Multi-image papers keep their page numbers, merge metadata across pages
(first non-empty value per key) and de-duplicate questions repeated
across overlapping photos.

`confidence` is a model self-report. It is never copied into the
canonical model (which has no such field) and never invented; a low value
is surfaced as a validation warning instead.

## 4. Validation

`validate_paper()` runs on every ingestion, on both paths, and reports
findings rather than rewriting data. Errors fail the ingestion; warnings
are logged and returned to the caller.

Errors (the result is not trustworthy): no questions at all; duplicate
question numbers (PDF path); identical wording under two numbers (PDF
path); numbers out of order; empty question text; a page outside the
document; marks outside `0..100`; a non-canonical question type; an empty
option string; a section that is not among the declared ones; a
`total_questions` that disagrees with the list.

Warnings (usable, but worth knowing): numbering gaps; numbering that
starts above 1; marks that could not be determined; an extracted marks
total that disagrees with the paper's own header; a question count that
disagrees with the section instructions; duplicate wording or duplicate
numbers on the image path; low vision confidence; `has_figure` without a
figure reference; a page with no questions.

Callers that want the findings:

```python
findings = []
paper = ingest_paper(path, diagnostics=findings)
```

The report compares the extraction against what the document itself
declared, which is an independent check: the marks in the header, and the
"Attempt N questions" section instructions. A mismatch is a warning, not
a silent pass.

## 5. Errors

```
IngestionError
├── DocumentError      unusable input (missing, empty, corrupt, no text layer)
├── ExtractionError    extraction ran but produced nothing trustworthy
│   └── TruncationError  model hit the token budget (never returned as data)
└── ValidationError    structurally invalid result; carries the full report
```

`ImageError` is a `DocumentError` and also a `ValueError`. A daily Groq
token/request limit is not retried - waiting cannot help - and comes back
as an `ExtractionError` with the provider's message.

## 6. Tests

```
pytest tests -m "not live" -q     # offline suite, no key, no network
GROQ_API_KEY=... pytest tests -m live -q
```

The offline suite runs the **real** 20-paper corpus in
`data/papers`: question counts and marks are checked against each paper's
own header and section instructions, every question is validated, and a
losslessness check asserts that no source line is dropped. The live suite
feeds real rendered pages to the real vision model and compares the
result with the PDF path.

## 7. Known limitations

* A PDF with no text layer is not OCR'd; render it to images and use the
  vision path. There is no local OCR fallback.
* The vision path may return sub-parts as their own questions numbered
  `1(i)`, while the PDF path keeps sub-parts inside the parent question's
  text. The canonical model has no sub-question field; both readings are
  lossless, but the two paths may segment a passage question differently.
* The vision output budget is capped by the provider tier
  (`VISION_MAX_TOKENS`, default 1000). Very dense pages are re-read as
  tiles; a page that still will not fit raises `TruncationError` rather
  than returning partial data.
* Figure detection is textual. The PDF path does not inspect embedded
  images, because a page image cannot be attributed to a specific
  question.
* Wrapped lines are joined with single spaces, so a double space inside a
  line is normalised. No other whitespace or wording is changed.
* Question types are heuristics (`mcq` when options are present, `long`
  for case studies and 5+ mark questions, `numerical` for
  answer-seeking wording, otherwise `short`).
