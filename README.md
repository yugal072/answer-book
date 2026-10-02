# answer-book

Turns a question paper into a worked solution book.

1. **Ingestion** - a question paper (PDF or a photo/scan) becomes a
   canonical `Paper` of `Question` objects, with sections, marks, options
   and metadata preserved.
2. **Solving** - each question is answered through the LangGraph solve
   pipeline.
3. **Output** - a solution book JSON per paper.

## Ingestion

The ingestion subsystem is deterministic for text PDFs and uses a vision
model only for images and scans. Both paths emit the same canonical model,
so the rest of the pipeline cannot tell them apart.

```
PDF   -> pypdf text layer -> structure-aware parser -> validation -> Paper
image -> Groq vision      -> tile/merge/dedupe      -> validation -> Paper
```

What it does, and where it refuses to guess, is documented in
**[docs/ingestion.md](docs/ingestion.md)**. The short version:

* question numbers, sections, marks, options, page numbers and chapter
  tags are extracted from the document, never invented;
* a numeric line only becomes a question when the layout says so, so
  wrapped text like `110 degrees.` never splits a question;
* marks are read per question and only fall back to a section's declared
  "each carries N marks" when the question states none of its own;
* MCQ options keep their labels and order, and options belonging to
  sub-questions stay in the wording instead of being re-attributed;
* every result is validated structurally and cross-checked against the
  paper's own header (total marks) and section instructions (question
  count). Findings are reported, not silently swallowed;
* a bad input, an unreadable PDF, a truncated model response or a
  structurally invalid result raises - it never returns an empty paper
  that looks successful.

### Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...        # only needed for the image path

python main.py data/papers/question_paper_455.pdf          # end to end
python -m app.ingestion.paper_loader data/papers          # ingest the corpus
```

The API exposes the same thing over HTTP:

```bash
curl -X POST "http://localhost:8000/api/v1/ingest?subject=&class_name=&board=" \
     -F "file=@data/papers/question_paper_455.pdf"
```

Unusable input returns `400`/`413`; a failed extraction or a result that
fails validation returns `422` with the reason.

### Tests

```bash
pip install -r requirements-dev.txt

pytest tests -m "not live" -q          # offline: real 20-paper corpus, no key
GROQ_API_KEY=... pytest tests -m live -q   # real vision calls, spend quota
```

The offline suite runs ingestion over the real papers in `data/papers`
and checks question counts, marks totals, sections, metadata, page
association, schema validity and that no source wording is lost.

## Layout

```
app/ingestion/   text_parser.py, paper_loader.py, image_loader.py,
                 validation.py, errors.py
app/models/      canonical Paper / Question
app/generate/    LangGraph solve pipeline
app/api/         FastAPI routes
data/papers/     the 20-paper test corpus
docs/ingestion.md
tests/
```
