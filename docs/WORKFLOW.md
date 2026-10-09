# Workflow architecture

How a `Paper` becomes a `SolvedPaper`. This describes what the code does today,
including its limits. Code: `app/workflow/`. Generated graph drawings:
**[workflow_graph.md](workflow_graph.md)** (`python -m app.workflow.visualize`).

```
FastAPI/CLI -> Ingestion (OCR+layout) -> Paper -> Graph Adapter -> GraphState
   paper graph:  validate_prepare -> fan-out (Send) -> question_pipeline x N -> aggregate_results
                 -> build_solved_paper -> assemble_output  --> JSON | SSE | PDF
   question_pipeline:  cache_lookup -> accepted? -> retrieve_stored ---------------------> finalize
                                   '-> solve_<type> -> verify -> passes? -> assess_confidence
                                                        '-> repair -> verify (bounded)     |
                                                    verified? -> store_verified -> finalize
```

## Diagram -> implementation

| Diagram stage | Implementation |
| --- | --- |
| FastAPI | `app/api/v1/solver.py` (`/process-paper`, `/solve-question`, `/solutions/{id}`, `/runs/{id}/resume`) |
| Ingestion: OCR + Layout -> Paper | `app/ingestion/*` (unchanged; upload validation added in `api/v1/ingestion.py`) |
| Graph Adapter (validate, normalize, `WorkItem`) | `app/workflow/adapter.py` |
| GraphState | `PaperState`, `QuestionState` in `schemas.py` (plain, serialisable dicts) |
| Validate / Prepare | `paper_graph.validate_prepare` |
| Fan-out Questions 1..N | conditional edge returning `Send("question_pipeline", ...)` per `WorkItem` |
| Knowledge Cache Lookup + acceptable match? | `question_graph.cache_lookup` + `cache_policy.py`; conditional edge `after_cache` |
| Retrieve Stored Solution | `question_graph.retrieve_stored` (re-validates the payload; falls back to the solver if invalid) |
| Solver Node (type-specific strategy + prompt) | conditional edge routes to `solve_mcq` / `solve_numerical` / `solve_short` / `solve_long`; prompts in `prompts.py`; calls via `llm.py` |
| Draft Solution | `SolverDraft` (typed per question type: `McqDraft`, `NumericalDraft`, `TextDraft`) |
| Verifier Node | `verify` -> `verification.Verifier` |
| Verification passes? | conditional edge `verification_passes` |
| Repair / Retry | `repair` -> back to `verify`; bounded by `max_repair_attempts` |
| Assess Confidence | `assess_confidence` -> `confidence.assess` |
| Verified Q+A -> PostgreSQL | `store_verified` -> `kb.SqlKnowledgeBase` (table `verified_solutions`) |
| Finalize Question | `finalize_question` (cache hits and fresh solves converge here) |
| Aggregate Results / Build SolvedPaper | `aggregate_results`, `build_solved_paper` |
| Assembly / Serving: JSON, SSE, PDF | `assemble_output` persists status + JSON export; `render.py` builds the PDF and SSE frames |

Deliberate differences from the drawing:
* The graph adapter runs *before* the graph (it builds the input state); `validate_prepare` is the first graph node.
* "Retrieve Stored Solution" has a fallback edge to the solver, which the drawing omits: a stored payload that fails re-validation must not be trusted.
* A repair that cannot run (provider failure) goes straight to `assess_confidence` with the previous draft, rather than looping.
* PDF/SSE are rendered by the serving layer from the finished `SolvedPaper`; `assemble_output` does persistence and the JSON export.

## Question outcomes (`status`)

| status | meaning |
| --- | --- |
| `verified` | freshly generated; verification passed **and** every gate in "Confidence" below was met. Stored in the knowledge base. |
| `cached` | an earlier *verified* result was accepted by the cache policy. No model call. |
| `unverified` | an answer exists but did not meet the verified criteria (failed verification after the retry limit, verifier unavailable, low confidence, marks unknown). **Teacher review required.** The draft and all diagnostics are kept. Never stored. |
| `failed` | no usable answer (provider outage, invalid question, internal error in that branch). |

Paper `outcome`: `complete` (all verified/cached) -> DB status `ready`; `needs_review` (some unverified); `partial` (some failed, some answered); `failed` (nothing answered). The `papers.status` column is `ready` **only** for `complete`.
Reaching the retry limit never turns a failed verification into a success.

## Verification semantics

Verification is a different task from solving (`verification.py`):

* **Deterministic (all types):** answer/steps present, mark-split entries positive and summing to the question's marks, plus type-specific checks: MCQ option letter exists and the answer text does not name a different letter; numerical `final_value` is finite and the solver's `arithmetic_expression` is re-evaluated by a whitelisted AST evaluator (`safe_math.py`, no `eval`, no model code ever runs) and must match.
* **Independent solve (MCQ, numerical):** a *different model family* (`GROQ_VERIFIER_MODEL`, default `openai/gpt-oss-20b`) answers the question blind (it never sees the draft) and the answers are compared.
* **Rubric review (short, long):** the verifier model reviews the draft against fixed criteria (`answers_the_question`, `factually_correct`, `complete_for_marks`, `mark_split_reasonable`). There is no ground truth for prose, so this is labelled `llm_review`, never "proven".
* Deterministic failures skip the (costly) model checks and go straight to repair with precise feedback.
* If the verifier itself cannot run the state is `inconclusive`: nothing is known, repair is not attempted, the answer is `unverified`.
* An MCQ whose first answer lost the cross-check is repaired, but the later "agreement" is not independent (the feedback told the solver it was wrong), so it cannot count as `cross_check` and the question ends `unverified`.

Evidence labels (`verified_by`): `none` < `structural` < `llm_review` < `cross_check` < `symbolic`.
`symbolic` means: independent solve agreed **and** the solver's own arithmetic expression re-evaluated to its stated value. It does not prove the expression models the question - that is what the independent solve covers. Nothing is ever called "proven" or "retrieved" unless that actually happened.

## Confidence and the verified-storage gate

`self_confidence` (the model's claim) is an input that can only lower the result. `final_confidence = min(self_confidence, cap[evidence])`, and `<= 0.40` if verification did not pass.

A result is `verified` (and stored) only if **all** hold (`confidence.py`, thresholds in `settings.py`, env-overridable):

| gate | default | env |
| --- | --- | --- |
| verification state is `passed` | - | - |
| evidence >= minimum for the type (mcq/numerical `cross_check`, short/long `llm_review`) | | |
| accuracy (share of checks passed, warnings weigh 0.5) | >= 0.80 | `WORKFLOW_ACCURACY_MIN` |
| final confidence | >= 0.75 | `WORKFLOW_STORE_MIN_CONFIDENCE` |
| marks stated in the source | required | - |
| evidence caps | none .30, structural .50, llm_review .80, cross_check .90, symbolic .95 | settings |
| unverified ceiling | 0.40 | `WORKFLOW_UNVERIFIED_CEILING` |
| repair attempts | 2 | `WORKFLOW_MAX_REPAIR_ATTEMPTS` |

Figure-dependent questions (`has_figure`) are answered and verified but never stored or reused: the text alone is not the whole question.

## Cache acceptance (`cache_policy.py`)

A database row is a candidate, not proof. It is reused only if all 12 checks pass: status `verified`; same schema version; accepted prompt version; no figure; current marks known; identical normalized text; same marks; same board, class and subject; same type; identical options in identical order; stored evidence/confidence meet the gates; and the stored solution **re-passes the deterministic checks against the current question**. Every rejection records its reason in the result (`cache.reasons`). The best accepted candidate is chosen by evidence, confidence, recency, then id.

`verified_solutions` has one row per `context_key` (signature + subject + type + options + figure flag + schema version). Writes are upserts inside a transaction with a savepoint, so repeated runs and concurrent writers cannot duplicate rows, and a weaker result never replaces a stronger one. Bump `PROMPT_VERSION` (settings.py) when prompts change materially; set `WORKFLOW_ACCEPTED_PROMPT_VERSIONS` to keep trusting older ones.

Existing tables are untouched. The knowledge base is a new additive table (`Base.metadata.create_all` / `python -m app.store.init_db`). Per-paper checkpoint rows continue in `solved_questions`; unverified answers are checkpointed there with `needs_teacher_check = true`, failed questions are not (a re-run retries them).

## Parallelism, state and failures

* Questions run in parallel via `Send` (`WORKFLOW_MAX_CONCURRENCY`, default 4). The only key shared between the paper graph and a question branch is `results`, merged **by `item_id`** (idempotent), and the final order is restored from the question index - never from arrival order.
* Every question node is wrapped so an unexpected exception becomes a diagnostic and a `failed` result **for that question only**.
* Graph state holds only plain data. API keys, DB sessions and clients live in closures (`deps.py`); tests assert the key never appears in results or checkpoints.
* Database failures are surfaced as diagnostics (`cache_unavailable`, `storage_failed`, `checkpoint_failed`, `persistence_failed`), never turned into success.

## Checkpointing and resume

Execution state (LangGraph checkpoints, `thread_id = run_id`) is separate from the knowledge base. With `DATABASE_URL` set to PostgreSQL the checkpointer is `PostgresSaver`; otherwise it is process-local `InMemorySaver`. Resume continues an interrupted run without redoing finished questions: `python -m app.workflow.cli --resume RUN_ID` or `POST /api/v1/runs/{run_id}/resume`. Without `DATABASE_URL` the knowledge base and checkpoints are also process-local (a warning is logged at start-up).

## LLM layer (`llm.py`)

Graph nodes see only the `LLMGateway` protocol. The production gateway is `prompt | ChatGroq.with_structured_output(<pydantic schema>, method="json_schema")`. Question text is untrusted: it is only ever passed as a template variable, wrapped in `<question>` tags with any closing tag neutralised; models have no tools and are never asked to run code. Only transient failures (HTTP 429/5xx, timeouts, connection errors) are retried, honouring `retry-after` / "try again in"; auth and bad-request errors are not; schema-invalid output gets `WORKFLOW_LLM_OUTPUT_RETRIES` (1) further attempt and then fails loudly. A sliding-window **token-budget limiter** (`WORKFLOW_TOKENS_PER_MINUTE`, default 6500 against this key's 8000 TPM) is shared by all threads and backs off together after a 429. Models: solver `GROQ_MODEL` (default `qwen/qwen3.8-27b`), verifier `GROQ_VERIFIER_MODEL` (default `openai/gpt-oss-20b`).

## HTTP API

| endpoint | notes |
| --- | --- |
| `POST /api/v1/process-paper?format=json\|pdf\|sse&limit=&subject=&class_name=&board=` | multipart `file` (PDF/PNG/JPEG/WEBP, <= 25 MB, extension and magic bytes checked before any model call). 200 JSON/PDF/stream; `502` + body when nothing could be answered; `503` no `GROQ_API_KEY`; `413/415/400/422` bad input. |
| `POST /api/v1/solve-question` | `{"question": {...}, "metadata": {...}}`; same graph and cache; no paper export. |
| `GET /api/v1/solutions/{paper_id}?format=json\|pdf` | stored book; PDF only for books produced by this workflow (`422` for legacy files). |
| `POST /api/v1/runs/{run_id}/resume?format=json\|sse` | continue from the checkpoint. |

JSON body: `SolvedPaper` (`schema_version`, `paper_id`, `status`, `outcome`, `metadata`, `summary`, `solutions` (contract-compatible list, in order, + `status`/`origin`), `results` (full per-question detail: attempts, verification, confidence assessment, cache decision, diagnostics), `diagnostics`).

SSE frames: `id: <n>` / `event: <name>` / `data: <json>`. Events: `run.started`, `run.prepared`, `question.stage` (`cache_lookup`, `cache_hit`, `solving`, `verifying`, `repairing`, `assessing`, `storing`), `question.completed` (`status`, `origin`, `verified_by`, `confidence`), `paper.completed`, then exactly one terminal `run.finished` (with `outcome`, `summary`, `result_url`) or `run.error`. Every event carries `run_id`. No prompts, solution bodies or configuration are streamed.

```bash
curl -N -X POST "localhost:8000/api/v1/process-paper?format=sse&limit=3" -F file=@test_papers/question_paper_455.pdf
curl -o book.pdf -X POST "localhost:8000/api/v1/process-paper?format=pdf&limit=3" -F file=@test_papers/question_paper_455.pdf
python -m app.workflow.cli test_papers/question_paper_455.pdf --limit 3 --format pdf --out book.pdf
```

## Running the tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests -m "not live and not db" -q                 # offline (what CI runs)
TEST_DATABASE_URL=postgresql://u:p@host/ab_test pytest tests/workflow/test_persistence.py -q   # real PostgreSQL (db name must contain "test")
GROQ_API_KEY=... pytest tests/workflow/test_live_groq.py -m live -q                      # real Groq, spends quota
```

## Known limitations

* Verification of prose answers is a second model's review, not proof. A shared blind spot between the two model families is possible.
* Questions that depend on a figure are answered from text only and are never cached; the solver may say information is missing.
* `symbolic` covers arithmetic only; there is no symbolic algebra/geometry prover.
* Resume across processes needs the PostgreSQL checkpointer; without `DATABASE_URL` it is in-process only.
* Groq free-tier limits (8,000 tokens/min on the key used in development) make a full 37-question paper take minutes.
* The older Gemini prototype (`app/answer_generation`, `app/graph/solve`, `app/verification`) and the Groq prototype (`app/generate`, `run_prototype.py`) are left in place unmodified; the API and CLI no longer call them.
