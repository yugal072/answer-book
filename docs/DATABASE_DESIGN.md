# Database Architecture & Question Reuse Design

This document details the database architecture, system design decisions, schema specifications, and edge-case recovery protocols for the **Paper to Solution Book** engine.

---

## 1. System Design Decisions & Technical Rationale

### Decision 1: Database Engine — PostgreSQL with SQLAlchemy
* **Decision:** We use **PostgreSQL** as the core database engine via **SQLAlchemy**, configured via `DATABASE_URL` in `.env`.
* **Local vs Cloud Flexibility:**
  * For local execution: connects to local native PostgreSQL (`localhost:5432`).
  * For team collaboration: connects to a shared cloud PostgreSQL instance (e.g. Neon.tech / Supabase) with zero code changes.
  * SQLAlchemy ensures robust connection pooling, automatic reconnection with `pool_pre_ping=True`, and native `JSONB` support.

### Decision 2: Layered Separation — Infrastructure (`app/store/`) vs Agent Tools (`app/tools/`)
* **Decision:** We strictly decouple database infrastructure from agent/pipeline tools:
  * **`app/store/` (Infrastructure Layer):** Engine connection (`connection.py`), table definitions (`models.py`), and table initialization (`init_db.py`). Zero business logic.
  * **`app/tools/` (Functional Tools Layer):** High-level tools called by LangGraph agents and `main.py`:
    - `hash_utils.py`: Text normalization and composite SHA-256 signatures.
    - `paper_cache_tool.py`: Recommendation A paper-level fingerprint lookup.
    - `question_lookup_tool.py`: Question signature lookup tool used by `planner_node`.
    - `question_save_tool.py`: Immediate per-question checkpoint committer for crash-resilience.

### Decision 3: Why Vector DB is Set Aside for Direct Answer Reuse
* **Decision:** We do **not** use a Vector Database (embeddings / cosine similarity) for 100% answer substitution.
* **The "Vector Trap" in Exam Questions:**
  * **Math Numbers Problem:** Two completely different math problems (e.g., $x^2 - 5x + 6 = 0$ vs $x^2 - 7x + 12 = 0$) share identical sentence structures and vocabulary. Embedding models give them a **96%+ cosine similarity score**. Relying on vector similarity could paste the answer of problem B into problem A.
  * **Marks & Depth Blindness:** Vector embeddings do not account for marks. A 2-mark question and a 5-mark question with the same text produce identical embeddings, but require drastically different solution depths.
* **The Solution:** We use **Deterministic Hashing (`sha256`)** for exact question-level caching. Vector search is reserved only for semantic textbook retrieval (RAG) in future phases.

### Decision 3: Separation of Storage vs. Indexing
* **Decision:** Large solution payloads (steps, mark splits, common mistakes) are stored in the relational database, not bloated into vector indices.
* **Benefit:** Keeps search indices ultra-compact in RAM while maintaining the relational database as the single source of truth.

---

## 2. Database Schema Design

The database consists of **two tables** linked in a Parent-Child relationship:
1. **`papers`**: Stores paper-level metadata, the SHA-256 fingerprint, and overall execution status.
2. **`solved_questions`**: Stores individual question data, the composite question signature, and the verified `SolutionContract` adhering strictly to Section 3 of the project brief.

```
┌─────────────────────────────────┐
│             papers              │
├─────────────────────────────────┤
│ paper_id (PK)                   │
│ fingerprint (UNIQUE INDEX)      │
│ status                          │
│ subject, class_name, board      │
│ total_questions, total_marks    │
│ sections, source_file           │
│ created_at, updated_at          │
└───────────────┬─────────────────┘
                │ 1 Paper has Many Questions
                ▼
┌─────────────────────────────────┐
│        solved_questions         │
├─────────────────────────────────┤
│ id (PK)                         │
│ paper_id (FK -> papers)         │
│ question_signature (INDEXED)    │
│ question_number, section        │
│ question_text, marks, type      │
│ options, has_figure             │
│ answer, steps, mark_split       │
│ common_mistakes, confidence     │
│ verified_by, needs_teacher_check│
└─────────────────────────────────┘
```

---

### Table 1: `papers` (Paper-Level Metadata)

| Column Name | Type | Constraints / Description |
| :--- | :--- | :--- |
| `paper_id` | `VARCHAR(64)` | **PRIMARY KEY**. e.g., `"pap_67c4fef7"` |
| `fingerprint` | `VARCHAR(64)` | **UNIQUE INDEX**. SHA-256 hash of raw input file bytes. |
| `status` | `VARCHAR(20)` | Execution status: `'parsing'`, `'solving'`, `'ready'`, or `'failed'`. |
| `subject` | `VARCHAR(100)` | Subject name (e.g., `"Mathematics"`, `"Science"`). |
| `class_name` | `VARCHAR(50)` | Grade/Class (e.g., `"Class 9"`). |
| `board` | `VARCHAR(50)` | Educational board (e.g., `"CBSE"`). |
| `total_questions`| `INTEGER` | Total number of questions in paper. |
| `total_marks` | `INTEGER` | Sum of maximum marks for the paper. |
| `sections` | `JSON` | List of section names: `["A", "B", "C", "D", "E"]`. |
| `source_file` | `VARCHAR(255)`| Path to input file (e.g., `"test_papers/question_paper_455.pdf"`). |
| `created_at` | `TIMESTAMP` | Auto-set to current timestamp on upload. |
| `updated_at` | `TIMESTAMP` | Updated every time a question or status updates. |

---

### Table 2: `solved_questions` (Question-Level Store & Cache)

| Column Name | Type | Constraints / Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` | **PRIMARY KEY** (Auto-incrementing integer). |
| `paper_id` | `VARCHAR(64)` | **FOREIGN KEY** $\to$ `papers(paper_id)` ON DELETE CASCADE. |
| `question_signature`| `VARCHAR(64)`| **INDEXED**. `sha256(normalized_text + marks + board + class_name)`. |
| `question_number` | `VARCHAR(20)` | Question identifier in exam (e.g., `"1"`, `"3(b)"`). |
| `section` | `VARCHAR(10)` | Section identifier (e.g., `"A"`, `"B"`). |
| `question_text` | `TEXT` | Full prompt text of the question. |
| `marks` | `INTEGER` | Marks allotted for this question. |
| `question_type` | `VARCHAR(20)` | Category: `'mcq'`, `'numerical'`, `'short'`, `'long'`. |
| `options` | `JSON` | List of choices if MCQ, otherwise `NULL`. |
| `has_figure` | `BOOLEAN` | `True` if diagram/figure is present, otherwise `False`. |
| `answer` | `TEXT` | Final concise answer string. |
| `steps` | `JSON` | List of step-by-step markdown derivations. |
| `mark_split` | `JSON` | List of mark allocations: `[{"marks": 1, "for": "Formula"}, ...]`. |
| `common_mistakes`| `JSON` | List of misconceptions: `[{"wrong": "...", "why": "..."}]`. |
| `confidence` | `FLOAT` | Confidence score between `0.0` and `1.0`. |
| `verified_by` | `VARCHAR(100)`| Solver/model identifier (`"groq:qwen/qwen3.8-27b"`). |
| `needs_teacher_check`|`BOOLEAN` | Safety flag: `True` if low confidence or ambiguous. |

---

## 3. Edge Cases & Solutions (Q&A)

### Q1: What if the exact same question appears in two papers with different marks (e.g., 2 marks in Paper 1 vs. 5 marks in Paper 2)?
* **The Problem:** A 2-mark answer only gives a brief definition, while a 5-mark answer requires full derivations and examples. Copying a 2-mark answer into a 5-mark question causes the student to lose marks and breaks the Section 3 rule where `mark_split` must sum to the question's total marks.
* **The Solution:** The question lookup uses a **Composite Question Signature**:
  $$\text{Signature} = \text{SHA-256}(\text{normalized\_text} + \mathbf{marks} + \text{board} + \text{class\_name})$$
  Because `marks` is part of the hash, a 2-mark question produces a completely different signature from a 5-mark question. The 5-mark question will register a **Cache Miss** and be routed to LangGraph to generate the full 5-mark depth.

---

### Q2: What happens if the system crashes midway (e.g., solves 20 of 38 questions and stops)? Will we lose the previously generated solutions?
* **The Problem:** If solutions are kept only in Python memory (`solved_solutions.append()`) until the end, a mid-run crash (power loss, terminal closed, rate limit error) loses all 20 solved questions.
* **The Solution: Immediate Per-Question Checkpointing.**
  * As soon as Question $i$ finishes solving and verifying, it is **immediately committed to the `solved_questions` table in SQLite**.
  * The `updated_at` column in `papers` is touched.
  * If a crash happens on Question 21, Questions 1 through 20 are **already permanently saved to disk**.

---

### Q3: What happens to the `status` column in the database during and after a crash?
* **During execution:** When the paper starts solving, `papers.status` is set to `'solving'`.
* **If a managed crash occurs (caught exception):** The code catches the error and marks `status = 'failed'` with an error log.
* **If a hard crash occurs (power loss / process kill):** The database row remains with `status = 'solving'` and `updated_at` frozen at the exact second the last question finished.

---

### Q4: What happens if a user re-uploads a paper whose execution previously stopped midway?
* **The Problem:** The user uploads the same PDF because they saw an error or timeout. We do not want to restart from Question 1 and waste tokens/time.
* **The Solution: The Smart Recovery Protocol.**
  1. The system computes the SHA-256 `fingerprint` and checks the `papers` table.
  2. If `status == 'ready'`: Returns the complete solution book immediately (0.01 seconds).
  3. If `status in ('solving', 'failed')`:
     * The system queries: `SELECT question_number FROM solved_questions WHERE paper_id = :id`.
     * It discovers Questions 1–20 are already solved!
     * It loads Questions 1–20 directly from the database into memory.
     * It logs: `[RESUMING] Loaded Q1-Q20 from database. Resuming solve from Q21!`
     * The LLM is called **only for Questions 21 to 38**.
     * When Question 38 finishes, `status` flips to `'ready'`.

---

### Q5: How do we prevent two workers from processing the same paper at the same time (Zombie / Duplicate Jobs)?
* **The Problem:** A user rapidly double-clicks the "Upload" button, causing two background workers to attempt solving Question 21 simultaneously.
* **The Solution: The 5-Minute Heartbeat Rule.**
  * When a paper is checked and `status == 'solving'`:
    * If `updated_at` was **less than 2 minutes ago**, an active worker is currently processing it. The system responds: *"Processing in progress, please wait."*
    * If `updated_at` was **more than 5 minutes ago**, the previous worker is dead (zombie job). The new worker takes ownership and safely resumes execution.

---

## 4. Execution Flow Summary

```
Upload Paper (PDF / Image)
           │
           ▼
[Step 1] Ingestion: Extract questions + calculate SHA-256 paper fingerprint
           │
           ▼
[Step 2] Check `papers` table by fingerprint:
           ├─► status == 'ready'   ──► Return complete Solution Book (Instant 1ms)
           └─► status != 'ready'   ──► Check `solved_questions` for already solved questions:
                                       • Load existing questions (skip LLM)
                                       • Resume solving remaining questions
                                       │
                                       ▼
[Step 3] For each remaining question:
           ├─► Check `solved_questions` by composite signature:
           │   `sha256(text + marks + board + class_name)`
           │   • Hit  ──► Reuse solution instantly
           │   • Miss ──► Run LangGraph Solver
           │
           └─► Immediately commit newly solved question to DB
                                       │
                                       ▼
[Step 4] All questions done ──────► Flip status = 'ready'
                                    Save complete JSON to `solutionPapers/`
```
