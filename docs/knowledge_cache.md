# Knowledge Cache and Persistence

## Purpose
The database can reduce repeated solving work, but a cache hit must not be treated as proof that a stored answer is correct.

## Cache lookup
1. Normalize question text consistently.
2. Compute a stable signature from the question and relevant context.
3. Retrieve candidate records.
4. Check contextual compatibility.
5. Check stored verification and review status.
6. Reuse only candidates satisfying current policy.

Relevant context may include subject, class, board/curriculum, marks, options, language, and figure dependencies. Use only available, meaningful fields; do not guess missing metadata. A text-only signature may be insufficient for figure-dependent questions.

## Cache eligibility
A candidate should generally be ineligible for trusted reuse if verification failed or is missing, `needs_teacher_check` is true, confidence is missing/invalid/below threshold, context is incompatible, or the record is incomplete or came from a failed workflow. Confidence alone is insufficient.

## Conditional storage
Store a generated solution only when the configured acceptance policy is satisfied, such as successful structural validation, required verification checks passing, adequate confidence, no unresolved teacher-review flag, and sufficient provenance.

A useful but ineligible result may be retained in an audit/evaluation store if policy permits, but it must not be presented as a trusted cache entry.

## Data integrity
Use conflict-safe handling for cache keys, transactions for related writes, explicit PostgreSQL migrations, appropriate indexes, and a policy for stale/superseded answers. Do not silently fall back from failed PostgreSQL connections to an in-memory database in production. ORM `create_all()` does not upgrade existing tables.

## Audit metadata
Where practical, record provider/model ID, prompt and workflow version, verification outcome/checks, confidence and its method, timestamps, attempt count, cache decision, and human-review outcome. Never persist credentials or sensitive provider headers.
