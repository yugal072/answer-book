# Per-Question Solving Workflow

## Goal
Implement a bounded, auditable workflow for solving each question independently while allowing the parent graph to fan out across questions.

## Intended stages

### 1. Prepare the work item
Validate required fields and attach paper context, including subject, class, board, section, chapter, marks, options, and figure references where available. Preserve the original question index and stable work-item ID.

### 2. Inspect requirements and plan
Assess whether visual input is needed, approximate complexity, reasoning depth, useful safe tools, output contract, mark allocation, eligible model tier, and prompt version. The planner selects capabilities and model configuration; it must not rely only on the `type` field.

### 3. Check the knowledge cache
Compute a normalized signature and retrieve possible matches. A signature match is only a retrieval signal. Compare context and enforce eligibility before reuse. Cache failures must not be mistaken for cache hits.

### 4. Generate a candidate
Build the prompt from a versioned template and provide only necessary context and tools. Request the agreed structured solution contract. Capture provider, model, prompt version, attempt number, and usage metadata when available.

### 5. Validate the candidate
Run deterministic checks first. Reject malformed data, invalid types, impossible mark splits, and missing required fields. Do not silently normalize malformed output into apparent success.

### 6. Verify independently
Use checks appropriate to the question: arithmetic consistency, MCQ option consistency, relevance/completeness, mark-split consistency, reasoning/answer consistency, and visual consistency when the actual figure is available. A different model/provider can help but is not proof of correctness.

### 7. Repair or retry
For repairable issues, send the candidate and concrete findings to a bounded repair attempt. Revalidate and reverify. Stop at the configured limit.

### 8. Decide final status
Accept only when required checks and policy thresholds pass. Mark `needs_review` when evidence is insufficient. Mark `failed` when no usable candidate is produced. Confidence is a heuristic unless calibrated against labelled data.

### 9. Store eligible results
Persist only results that satisfy the storage policy. Record verification outcome, model/provider, prompt version, and provenance. Do not save low-confidence or teacher-review results as trusted cache entries.

### 10. Return a stable result
Return work-item ID, original index, question number, solution if available, final status, verification summary, and error details.

## Parent graph behavior
The parent graph prepares the paper, dispatches work items, collects each worker result exactly once, detects missing/duplicate results, and restores original order. Aggregation must tolerate worker failures.

## Failure handling
Expected failures include provider errors, malformed output, unsupported vision requirements, cache/database errors, verification failures, and exhausted repair attempts. Represent failures explicitly; do not return a success-shaped placeholder answer when solving failed.
