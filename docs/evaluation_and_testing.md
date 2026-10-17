# Evaluation, Testing, and Rollout

## Goal
Evaluate solver quality separately from whether the graph executes successfully. A successful API response is not evidence that an answer is correct.

## Test layers

### Unit tests
Use fake providers and repositories. Test state/output validation, malformed JSON, missing fields, confidence edge cases, mark-split constraints, cache hit/miss/error and eligibility, verification outcomes, repair limits, provider selection/capabilities, aggregation failures, and tool safety. Unit tests should not require API keys, network access, or a live database.

### Adapter tests
Test each adapter's request/response normalization, configuration validation, capability reporting, error mapping, and timeout behavior. Use mocks or approved test mechanisms.

### Integration tests
Use fake providers and a disposable database to exercise prepare, dispatch, cache lookup, generation, validation, verification, repair, finalization, persistence, and aggregation.

### Live model tests
Keep real-provider tests separate because they can incur costs, rate limits, and nondeterminism.

### Dataset evaluation
Use existing exam papers and corresponding answer papers as labelled data. Keep a fixed held-out subset for fair comparison. Avoid tuning prompts on every example and then reporting those same examples as independent evaluation.

## Baseline metrics
Record:
- answer correctness under a defined rubric;
- completion and teacher-review rates;
- verification precision and false-acceptance rate;
- schema/validation failure rate;
- cache hit and cache false-acceptance rates;
- average and percentile latency;
- provider failures, retries, and rate-limit events;
- estimated cost or quota consumption per paper/question.

For theory answers, define a rubric covering required concepts, factual correctness, completeness, and mark allocation.

## Comparing prompts and models
Version prompts explicitly. Record dataset version/question IDs, provider/model IDs, prompt version, configuration/thresholds, metrics, and representative failures. Compare models on the same examples and rubric.

## Rollout sequence
1. Preserve the current working implementation.
2. Add provider interfaces and fake-provider tests.
3. Add the per-question workflow behind a controlled integration point.
4. Run unit and integration tests.
5. Run a small live-model sample.
6. Evaluate against labelled exam papers.
7. Compare against baseline and inspect false accepts.
8. Enable broader use only after acceptance thresholds are met.

## Release gate
Before production, confirm state/import contracts, retry and timeout limits, verification acceptance rules, cache eligibility and migrations, missing-figure behavior, evaluation thresholds, and safe diagnostic logging.
