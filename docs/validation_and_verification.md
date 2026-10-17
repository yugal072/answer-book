# Answer Validation and Verification

## Purpose
Validation and verification are different responsibilities. Both are required before a generated answer is accepted or stored as a trusted cache entry.

## Validation: is the output well-formed?
Validation should be deterministic wherever possible.

Check:
- required keys and field types;
- question number matches the work item;
- required answer and steps are present;
- confidence is numeric and within `[0, 1]`;
- `needs_teacher_check` is boolean;
- verification metadata follows the contract;
- mark-split entries have valid marks and descriptions;
- mark-split total does not exceed available marks;
- strings and lists have reasonable size limits.

Do not assign high confidence or mark an answer verified when the model omitted those fields.

## Semantic consistency checks
Where feasible, validate:
- MCQ answer matches a supplied option;
- final numerical answer agrees with computed intermediate results;
- steps do not contradict the final answer;
- mark allocation is consistent with the question;
- the answer addresses all requested subparts.

Record which checks ran, passed, failed, or were unavailable.

## Verification: is the answer likely correct?
Verification may combine deterministic/domain-specific checks, independent model review, trusted reference material, and human review for unresolved cases. The verifier should return structured findings: outcome, reasons, checks performed, and repair instructions. Approval is evidence, not a guarantee.

## Repair policy
A repair request should include the original question, candidate answer, validation errors, verification findings, and output contract. After repair:
1. validate again;
2. verify again;
3. increment the attempt counter;
4. stop at the configured maximum;
5. finalize as accepted, needs review, or failed.

Repair must never bypass validation.

## Confidence policy
Confidence must be finite and within `[0, 1]`. Missing confidence is unknown; it must not be replaced with `1.0`. `0.0` is valid and must not be treated as missing.

Initially, confidence may be a heuristic derived from explicit signals. To interpret it as a probability, calibrate it on labelled questions and measure calibration on held-out data. Do not base cache eligibility only on model self-reported confidence.

## Suggested verification outcomes
- `passed`: required checks passed;
- `failed`: one or more required checks failed;
- `needs_review`: checks were inconclusive or unavailable;
- `not_checked`: verification has not run.

Keep the outcome separate from the reasons and check details.

## Teacher review
Require review when a required capability is unavailable, a substantive issue remains after repair, critical context is missing, attempts are exhausted without acceptance, or evidence is insufficient for a safe decision.
