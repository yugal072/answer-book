"""Type-specific verification of a draft solution.

Verification is a different task from solving:

* **Deterministic checks** (all types): shape, mark-split arithmetic, option
  validity, answer/choice consistency, and - for numerical questions - a
  re-evaluation of the solver's arithmetic expression with a whitelisted AST
  evaluator (no ``eval``, no model-written code is run).
* **Independent solve** (MCQ, numerical): a *different model* answers the
  question blind - it never sees the draft - and the two answers are compared.
* **Rubric review** (short, long): a different model reviews the draft against
  fixed criteria. There is no ground truth for prose, so this is the strongest
  evidence available and is labelled ``llm_review``, never "proven".

Evidence labels (see settings.EVIDENCE_RANK) record what actually happened:
``symbolic`` is only reported when an independent solve AGREED and the
solver's arithmetic expression re-evaluated to its own final value.

Deterministic failures short-circuit the (token-costly) model checks: the
draft goes straight to repair with precise feedback.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.workflow.llm import LLMError, LLMGateway
from app.workflow.mcq import normalize_letter, parse_options
from app.workflow.prompts import REVIEW_CHECKS
from app.workflow.safe_math import UnsafeExpression, numbers_close, safe_eval
from app.workflow.schemas import Finding, PaperContext, SolverDraft, VerificationReport, WorkItem
from app.workflow.settings import WorkflowSettings

_BLOCKING_REVIEW = {"answers_the_question", "factually_correct", "complete_for_marks"}
_MARK_EPS = 1e-6


def _structural(item: WorkItem, draft: SolverDraft) -> List[Finding]:
    f: List[Finding] = []
    f.append(Finding(check="answer_present", passed=bool(draft.answer.strip()), message="answer is empty"))
    f.append(Finding(check="steps_present", passed=bool(draft.steps), message="no solution steps"))

    total = sum(float(m.get("marks", 0)) for m in draft.mark_split)
    ok_positive = all(float(m.get("marks", 0)) > 0 for m in draft.mark_split)
    f.append(Finding(check="mark_split_positive", passed=ok_positive, message="mark_split has a non-positive entry"))
    if item.marks_assumed:
        f.append(
            Finding(
                check="mark_split_sum",
                passed=True,
                severity="info",
                message="source did not state marks; sum not enforced",
            )
        )
    else:
        f.append(
            Finding(
                check="mark_split_sum",
                passed=abs(total - item.marks) < _MARK_EPS,
                message=f"mark_split sums to {total:g} but the question carries {item.marks} mark(s)",
            )
        )
    if item.type == "long" and item.marks >= 4:
        f.append(
            Finding(
                check="depth_for_marks",
                passed=len(draft.steps) >= 3,
                severity="warning",
                message=f"only {len(draft.steps)} point(s) for a {item.marks}-mark descriptive answer",
            )
        )
    return f


def _mcq_deterministic(item: WorkItem, draft: SolverDraft) -> List[Finding]:
    labels = [lab for lab, _ in parse_options(item.options)]
    chosen = normalize_letter(draft.chosen_option)
    out = [
        Finding(
            check="option_valid",
            passed=chosen in labels,
            message=f"chosen option {draft.chosen_option!r} is not one of {labels}",
        )
    ]
    if chosen in labels:
        in_answer = re.search(r"\(\s*([A-Za-z])\s*\)", draft.answer)
        if in_answer and in_answer.group(1).upper() != chosen:
            out.append(
                Finding(
                    check="answer_matches_choice",
                    passed=False,
                    message=f"answer text names ({in_answer.group(1).upper()}) but chosen_option is {chosen}",
                )
            )
        else:
            out.append(Finding(check="answer_matches_choice", passed=True))
    return out


def _numeric_deterministic(item: WorkItem, draft: SolverDraft, rel_tol: float) -> Tuple[List[Finding], bool]:
    """Returns (findings, recomputed_ok). ``recomputed_ok`` is True only if the
    expression was actually evaluated and matched the stated value."""
    out: List[Finding] = []
    value = draft.final_value
    if value is None or not math.isfinite(value):
        return [Finding(check="final_value_present", passed=False, message="no finite numeric final_value")], False
    out.append(Finding(check="final_value_present", passed=True))
    expr = (draft.arithmetic_expression or "").strip()
    if not expr:
        out.append(
            Finding(
                check="arithmetic_recomputed",
                passed=True,
                severity="info",
                message="no arithmetic expression supplied; deterministic recomputation skipped",
            )
        )
        return out, False
    try:
        computed = safe_eval(expr)
    except UnsafeExpression as exc:
        out.append(
            Finding(
                check="arithmetic_recomputed",
                passed=True,
                severity="info",
                message=f"expression not evaluable ({exc}); deterministic recomputation skipped",
            )
        )
        return out, False
    ok = numbers_close(computed, value, rel_tol=max(rel_tol, 1e-6))
    out.append(
        Finding(
            check="arithmetic_recomputed",
            passed=ok,
            message=f"expression evaluates to {computed:g} but final_value is {value:g}",
        )
    )
    return out, ok


def deterministic_checks(item: WorkItem, draft: SolverDraft, rel_tol: float) -> Tuple[List[Finding], bool]:
    """All deterministic checks for item.type. Returns (findings, arithmetic_recomputed_ok).

    Shared by the verifier node and by the cache policy, which re-validates
    stored solutions against the *current* question with exactly these checks.
    """
    findings = _structural(item, draft)
    recomputed_ok = False
    if item.type == "mcq":
        findings += _mcq_deterministic(item, draft)
    elif item.type == "numerical":
        nf, recomputed_ok = _numeric_deterministic(item, draft, rel_tol)
        findings += nf
    return findings, recomputed_ok


class Verifier:
    def __init__(self, gateway: LLMGateway, settings: WorkflowSettings) -> None:
        self.gateway = gateway
        self.settings = settings

    def verify(
        self,
        item: WorkItem,
        ctx: PaperContext,
        draft: SolverDraft,
        *,
        independent: Optional[Dict[str, Any]] = None,
        tainted: bool = False,
    ) -> Tuple[VerificationReport, Optional[Dict[str, Any]]]:
        """Returns the report and the (possibly newly computed) independent solve."""
        findings, recomputed_ok = deterministic_checks(item, draft, self.settings.numeric_rel_tol)

        det_errors = [f for f in findings if not f.passed and f.severity == "error"]
        if det_errors:
            # Cheap failure: skip model checks, go to repair with precise feedback.
            return self._report("failed", findings, evidence="none"), independent

        evidence = "structural"
        try:
            if item.type == "mcq":
                if independent is None:
                    r = self.gateway.blind_solve_mcq(item, ctx)
                    independent = {"kind": "mcq", "chosen_option": normalize_letter(r.chosen_option)}
                agree = independent.get("chosen_option") == normalize_letter(draft.chosen_option)
                findings.append(
                    Finding(
                        check="independent_solve",
                        passed=agree,
                        method="independent_solve",
                        message=(
                            "independent examiner (different model, blind) agrees"
                            if agree
                            else "an independent examiner solving the question blind chose a different option"
                        ),
                    )
                )
                if agree and not tainted:
                    evidence = "cross_check"
            elif item.type == "numerical":
                if independent is None:
                    r = self.gateway.blind_solve_numeric(item, ctx)
                    independent = {"kind": "numerical", "final_value": r.final_value}
                other = independent["final_value"]
                agree = draft.final_value is not None and numbers_close(
                    other, draft.final_value, rel_tol=self.settings.numeric_rel_tol
                )
                findings.append(
                    Finding(
                        check="independent_solve",
                        passed=agree,
                        method="independent_solve",
                        message=(
                            "independent examiner (different model, blind) reached the same value"
                            if agree
                            else "an independent examiner solving the question blind reached a different value"
                        ),
                    )
                )
                if agree:
                    evidence = "symbolic" if recomputed_ok else "cross_check"
            else:
                review = self.gateway.review(item, ctx, draft)
                got = {c.name: c for c in review.checks}
                missing = [n for n in REVIEW_CHECKS if n not in got]
                if missing:
                    findings.append(
                        Finding(
                            check="review_complete",
                            passed=False,
                            method="llm_review",
                            message=f"reviewer omitted required checks: {', '.join(missing)}",
                        )
                    )
                else:
                    for name in REVIEW_CHECKS:
                        c = got[name]
                        findings.append(
                            Finding(
                                check=name,
                                passed=c.passed,
                                severity="error" if name in _BLOCKING_REVIEW else "warning",
                                method="llm_review",
                                message=c.note or "reviewer marked this criterion as not met",
                            )
                        )
                    for issue in review.issues[:5]:
                        findings.append(
                            Finding(check="reviewer_issue", passed=True, severity="info", method="llm_review", message=issue)
                        )
                    evidence = "llm_review"
        except LLMError as exc:
            findings.append(
                Finding(
                    check="verifier_available",
                    passed=False,
                    method="independent_solve" if item.type in ("mcq", "numerical") else "llm_review",
                    message=f"verifier could not run: {type(exc).__name__}",
                )
            )
            return self._report("inconclusive", findings, evidence="structural", error=str(exc)), independent

        errors = [f for f in findings if not f.passed and f.severity == "error"]
        state = "failed" if errors else "passed"
        return self._report(state, findings, evidence=evidence), independent

    def _report(self, state: str, findings: List[Finding], *, evidence: str, error: Optional[str] = None) -> VerificationReport:
        counted = [f for f in findings if f.severity != "info"]
        total = sum(1.0 if f.severity == "error" else 0.5 for f in counted)
        passed = sum((1.0 if f.severity == "error" else 0.5) for f in counted if f.passed)
        score = round(passed / total, 4) if total else 0.0
        return VerificationReport(
            state=state,  # type: ignore[arg-type]
            evidence=evidence,  # type: ignore[arg-type]
            findings=findings,
            score=score,
            verifier_model=getattr(self.gateway, "verifier_model", ""),
            error=error,
        )
