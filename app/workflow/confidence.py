"""Confidence / accuracy assessment and the "safe to store as verified" decision.

Four distinct things are kept apart:

1. ``self_confidence``  - what the model said about itself. Treated as an
   input that can only LOWER the result; it is never evidence.
2. ``evidence``         - what verification actually did (settings.EVIDENCE_RANK).
3. ``verification_score`` ("accuracy") - share of verification checks passed.
4. ``verified``         - the decision: verification passed AND evidence is at
   least the minimum for this question type AND the accuracy and confidence
   gates are met.

Reaching the retry limit never makes a failed verification verified.
"""
from __future__ import annotations

from typing import List

from app.workflow.schemas import ConfidenceAssessment, SolverDraft, VerificationReport, WorkItem
from app.workflow.settings import WorkflowSettings


def assess(item: WorkItem, draft: SolverDraft, report: VerificationReport, s: WorkflowSettings) -> ConfidenceAssessment:
    cap = s.evidence_confidence_cap.get(report.evidence, 0.30)
    final = min(draft.self_confidence, cap)
    reasons: List[str] = []

    if report.state != "passed":
        final = min(final, s.unverified_confidence_ceiling)
        reasons.append(f"verification {report.state}")
    need = s.min_evidence.get(item.type, "cross_check")
    if not s.evidence_at_least(report.evidence, need):
        reasons.append(f"evidence '{report.evidence}' is below the '{need}' required for {item.type} questions")
    if item.marks_assumed:
        reasons.append("marks were not stated in the source, so the mark split cannot be validated or reused")
    if report.score < s.accuracy_min:
        reasons.append(f"accuracy {report.score:.2f} < required {s.accuracy_min:.2f}")
    if final < s.store_min_confidence:
        reasons.append(f"final confidence {final:.2f} < required {s.store_min_confidence:.2f}")

    return ConfidenceAssessment(
        self_confidence=draft.self_confidence,
        evidence=report.evidence,
        evidence_cap=cap,
        verification_state=report.state,
        verification_score=report.score,
        final_confidence=round(final, 4),
        verified=not reasons,
        reasons=reasons,
    )
