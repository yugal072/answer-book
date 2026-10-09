"""Per-question sub-graph (Stages B-E of the architecture diagram).

    START -> cache_lookup -+-hit-> retrieve_stored -+-> finalize_question
                           |                        '-fallback-> (solver)
                           '-miss-> solve_{mcq|numerical|short|long}
    solve_* -> verify -> [verification passes?]
         passed / inconclusive / retries exhausted -> assess_confidence
         failed and retries remain -> repair -> verify
    assess_confidence -> [verified?] -> store_verified -> finalize_question
                                    '-> finalize_question (unverified)
    finalize_question -> END

Every node catches its own failures: a problem in one question becomes a
diagnostic and a ``failed``/``unverified`` result for THAT question only.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from app.workflow import cache_policy
from app.workflow.confidence import assess
from app.workflow.deps import WorkflowDeps
from app.workflow.events import emit
from app.workflow.llm import LLMError
from app.workflow.prompts import RepairContext
from app.workflow.schemas import (
    AttemptRecord,
    CacheDecision,
    ConfidenceAssessment,
    Diagnostic,
    PaperContext,
    QuestionResult,
    QuestionState,
    SolutionOut,
    SolverDraft,
    VerificationReport,
    WorkItem,
)
from app.workflow.settings import PROMPT_VERSION
from app.workflow.verification import Verifier

log = logging.getLogger(__name__)


def _diag(stage: str, severity: str, code: str, message: str) -> dict:
    return Diagnostic(stage=stage, severity=severity, code=code, message=message).model_dump()  # type: ignore[arg-type]


def _item(state: QuestionState) -> WorkItem:
    return WorkItem.model_validate(state["item"])


def _ctx(state: QuestionState) -> PaperContext:
    return PaperContext.model_validate(state["ctx"])


def safe_node(stage: str) -> Callable:
    """Turn any unexpected exception into a diagnostic + ``fatal`` for this question."""

    def deco(fn: Callable[[QuestionState], Dict[str, Any]]) -> Callable[[QuestionState], Dict[str, Any]]:
        @functools.wraps(fn)
        def wrapper(state: QuestionState) -> Dict[str, Any]:
            try:
                return fn(state)
            except Exception as exc:  # noqa: BLE001
                log.exception("question node %s failed", stage)
                return {
                    "fatal": True,
                    "diagnostics": [_diag(stage, "error", "node_error", f"{type(exc).__name__}: {str(exc)[:200]}")],
                }

        return wrapper

    return deco


def solution_from(
    item: WorkItem, draft: SolverDraft, assessment: ConfidenceAssessment, *, verified_by: Optional[str] = None
) -> SolutionOut:
    return SolutionOut(
        question_number=item.number,
        answer=draft.answer,
        steps=list(draft.steps),
        mark_split=[{"marks": m["marks"], "for": m["for"]} for m in draft.mark_split],
        common_mistakes=list(draft.common_mistakes),
        confidence=assessment.final_confidence,
        verified_by=verified_by or (assessment.evidence if assessment.verified else "none"),
        needs_teacher_check=not assessment.verified,
    )


def build_question_graph(deps: WorkflowDeps):
    s = deps.settings
    verifier = Verifier(deps.llm, s)

    # ------------------------------------------------------------------ B: cache
    @safe_node("cache_lookup")
    def cache_lookup(state: QuestionState) -> Dict[str, Any]:
        item, ctx = _item(state), _ctx(state)
        emit("question.stage", item_id=item.item_id, number=item.number, stage="cache_lookup")
        diags: List[dict] = []
        try:
            candidates = deps.kb.find_candidates(item.signature)
        except Exception as exc:  # noqa: BLE001 - database failure: surface it, then solve without cache
            decision = CacheDecision(looked_up=False, accepted=False, error=f"{type(exc).__name__}: {str(exc)[:160]}")
            decision.reasons.append("cache unavailable; solving without it")
            diags.append(_diag("cache_lookup", "warning", "cache_unavailable", decision.error or ""))
            return {"cache": decision.model_dump(), "stored": None, "diagnostics": diags}
        decision, best = cache_policy.decide(item, ctx, candidates, s)
        if decision.accepted:
            emit("question.stage", item_id=item.item_id, number=item.number, stage="cache_hit")
        return {"cache": decision.model_dump(), "stored": best.model_dump() if best else None, "diagnostics": diags}

    def after_cache(state: QuestionState) -> str:
        if state.get("fatal"):
            return "failed"
        if (state.get("cache") or {}).get("accepted") and state.get("stored"):
            return "hit"
        return _item(state).type

    @safe_node("retrieve_stored")
    def retrieve_stored(state: QuestionState) -> Dict[str, Any]:
        item = _item(state)
        stored = cache_policy.StoredSolution.model_validate(state["stored"])
        # Output-contract validation of the stored payload before it is trusted.
        draft = SolverDraft.model_validate(stored.draft)
        sol = SolutionOut.model_validate(stored.solution)
        if not sol.answer.strip():
            raise ValueError("stored solution has an empty answer")
        assessment = ConfidenceAssessment(
            self_confidence=stored.self_confidence,
            evidence=stored.evidence,  # type: ignore[arg-type]
            evidence_cap=stored.confidence,
            verification_state="passed",
            verification_score=stored.verification_score,
            final_confidence=stored.confidence,
            verified=True,
            reasons=[],
        )
        return {"draft": draft.model_dump(), "confidence": assessment.model_dump()}

    def after_retrieve(state: QuestionState) -> str:
        return "failed_retrieval" if state.get("fatal") else "ok"

    @safe_node("retrieve_stored")
    def retrieval_fallback(state: QuestionState) -> Dict[str, Any]:
        """Stored payload failed re-validation: drop the cache hit and solve normally."""
        cache = dict(state.get("cache") or {})
        cache["accepted"] = False
        cache.setdefault("reasons", []).append("stored payload failed output-contract validation on retrieval")
        return {
            "cache": cache,
            "stored": None,
            "fatal": False,
            "diagnostics": [_diag("retrieve_stored", "warning", "stored_invalid", "stored solution rejected on retrieval")],
        }

    # ------------------------------------------------------------------ C: solve
    def make_solver(strategy: str) -> Callable:
        @safe_node(f"solve_{strategy}")
        def solve(state: QuestionState) -> Dict[str, Any]:
            item, ctx = _item(state), _ctx(state)
            emit("question.stage", item_id=item.item_id, number=item.number, stage="solving", strategy=strategy)
            try:
                draft = deps.llm.draft(item, ctx, None)
            except LLMError as exc:
                return {
                    "draft": None,
                    "fatal": True,
                    "diagnostics": [_diag(f"solve_{strategy}", "error", type(exc).__name__, str(exc)[:240])],
                }
            return {"draft": draft.model_dump()}

        return solve

    def after_draft(state: QuestionState) -> str:
        return "failed" if state.get("fatal") or not state.get("draft") else "verify"

    # ------------------------------------------------------------------ D: verify / repair
    @safe_node("verify")
    def verify(state: QuestionState) -> Dict[str, Any]:
        item, ctx = _item(state), _ctx(state)
        draft = SolverDraft.model_validate(state["draft"])
        emit("question.stage", item_id=item.item_id, number=item.number, stage="verifying")
        report, independent = verifier.verify(
            item, ctx, draft, independent=state.get("independent"), tainted=bool(state.get("tainted"))
        )
        attempt = AttemptRecord(
            attempt=int(state.get("repair_count", 0)) + 1, draft=draft.model_dump(), verification=report.model_dump()
        )
        out: Dict[str, Any] = {"verification": report.model_dump(), "attempts": [attempt.model_dump()]}
        if independent is not None:
            out["independent"] = independent
        if report.state == "inconclusive":
            out["diagnostics"] = [_diag("verify", "warning", "verifier_unavailable", report.error or "")]
        return out

    def verification_passes(state: QuestionState) -> str:
        if state.get("fatal") or not state.get("verification"):
            return "failed"  # finalize_question reports whatever exists (draft => unverified, none => failed)
        rep = state.get("verification") or {}
        if rep.get("state") == "failed" and int(state.get("repair_count", 0)) < s.max_repair_attempts:
            return "repair"
        return "assess"  # passed, inconclusive, or retries exhausted - all go to assessment

    @safe_node("repair")
    def repair(state: QuestionState) -> Dict[str, Any]:
        item, ctx = _item(state), _ctx(state)
        report = VerificationReport.model_validate(state["verification"])
        prev = SolverDraft.model_validate(state["draft"])
        findings = report.failed_messages()
        n = int(state.get("repair_count", 0)) + 1
        emit("question.stage", item_id=item.item_id, number=item.number, stage="repairing", attempt=n)
        # An MCQ whose first answer lost the blind cross-check learned something about
        # the verifier's answer from our feedback, so later agreement is not independent.
        tainted = bool(state.get("tainted")) or (
            item.type == "mcq" and any(f.check == "independent_solve" and not f.passed for f in report.findings)
        )
        try:
            new = deps.llm.draft(item, ctx, RepairContext(prev.answer, findings))
        except LLMError as exc:
            return {
                "repair_failed": True,
                "repair_count": n,
                "repair_feedback": findings,
                "diagnostics": [_diag("repair", "error", type(exc).__name__, str(exc)[:240])],
            }
        return {
            "draft": new.model_dump(),
            "repair_count": n,
            "repair_feedback": findings,
            "repair_failed": False,
            "tainted": tainted,
        }

    def after_repair(state: QuestionState) -> str:
        if state.get("fatal"):
            return "failed"
        return "assess" if state.get("repair_failed") else "verify"

    # ------------------------------------------------------------------ E: assess / store
    @safe_node("assess_confidence")
    def assess_confidence(state: QuestionState) -> Dict[str, Any]:
        item = _item(state)
        emit("question.stage", item_id=item.item_id, number=item.number, stage="assessing")
        draft = SolverDraft.model_validate(state["draft"])
        report = VerificationReport.model_validate(state["verification"])
        a = assess(item, draft, report, s)
        out: Dict[str, Any] = {"confidence": a.model_dump()}
        if not a.verified and report.state == "failed":
            out["diagnostics"] = [_diag("assess_confidence", "warning", "retries_exhausted",
                                         f"verification still failing after {state.get('repair_count', 0)} repair attempt(s)")]
        return out

    def store_decision(state: QuestionState) -> str:
        if state.get("fatal") or _item(state).has_figure:
            return "finalize"  # a figure-dependent question is never reusable from text alone, so it is not stored
        return "store" if (state.get("confidence") or {}).get("verified") else "finalize"

    @safe_node("store_verified")
    def store_verified(state: QuestionState) -> Dict[str, Any]:
        item, ctx = _item(state), _ctx(state)
        emit("question.stage", item_id=item.item_id, number=item.number, stage="storing")
        draft = SolverDraft.model_validate(state["draft"])
        a = ConfidenceAssessment.model_validate(state["confidence"])
        report = VerificationReport.model_validate(state["verification"])
        solution = solution_from(item, draft, a).model_dump()
        try:
            outcome = deps.kb.store_verified(item, ctx, draft, solution, a, report)
        except Exception as exc:  # noqa: BLE001
            return {
                "stored_in_kb": False,
                "diagnostics": [_diag("store_verified", "error", "storage_failed", f"{type(exc).__name__}: {str(exc)[:200]}")],
            }
        cache = dict(state.get("cache") or {})
        cache["kb_id"] = outcome.kb_id
        return {
            "stored_in_kb": outcome.action in ("inserted", "updated", "kept_existing"),
            "cache": cache,
            "diagnostics": [_diag("store_verified", "info", f"kb_{outcome.action}", f"knowledge base id {outcome.kb_id}")],
        }

    # ------------------------------------------------------------------ finalize
    def finalize_question(state: QuestionState) -> Dict[str, Any]:
        item, ctx = _item(state), _ctx(state)
        try:
            result = _build_result(state, item)
        except Exception as exc:  # noqa: BLE001 - last line of defence: never lose the branch
            log.exception("finalize failed")
            result = QuestionResult(
                item_id=item.item_id,
                index=item.index,
                question=item.model_dump(),
                status="failed",
                origin="none",
                solution=SolutionOut(question_number=item.number),
                diagnostics=[Diagnostic(stage="finalize", severity="error", code="finalize_error", message=str(exc)[:240])],
            )
        try:
            if not ctx.ephemeral:
                deps.papers.save_question(ctx, result)  # per-question checkpoint (crash recovery)
        except Exception as exc:  # noqa: BLE001
            result.diagnostics.append(
                Diagnostic(stage="finalize", severity="error", code="checkpoint_failed", message=f"{type(exc).__name__}: {str(exc)[:160]}")
            )
        emit(
            "question.completed",
            item_id=item.item_id,
            number=item.number,
            index=item.index,
            status=result.status,
            origin=result.origin,
            verified_by=result.solution.verified_by,
            confidence=result.solution.confidence,
        )
        return {"results": [result.model_dump()]}

    def _build_result(state: QuestionState, item: WorkItem) -> QuestionResult:
        diags = [Diagnostic.model_validate(d) for d in state.get("diagnostics", [])]
        diags += [Diagnostic(stage="adapter", severity="warning", code="normalized", message=w) for w in item.warnings]
        cache = CacheDecision.model_validate(state.get("cache") or {})
        attempts = [AttemptRecord.model_validate(a) for a in state.get("attempts", [])]
        report = VerificationReport.model_validate(state["verification"]) if state.get("verification") else None
        base = dict(
            item_id=item.item_id, index=item.index, question=item.model_dump(), cache=cache, attempts=attempts,
            verification=report, diagnostics=diags, prompt_version=PROMPT_VERSION,
        )
        draft = SolverDraft.model_validate(state["draft"]) if state.get("draft") else None
        if draft is None:
            return QuestionResult(status="failed", origin="none", solution=SolutionOut(question_number=item.number), **base)

        if state.get("stored") and cache.accepted:
            a = ConfidenceAssessment.model_validate(state["confidence"])
            stored = cache_policy.StoredSolution.model_validate(state["stored"])
            sol = solution_from(item, draft, a, verified_by=f"cache:kb:{stored.evidence}")
            return QuestionResult(status="cached", origin="cache", solution=sol, confidence_assessment=a, **base)

        if state.get("confidence"):
            a = ConfidenceAssessment.model_validate(state["confidence"])
        else:  # a node failed after drafting: nothing was established about the draft
            a = ConfidenceAssessment(
                self_confidence=draft.self_confidence, evidence="none", evidence_cap=s.evidence_confidence_cap["none"],
                verification_state="inconclusive", verification_score=0.0,
                final_confidence=min(draft.self_confidence, s.unverified_confidence_ceiling),
                verified=False, reasons=["pipeline error before assessment"],
            )
        sol = solution_from(item, draft, a)
        return QuestionResult(
            status="verified" if a.verified else "unverified",
            origin="generated",
            solution=sol,
            confidence_assessment=a,
            stored_in_kb=bool(state.get("stored_in_kb")),
            **base,
        )

    # ------------------------------------------------------------------ wiring
    g = StateGraph(QuestionState)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("retrieve_stored", retrieve_stored)
    g.add_node("retrieval_fallback", retrieval_fallback)
    for t in ("mcq", "numerical", "short", "long"):
        g.add_node(f"solve_{t}", make_solver(t))
    g.add_node("verify", verify)
    g.add_node("repair", repair)
    g.add_node("assess_confidence", assess_confidence)
    g.add_node("store_verified", store_verified)
    g.add_node("finalize_question", finalize_question)

    g.add_edge(START, "cache_lookup")
    g.add_conditional_edges(
        "cache_lookup",
        after_cache,
        {
            "hit": "retrieve_stored",
            "mcq": "solve_mcq",
            "numerical": "solve_numerical",
            "short": "solve_short",
            "long": "solve_long",
            "failed": "finalize_question",
        },
    )
    g.add_conditional_edges(
        "retrieve_stored", after_retrieve, {"ok": "finalize_question", "failed_retrieval": "retrieval_fallback"}
    )
    g.add_conditional_edges(
        "retrieval_fallback",
        lambda st: _item(st).type,
        {"mcq": "solve_mcq", "numerical": "solve_numerical", "short": "solve_short", "long": "solve_long"},
    )
    for t in ("mcq", "numerical", "short", "long"):
        g.add_conditional_edges(f"solve_{t}", after_draft, {"verify": "verify", "failed": "finalize_question"})
    g.add_conditional_edges(
        "verify", verification_passes, {"repair": "repair", "assess": "assess_confidence", "failed": "finalize_question"}
    )
    g.add_conditional_edges(
        "repair", after_repair, {"verify": "verify", "assess": "assess_confidence", "failed": "finalize_question"}
    )
    g.add_conditional_edges("assess_confidence", store_decision, {"store": "store_verified", "finalize": "finalize_question"})
    g.add_edge("store_verified", "finalize_question")
    g.add_edge("finalize_question", END)
    return g.compile()
