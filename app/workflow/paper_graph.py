"""Whole-paper graph (Stages A and F of the architecture diagram).

    START -> validate_prepare -> [fan-out via Send] -> question_pipeline (xN, parallel)
          -> aggregate_results -> build_solved_paper -> assemble_output -> END

``question_pipeline`` is the compiled per-question sub-graph used as a node.
The only state key shared with it is ``results`` (merged by item_id), so
parallel branches can never overwrite each other.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.workflow.deps import WorkflowDeps
from app.workflow.events import emit
from app.workflow.question_graph import build_question_graph
from app.workflow.schemas import (
    Diagnostic,
    PaperContext,
    PaperState,
    PaperSummary,
    QuestionResult,
    SolutionOut,
    SolvedPaper,
    WorkItem,
)

log = logging.getLogger(__name__)

#: Status written to ``papers.status``. Only a fully verified/cached paper is
#: "ready" - the legacy paper-level cache serves "ready" papers only.
DB_STATUS = {"complete": "ready", "needs_review": "needs_review", "partial": "partial", "failed": "failed"}


def paper_outcome(results: List[QuestionResult]) -> str:
    if not results:
        return "failed"
    failed = sum(r.status == "failed" for r in results)
    unverified = sum(r.status == "unverified" for r in results)
    if failed == len(results):
        return "failed"
    if failed:
        return "partial"
    if unverified:
        return "needs_review"
    return "complete"


def summarize(results: List[QuestionResult]) -> PaperSummary:
    c = lambda st: sum(r.status == st for r in results)  # noqa: E731
    return PaperSummary(total=len(results), verified=c("verified"), cached=c("cached"), unverified=c("unverified"), failed=c("failed"))


def build_solved_paper(
    ctx: PaperContext, results: List[QuestionResult], diagnostics: List[Diagnostic]
) -> SolvedPaper:
    ordered = sorted(results, key=lambda r: r.index)
    outcome = paper_outcome(ordered)
    return SolvedPaper(
        paper_id=ctx.paper_id,
        fingerprint=ctx.fingerprint,
        status=DB_STATUS[outcome],
        outcome=outcome,  # type: ignore[arg-type]
        metadata={
            "paper_id": ctx.paper_id,
            "fingerprint": ctx.fingerprint,
            "subject": ctx.subject,
            "class": ctx.class_name,
            "class_name": ctx.class_name,
            "board": ctx.board,
            "source_file": ctx.source_file,
        },
        total_questions=len(ordered),
        total_marks=ctx.total_marks,
        sections=ctx.sections,
        summary=summarize(ordered),
        solutions=[
            {**r.solution.model_dump(), "status": r.status, "origin": r.origin, "item_id": r.item_id}
            for r in ordered
        ],
        results=ordered,
        diagnostics=diagnostics,
    )


def build_paper_graph(deps: WorkflowDeps, checkpointer: Any = None):
    question_graph = build_question_graph(deps)

    def validate_prepare(state: PaperState) -> Dict[str, Any]:
        ctx = PaperContext.model_validate(state["paper_context"])
        items = [WorkItem.model_validate(i) for i in state.get("work_items", [])]
        diags: List[dict] = []
        valid: List[dict] = []
        early: List[dict] = []
        seen = set()
        for it in items:
            problem = None
            if it.item_id in seen:
                problem = f"duplicate work item id {it.item_id}"
            elif not it.text.strip():
                problem = "question text is empty"
            seen.add(it.item_id)
            if problem:
                diags.append(Diagnostic(stage="validate_prepare", severity="error", code="invalid_item",
                                        message=f"Q{it.number}: {problem}").model_dump())
                early.append(
                    QuestionResult(
                        item_id=it.item_id, index=it.index, question=it.model_dump(), status="failed", origin="none",
                        solution=SolutionOut(question_number=it.number),
                        diagnostics=[Diagnostic(stage="validate_prepare", severity="error", code="invalid_item", message=problem)],
                    ).model_dump()
                )
            else:
                valid.append(it.model_dump())
        try:
            if not ctx.ephemeral:
                deps.papers.begin(ctx, len(items))
        except Exception as exc:  # noqa: BLE001
            diags.append(Diagnostic(stage="validate_prepare", severity="error", code="persistence_failed",
                                    message=f"paper bookkeeping unavailable: {type(exc).__name__}: {str(exc)[:160]}").model_dump())
        emit("run.prepared", paper_id=ctx.paper_id, total_questions=len(items), runnable=len(valid))
        return {"work_items": valid, "results": early, "expected_ids": [i.item_id for i in items], "paper_diagnostics": diags}

    def fan_out(state: PaperState):
        sends = [Send("question_pipeline", {"item": it, "ctx": state["paper_context"]}) for it in state.get("work_items", [])]
        return sends or "aggregate_results"

    def aggregate_results(state: PaperState) -> Dict[str, Any]:
        have = {r["item_id"] for r in state.get("results", [])}
        extra: List[dict] = []
        diags: List[dict] = []
        for raw in state.get("work_items", []):
            if raw["item_id"] not in have:  # a branch ended without a result - never lose a question silently
                it = WorkItem.model_validate(raw)
                msg = "question branch returned no result"
                diags.append(Diagnostic(stage="aggregate_results", severity="error", code="missing_result", message=f"Q{it.number}: {msg}").model_dump())
                extra.append(
                    QuestionResult(item_id=it.item_id, index=it.index, question=it.model_dump(), status="failed", origin="none",
                                   solution=SolutionOut(question_number=it.number),
                                   diagnostics=[Diagnostic(stage="aggregate_results", severity="error", code="missing_result", message=msg)]).model_dump()
                )
        return {"results": extra, "paper_diagnostics": diags}

    def build_solved(state: PaperState) -> Dict[str, Any]:
        ctx = PaperContext.model_validate(state["paper_context"])
        results = [QuestionResult.model_validate(r) for r in state.get("results", [])]
        diags = [Diagnostic.model_validate(d) for d in state.get("paper_diagnostics", [])]
        solved = build_solved_paper(ctx, results, diags)
        return {"solved_paper": solved.model_dump(mode="json", by_alias=True)}

    def assemble_output(state: PaperState) -> Dict[str, Any]:
        """Assembly / Serving: persist status and export the JSON book.

        PDF and SSE are rendered from the same SolvedPaper by the serving layer.
        """
        ctx = PaperContext.model_validate(state["paper_context"])
        solved = state["solved_paper"]
        diags: List[dict] = []
        persisted = True
        try:
            if not ctx.ephemeral:
                deps.papers.finish(ctx, solved["status"], solved["total_questions"])
        except Exception as exc:  # noqa: BLE001
            persisted = False
            diags.append(Diagnostic(stage="assemble_output", severity="error", code="persistence_failed",
                                    message=f"{type(exc).__name__}: {str(exc)[:160]}").model_dump())
        path = None
        try:
            from app.workflow.render import write_json  # local import: render is a serving concern

            if not ctx.ephemeral:
                path = str(write_json(solved, deps.settings.output_dir))
        except Exception as exc:  # noqa: BLE001
            diags.append(Diagnostic(stage="assemble_output", severity="error", code="export_failed",
                                    message=f"{type(exc).__name__}: {str(exc)[:160]}").model_dump())
        if diags:
            solved = {**solved, "diagnostics": list(solved["diagnostics"]) + diags}
        emit("paper.completed", paper_id=ctx.paper_id, outcome=solved["outcome"], summary=solved["summary"])
        return {"solved_paper": solved, "output": {"persisted": persisted, "json_path": path}, "paper_diagnostics": diags}

    g = StateGraph(PaperState)
    g.add_node("validate_prepare", validate_prepare)
    g.add_node("question_pipeline", question_graph)
    g.add_node("aggregate_results", aggregate_results)
    g.add_node("build_solved_paper", build_solved)
    g.add_node("assemble_output", assemble_output)
    g.add_edge(START, "validate_prepare")
    g.add_conditional_edges("validate_prepare", fan_out, ["question_pipeline", "aggregate_results"])
    g.add_edge("question_pipeline", "aggregate_results")
    g.add_edge("aggregate_results", "build_solved_paper")
    g.add_edge("build_solved_paper", "assemble_output")
    g.add_edge("assemble_output", END)
    return g.compile(checkpointer=checkpointer)
