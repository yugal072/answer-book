"""AnswerBookService: the single entry point the API and CLI use.

    Paper --adapter--> GraphState --> paper graph --> SolvedPaper

The service owns run identity (``run_id`` = LangGraph ``thread_id``), the
checkpointer, event sequencing for SSE, and resume. It never puts clients,
sessions or credentials into graph state.

Checkpointing (execution state, per run) is deliberately separate from the
knowledge base (reusable verified solutions, across runs): a checkpoint lets
an interrupted run continue; the KB lets a *new* run skip work that is
already verified.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from app.models.loaders_models import Paper
from app.workflow.adapter import AdaptedPaper, adapt_paper
from app.workflow.deps import WorkflowDeps
from app.workflow.paper_graph import build_paper_graph
from app.workflow.schemas import SolvedPaper
from app.workflow.settings import WorkflowSettings

log = logging.getLogger(__name__)


class RunNotFound(KeyError):
    """No checkpointed run with that id."""


@dataclass
class RunHandle:
    run_id: str
    config: Dict[str, Any]


class AnswerBookService:
    def __init__(self, deps: WorkflowDeps, checkpointer: Any = None) -> None:
        self.deps = deps
        self.settings: WorkflowSettings = deps.settings
        self.checkpointer = checkpointer
        self.graph = build_paper_graph(deps, checkpointer)

    # ------------------------------------------------------------------ helpers
    def adapt(self, paper: Paper, *, limit: Optional[int] = None, source_file: Optional[str] = None) -> AdaptedPaper:
        return adapt_paper(paper, limit=limit, source_file=source_file)

    def _config(self, run_id: str) -> Dict[str, Any]:
        s = self.settings
        return {
            "configurable": {"thread_id": run_id},
            "recursion_limit": max(s.question_recursion_limit(), 25),
            "max_concurrency": s.max_concurrency,
        }

    @staticmethod
    def _graph_input(adapted: AdaptedPaper) -> Dict[str, Any]:
        return {
            "paper_context": adapted.context.model_dump(),
            "work_items": [i.model_dump() for i in adapted.items],
            "paper_diagnostics": [d.model_dump() for d in adapted.diagnostics],
        }

    # ------------------------------------------------------------------ run
    def run(
        self,
        paper: Paper,
        *,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> SolvedPaper:
        adapted = self.adapt(paper, limit=limit, source_file=source_file)
        handle = RunHandle(run_id or uuid.uuid4().hex, {})
        final = self.graph.invoke(self._graph_input(adapted), self._config(handle.run_id))
        return SolvedPaper.model_validate(final["solved_paper"])

    def solve_single(self, question: Dict[str, Any], metadata: Dict[str, Any]) -> SolvedPaper:
        """Solve one ad-hoc question (no paper rows, no export; the KB is still used)."""
        import hashlib

        from app.models.loaders_models import Question

        q = Question.model_validate({k: v for k, v in question.items() if k in Question.model_fields})
        fp = hashlib.sha256(f"{q.number}|{q.text}".encode("utf-8")).hexdigest()
        paper = Paper(
            paper_id=f"adhoc_{fp[:8]}", fingerprint=fp, status="solving",
            subject=metadata.get("subject"), class_name=metadata.get("class") or metadata.get("class_name"),
            board=metadata.get("board"), questions=[q], total_questions=1, total_marks=q.marks,
        )
        adapted = self.adapt(paper)
        adapted.context.ephemeral = True
        final = self.graph.invoke(self._graph_input(adapted), self._config(uuid.uuid4().hex))
        return SolvedPaper.model_validate(final["solved_paper"])

    def resume(self, run_id: str) -> SolvedPaper:
        """Continue an interrupted run from its last checkpoint (needs a checkpointer)."""
        if self.checkpointer is None:
            raise RunNotFound(run_id)
        cfg = self._config(run_id)
        snap = self.graph.get_state(cfg)
        if not snap or not snap.values:
            raise RunNotFound(run_id)
        if snap.next:
            final = self.graph.invoke(None, cfg)
        else:
            final = snap.values
        return SolvedPaper.model_validate(final["solved_paper"])

    # ------------------------------------------------------------------ stream
    def stream(
        self,
        paper: Paper,
        *,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield SSE-ready events: ``{"event": name, "data": {...}}`` with a stable contract.

        Events: run.started, run.prepared, question.stage, question.completed,
        paper.completed, then exactly one terminal event: run.finished | run.error.
        """
        rid = run_id or uuid.uuid4().hex
        try:
            adapted = self.adapt(paper, limit=limit, source_file=source_file)
        except Exception as exc:  # noqa: BLE001
            yield {"event": "run.error", "data": {"run_id": rid, "code": type(exc).__name__, "message": str(exc)[:300]}}
            return
        yield {"event": "run.started", "data": {"run_id": rid, "paper_id": adapted.context.paper_id,
                                                "total_questions": len(adapted.items)}}
        yield from self._pump(self._graph_input(adapted), rid, adapted.context.paper_id)

    def resume_stream(self, run_id: str) -> Iterator[Dict[str, Any]]:
        if self.checkpointer is None or not (self.graph.get_state(self._config(run_id)) or None):
            yield {"event": "run.error", "data": {"run_id": run_id, "code": "RunNotFound", "message": "no such run"}}
            return
        snap = self.graph.get_state(self._config(run_id))
        if not snap.values:
            yield {"event": "run.error", "data": {"run_id": run_id, "code": "RunNotFound", "message": "no such run"}}
            return
        yield {"event": "run.started", "data": {"run_id": run_id, "resumed": True,
                                                "paper_id": snap.values["paper_context"]["paper_id"]}}
        yield from self._pump(None, run_id, snap.values["paper_context"]["paper_id"])

    def _pump(self, graph_input: Optional[dict], run_id: str, paper_id: str) -> Iterator[Dict[str, Any]]:
        solved: Optional[dict] = None
        t0 = time.monotonic()
        try:
            # subgraphs=True: the per-question progress events come from inside the sub-graph
            for namespace, mode, chunk in self.graph.stream(
                graph_input, self._config(run_id), stream_mode=["custom", "updates"], subgraphs=True
            ):
                if mode == "custom":
                    name = chunk.get("event", "progress")
                    yield {"event": name, "data": {"run_id": run_id, **{k: v for k, v in chunk.items() if k != "event"}}}
                elif mode == "updates" and not namespace and isinstance(chunk, dict):
                    sp = (chunk.get("build_solved_paper") or {}).get("solved_paper")
                    if sp:
                        solved = sp
                    sp2 = (chunk.get("assemble_output") or {}).get("solved_paper")
                    if sp2:
                        solved = sp2
        except GeneratorExit:  # client went away: stop quietly, the checkpoint keeps the progress
            log.info("stream for run %s closed by client", run_id)
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("run %s failed", run_id)
            yield {"event": "run.error", "data": {"run_id": run_id, "code": type(exc).__name__, "message": str(exc)[:300]}}
            return
        if solved is None:
            yield {"event": "run.error", "data": {"run_id": run_id, "code": "NoResult", "message": "graph ended without a solved paper"}}
            return
        yield {"event": "run.finished", "data": {
            "run_id": run_id, "paper_id": paper_id, "outcome": solved["outcome"], "status": solved["status"],
            "summary": solved["summary"], "elapsed_s": round(time.monotonic() - t0, 2),
            "result_url": f"/api/v1/solutions/{paper_id}",
        }}
