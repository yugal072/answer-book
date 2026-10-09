"""Deterministic test doubles and builders for the workflow tests.

``ScriptedLLM`` implements the LLMGateway protocol. It is a TEST DOUBLE for the
model: every graph-routing test below exercises the real compiled LangGraph,
real verifier, real confidence logic, real cache policy - only the provider
call is scripted.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

from app.models.loaders_models import Paper, Question
from app.workflow.deps import WorkflowDeps
from app.workflow.kb import InMemoryKnowledgeBase, InMemoryPaperRepository
from app.workflow.llm import LLMUnavailableError
from app.workflow.prompts import RepairContext
from app.workflow.schemas import BlindMcq, BlindNumeric, PaperContext, ReviewCheck, ReviewResult, SolverDraft, WorkItem
from app.workflow.service import AnswerBookService
from app.workflow.settings import PROMPT_VERSION, WorkflowSettings


class SimulatedCrash(BaseException):
    """Stands in for a process dying mid-run (deliberately not an Exception)."""


Step = Union[SolverDraft, Exception, BaseException, Callable[[WorkItem], Any]]


def good_draft(item: WorkItem, **over: Any) -> SolverDraft:
    base: Dict[str, Any] = dict(
        strategy=item.type,
        answer=f"answer to {item.number}",
        steps=["step one", "step two", "step three"],
        mark_split=[{"marks": float(item.marks), "for": "complete correct solution"}],
        common_mistakes=[{"wrong": "w", "why": "y"}],
        self_confidence=0.95,
        prompt_version=PROMPT_VERSION,
        model="fake-solver",
    )
    if item.type == "mcq":
        base.update(chosen_option="B", answer="(B) second")
    if item.type == "numerical":
        base.update(final_value=4.0, unit="ohm", arithmetic_expression="(6*12)/(6+12)", answer="4 ohm")
    base.update(over)
    return SolverDraft(**base)


class ScriptedLLM:
    """Per-question-number scripts; each list is consumed in order, the last entry repeats."""

    solver_model = "fake-solver"
    verifier_model = "fake-verifier"

    def __init__(
        self,
        drafts: Optional[Dict[str, List[Step]]] = None,
        blind_mcq: Union[str, Dict[str, Union[str, Exception]]] = "B",
        blind_num: Union[float, Dict[str, Union[float, Exception]]] = 4.0,
        review: Optional[Dict[str, Union[ReviewResult, Exception]]] = None,
        delay: float = 0.0,
    ) -> None:
        self.drafts = drafts or {}
        self.blind_mcq_cfg, self.blind_num_cfg = blind_mcq, blind_num
        self.review_cfg = review or {}
        self.delay = delay
        self._pos: Dict[str, int] = {}
        self._lock = threading.Lock()
        self.draft_calls: List[str] = []
        self.repair_calls: List[str] = []
        self.blind_calls: List[str] = []
        self.review_calls: List[str] = []
        self.active = 0
        self.peak = 0

    def _enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        if self.delay:
            time.sleep(self.delay)

    def _exit(self) -> None:
        with self._lock:
            self.active -= 1

    def draft(self, item: WorkItem, ctx: PaperContext, repair: Optional[RepairContext] = None) -> SolverDraft:
        self._enter()
        try:
            with self._lock:
                self.draft_calls.append(item.number)
                if repair is not None:
                    self.repair_calls.append(item.number)
                steps = self.drafts.get(item.number)
                if not steps:
                    step: Step = lambda it: good_draft(it)  # noqa: E731
                else:
                    i = self._pos.get(item.number, 0)
                    step = steps[min(i, len(steps) - 1)]
                    self._pos[item.number] = i + 1
            if isinstance(step, (Exception, BaseException)):
                raise step
            return step(item) if callable(step) else step
        finally:
            self._exit()

    def _pick(self, cfg: Any, number: str) -> Any:
        return cfg.get(number, cfg.get("*")) if isinstance(cfg, dict) else cfg

    def blind_solve_mcq(self, item: WorkItem, ctx: PaperContext) -> BlindMcq:
        self._enter()
        try:
            self.blind_calls.append(item.number)
            v = self._pick(self.blind_mcq_cfg, item.number)
            if isinstance(v, Exception):
                raise v
            return BlindMcq(chosen_option=v or "B")
        finally:
            self._exit()

    def blind_solve_numeric(self, item: WorkItem, ctx: PaperContext) -> BlindNumeric:
        self._enter()
        try:
            self.blind_calls.append(item.number)
            v = self._pick(self.blind_num_cfg, item.number)
            if isinstance(v, Exception):
                raise v
            return BlindNumeric(final_value=4.0 if v is None else v)
        finally:
            self._exit()

    def review(self, item: WorkItem, ctx: PaperContext, draft: SolverDraft) -> ReviewResult:
        self._enter()
        try:
            self.review_calls.append(item.number)
            v = self.review_cfg.get(item.number)
            if isinstance(v, Exception):
                raise v
            if v is not None:
                return v
            return passing_review()
        finally:
            self._exit()


def passing_review() -> ReviewResult:
    from app.workflow.prompts import REVIEW_CHECKS

    return ReviewResult(checks=[ReviewCheck(name=n, passed=True, note="ok") for n in REVIEW_CHECKS])


def failing_review(failing: str = "factually_correct", note: str = "wrong fact") -> ReviewResult:
    from app.workflow.prompts import REVIEW_CHECKS

    return ReviewResult(
        checks=[ReviewCheck(name=n, passed=(n != failing), note=note if n == failing else "ok") for n in REVIEW_CHECKS],
        issues=[note],
    )


OPTS = ["(A) first", "(B) second", "(C) third", "(D) fourth"]


def q(number: str, qtype: str = "short", text: Optional[str] = None, marks: int = 2, **kw: Any) -> Question:
    opts = kw.pop("options", OPTS if qtype == "mcq" else None)
    return Question(number=number, text=text or f"Question text number {number} for type {qtype}?", marks=marks,
                    type=qtype, options=opts, section=kw.pop("section", "A"), **kw)


def make_paper(questions: List[Question], paper_id: str = "pap_test0001", **kw: Any) -> Paper:
    return Paper(paper_id=paper_id, fingerprint=kw.pop("fingerprint", "f" * 64), status="solving",
                 subject=kw.pop("subject", "Mathematics"), class_name=kw.pop("class_name", "Class 9"),
                 board=kw.pop("board", "CBSE"), questions=questions, total_questions=len(questions),
                 total_marks=sum(x.marks or 0 for x in questions), sections=["A"], **kw)


def settings(tmp_path, **over: Any) -> WorkflowSettings:
    base = dict(output_dir=tmp_path / "out", max_repair_attempts=2, max_concurrency=4)
    base.update(over)
    return WorkflowSettings(**base)


def make_service(llm: ScriptedLLM, tmp_path, *, kb=None, papers=None, checkpointer=None, **over: Any):
    kb = kb if kb is not None else InMemoryKnowledgeBase()
    papers = papers if papers is not None else InMemoryPaperRepository()
    svc = AnswerBookService(WorkflowDeps(kb=kb, papers=papers, llm=llm, settings=settings(tmp_path, **over)), checkpointer)
    return svc, kb, papers


MIXED = lambda: [  # noqa: E731
    q("1", "mcq", marks=1),
    q("2", "numerical", marks=3, text="Calculate the resistance of 6 ohm and 12 ohm resistors in parallel."),
    q("3", "short", marks=2),
    q("4", "long", marks=5),
]
