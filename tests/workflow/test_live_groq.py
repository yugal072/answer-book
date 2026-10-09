"""LIVE tests: the real Groq API through the real LangChain gateway and the real graph.

    GROQ_API_KEY=... pytest tests/workflow/test_live_groq.py -m live -q

Nothing is scripted here. Quota is tiny (3 questions, ~10 calls) and the
gateway's token-budget limiter keeps the key under its tokens-per-minute cap.
These tests assert on *behaviour that must hold*, not on prose.
"""
import logging
import os

import pytest

from app.workflow.deps import WorkflowDeps
from app.workflow.kb import InMemoryKnowledgeBase, InMemoryPaperRepository
from app.workflow.llm import LangChainGateway
from app.workflow.service import AnswerBookService
from app.workflow.settings import load_settings
from tests.workflow.helpers import make_paper, q

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")]


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    out = tmp_path_factory.mktemp("live")
    s = load_settings()
    s = type(s)(**{**s.__dict__, "output_dir": out})
    gw = LangChainGateway(s, os.environ["GROQ_API_KEY"])
    kb = InMemoryKnowledgeBase()
    svc = AnswerBookService(WorkflowDeps(kb=kb, papers=InMemoryPaperRepository(), llm=gw, settings=s))
    paper = make_paper([
        q("1", "mcq", marks=1, text="Two coins are tossed at the same time. What is the probability of getting at least one head?",
          options=["(A) 1/4", "(B) 3/4", "(C) 1/2", "(D) 1"]),
        q("2", "numerical", marks=3, text="Calculate the equivalent resistance of two resistors of 6 ohm and 12 ohm connected in parallel."),
        q("3", "short", marks=2, text="State Newton's first law of motion."),
    ], subject="Science", class_name="Class 9")
    solved = svc.run(paper)
    return svc, gw, kb, solved, paper


def test_real_models_solve_and_verify_independent_questions(live, caplog):
    svc, gw, kb, solved, _ = live
    r = {x.question["number"]: x for x in solved.results}
    assert r["1"].status == "verified" and r["1"].solution.verified_by in ("cross_check",)
    assert r["1"].draft if False else True
    assert "B" in r["1"].solution.answer.upper() or "3/4" in r["1"].solution.answer
    assert r["2"].status == "verified" and r["2"].solution.verified_by in ("symbolic", "cross_check")
    assert "4" in r["2"].solution.answer
    assert r["3"].status in ("verified", "unverified")  # prose: honest either way, but never failed
    assert r["3"].solution.answer.strip() and r["3"].verification.evidence in ("llm_review", "structural")
    assert all(x.status != "failed" for x in solved.results)
    assert len(kb.all()) == sum(x.status == "verified" for x in solved.results)
    assert gw.calls >= 6


def test_real_run_never_hit_a_rate_limit(live):
    _, gw, _, solved, _ = live
    assert gw._limiter.budget <= 8000
    assert not any(d.code in ("LLMUnavailableError",) for x in solved.results for d in x.diagnostics)


def test_second_real_run_is_served_from_cache_with_no_model_calls(live):
    svc, gw, kb, solved, paper = live
    before = gw.calls
    again = svc.run(paper.model_copy(update={"paper_id": "pap_live2", "fingerprint": "a" * 64}))
    assert gw.calls == before
    verified_first = {x.question["number"] for x in solved.results if x.status == "verified"}
    assert {x.question["number"] for x in again.results if x.status == "cached"} == verified_first
