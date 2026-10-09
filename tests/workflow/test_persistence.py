"""Knowledge base, paper bookkeeping and checkpointing against a real SQL database.

Runs on SQLite always (fast, but SQLite is NOT PostgreSQL - treat it only as
a logic check) and on real PostgreSQL when TEST_DATABASE_URL points at a
database whose name contains 'test'. Those cases are marked ``db``; without the
variable they are reported as SKIPPED, never as passed.
"""
import os
import threading

import pytest
from sqlalchemy import create_engine, inspect, func
from sqlalchemy.orm import sessionmaker

from app.store.connection import make_session_scope
from app.store.knowledge import VerifiedSolutionTable
from app.store.models import Base, PaperTable, SolvedQuestionTable
from app.workflow.adapter import adapt_paper
from app.workflow.deps import WorkflowDeps
from app.workflow.factory import make_checkpointer, normalize_db_url
from app.workflow.kb import SqlKnowledgeBase, SqlPaperRepository
from app.workflow.schemas import ConfidenceAssessment, VerificationReport
from app.workflow.service import AnswerBookService
from tests.workflow.helpers import MIXED, ScriptedLLM, SimulatedCrash, good_draft, make_paper, q, settings

PG_URL = os.getenv("TEST_DATABASE_URL", "")

BACKENDS = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param("postgres", id="postgres", marks=[
        pytest.mark.db,
        pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set: real PostgreSQL NOT exercised"),
    ]),
]


@pytest.fixture(params=BACKENDS)
def env(request, tmp_path):
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path / 'kb.db'}"
    else:
        assert "test" in PG_URL.rsplit("/", 1)[-1].lower(), "refusing to drop tables in a non-test database"
        url = normalize_db_url(PG_URL)
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    scope = make_session_scope(sessionmaker(bind=engine, autoflush=False))
    yield type("Env", (), dict(engine=engine, scope=scope, kb=SqlKnowledgeBase(scope), papers=SqlPaperRepository(scope),
                               url=url, backend=request.param))
    Base.metadata.drop_all(engine)
    engine.dispose()


def count(env, table):
    with env.scope() as s:
        return s.query(func.count()).select_from(table).scalar()


def parts(paper=None, **draft_over):
    ad = adapt_paper(paper or make_paper([q("1", "mcq", marks=1)]))
    item = ad.items[0]
    draft = good_draft(item, **draft_over)
    a = ConfidenceAssessment(self_confidence=0.95, evidence="cross_check", evidence_cap=0.9, verification_state="passed",
                             verification_score=1.0, final_confidence=0.9, verified=True)
    rep = VerificationReport(state="passed", evidence="cross_check", score=1.0, verifier_model="v")
    sol = {"question_number": item.number, "answer": draft.answer, "steps": draft.steps, "mark_split": draft.mark_split,
           "common_mistakes": draft.common_mistakes, "confidence": 0.9, "verified_by": "cross_check", "needs_teacher_check": False}
    return item, ad.context, draft, sol, a, rep


# ------------------------------------------------------------------ knowledge base
def test_schema_is_additive_and_legacy_tables_are_unchanged(env):
    insp = inspect(env.engine)
    assert {"papers", "solved_questions", "verified_solutions"} <= set(insp.get_table_names())
    legacy = {c["name"] for c in insp.get_columns("solved_questions")}
    assert legacy == {c.name for c in SolvedQuestionTable.__table__.columns}
    uniques = [i for i in insp.get_indexes("verified_solutions") if i["unique"]]
    assert any(i["column_names"] == ["context_key"] for i in uniques)


def test_store_then_find_roundtrip(env):
    item, ctx, draft, sol, a, rep = parts()
    out = env.kb.store_verified(item, ctx, draft, sol, a, rep)
    assert out.action == "inserted" and out.kb_id
    got = env.kb.find_candidates(item.signature)
    assert len(got) == 1 and got[0].id == out.kb_id
    assert got[0].verification_status == "verified" and got[0].evidence == "cross_check"
    assert got[0].source_paper_id == ctx.paper_id and got[0].options == item.options
    assert got[0].draft["chosen_option"] == "B" and env.kb.find_candidates("0" * 64) == []


def test_store_is_idempotent_and_never_downgrades(env):
    item, ctx, draft, sol, a, rep = parts()
    first = env.kb.store_verified(item, ctx, draft, sol, a, rep)
    again = env.kb.store_verified(item, ctx, draft, sol, a, rep)
    assert again.kb_id == first.kb_id and again.action == "updated" and count(env, VerifiedSolutionTable) == 1
    weaker = a.model_copy(update={"evidence": "llm_review", "final_confidence": 0.5})
    kept = env.kb.store_verified(item, ctx, draft, sol, weaker, rep)
    assert kept.action == "kept_existing"
    assert env.kb.find_candidates(item.signature)[0].evidence == "cross_check"
    stronger = a.model_copy(update={"evidence": "symbolic", "final_confidence": 0.95})
    assert env.kb.store_verified(item, ctx, draft, sol, stronger, rep).action == "updated"
    assert env.kb.find_candidates(item.signature)[0].evidence == "symbolic" and count(env, VerifiedSolutionTable) == 1


def test_different_context_gets_a_different_row(env):
    for subject in ("Mathematics", "Physics"):
        item, ctx, draft, sol, a, rep = parts(make_paper([q("1", "mcq", marks=1)], subject=subject))
        env.kb.store_verified(item, ctx, draft, sol, a, rep)
    assert count(env, VerifiedSolutionTable) == 2


def test_failed_write_rolls_back_completely(env):
    item, ctx, draft, sol, a, rep = parts()
    sol["answer"] = object()  # not JSON serialisable -> the INSERT must fail
    with pytest.raises(Exception):
        env.kb.store_verified(item, ctx, draft, sol, a, rep)
    assert count(env, VerifiedSolutionTable) == 0
    item, ctx, draft, sol, a, rep = parts()
    assert env.kb.store_verified(item, ctx, draft, sol, a, rep).action == "inserted"  # the table is still usable


@pytest.mark.db
@pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set: real PostgreSQL NOT exercised")
def test_concurrent_writers_produce_exactly_one_row_on_postgres(env):
    if env.backend != "postgres":
        pytest.skip("postgres-only")
    item, ctx, draft, sol, a, rep = parts()
    results, errors = [], []

    def work():
        try:
            results.append(env.kb.store_verified(item, ctx, draft, sol, a, rep))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors
    assert count(env, VerifiedSolutionTable) == 1
    assert sum(r.action == "inserted" for r in results) == 1 and len({r.kb_id for r in results}) == 1


# ------------------------------------------------------------------ paper bookkeeping
def test_paper_repository_lifecycle_and_idempotent_checkpoints(env, tmp_path):
    llm = ScriptedLLM()
    svc = AnswerBookService(WorkflowDeps(kb=env.kb, papers=env.papers, llm=llm, settings=settings(tmp_path)))
    paper = make_paper(MIXED())
    for _ in range(2):
        solved = svc.run(paper)
    assert solved.outcome == "complete"
    with env.scope() as s:
        row = s.query(PaperTable).filter_by(paper_id=paper.paper_id).one()
        assert (row.status, row.total_questions, row.subject, row.board) == ("ready", 4, "Mathematics", "CBSE")
    assert count(env, SolvedQuestionTable) == 4  # 2 runs, still 4 rows
    assert count(env, PaperTable) == 1 and count(env, VerifiedSolutionTable) == 4
    with env.scope() as s:
        r = s.query(SolvedQuestionTable).filter_by(question_number="1").one()
        assert r.needs_teacher_check is False and r.verified_by.startswith("cache:kb:")  # latest run was served from cache


def test_second_run_is_served_from_the_database_cache(env, tmp_path):
    llm = ScriptedLLM()
    svc = AnswerBookService(WorkflowDeps(kb=env.kb, papers=env.papers, llm=llm, settings=settings(tmp_path)))
    svc.run(make_paper(MIXED()))
    n = len(llm.draft_calls)
    again = svc.run(make_paper(MIXED(), paper_id="pap_second", fingerprint="9" * 64))
    assert len(llm.draft_calls) == n and again.summary.cached == 4 and all(r.cache.kb_id for r in again.results)


def test_partial_paper_is_never_marked_ready_in_the_database(env, tmp_path):
    from app.workflow.llm import LLMUnavailableError

    llm = ScriptedLLM(drafts={"3": [LLMUnavailableError("down")]})
    svc = AnswerBookService(WorkflowDeps(kb=env.kb, papers=env.papers, llm=llm, settings=settings(tmp_path)))
    solved = svc.run(make_paper(MIXED()))
    with env.scope() as s:
        assert s.query(PaperTable).one().status == "partial" == solved.status
    assert count(env, SolvedQuestionTable) == 3  # the failed question has no row, so a re-run retries it


def test_unverified_answers_are_checkpointed_but_not_added_to_the_knowledge_base(env, tmp_path):
    llm = ScriptedLLM(drafts={"3": [lambda it: good_draft(it, self_confidence=0.1)]})
    svc = AnswerBookService(WorkflowDeps(kb=env.kb, papers=env.papers, llm=llm, settings=settings(tmp_path)))
    svc.run(make_paper(MIXED()))
    assert count(env, VerifiedSolutionTable) == 3
    with env.scope() as s:
        row = s.query(SolvedQuestionTable).filter_by(question_number="3").one()
        assert row.needs_teacher_check is True and row.verified_by == "none"
        assert s.query(PaperTable).one().status == "needs_review"


def test_database_outage_is_surfaced_not_masked(env, tmp_path):
    class Down:
        def find_candidates(self, sig):
            raise ConnectionError("db down")

        def store_verified(self, *a, **k):
            raise ConnectionError("db down")

    class DownPapers:
        def begin(self, *a): raise ConnectionError("db down")
        def save_question(self, *a): raise ConnectionError("db down")
        def finish(self, *a): raise ConnectionError("db down")

    svc = AnswerBookService(WorkflowDeps(kb=Down(), papers=DownPapers(), llm=ScriptedLLM(), settings=settings(tmp_path)))
    solved = svc.run(make_paper([q("3")]))
    codes = {d.code for d in solved.diagnostics} | {d.code for r in solved.results for d in r.diagnostics}
    assert {"persistence_failed", "cache_unavailable", "storage_failed", "checkpoint_failed"} <= codes
    assert solved.results[0].stored_in_kb is False


# ------------------------------------------------------------------ checkpointer
@pytest.mark.db
@pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set: real PostgreSQL NOT exercised")
def test_interrupted_run_resumes_from_postgres_checkpoints_in_a_fresh_service(env, tmp_path):
    if env.backend != "postgres":
        pytest.skip("postgres-only")
    saver = make_checkpointer(PG_URL)
    llm1 = ScriptedLLM(drafts={"3": [SimulatedCrash("killed")]})
    svc1 = AnswerBookService(WorkflowDeps(kb=env.kb, papers=env.papers, llm=llm1, settings=settings(tmp_path, max_concurrency=1)), saver)
    paper = make_paper([q("1"), q("2"), q("3"), q("4")])
    with pytest.raises(SimulatedCrash):
        svc1.run(paper, run_id="pg-run-1")
    # "restart": a brand-new service and a brand-new checkpointer connection pool, same database
    llm2 = ScriptedLLM()
    svc2 = AnswerBookService(WorkflowDeps(kb=env.kb, papers=env.papers, llm=llm2, settings=settings(tmp_path)), make_checkpointer(PG_URL))
    solved = svc2.resume("pg-run-1")
    assert solved.outcome == "complete" and [r.question["number"] for r in solved.results] == ["1", "2", "3", "4"]
    assert "1" not in llm2.draft_calls and "2" not in llm2.draft_calls  # finished questions were not redone
    assert "3" in llm2.draft_calls
