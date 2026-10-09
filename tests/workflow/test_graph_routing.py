"""Routing/branch tests against the REAL compiled graphs (provider calls scripted)."""
import pytest

from app.workflow.cache_policy import StoredSolution
from app.workflow.llm import LLMUnavailableError
from app.workflow.schemas import QuestionResult
from app.workflow.paper_graph import paper_outcome
from tests.workflow.helpers import (
    MIXED, ScriptedLLM, SimulatedCrash, failing_review, good_draft, make_paper, make_service, q,
)


def by_number(solved):
    return {r.question["number"]: r for r in solved.results}


# ----------------------------------------------------------------- happy path
def test_all_four_types_verified_and_stored(tmp_path):
    llm = ScriptedLLM()
    svc, kb, papers = make_service(llm, tmp_path)
    solved = svc.run(make_paper(MIXED()))
    r = by_number(solved)
    assert solved.outcome == "complete" and solved.status == "ready"
    assert [x.question["number"] for x in solved.results] == ["1", "2", "3", "4"]
    assert {k: v.status for k, v in r.items()} == {"1": "verified", "2": "verified", "3": "verified", "4": "verified"}
    # evidence actually obtained, per type
    assert r["1"].solution.verified_by == "cross_check"
    assert r["2"].solution.verified_by == "symbolic"
    assert r["3"].solution.verified_by == "llm_review"
    assert llm.blind_calls == ["1", "2"] and llm.review_calls == ["3", "4"]
    assert all(x.stored_in_kb and x.origin == "generated" for x in solved.results)
    assert len(kb.all()) == 4
    assert papers.papers["pap_test0001"]["status"] == "ready"
    assert solved.summary.model_dump() == dict(total=4, verified=4, cached=0, unverified=0, failed=0)
    # contract-compatible list, in paper order, with explicit status
    assert [s["question_number"] for s in solved.solutions] == ["1", "2", "3", "4"]
    assert all(s["needs_teacher_check"] is False for s in solved.solutions)


def test_original_question_context_is_preserved(tmp_path):
    llm = ScriptedLLM()
    svc, *_ = make_service(llm, tmp_path)
    qq = q("7", "short", marks=3, chapter="Ch 2: X", section="B", page=4, choice_group="7")
    solved = svc.run(make_paper([qq]))
    got = solved.results[0].question
    assert (got["marks"], got["chapter"], got["section"], got["page"], got["choice_group"]) == (3, "Ch 2: X", "B", 4, "7")
    assert solved.metadata["subject"] == "Mathematics" and solved.metadata["board"] == "CBSE"


# ----------------------------------------------------------------- cache
def test_second_run_is_served_from_cache_with_zero_llm_calls(tmp_path):
    llm = ScriptedLLM()
    svc, kb, _ = make_service(llm, tmp_path)
    svc.run(make_paper(MIXED()))
    drafts_before = len(llm.draft_calls)
    again = svc.run(make_paper(MIXED(), paper_id="pap_test0002", fingerprint="e" * 64))
    assert len(llm.draft_calls) == drafts_before and len(llm.blind_calls) == 2
    assert all(x.status == "cached" and x.origin == "cache" for x in again.results)
    assert all(x.solution.verified_by.startswith("cache:kb:") for x in again.results)
    assert all(x.cache.accepted and x.cache.kb_id for x in again.results)
    assert again.outcome == "complete" and again.summary.cached == 4
    assert len(kb.all()) == 4  # no duplicate rows


def test_rerun_is_idempotent(tmp_path):
    svc, kb, _ = make_service(ScriptedLLM(), tmp_path)
    for _ in range(3):
        svc.run(make_paper(MIXED()))
    assert len(kb.all()) == 4


@pytest.mark.parametrize("override", [{"subject": "Physics"}, {"board": "ICSE"}, {"class_name": "Class 10"}])
def test_different_educational_context_is_a_cache_miss(tmp_path, override):
    llm = ScriptedLLM()
    svc, kb, _ = make_service(llm, tmp_path)
    svc.run(make_paper([q("3", "short")]))
    n = len(llm.draft_calls)
    solved = svc.run(make_paper([q("3", "short")], paper_id="pap_other", **override))
    assert len(llm.draft_calls) == n + 1
    assert solved.results[0].status == "verified" and solved.results[0].origin == "generated"


def test_stored_row_that_is_not_verified_is_rejected_and_solved_fresh(tmp_path):
    llm = ScriptedLLM()
    svc, kb, _ = make_service(llm, tmp_path)
    svc.run(make_paper([q("3", "short")]))
    row = kb.all()[0]
    key = next(k for k, v in kb._rows.items() if v is row)
    kb._rows[key] = row.model_copy(update={"verification_status": "unverified"})
    solved = svc.run(make_paper([q("3", "short")], paper_id="pap_b"))
    r = solved.results[0]
    assert r.origin == "generated" and r.status == "verified"
    assert not r.cache.accepted and "not verified" in " ".join(r.cache.reasons)
    assert len(llm.draft_calls) == 2


def test_figure_question_is_neither_stored_nor_reused(tmp_path):
    llm = ScriptedLLM()
    svc, kb, _ = make_service(llm, tmp_path)
    p = lambda pid: make_paper([q("5", "short", has_figure=True)], paper_id=pid)  # noqa: E731
    first = svc.run(p("pap_f1"))
    assert first.results[0].status == "verified" and kb.all() == []
    svc.run(p("pap_f2"))
    assert len(llm.draft_calls) == 2


def test_mcq_without_options_is_routed_to_short_strategy_with_warning(tmp_path):
    llm = ScriptedLLM()
    svc, *_ = make_service(llm, tmp_path)
    solved = svc.run(make_paper([q("1", "mcq", options=[])]))
    r = solved.results[0]
    assert r.question["type"] == "short" and r.question["declared_type"] == "mcq"
    assert llm.review_calls == ["1"] and llm.blind_calls == []
    assert any("routed to the 'short' strategy" in d.message for d in r.diagnostics)


# ----------------------------------------------------------------- verification / repair
def bad_numeric(item):
    return good_draft(item, final_value=5.0, answer="5 ohm")  # expression still evaluates to 4


def test_failed_verification_triggers_repair_and_reverification(tmp_path):
    llm = ScriptedLLM(drafts={"2": [bad_numeric, lambda it: good_draft(it)]})
    svc, kb, _ = make_service(llm, tmp_path)
    solved = svc.run(make_paper([MIXED()[1]]))
    r = solved.results[0]
    assert r.status == "verified" and r.solution.verified_by == "symbolic"
    assert llm.repair_calls == ["2"]
    assert [a.attempt for a in r.attempts] == [1, 2]
    assert r.attempts[0].verification["state"] == "failed"  # earlier diagnostics are preserved
    assert any("arithmetic_recomputed" in f["check"] and not f["passed"] for f in r.attempts[0].verification["findings"])
    assert r.attempts[1].verification["state"] == "passed"
    assert len(llm.blind_calls) == 1  # the paid independent solve ran once, not per attempt


def test_retry_exhaustion_is_unverified_and_never_stored(tmp_path):
    llm = ScriptedLLM(drafts={"2": [bad_numeric]})
    svc, kb, _ = make_service(llm, tmp_path, max_repair_attempts=2)
    solved = svc.run(make_paper([MIXED()[1]]))
    r = solved.results[0]
    assert len(llm.draft_calls) == 3 and len(r.attempts) == 3  # 1 draft + 2 bounded repairs
    assert r.status == "unverified" and r.origin == "generated"
    assert r.solution.needs_teacher_check and r.solution.verified_by == "none"
    assert r.solution.confidence <= 0.40
    assert not r.stored_in_kb and kb.all() == []
    assert any(d.code == "retries_exhausted" for d in r.diagnostics)
    assert r.solution.answer == "5 ohm"  # the draft is kept, clearly labelled, not hidden
    assert solved.outcome == "needs_review" and solved.status == "needs_review"


@pytest.mark.parametrize("repairs", [0, 1, 3])
def test_repair_limit_is_configurable(tmp_path, repairs):
    llm = ScriptedLLM(drafts={"2": [bad_numeric]})
    svc, *_ = make_service(llm, tmp_path, max_repair_attempts=repairs)
    svc.run(make_paper([MIXED()[1]]))
    assert len(llm.draft_calls) == 1 + repairs


def test_independent_disagreement_on_mcq_repairs_but_cannot_become_verified(tmp_path):
    """After the verifier's disagreement is fed back, a later 'agreement' is not independent evidence."""
    llm = ScriptedLLM(
        drafts={"1": [lambda it: good_draft(it, chosen_option="C", answer="(C) third"), lambda it: good_draft(it)]},
        blind_mcq="B",
    )
    svc, kb, _ = make_service(llm, tmp_path)
    r = svc.run(make_paper([MIXED()[0]])).results[0]
    assert llm.repair_calls == ["1"]
    assert r.attempts[1].verification["state"] == "passed"
    assert r.status == "unverified" and kb.all() == []
    assert any("evidence" in x for x in r.confidence_assessment.reasons)


def test_numeric_independent_disagreement_blocks_verification(tmp_path):
    llm = ScriptedLLM(blind_num=9.0)
    svc, kb, _ = make_service(llm, tmp_path, max_repair_attempts=1)
    r = svc.run(make_paper([MIXED()[1]])).results[0]
    assert r.status == "unverified" and kb.all() == []
    assert any(f["check"] == "independent_solve" and not f["passed"] for f in r.attempts[-1].verification["findings"])


def test_review_failure_on_text_answer_triggers_repair(tmp_path):
    llm = ScriptedLLM(review={"3": failing_review()})
    svc, kb, _ = make_service(llm, tmp_path, max_repair_attempts=1)
    r = svc.run(make_paper([q("3", "short")])).results[0]
    assert llm.repair_calls == ["3"] and r.status == "unverified" and kb.all() == []


def test_mark_split_that_does_not_sum_is_caught_deterministically(tmp_path):
    bad = lambda it: good_draft(it, mark_split=[{"marks": 1.0, "for": "x"}])  # noqa: E731
    llm = ScriptedLLM(drafts={"3": [bad, lambda it: good_draft(it)]})
    svc, *_ = make_service(llm, tmp_path)
    r = svc.run(make_paper([q("3", "short", marks=2)])).results[0]
    assert r.status == "verified" and len(r.attempts) == 2
    assert llm.review_calls == ["3"]  # model review skipped for the deterministic failure


# ----------------------------------------------------------------- confidence
def test_self_reported_confidence_is_not_evidence(tmp_path):
    llm = ScriptedLLM(drafts={"3": [lambda it: good_draft(it, self_confidence=1.0)]})
    svc, *_ = make_service(llm, tmp_path)
    r = svc.run(make_paper([q("3", "short")])).results[0]
    assert r.confidence_assessment.final_confidence == 0.8  # capped by llm_review evidence, not 1.0


def test_low_self_confidence_blocks_verified_even_when_checks_pass(tmp_path):
    llm = ScriptedLLM(drafts={"3": [lambda it: good_draft(it, self_confidence=0.2)]})
    svc, kb, _ = make_service(llm, tmp_path)
    r = svc.run(make_paper([q("3", "short")])).results[0]
    assert r.verification.state == "passed" and r.status == "unverified"
    assert kb.all() == [] and any("confidence" in x for x in r.confidence_assessment.reasons)


def test_missing_marks_are_flagged_and_not_stored(tmp_path):
    llm = ScriptedLLM()
    svc, kb, _ = make_service(llm, tmp_path)
    r = svc.run(make_paper([q("3", "short", marks=None)])).results[0]
    assert r.question["marks_assumed"] and r.status == "unverified" and kb.all() == []


# ----------------------------------------------------------------- failures are isolated
def test_provider_failure_in_one_question_does_not_affect_others(tmp_path):
    llm = ScriptedLLM(drafts={"3": [LLMUnavailableError("provider down")]})
    svc, *_ = make_service(llm, tmp_path)
    solved = svc.run(make_paper(MIXED()))
    r = by_number(solved)
    assert r["3"].status == "failed" and r["3"].solution.answer == "" and r["3"].solution.needs_teacher_check
    assert any(d.code == "LLMUnavailableError" for d in r["3"].diagnostics)
    assert [r[k].status for k in ("1", "2", "4")] == ["verified"] * 3
    assert solved.outcome == "partial" and solved.status == "partial"
    assert solved.summary.failed == 1


def test_all_questions_failing_is_outcome_failed(tmp_path):
    llm = ScriptedLLM(drafts={"*": []}, )
    llm.drafts = {n: [LLMUnavailableError("down")] for n in ("1", "2")}
    svc, *_ = make_service(llm, tmp_path)
    solved = svc.run(make_paper([q("1"), q("2")]))
    assert solved.outcome == "failed" and solved.summary.failed == 2


def test_unexpected_bug_in_a_branch_is_contained(tmp_path):
    llm = ScriptedLLM(drafts={"3": [RuntimeError("boom")]})
    svc, *_ = make_service(llm, tmp_path)
    solved = svc.run(make_paper(MIXED()))
    r = by_number(solved)
    assert r["3"].status == "failed" and any(d.code == "node_error" for d in r["3"].diagnostics)
    assert solved.summary.verified == 3


def test_verifier_outage_yields_unverified_without_wasting_repairs(tmp_path):
    llm = ScriptedLLM(blind_mcq=LLMUnavailableError("verifier down"))
    svc, kb, _ = make_service(llm, tmp_path)
    r = svc.run(make_paper([MIXED()[0]])).results[0]
    assert r.verification.state == "inconclusive" and r.status == "unverified"
    assert llm.repair_calls == [] and kb.all() == []
    assert r.solution.verified_by == "none"


def test_empty_question_text_fails_that_question_only(tmp_path):
    llm = ScriptedLLM()
    svc, *_ = make_service(llm, tmp_path)
    solved = svc.run(make_paper([q("1", text=" "), q("2")]))
    r = by_number(solved)
    assert r["1"].status == "failed" and r["2"].status == "verified"
    assert llm.draft_calls == ["2"]


def test_cache_database_failure_is_surfaced_and_solving_continues(tmp_path):
    kb = __import__("app.workflow.kb", fromlist=["x"]).InMemoryKnowledgeBase()
    kb.fail_on_find = RuntimeError("db unreachable")
    svc, *_ = make_service(ScriptedLLM(), tmp_path, kb=kb)
    r = svc.run(make_paper([q("3")])).results[0]
    assert r.status == "verified" and r.cache.error and not r.cache.looked_up
    assert any(d.code == "cache_unavailable" for d in r.diagnostics)


def test_storage_failure_is_surfaced_not_hidden(tmp_path):
    kb = __import__("app.workflow.kb", fromlist=["x"]).InMemoryKnowledgeBase()
    kb.fail_on_store = RuntimeError("disk full")
    svc, *_ = make_service(ScriptedLLM(), tmp_path, kb=kb)
    r = svc.run(make_paper([q("3")])).results[0]
    assert r.status == "verified" and r.stored_in_kb is False
    assert any(d.code == "storage_failed" and d.severity == "error" for d in r.diagnostics)


# ----------------------------------------------------------------- fan-out
def test_parallel_fan_out_is_safe_and_ordered(tmp_path):
    n = 14
    qs = [q(str(i), "short") for i in range(1, n + 1)]
    llm = ScriptedLLM(delay=0.03)
    svc, kb, _ = make_service(llm, tmp_path, max_concurrency=5)
    solved = svc.run(make_paper(qs))
    assert llm.peak > 1, "questions were not processed concurrently"
    assert [r.question["number"] for r in solved.results] == [str(i) for i in range(1, n + 1)]
    assert solved.summary.verified == n and len(kb.all()) == n
    assert len({r.item_id for r in solved.results}) == n


def test_duplicate_question_numbers_are_kept_separate(tmp_path):
    qs = [q("12", "short", text="Option one text?", choice_group="12"), q("12", "short", text="Option two text?", choice_group="12")]
    solved = make_service(ScriptedLLM(), tmp_path)[0].run(make_paper(qs))
    assert len(solved.results) == 2 and {r.item_id for r in solved.results} == {"q001", "q002"}


# ----------------------------------------------------------------- aggregation
def _res(i, status):
    from app.workflow.schemas import SolutionOut

    return QuestionResult(item_id=f"q{i}", index=i, question={}, status=status, origin="none",
                          solution=SolutionOut(question_number=str(i)))


@pytest.mark.parametrize("statuses,expected", [
    (["verified", "cached"], "complete"),
    (["verified", "unverified"], "needs_review"),
    (["verified", "failed"], "partial"),
    (["unverified", "failed"], "partial"),
    (["failed", "failed"], "failed"),
    ([], "failed"),
])
def test_paper_outcome_matrix(statuses, expected):
    assert paper_outcome([_res(i, s) for i, s in enumerate(statuses)]) == expected


def test_papers_table_records_non_ready_status_for_mixed_outcomes(tmp_path):
    llm = ScriptedLLM(drafts={"3": [LLMUnavailableError("down")]})
    svc, _, papers = make_service(llm, tmp_path)
    svc.run(make_paper(MIXED()))
    assert papers.papers["pap_test0001"]["status"] == "partial"  # never 'ready' with a failed question


def test_json_export_matches_solved_paper(tmp_path):
    import json

    svc, *_ = make_service(ScriptedLLM(), tmp_path)
    solved = svc.run(make_paper(MIXED()))
    data = json.loads((tmp_path / "out" / "pap_test0001.json").read_text())
    assert data["outcome"] == solved.outcome and len(data["results"]) == 4
    assert [s["question_number"] for s in data["solutions"]] == ["1", "2", "3", "4"]


# ----------------------------------------------------------------- checkpoint / resume
def test_interrupted_run_resumes_from_checkpoint_without_redoing_finished_work(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    llm = ScriptedLLM(drafts={"3": [SimulatedCrash("process died"), lambda it: good_draft(it)]})
    svc, kb, _ = make_service(llm, tmp_path, checkpointer=saver, max_concurrency=1)
    paper = make_paper([q("1"), q("2"), q("3"), q("4")])
    with pytest.raises(SimulatedCrash):
        svc.run(paper, run_id="run-1")
    done_before = list(llm.draft_calls)
    solved = svc.resume("run-1")
    assert solved.outcome == "complete" and [r.question["number"] for r in solved.results] == ["1", "2", "3", "4"]
    assert llm.draft_calls.count("1") == 1 and llm.draft_calls.count("2") == 1, (done_before, llm.draft_calls)
    assert llm.draft_calls.count("3") == 2  # the crashed branch re-ran once
    # a finished run resumes to the same answer without work
    n = len(llm.draft_calls)
    assert svc.resume("run-1").outcome == "complete" and len(llm.draft_calls) == n


def test_resume_unknown_run_raises(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver
    from app.workflow.service import RunNotFound

    svc, *_ = make_service(ScriptedLLM(), tmp_path, checkpointer=InMemorySaver())
    with pytest.raises(RunNotFound):
        svc.resume("nope")


# ----------------------------------------------------------------- graph shape
def test_compiled_graphs_expose_the_diagram_nodes(tmp_path):
    svc, *_ = make_service(ScriptedLLM(), tmp_path)
    paper_nodes = set(svc.graph.get_graph().nodes)
    assert {"validate_prepare", "question_pipeline", "aggregate_results", "build_solved_paper", "assemble_output"} <= paper_nodes
    from app.workflow.question_graph import build_question_graph

    q_nodes = set(build_question_graph(svc.deps).get_graph().nodes)
    assert {"cache_lookup", "retrieve_stored", "solve_mcq", "solve_numerical", "solve_short", "solve_long", "verify",
            "repair", "assess_confidence", "store_verified", "finalize_question"} <= q_nodes
