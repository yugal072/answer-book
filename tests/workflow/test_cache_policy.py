"""Cache-acceptance policy: every rule, in isolation (pure functions)."""
import pytest

from app.workflow import cache_policy as cp
from app.workflow.adapter import adapt_paper
from app.workflow.schemas import PaperContext
from app.workflow.settings import SCHEMA_VERSION, WorkflowSettings
from tests.workflow.helpers import ScriptedLLM, good_draft, make_paper, q, make_service

S = WorkflowSettings()


@pytest.fixture()
def stored_and_item(tmp_path):
    """A genuine verified row produced by a real run, plus the matching item/context."""
    svc, kb, _ = make_service(ScriptedLLM(), tmp_path)
    paper = make_paper([q("1", "mcq", marks=1)])
    svc.run(paper)
    ad = adapt_paper(paper)
    return kb.all()[0], ad.items[0], ad.context


def test_genuine_row_is_accepted(stored_and_item):
    stored, item, ctx = stored_and_item
    assert cp.evaluate_candidate(item, ctx, stored, S) == []
    dec, best = cp.decide(item, ctx, [stored], S)
    assert dec.accepted and best.id == stored.id and dec.kb_id == stored.id


@pytest.mark.parametrize("field,value,needle", [
    ("verification_status", "unverified", "not verified"),
    ("schema_version", "0", "schema_version"),
    ("prompt_version", "ancient", "prompt_version"),
    ("marks", 7, "marks differ"),
    ("board", "ICSE", "board differs"),
    ("class_name", "Class 12", "class differs"),
    ("subject", "Chemistry", "subject differs"),
    ("question_type", "short", "type differs"),
    ("options", ["(A) x", "(B) y"], "options differ"),
    ("options", ["(B) second", "(A) first", "(C) third", "(D) fourth"], "options differ"),  # order matters: letters must map identically
    ("question_text_norm", "something else entirely", "text differs"),
    ("evidence", "structural", "below"),
    ("confidence", 0.2, "confidence"),
    ("has_figure", True, "figure"),
])
def test_each_rule_rejects(stored_and_item, field, value, needle):
    stored, item, ctx = stored_and_item
    bad = stored.model_copy(update={field: value})
    why = cp.evaluate_candidate(item, ctx, bad, S)
    assert why and any(needle in w for w in why), why
    dec, best = cp.decide(item, ctx, [bad], S)
    assert not dec.accepted and best is None and dec.candidates == 1


def test_current_question_with_unknown_marks_never_reuses(stored_and_item):
    stored, item, ctx = stored_and_item
    item = item.model_copy(update={"marks_assumed": True})
    assert any("does not state its marks" in w for w in cp.evaluate_candidate(item, ctx, stored, S))


def test_corrupted_stored_solution_fails_contract_revalidation(stored_and_item):
    stored, item, ctx = stored_and_item
    draft = dict(stored.draft)
    draft["mark_split"] = [{"marks": 99.0, "for": "x"}]  # no longer sums to the question's marks
    why = cp.evaluate_candidate(item, ctx, stored.model_copy(update={"draft": draft}), S)
    assert any("deterministic checks" in w and "mark_split_sum" in w for w in why)


def test_stored_mcq_letter_must_exist_in_current_options(stored_and_item):
    stored, item, ctx = stored_and_item
    draft = dict(stored.draft, chosen_option="Z")
    why = cp.evaluate_candidate(item, ctx, stored.model_copy(update={"draft": draft}), S)
    assert any("option_valid" in w for w in why)


def test_unparseable_stored_draft_is_rejected(stored_and_item):
    stored, item, ctx = stored_and_item
    why = cp.evaluate_candidate(item, ctx, stored.model_copy(update={"draft": {"nonsense": 1}}), S)
    assert why and "output contract" in why[0]


def test_best_candidate_selection_is_deterministic(stored_and_item):
    stored, item, ctx = stored_and_item
    weaker = stored.model_copy(update={"id": 90, "evidence": "cross_check", "confidence": 0.8})
    stronger = stored.model_copy(update={"id": 91, "evidence": "symbolic", "confidence": 0.8})
    # mcq only needs cross_check; symbolic outranks it
    for order in ([weaker, stronger], [stronger, weaker]):
        dec, best = cp.decide(item, ctx, order, S)
        assert best.id == 91
    tie_a = stored.model_copy(update={"id": 5, "updated_at": "7"})
    tie_b = stored.model_copy(update={"id": 6, "updated_at": "7"})
    assert cp.decide(item, ctx, [tie_b, tie_a], S)[1].id == 5


def test_decide_without_candidates():
    item = adapt_paper(make_paper([q("1")])).items[0]
    dec, best = cp.decide(item, PaperContext(paper_id="p", fingerprint="f"), [], S)
    assert not dec.accepted and best is None and dec.looked_up and dec.candidates == 0


def test_context_key_depends_on_context_not_on_paper(stored_and_item):
    stored, item, ctx = stored_and_item
    other_paper_ctx = ctx.model_copy(update={"paper_id": "pap_zzz", "fingerprint": "z" * 64})
    assert cp.context_key(item, ctx) == cp.context_key(item, other_paper_ctx)
    assert cp.context_key(item, ctx) != cp.context_key(item, ctx.model_copy(update={"subject": "Physics"}))
    assert SCHEMA_VERSION == stored.schema_version
