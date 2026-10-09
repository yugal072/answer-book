"""The real LangChainGateway class with a fake chat model (no network).

Covers: transient retry/back-off, non-retry of permanent errors, malformed
output handling, rate limiter behaviour, and that secrets never reach state.
"""
import json
from types import SimpleNamespace

import pytest

from app.workflow.llm import (
    LangChainGateway, LLMConfigError, LLMOutputError, LLMUnavailableError, TokenBudgetLimiter, is_transient,
)
from app.workflow.schemas import McqDraft, NumericalDraft, TextDraft, WorkItem
from app.workflow.adapter import adapt_paper
from tests.workflow.helpers import make_paper, q, settings, make_service
from app.workflow.kb import InMemoryKnowledgeBase, InMemoryPaperRepository
from app.workflow.deps import WorkflowDeps
from app.workflow.service import AnswerBookService

MCQ_PAYLOAD = dict(answer="(B) second", steps=["a"], mark_split=[{"marks": 1, "for": "x"}], common_mistakes=[],
                   self_confidence=0.9, chosen_option="B")


class HTTPErr(Exception):
    def __init__(self, status, msg="err", headers=None):
        super().__init__(msg)
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


class APITimeoutError(Exception):
    pass


class FakeChat:
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def with_structured_output(self, schema, method=None, include_raw=False):
        outer = self

        class Chain:
            def invoke(self, messages):
                outer.calls += 1
                step = outer.script.pop(0) if len(outer.script) > 1 else outer.script[0]
                if isinstance(step, Exception):
                    raise step
                raw = SimpleNamespace(usage_metadata={"total_tokens": 100})
                if step == "BAD":
                    return {"raw": raw, "parsed": None, "parsing_error": ValueError("schema mismatch")}
                return {"raw": raw, "parsed": schema.model_validate(step), "parsing_error": None}

        return Chain()


class _Clock:
    """Fake time shared by the gateway and its limiter so waits advance instead of spinning."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def gw(script, tmp_path, sleeps=None, **over):
    clock = _Clock()
    chat = FakeChat(script)
    lim = TokenBudgetLimiter(10**9, clock=clock.now, sleep=clock.sleep)
    g = LangChainGateway(settings(tmp_path, **over), "key", limiter=lim, model_factory=lambda name: chat, sleep=clock.sleep)
    return g, chat, clock.sleeps


@pytest.fixture()
def mcq():
    ad = adapt_paper(make_paper([q("1", "mcq", marks=1)]))
    return ad.items[0], ad.context


def test_success_maps_to_solver_draft(tmp_path, mcq):
    g, chat, _ = gw([MCQ_PAYLOAD], tmp_path)
    d = g.draft(*mcq)
    assert d.strategy == "mcq" and d.chosen_option == "B" and d.mark_split == [{"marks": 1.0, "for": "x"}]
    assert d.model == "fake-" or d.model == g.solver_model


def test_rate_limit_is_retried_and_retry_after_is_honoured(tmp_path, mcq):
    g, chat, sleeps = gw([HTTPErr(429, headers={"retry-after": "3"}), MCQ_PAYLOAD], tmp_path)
    g.draft(*mcq)
    assert chat.calls == 2 and any(s >= 3.0 for s in sleeps)


def test_try_again_in_hint_in_message_is_honoured(tmp_path, mcq):
    g, chat, sleeps = gw([HTTPErr(429, "Rate limit reached. Please try again in 7.5s."), MCQ_PAYLOAD], tmp_path)
    g.draft(*mcq)
    assert any(s >= 7.5 for s in sleeps)


def test_transient_retries_are_bounded(tmp_path, mcq):
    g, chat, sleeps = gw([HTTPErr(503)], tmp_path, llm_transient_retries=2)
    with pytest.raises(LLMUnavailableError):
        g.draft(*mcq)
    assert chat.calls == 3 and len(sleeps) == 2


def test_rate_limits_get_their_own_larger_patient_budget(tmp_path, mcq):
    # 6 consecutive 429s exceed the generic transient budget (2) but not the rate-limit budget (8)
    g, chat, sleeps = gw([HTTPErr(429)] * 6 + [MCQ_PAYLOAD], tmp_path, llm_transient_retries=2, llm_rate_limit_retries=8)
    assert g.draft(*mcq).chosen_option == "B" and chat.calls == 7
    assert all(s >= 5.0 for s in sleeps)  # never hammers the provider faster than every 5 s


def test_rate_limit_retries_are_still_bounded(tmp_path, mcq):
    g, chat, _ = gw([HTTPErr(429)], tmp_path, llm_rate_limit_retries=3)
    with pytest.raises(LLMUnavailableError):
        g.draft(*mcq)
    assert chat.calls == 4


def test_timeouts_and_connection_errors_are_transient(tmp_path, mcq):
    assert is_transient(APITimeoutError()) and not is_transient(ValueError("x"))
    g, chat, _ = gw([APITimeoutError(), MCQ_PAYLOAD], tmp_path)
    g.draft(*mcq)
    assert chat.calls == 2


@pytest.mark.parametrize("status,exc", [(400, LLMUnavailableError), (404, LLMUnavailableError), (401, LLMConfigError), (403, LLMConfigError)])
def test_permanent_errors_are_not_retried(tmp_path, mcq, status, exc):
    g, chat, sleeps = gw([HTTPErr(status)], tmp_path)
    with pytest.raises(exc):
        g.draft(*mcq)
    assert chat.calls == 1 and sleeps == []


def test_unknown_exceptions_are_not_retried(tmp_path, mcq):
    g, chat, _ = gw([KeyError("bug")], tmp_path)
    with pytest.raises(LLMUnavailableError):
        g.draft(*mcq)
    assert chat.calls == 1


def test_malformed_output_gets_one_reparse_then_fails_loudly(tmp_path, mcq):
    g, chat, _ = gw(["BAD"], tmp_path, llm_output_retries=1)
    with pytest.raises(LLMOutputError):
        g.draft(*mcq)
    assert chat.calls == 2


def test_malformed_then_valid_recovers(tmp_path, mcq):
    g, chat, _ = gw(["BAD", MCQ_PAYLOAD], tmp_path)
    assert g.draft(*mcq).chosen_option == "B" and chat.calls == 2


def test_provider_json_validation_400_counts_as_output_failure(tmp_path, mcq):
    g, chat, _ = gw([HTTPErr(400, "Failed to validate JSON. failed_generation: ..."), MCQ_PAYLOAD], tmp_path)
    assert g.draft(*mcq).chosen_option == "B"


def test_schema_rejects_out_of_contract_values():
    with pytest.raises(Exception):
        McqDraft.model_validate({**MCQ_PAYLOAD, "self_confidence": 7})
    with pytest.raises(Exception):
        McqDraft.model_validate({**MCQ_PAYLOAD, "steps": ["  "]})
    with pytest.raises(Exception):
        McqDraft.model_validate({**MCQ_PAYLOAD, "mark_split": []})
    with pytest.raises(Exception):
        NumericalDraft.model_validate({k: v for k, v in MCQ_PAYLOAD.items() if k != "chosen_option"})  # no final_value
    TextDraft.model_validate({k: v for k, v in MCQ_PAYLOAD.items() if k != "chosen_option"})


def test_missing_api_key_is_a_config_error():
    with pytest.raises(LLMConfigError):
        LangChainGateway(settings(__import__("pathlib").Path("/tmp")), "")


def test_api_key_never_enters_results_or_checkpoints(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    secret = "gsk_SECRET_DO_NOT_LEAK"
    chat = FakeChat([{**MCQ_PAYLOAD}])
    # blind solve + draft both go through this fake; MCQ blind uses BlindMcq schema, so script both shapes
    class Multi(FakeChat):
        def with_structured_output(self, schema, method=None, include_raw=False):
            payload = {"chosen_option": "B", "reasoning": "r"} if schema.__name__ == "BlindMcq" else MCQ_PAYLOAD
            return FakeChat([payload]).with_structured_output(schema, method, include_raw)

    g = LangChainGateway(settings(tmp_path), secret, model_factory=lambda n: Multi([]), sleep=lambda s: None)
    saver = InMemorySaver()
    svc = AnswerBookService(WorkflowDeps(kb=InMemoryKnowledgeBase(), papers=InMemoryPaperRepository(), llm=g, settings=settings(tmp_path)), saver)
    solved = svc.run(make_paper([q("1", "mcq", marks=1)]), run_id="r1")
    assert solved.results[0].status == "verified"
    assert secret not in json.dumps(solved.model_dump(mode="json"))
    assert secret not in repr(saver.storage) and secret.encode() not in repr(saver.blobs).encode()


# ----------------------------------------------------------------- limiter
class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_limiter_blocks_until_window_frees_up():
    c = Clock()
    lim = TokenBudgetLimiter(1000, clock=c.now, sleep=c.sleep)
    lim.acquire(600)
    lim.acquire(300)
    assert c.t == 0
    lim.acquire(600)  # 1500 > 1000 -> must wait for the first reservation to age out
    assert c.t >= 60 - 1e-6 and c.sleeps


def test_limiter_settle_returns_unused_budget():
    c = Clock()
    lim = TokenBudgetLimiter(1000, clock=c.now, sleep=c.sleep)
    ev = lim.acquire(900)
    lim.settle(ev, 100)
    lim.acquire(800)
    assert c.t == 0


def test_limiter_penalize_blocks_all_callers():
    c = Clock()
    lim = TokenBudgetLimiter(1000, clock=c.now, sleep=c.sleep)
    lim.penalize(12)
    lim.acquire(10)
    assert c.t >= 12


def test_oversized_request_is_clamped_not_deadlocked():
    c = Clock()
    lim = TokenBudgetLimiter(500, clock=c.now, sleep=c.sleep)
    lim.acquire(10_000)
    assert c.t == 0
