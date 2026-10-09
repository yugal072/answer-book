"""LLM gateway: the only place the workflow talks to a model provider.

Graph nodes depend on the :class:`LLMGateway` protocol; they never see
LangChain, Groq, API keys or HTTP details. :class:`LangChainGateway` is the
production implementation (prompt template | chat model with structured
output). Tests inject a deterministic fake that implements the same protocol.

Failure policy
--------------
* Transient provider failures (HTTP 429, 5xx, timeouts, connection errors)
  are retried a bounded number of times with back-off that honours the
  provider's ``retry-after`` / "try again in" hints.
* Everything else (auth, bad request, unknown) is NOT retried.
* Output that fails schema validation gets ``llm_output_retries`` further
  attempts, then :class:`LLMOutputError`. It is never turned into an answer.
* Any final failure raises a typed :class:`LLMError`; nothing is faked.
"""
from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Protocol, Tuple, Type, TypeVar

from pydantic import BaseModel

from app.workflow.prompts import (
    BLIND_MCQ_PROMPT,
    BLIND_NUMERIC_PROMPT,
    REVIEW_CHECKS,
    REVIEW_PROMPT,
    SOLVE_PROMPT,
    RepairContext,
    base_variables,
    review_variables,
    solve_variables,
)
from app.workflow.schemas import (
    BlindMcq,
    BlindNumeric,
    McqDraft,
    NumericalDraft,
    PaperContext,
    ReviewResult,
    SolverDraft,
    TextDraft,
    WorkItem,
)
from app.workflow.settings import PROMPT_VERSION, WorkflowSettings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


# --------------------------------------------------------------------------- errors
class LLMError(RuntimeError):
    """Base class for all gateway failures."""


class LLMConfigError(LLMError):
    """No usable credentials / model configuration."""


class LLMUnavailableError(LLMError):
    """Provider unreachable, rate-limited or erroring after bounded retries."""


class LLMOutputError(LLMError):
    """Provider answered, but the output did not satisfy the schema."""


# --------------------------------------------------------------------------- protocol
class LLMGateway(Protocol):
    solver_model: str
    verifier_model: str

    def draft(self, item: WorkItem, ctx: PaperContext, repair: Optional[RepairContext] = None) -> SolverDraft: ...
    def blind_solve_mcq(self, item: WorkItem, ctx: PaperContext) -> BlindMcq: ...
    def blind_solve_numeric(self, item: WorkItem, ctx: PaperContext) -> BlindNumeric: ...
    def review(self, item: WorkItem, ctx: PaperContext, draft: SolverDraft) -> ReviewResult: ...


# --------------------------------------------------------------------------- rate limiting
class TokenBudgetLimiter:
    """Sliding 60 s window over estimated/actual token usage (thread-safe).

    ``acquire`` reserves an estimate and blocks until it fits; ``settle``
    replaces the estimate with the provider-reported usage; ``penalize`` is
    called on a 429 so every thread backs off together.
    """

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.budget = max(1, tokens_per_minute)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._events: Deque[list] = deque()  # [timestamp, tokens]
        self._blocked_until = 0.0
        self.waited_s = 0.0

    def _purge(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= 60.0:
            self._events.popleft()

    def acquire(self, tokens: int) -> list:
        tokens = min(max(1, tokens), self.budget)  # one call may never exceed the whole budget
        while True:
            with self._lock:
                now = self._clock()
                self._purge(now)
                used = sum(e[1] for e in self._events)
                wait = max(0.0, self._blocked_until - now)
                if wait == 0.0 and used + tokens <= self.budget:
                    event = [now, tokens]
                    self._events.append(event)
                    return event
                if wait == 0.0 and self._events:
                    wait = max(0.05, 60.0 - (now - self._events[0][0]))
            self.waited_s += wait
            self._sleep(min(wait, 30.0))

    def settle(self, event: list, actual_tokens: Optional[int]) -> None:
        if actual_tokens is not None:
            with self._lock:
                event[1] = max(1, int(actual_tokens))

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + max(0.0, seconds))


# --------------------------------------------------------------------------- error classification
_TRY_AGAIN = re.compile(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)\s*(ms|s)", re.I)


def _status_of(exc: BaseException) -> Optional[int]:
    return getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)


def is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    status = _status_of(exc)
    if status is not None:
        return status == 429 or status >= 500
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadTimeout",
        "ConnectTimeout",
        "TimeoutException",
        "TimeoutError",
        "RemoteProtocolError",
    }


def _is_output_failure(exc: BaseException) -> bool:
    """Groq returns HTTP 400 when the model's JSON fails schema validation."""
    return _status_of(exc) == 400 and "failed_generation" in str(exc).lower() or (
        _status_of(exc) == 400 and "validate json" in str(exc).lower()
    )


def retry_delay(exc: BaseException, attempt: int, rng: random.Random) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        ra = headers.get("retry-after")
        if ra is not None:
            return float(ra) + 0.5
    except (TypeError, ValueError):
        pass
    m = _TRY_AGAIN.search(str(exc))
    if m:
        mins = float(m.group(1) or 0)
        val = float(m.group(2))
        secs = val / 1000.0 if m.group(3).lower() == "ms" else val
        return mins * 60 + secs + 0.5
    return min(30.0, 2.0 ** (attempt + 1)) + rng.uniform(0, 1.0)


# --------------------------------------------------------------------------- production gateway
class LangChainGateway:
    """prompt | ChatGroq.with_structured_output(schema) with limits and typed errors."""

    def __init__(
        self,
        settings: WorkflowSettings,
        api_key: str,
        *,
        limiter: Optional[TokenBudgetLimiter] = None,
        model_factory: Optional[Callable[[str], Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key and model_factory is None:
            raise LLMConfigError("GROQ_API_KEY is not set; the solver and verifier cannot run.")
        self.settings = settings
        self.solver_model = settings.solver_model
        self.verifier_model = settings.verifier_model
        self._api_key = api_key  # held privately; never placed in graph state
        self._limiter = limiter or TokenBudgetLimiter(settings.tokens_per_minute, sleep=sleep)
        self._model_factory = model_factory
        self._sleep = sleep
        self._rng = random.Random()
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.calls = 0

    # -- model construction ----------------------------------------------------
    def _chat_model(self, name: str) -> Any:
        with self._lock:
            if name not in self._models:
                if self._model_factory is not None:
                    self._models[name] = self._model_factory(name)
                else:
                    from langchain_groq import ChatGroq  # local import: optional at import time

                    self._models[name] = ChatGroq(
                        api_key=self._api_key,
                        model=name,
                        temperature=self.settings.temperature,
                        timeout=self.settings.llm_timeout_s,
                        max_retries=0,  # retries are ours: bounded and classified
                        max_tokens=self.settings.max_output_tokens,
                    )
            return self._models[name]

    # -- core call -------------------------------------------------------------
    def _run(self, prompt: Any, variables: Dict[str, str], schema: Type[T], model: str) -> T:
        messages = prompt.format_messages(**variables)
        est_in = sum(len(str(m.content)) for m in messages) // 3 + 64
        reserve = est_in + self.settings.max_output_tokens
        chain = self._chat_model(model).with_structured_output(schema, method="json_schema", include_raw=True)

        output_failures = 0
        transient = 0
        rate_limited = 0
        last_problem = ""
        while True:
            event = self._limiter.acquire(reserve)
            try:
                self.calls += 1
                out = chain.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - classified below
                self._limiter.settle(event, 1)
                if _is_output_failure(exc):
                    output_failures += 1
                    last_problem = f"provider rejected output: {str(exc)[:160]}"
                    if output_failures > self.settings.llm_output_retries:
                        raise LLMOutputError(last_problem) from exc
                    continue
                if is_transient(exc):
                    is_429 = _status_of(exc) == 429
                    if is_429:
                        rate_limited += 1
                        attempt, budget = rate_limited, self.settings.llm_rate_limit_retries
                        delay = min(60.0, max(5.0, retry_delay(exc, attempt, self._rng)))
                    else:
                        transient += 1
                        attempt, budget = transient, self.settings.llm_transient_retries
                        delay = retry_delay(exc, attempt, self._rng)
                    self._limiter.penalize(delay)
                    log.warning("LLM transient failure (%s), retry %d/%d in %.1fs", type(exc).__name__, attempt, budget, delay)
                    if attempt > budget:
                        raise LLMUnavailableError(
                            f"{model}: provider unavailable after {attempt - 1} retries "
                            f"({type(exc).__name__})"
                        ) from exc
                    self._sleep(delay)
                    continue
                if _status_of(exc) in (401, 403):
                    raise LLMConfigError(f"{model}: provider rejected the credentials (HTTP {_status_of(exc)})") from exc
                raise LLMUnavailableError(f"{model}: {type(exc).__name__}: {str(exc)[:160]}") from exc

            raw = out.get("raw") if isinstance(out, dict) else None
            usage = getattr(raw, "usage_metadata", None) or {}
            self._limiter.settle(event, usage.get("total_tokens"))
            parsed = out.get("parsed") if isinstance(out, dict) else out
            if parsed is not None:
                return parsed
            output_failures += 1
            last_problem = f"output failed schema validation: {str(out.get('parsing_error'))[:200]}"
            if output_failures > self.settings.llm_output_retries:
                raise LLMOutputError(f"{model}: {last_problem}")

    # -- protocol methods ------------------------------------------------------
    def draft(self, item: WorkItem, ctx: PaperContext, repair: Optional[RepairContext] = None) -> SolverDraft:
        schema = {"mcq": McqDraft, "numerical": NumericalDraft}.get(item.type, TextDraft)
        parsed = self._run(SOLVE_PROMPT, solve_variables(item, ctx, repair), schema, self.solver_model)
        return SolverDraft(
            strategy=item.type,
            answer=parsed.answer.strip(),
            steps=list(parsed.steps),
            mark_split=[{"marks": m.marks, "for": m.for_} for m in parsed.mark_split],
            common_mistakes=[c.model_dump() for c in parsed.common_mistakes],
            self_confidence=parsed.self_confidence,
            chosen_option=getattr(parsed, "chosen_option", None),
            final_value=getattr(parsed, "final_value", None),
            unit=getattr(parsed, "unit", None),
            arithmetic_expression=getattr(parsed, "arithmetic_expression", None),
            prompt_version=PROMPT_VERSION,
            model=self.solver_model,
        )

    def blind_solve_mcq(self, item: WorkItem, ctx: PaperContext) -> BlindMcq:
        return self._run(BLIND_MCQ_PROMPT, base_variables(item, ctx), BlindMcq, self.verifier_model)

    def blind_solve_numeric(self, item: WorkItem, ctx: PaperContext) -> BlindNumeric:
        return self._run(BLIND_NUMERIC_PROMPT, base_variables(item, ctx), BlindNumeric, self.verifier_model)

    def review(self, item: WorkItem, ctx: PaperContext, draft: SolverDraft) -> ReviewResult:
        vars_ = review_variables(item, ctx, draft.answer, draft.steps, draft.mark_split)
        return self._run(REVIEW_PROMPT, vars_, ReviewResult, self.verifier_model)


REQUIRED_REVIEW_CHECKS: Tuple[str, ...] = REVIEW_CHECKS
