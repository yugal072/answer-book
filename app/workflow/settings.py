"""Workflow configuration: every threshold the graph uses, in one place.

All values are read from environment variables (with the defaults below) so a
deployment can tune them without code changes. See ``docs/WORKFLOW.md`` for
the meaning of each threshold.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# Version of the *output contract* (SolvedQuestion / SolvedPaper). A stored
# solution is only reusable when its schema_version matches.
SCHEMA_VERSION = "1"
# Version of the solver/verifier prompt set. Bump when prompts change in a way
# that could change answers; stored solutions made under an unaccepted prompt
# version are not reused.
PROMPT_VERSION = "2026-10-09.2"

# Evidence ladder: how strong was the check that backed a verification?
# Higher is stronger. See docs/WORKFLOW.md section "Verification semantics".
EVIDENCE_RANK: Dict[str, int] = {
    "none": 0,
    "structural": 1,   # deterministic shape/consistency checks only
    "llm_review": 2,   # a second model reviewed the draft against a rubric
    "cross_check": 3,  # an independent blind re-solve agreed with the draft
    "symbolic": 4,     # cross_check + deterministic re-evaluation of the arithmetic
}


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class WorkflowSettings:
    # --- retry / concurrency -------------------------------------------------
    #: Repair attempts after the first draft. Total drafts <= 1 + this.
    max_repair_attempts: int = field(default_factory=lambda: _i("WORKFLOW_MAX_REPAIR_ATTEMPTS", 2))
    #: Parallel question branches (the LLM gateway additionally rate-limits).
    max_concurrency: int = field(default_factory=lambda: _i("WORKFLOW_MAX_CONCURRENCY", 4))

    # --- confidence / storage gates -----------------------------------------
    #: A verified result must reach this final confidence to enter the KB.
    store_min_confidence: float = field(default_factory=lambda: _f("WORKFLOW_STORE_MIN_CONFIDENCE", 0.75))
    #: Share of verification checks that must pass (error-level checks must
    #: ALL pass regardless; this additionally bounds warning-level failures).
    accuracy_min: float = field(default_factory=lambda: _f("WORKFLOW_ACCURACY_MIN", 0.80))
    #: Final confidence is never above this when verification did not pass.
    unverified_confidence_ceiling: float = field(
        default_factory=lambda: _f("WORKFLOW_UNVERIFIED_CEILING", 0.40)
    )
    #: Relative tolerance when two numeric answers are compared.
    numeric_rel_tol: float = field(default_factory=lambda: _f("WORKFLOW_NUMERIC_REL_TOL", 1e-3))

    # --- evidence requirements / caps ---------------------------------------
    #: Minimum evidence level that makes a result "verified", per question type.
    min_evidence: Dict[str, str] = field(
        default_factory=lambda: {
            "mcq": "cross_check",
            "numerical": "cross_check",
            "short": "llm_review",
            "long": "llm_review",
        }
    )
    #: The model's self-reported confidence is capped by how strong the
    #: evidence was; self-confidence alone can never raise the final value.
    evidence_confidence_cap: Dict[str, float] = field(
        default_factory=lambda: {
            "none": 0.30,
            "structural": 0.50,
            "llm_review": 0.80,
            "cross_check": 0.90,
            "symbolic": 0.95,
        }
    )

    # --- cache acceptance ----------------------------------------------------
    accepted_prompt_versions: Tuple[str, ...] = field(
        default_factory=lambda: tuple(
            v.strip()
            for v in os.getenv("WORKFLOW_ACCEPTED_PROMPT_VERSIONS", PROMPT_VERSION).split(",")
            if v.strip()
        )
    )

    # --- LLM -----------------------------------------------------------------
    solver_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
    #: A different model family for verification so errors are less correlated.
    verifier_model: str = field(default_factory=lambda: os.getenv("GROQ_VERIFIER_MODEL", "openai/gpt-oss-20b"))
    temperature: float = field(default_factory=lambda: _f("GROQ_TEMPERATURE", 0.1))
    llm_timeout_s: float = field(default_factory=lambda: _f("WORKFLOW_LLM_TIMEOUT_S", 90.0))
    #: Retries for *transient* provider failures only (429/5xx/timeout/connect).
    llm_transient_retries: int = field(default_factory=lambda: _i("WORKFLOW_LLM_TRANSIENT_RETRIES", 4))
    #: Rate limits (429) are "wait, then retry", not failures: a separate, larger
    #: bounded budget. Other processes sharing the key (or a just-killed run) can
    #: use up the provider's window without this process knowing.
    llm_rate_limit_retries: int = field(default_factory=lambda: _i("WORKFLOW_LLM_RATE_LIMIT_RETRIES", 8))
    #: Extra attempts when the model returns output that fails validation.
    llm_output_retries: int = field(default_factory=lambda: _i("WORKFLOW_LLM_OUTPUT_RETRIES", 1))
    max_output_tokens: int = field(default_factory=lambda: _i("WORKFLOW_MAX_OUTPUT_TOKENS", 2400))
    #: Groq limits this key to 8000 tokens/minute; stay under it with headroom.
    tokens_per_minute: int = field(default_factory=lambda: _i("WORKFLOW_TOKENS_PER_MINUTE", 6500))

    # --- I/O -----------------------------------------------------------------
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("WORKFLOW_OUTPUT_DIR", "solutionPapers")))
    max_upload_bytes: int = field(default_factory=lambda: _i("WORKFLOW_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))

    def question_recursion_limit(self) -> int:
        """Graph steps one question can need: fixed nodes + 2 per repair cycle."""
        return 16 + 2 * (self.max_repair_attempts + 1)

    def evidence_at_least(self, have: str, need: str) -> bool:
        return EVIDENCE_RANK.get(have, 0) >= EVIDENCE_RANK.get(need, 99)


def load_settings() -> WorkflowSettings:
    return WorkflowSettings()
