"""Question-Solution Knowledge Base table.

An additive table (existing ``papers`` / ``solved_questions`` are untouched).
It holds ONLY solutions that passed the workflow's verified-storage criteria;
per-paper checkpoint rows keep living in ``solved_questions``.

``context_key`` is the idempotency key: one row per
(question signature, subject, type, options, figure flag, schema version).
Re-running a paper updates that row instead of adding a duplicate.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.store.models import Base


class VerifiedSolutionTable(Base):
    __tablename__ = "verified_solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    context_key: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    question_signature: Mapped[str] = mapped_column(String(64), index=True, nullable=False)

    # question context (what makes a reuse legitimate)
    question_text_norm: Mapped[str] = mapped_column(Text, nullable=False)
    marks: Mapped[int] = mapped_column(Integer, nullable=False)
    board: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    class_name: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    subject: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    question_type: Mapped[str] = mapped_column(String(20), nullable=False)
    options: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    has_figure: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # the solution and how it earned "verified"
    solution: Mapped[Any] = mapped_column(JSON, nullable=False)  # contract fields
    draft: Mapped[Any] = mapped_column(JSON, nullable=False)  # full SolverDraft (typed fields)
    verification_status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'verified'
    evidence: Mapped[str] = mapped_column(String(20), nullable=False)
    verification_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    self_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(10), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(40), nullable=False)
    solver_model: Mapped[str] = mapped_column(String(100), default="", nullable=False)
    verifier_model: Mapped[str] = mapped_column(String(100), default="", nullable=False)

    # provenance
    source_paper_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_question_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
