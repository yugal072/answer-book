"""SQLAlchemy database models for papers and solved questions.

Defines the relational schema:
- PaperTable: High-level paper metadata, fingerprint, and execution status.
- SolvedQuestionTable: Individual question solutions, composite signatures, and verification attributes.
"""

from datetime import datetime
from typing import Any, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class PaperTable(Base):
    """Stores paper-level metadata, fingerprint, and pipeline execution status."""

    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="parsing", nullable=False)
    subject: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    class_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    board: Mapped[Optional[str]] = mapped_column(String(50), default="CBSE", nullable=True)
    total_questions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_marks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sections: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    source_file: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # 1 Paper -> Many Solved Questions (Cascade delete ensures children are removed if paper is deleted)
    questions: Mapped[List["SolvedQuestionTable"]] = relationship(
        "SolvedQuestionTable", back_populates="paper", cascade="all, delete-orphan"
    )


class SolvedQuestionTable(Base):
    """Stores individual solved question solutions, signatures, and verification attributes."""

    __tablename__ = "solved_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("papers.paper_id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_signature: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    question_number: Mapped[str] = mapped_column(String(20), nullable=False)
    section: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    marks: Mapped[int] = mapped_column(Integer, nullable=False)
    question_type: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    options: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    has_figure: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Solution Contract fields (Adheres strictly to Section 3 of the Brief)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    steps: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    mark_split: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    common_mistakes: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verified_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    needs_teacher_check: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    paper: Mapped["PaperTable"] = relationship("PaperTable", back_populates="questions")
