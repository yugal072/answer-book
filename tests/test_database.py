"""Automated Unit Tests for PostgreSQL Database Persistence and Caching.

Covers:
1. Database connectivity and table schema verification.
2. Saving and pulling question checkpoints (crash recovery).
3. Question-level composite signature cache lookup.
4. Recommendation A paper-level pre-flight cache checks.
"""

import sys
from pathlib import Path
import pytest

# 1. Skip entire file if sqlalchemy is not installed (e.g. offline CI runner)
pytest.importorskip("sqlalchemy")

# 2. Mark as live/db test so `-m "not live"` in GitHub Actions excludes it
pytestmark = [pytest.mark.live, pytest.mark.db]

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 3. Gracefully skip if database/environment config is not present
try:
    from sqlalchemy import inspect
    from app.store.connection import engine, get_db_session
    from app.store.models import PaperTable, SolvedQuestionTable
    from app.tools.hash_utils import generate_question_signature, normalize_text
    from app.tools.paper_cache_tool import (
        check_paper_cache,
        get_paper_resume_info,
        mark_paper_status,
        upsert_paper_record,
    )
    from app.tools.question_lookup_tool import lookup_cached_question
    from app.tools.question_save_tool import save_question_checkpoint
except Exception as e:
    pytest.skip(f"Database environment not configured: {e}", allow_module_level=True)

TEST_PAPER_ID = "test_pytest_paper_001"
TEST_FINGERPRINT = "test_pytest_fp_abcdef123456"


@pytest.fixture(autouse=True)
def clean_test_records():
    """Pytest fixture to clean up test database rows before and after each test."""
    def _cleanup():
        with get_db_session() as session:
            existing = session.query(PaperTable).filter_by(paper_id=TEST_PAPER_ID).first()
            if existing:
                session.delete(existing)

    _cleanup()
    yield
    _cleanup()


def test_db_tables_exist():
    """Verify that PostgreSQL is connected and tables exist."""
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "papers" in tables, "Table 'papers' must exist in PostgreSQL"
    assert "solved_questions" in tables, "Table 'solved_questions' must exist in PostgreSQL"


def test_save_and_pull_question_checkpoint():
    """Test saving a question checkpoint and pulling it back for resume."""
    # 1. Initialize paper in database
    upsert_paper_record(
        paper_id=TEST_PAPER_ID,
        fingerprint=TEST_FINGERPRINT,
        status="solving",
        subject="Mathematics Part 1",
        class_name="Class 9",
        board="CBSE",
        total_questions=1,
        total_marks=2,
    )

    # 2. Save a question checkpoint
    q_item = {
        "number": "1",
        "text": "Find the value of x if 2x + 4 = 10.",
        "marks": 2,
        "section": "A",
        "type": "numerical",
        "options": None,
        "has_figure": False,
    }
    sig = generate_question_signature(q_item["text"], q_item["marks"], "CBSE", "Class 9")
    solution_contract = {
        "question_number": "1",
        "answer": "x = 3",
        "steps": ["2x = 10 - 4", "2x = 6", "x = 3"],
        "mark_split": [{"marks": 1, "for": "isolating x"}, {"marks": 1, "for": "final value"}],
        "common_mistakes": [{"wrong": "x = 7", "why": "added 4 instead of subtracting"}],
        "confidence": 0.99,
        "verified_by": "groq:qwen/qwen3.8-27b",
        "needs_teacher_check": False,
    }

    saved = save_question_checkpoint(TEST_PAPER_ID, sig, q_item, solution_contract)
    assert saved is True, "Question checkpoint must return True on save"

    # 3. Pull it back from DB (Crash Recovery test)
    existing_solutions, solved_numbers = get_paper_resume_info(TEST_PAPER_ID)
    assert "1" in solved_numbers, "Question 1 must be registered in solved_numbers"
    assert len(existing_solutions) == 1, "Must pull exactly 1 solved question"

    pulled_sol = existing_solutions[0]
    assert pulled_sol["question_number"] == "1"
    assert pulled_sol["answer"] == "x = 3"
    assert len(pulled_sol["steps"]) == 3
    assert len(pulled_sol["mark_split"]) == 2
    assert pulled_sol["confidence"] == 0.99
    assert pulled_sol["needs_teacher_check"] is False


def test_question_signature_cache_lookup():
    """Test question-level cache lookup by composite signature."""
    # 1. Insert seed question
    upsert_paper_record(TEST_PAPER_ID, TEST_FINGERPRINT, status="solving")
    q_item = {
        "number": "5",
        "text": "What is the capital of France?",
        "marks": 1,
        "section": "A",
        "type": "short",
    }
    sig = generate_question_signature(q_item["text"], q_item["marks"], "CBSE", "Class 9")
    sol = {
        "question_number": "5",
        "answer": "Paris",
        "steps": ["Recall capital city of France."],
        "mark_split": [{"marks": 1, "for": "correct city"}],
        "common_mistakes": [],
        "confidence": 1.0,
        "verified_by": "groq:test",
        "needs_teacher_check": False,
    }
    save_question_checkpoint(TEST_PAPER_ID, sig, q_item, sol)

    # 2. Lookup with new question number (e.g. Q14 in a different paper)
    cached = lookup_cached_question(sig, target_question_number="14")
    assert cached is not None, "Question should be found in cache by signature"
    assert cached["question_number"] == "14", "Cached question number must be mapped to current target"
    assert cached["answer"] == "Paris"

    # 3. Lookup non-existent signature
    assert lookup_cached_question("non_existent_signature_hash") is None


def test_paper_level_cache_preflight():
    """Test Recommendation A pre-flight cache hit and status transitions."""
    # 1. When status is 'solving', pre-flight cache check must return None
    upsert_paper_record(TEST_PAPER_ID, TEST_FINGERPRINT, status="solving", subject="Science")
    assert check_paper_cache(TEST_FINGERPRINT) is None

    # 2. Add a question
    q_item = {"number": "1", "text": "Define inertia.", "marks": 2}
    sig = generate_question_signature(q_item["text"], 2, "CBSE", "Class 9")
    sol = {
        "question_number": "1",
        "answer": "Inertia is the resistance of an object to changes in its state of motion.",
        "steps": ["State definition."],
        "mark_split": [{"marks": 2, "for": "definition"}],
        "common_mistakes": [],
        "confidence": 0.95,
        "verified_by": "groq:test",
        "needs_teacher_check": False,
    }
    save_question_checkpoint(TEST_PAPER_ID, sig, q_item, sol)

    # 3. Mark paper as 'ready'
    mark_paper_status(TEST_PAPER_ID, "ready")

    # 4. Now pre-flight check must return the full Solution Book payload
    cached_payload = check_paper_cache(TEST_FINGERPRINT)
    assert cached_payload is not None, "Completed paper must be cached"
    assert cached_payload["paper_id"] == TEST_PAPER_ID
    assert cached_payload["status"] == "ready"
    assert len(cached_payload["solutions"]) == 1
    assert cached_payload["solutions"][0]["answer"] == sol["answer"]
