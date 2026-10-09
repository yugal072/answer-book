"""Production wiring: settings + environment -> a ready :class:`AnswerBookService`."""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.core.config import settings as app_settings
from app.workflow.deps import WorkflowDeps
from app.workflow.kb import InMemoryKnowledgeBase, InMemoryPaperRepository, SqlKnowledgeBase, SqlPaperRepository
from app.workflow.llm import LangChainGateway
from app.workflow.service import AnswerBookService
from app.workflow.settings import WorkflowSettings, load_settings

log = logging.getLogger(__name__)


def normalize_db_url(url: str) -> str:
    """Use the psycopg (v3) driver for plain postgres URLs (what requirements.txt installs)."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def make_checkpointer(db_url: str) -> Any:
    """PostgresSaver for a Postgres URL, otherwise a process-local InMemorySaver."""
    if db_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        conninfo = db_url.replace("postgresql+psycopg://", "postgresql://", 1).replace("postgres://", "postgresql://", 1)
        pool = ConnectionPool(
            conninfo=conninfo, max_size=8, open=True,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        saver = PostgresSaver(pool)
        saver.setup()  # creates the checkpoint tables if missing (idempotent)
        return saver
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def build_default_service(settings: Optional[WorkflowSettings] = None) -> AnswerBookService:
    s = settings or load_settings()
    db_url = app_settings.DATABASE_URL
    if db_url:
        from app.store.connection import engine, get_db_session
        from app.store.models import Base
        import app.store.knowledge  # noqa: F401  (registers the knowledge-base table)

        Base.metadata.create_all(bind=engine)  # additive and idempotent
        kb, papers = SqlKnowledgeBase(get_db_session), SqlPaperRepository(get_db_session)
        checkpointer = make_checkpointer(db_url)
        log.info("workflow storage: SQL database + %s checkpointer", type(checkpointer).__name__)
    else:
        log.warning("DATABASE_URL is not set: the knowledge base and checkpoints are PROCESS-LOCAL and "
                    "will be lost on restart. Set DATABASE_URL to a PostgreSQL URL for persistence.")
        kb, papers = InMemoryKnowledgeBase(), InMemoryPaperRepository()
        checkpointer = make_checkpointer("")
    llm = LangChainGateway(s, app_settings.GROQ_API_KEY)
    return AnswerBookService(WorkflowDeps(kb=kb, papers=papers, llm=llm, settings=s), checkpointer)


def storage_mode() -> str:
    return "postgres" if app_settings.DATABASE_URL.startswith(("postgres", "postgresql")) else (
        "sql" if app_settings.DATABASE_URL else "memory"
    )
