"""Database connection and session management for PostgreSQL.

Configures the SQLAlchemy engine with connection pooling and provides
the get_db_session context manager for safe transactional database operations.
"""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Engine configured with pool_pre_ping to automatically recover from dropped connections,
# and pool_recycle=3600 to refresh stale connections every hour.
def _normalize(url: str) -> str:
    """Plain ``postgres://`` / ``postgresql://`` URLs use the psycopg (v3) driver from requirements.txt."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


db_url = _normalize(settings.DATABASE_URL) if settings.DATABASE_URL else "sqlite:///:memory:"
engine = create_engine(
    db_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def make_session_scope(maker: sessionmaker):
    """Build a transactional session context manager bound to ``maker``.

    Guarantees:
        - Commits automatically if the block completes without errors.
        - Rolls back automatically if an unhandled exception occurs.
        - Closes and returns the connection to the pool in all cases.
    """

    @contextmanager
    def session_scope() -> Generator[Session, None, None]:
        session = maker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return session_scope


#: Context manager yielding a session on the application engine.
get_db_session = make_session_scope(SessionLocal)
