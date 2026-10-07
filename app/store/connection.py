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
db_url = settings.DATABASE_URL or "sqlite:///:memory:"
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


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for acquiring a database session.

    Yields:
        Session: Active SQLAlchemy session.

    Guarantees:
        - Commits automatically if the block completes without errors.
        - Rolls back automatically if an unhandled exception occurs.
        - Closes and returns the connection to the pool in all cases.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
