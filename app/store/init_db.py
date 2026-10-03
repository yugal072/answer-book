"""Database initialization script.

Creates all tables defined in models.py (papers, solved_questions) in the
configured PostgreSQL database. Safe to run multiple times (idempotent).

Usage:
    python -m app.store.init_db
"""

import sys
from sqlalchemy import inspect
from app.store.connection import engine
from app.store.models import Base


def init_database() -> bool:
    """Creates all database tables and verifies the schema."""
    print("=" * 70)
    print("        POSTGRESQL DATABASE INITIALIZATION (ANSWER BOOK)")
    print("=" * 70)

    try:
        print("[1/3] Connecting to PostgreSQL database...")
        # Verify connection
        with engine.connect() as conn:
            print("      Connection established successfully.")

        print("[2/3] Creating tables if they do not exist...")
        Base.metadata.create_all(bind=engine)
        print("      Base.metadata.create_all completed.")

        print("[3/3] Inspecting database tables...")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        print(f"\nDiscovered {len(table_names)} tables in database:")
        for tbl in table_names:
            columns = inspector.get_columns(tbl)
            col_names = [col["name"] for col in columns]
            print(f"  • Table '{tbl}': {len(col_names)} columns -> {', '.join(col_names[:6])}...")

        print("\n" + "=" * 70)
        print("  Database initialization completed successfully! (Status: READY)")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to initialize database: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = init_database()
    if not success:
        sys.exit(1)
