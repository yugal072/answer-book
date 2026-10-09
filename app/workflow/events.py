"""Progress events emitted by graph nodes through LangGraph's custom stream.

Only whitelisted, non-sensitive fields ever leave a node this way (no prompts,
no raw state, no credentials). Outside a streaming run the writer is a no-op.
"""
from __future__ import annotations

from typing import Any, Dict


def emit(event: str, **fields: Any) -> None:
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not inside a graph run (e.g. direct node call in a unit test)
        return
    payload: Dict[str, Any] = {"event": event}
    payload.update({k: v for k, v in fields.items() if v is not None})
    writer(payload)
