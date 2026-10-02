"""Shared exception hierarchy for the ingestion subsystem.

Every failure mode in ingestion is one of four things, and the type says
which one, so callers (API layer, pipeline, tests) can react without
string-matching messages:

``DocumentError``
    The *input* is unusable: missing, empty, unsupported, corrupt,
    unreadable, or a PDF with no usable text layer. Nothing was extracted.
``ExtractionError``
    Extraction was attempted but did not produce trustworthy data
    (model transport failure, unparsable model response, unreadable image).
``ValidationError``
    Extraction produced data that failed structural validation. Carries the
    full :class:`~app.ingestion.validation.ValidationReport` so the caller
    gets diagnostics, not just a string.
``IngestionError``
    Base class for all of the above.

Design rules that these types exist to enforce:

* A failure is never disguised as a successful, empty ``Paper``.
* Truncated model output is an error (``TruncationError``), never a
  partial result.
* Callers can catch ``IngestionError`` to handle "ingestion did not work"
  in one place.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.ingestion.validation import Diagnostic


class IngestionError(Exception):
    """Base class for every ingestion failure."""


class DocumentError(IngestionError):
    """The input document/image cannot be read at all."""


class ExtractionError(IngestionError):
    """Extraction ran but did not yield trustworthy data."""


class TruncationError(ExtractionError):
    """Model output hit the token budget; partial data is never returned."""


class ValidationError(IngestionError):
    """Extraction produced structurally invalid data.

    ``diagnostics`` holds the full report (see
    :mod:`app.ingestion.validation`) so the reason is never just a message.
    """

    def __init__(
        self,
        message: str,
        diagnostics: Optional[List["Diagnostic"]] = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = list(diagnostics or [])

    def __str__(self) -> str:  # pragma: no cover - formatting only
        base = super().__str__()
        if not self.diagnostics:
            return base
        lines = [base]
        for d in self.diagnostics[:20]:
            lines.append(f"  - [{d.severity}] {d.code}: {d.message}")
        if len(self.diagnostics) > 20:
            lines.append(f"  - ... and {len(self.diagnostics) - 20} more")
        return "\n".join(lines)
