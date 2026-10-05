"""Ingestion Pipeline orchestrator

Routes input (PDF, image file, multi-page image list, or raw bytes) to the appropriate
loader (paper_loader or image_loader), performs input validation and error handling,
applies optional metadata overrides, and returns the canonical Paper object.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from app.ingestion.errors import DocumentError, IngestionError
from app.ingestion.image_loader import (
    SUPPORTED_EXTENSIONS as IMAGE_EXTENSIONS,
    extract_paper_from_image_bytes,
    extract_paper_from_image_file,
    extract_paper_from_images,
)
from app.ingestion.paper_loader import ingest_paper, ingest_paper_with_report
from app.models.loaders_models import Paper

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf"}.union(IMAGE_EXTENSIONS)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

def ingest_file(
    input_source: Union[str, Path, bytes, List[Union[str, Path]]],
    *,
    subject: Optional[str] = None,
    class_name: Optional[str] = None,
    board: Optional[str] = None,
    diagnostics: Optional[list] = None,
    model: Optional[str] = None,
) -> Paper:
    """Run the complete ingestion pipeline for a given input source.

    Args:
        input_source: File path (str/Path), image list, or raw file bytes.
        subject: Optional override for subject metadata.
        class_name: Optional override for class metadata.
        board: Optional override for board metadata.
        diagnostics: Optional list appended with validation Diagnostic findings.
        model: Optional model override for vision extraction.

    Returns:
        Canonical Paper instance.

    Raises:
        DocumentError: If input is missing, corrupt, unsupported, or unreadable.
        ExtractionError: If vision extraction fails or returns unparsable results.
        ValidationError: If extracted paper fails structural validation checks.
    """
    paper: Paper

    # 1. Handle list of image paths (multi-page image paper)
    if isinstance(input_source, list):
        if not input_source:
            raise DocumentError("Empty input list provided for ingestion.")
        paths = [Path(p) for p in input_source]
        for p in paths:
            if p.suffix.lower() not in IMAGE_EXTENSIONS:
                raise DocumentError(
                    f"Unsupported image extension '{p.suffix}' in multi-page input. "
                    f"Supported: {sorted(IMAGE_EXTENSIONS)}"
                )
        paper = extract_paper_from_images(paths, model=model, diagnostics=diagnostics)

    # 2. Handle raw bytes input
    elif isinstance(input_source, bytes):
        if not input_source:
            raise DocumentError("Provided file bytes are empty.")
        if len(input_source) > MAX_UPLOAD_BYTES:
            raise DocumentError(
                f"File size exceeds maximum allowed threshold "
                f"({len(input_source) / 1e6:.1f} MB > {MAX_UPLOAD_BYTES / 1e6:.0f} MB)."
            )

        # Detect PDF by magic numbers
        if input_source.startswith(b"%PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(input_source)
                tmp_path = Path(tmp.name)
            try:
                paper = ingest_paper(tmp_path, diagnostics=diagnostics)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            paper = extract_paper_from_image_bytes(
                input_source, model=model, diagnostics=diagnostics
            )

    # 3. Handle single file path (str or Path)
    elif isinstance(input_source, (str, Path)):
        path = Path(input_source)
        if not path.exists():
            raise DocumentError(f"Input file not found: {path}")
        if not path.is_file():
            raise DocumentError(f"Input path is not a file: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise DocumentError(
                f"Unsupported file format '{suffix or '(none)'}'. "
                f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".pdf":
            paper = ingest_paper(path, diagnostics=diagnostics)
        else:
            paper = extract_paper_from_image_file(path, model=model, diagnostics=diagnostics)

    else:
        raise DocumentError(f"Unsupported input_source type: {type(input_source).__name__}")

    # Apply metadata overrides if provided
    if subject:
        paper.subject = subject
    if class_name:
        paper.class_name = class_name
    if board:
        paper.board = board

    return paper


class IngestionPipeline:
    """Object-oriented wrapper around the ingestion pipeline."""

    def __init__(
        self,
        default_model: Optional[str] = None,
        collect_diagnostics: bool = True,
    ) -> None:
        self.default_model = default_model
        self.collect_diagnostics = collect_diagnostics

    def run(
        self,
        input_source: Union[str, Path, bytes, List[Union[str, Path]]],
        subject: Optional[str] = None,
        class_name: Optional[str] = None,
        board: Optional[str] = None,
    ) -> tuple[Paper, list]:
        """Runs ingestion pipeline and returns (Paper, diagnostics_list)."""
        diagnostics: list = []
        paper = ingest_file(
            input_source,
            subject=subject,
            class_name=class_name,
            board=board,
            diagnostics=diagnostics if self.collect_diagnostics else None,
            model=self.default_model,
        )
        return paper, diagnostics