"""Ingestion Subsystem Package

Provides unified question paper ingestion from PDFs or Images into canonical Paper objects
"""

from app.ingestion.errors import (
    DocumentError,
    ExtractionError,
    IngestionError,
    TruncationError,
    ValidationError,
)
from app.ingestion.image_loader import (
    extract_paper_from_image_bytes,
    extract_paper_from_image_file,
    extract_paper_from_images,
)

from app.ingestion.paper_loader import ingest_paper, ingest_paper_with_report
from app.ingestion.pipeline import IngestionPipeline, ingest_file
from app.ingestion.validation import ValidationReport, validate_paper

__all__= [
    "ingest_file",
    "IngestionPipeline",
    "ingest_paper",
    "ingest_paper_with_report",
    "extract_paper_from_image_file",
    "extract_paper_from_image_bytes",
    "extract_paper_from_images",
    "validate_paper",
    "ValidationReport",
    "IngestionError",
    "DocumentError",
    "ExtractionError",
    "TruncationError",
    "ValidationError",
]