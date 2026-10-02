import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.ingestion.errors import IngestionError
from app.ingestion.image_loader import extract_paper_from_image_bytes
from app.ingestion.paper_loader import ingest_paper
from app.models.loaders_models import Paper

router = APIRouter(tags=["Ingestion"])

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def process_file_ingestion(
    file: UploadFile,
    content: bytes,
    subject: Optional[str] = None,
    class_name: Optional[str] = None,
    board: Optional[str] = None,
) -> Paper:
    """Core file ingestion logic shared between endpoints.

    Unusable input and failed extraction are reported as HTTP 422 with the
    reason; they are never returned as an empty but "successful" paper.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix or '(none)'}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    if not content:
        raise HTTPException(
            status_code=400, detail=f"Uploaded file '{file.filename}' is empty."
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Uploaded file is too large "
                f"({len(content) / 1e6:.1f} MB, limit "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB)."
            ),
        )

    try:
        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)
            try:
                paper = ingest_paper(file_path=tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            paper = extract_paper_from_image_bytes(content)
    except IngestionError as exc:
        # DocumentError / ExtractionError / ValidationError: the input or
        # the extraction failed, and the caller deserves the reason.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if subject:
        paper.subject = subject
    if class_name:
        paper.class_name = class_name
    if board:
        paper.board = board

    return paper


@router.post("/ingest", response_model=Paper)
def ingest_paper_endpoint(
    file: UploadFile = File(...),
    subject: Optional[str] = Query(None),
    class_name: Optional[str] = Query(None),
    board: Optional[str] = Query(None),
):
    """Ingests a PDF or Image question paper and returns extracted Paper metadata & questions."""
    content = file.file.read()
    return process_file_ingestion(
        file=file,
        content=content,
        subject=subject,
        class_name=class_name,
        board=board,
    )