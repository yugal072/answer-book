import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.ingestion import IngestionError, ingest_file
from app.ingestion.pipeline import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS
from app.models.loaders_models import Paper

router = APIRouter(tags=["Ingestion"])

_MAGIC = {
    ".pdf": (b"%PDF",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".webp": (b"RIFF",),
}


def validate_upload(filename: Optional[str], content: bytes) -> None:
    """Reject bad uploads BEFORE ingestion (which can spend model quota).

    Checks size, extension, and that the leading bytes really are that format.
    Raises HTTPException: 400 empty, 413 too large, 415 unsupported/mismatched.
    """
    name = filename or "upload"
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{name}' is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(content) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type '{ext or '(none)'}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if not any(content.startswith(m) for m in _MAGIC[ext]) or (ext == ".webp" and content[8:12] != b"WEBP"):
        raise HTTPException(status_code=415, detail=f"File content does not look like a {ext} file.")


def process_file_ingestion(
    file: UploadFile,
    content: bytes,
    subject: Optional[str] = None,
    class_name: Optional[str] = None,
    board: Optional[str] = None,
) -> Paper:
    """Core file ingestion logic shared between endpoints."""
    validate_upload(file.filename, content)

    try:
        return ingest_file(
            input_source=content,
            subject=subject,
            class_name=class_name,
            board=board,
        )
    except IngestionError as exc:
        # DocumentError / ExtractionError / ValidationError: reported as 422 with reason
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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