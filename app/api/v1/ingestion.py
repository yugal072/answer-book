import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.ingestion import IngestionError, ingest_file
from app.models.loaders_models import Paper

router = APIRouter(tags=["Ingestion"])

def process_file_ingestion(
    file: UploadFile,
    content: bytes,
    subject: Optional[str] = None,
    class_name: Optional[str] = None,
    board: Optional[str] = None,
) -> Paper:
    """Core file ingestion logic shared between endpoints."""
    if not content:
        raise HTTPException(
            status_code=400, detail=f"Uploaded file '{file.filename}' is empty."
        )

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