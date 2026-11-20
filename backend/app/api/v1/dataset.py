"""
Dataset management API endpoints.

GET  /api/v1/datasets                        — dataset catalog (all supported datasets)
GET  /api/v1/datasets/{name}/stats           — split statistics from dataset_stats.json
GET  /api/v1/datasets/{name}/analytics       — word freq, length dist, quality report
GET  /api/v1/datasets/{name}/samples         — random sample rows from a split
GET  /api/v1/datasets/{name}/audit           — preprocessing audit log
POST /api/v1/datasets/upload                 — upload a CSV/TSV/JSON dataset file
POST /api/v1/datasets/preprocess             — trigger preprocessing pipeline
GET  /api/v1/datasets/{name}/export/{split}  — download processed CSV

All statistics are computed from actual file contents.
No hardcoded numbers anywhere in this file.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, status
from fastapi.responses import Response

from app.schemas.dataset import (
    DatasetCatalogResponse,
    DatasetStats,
    DatasetAnalytics,
    DatasetSamples,
    PreprocessingAudit,
    DatasetUploadResponse,
    PreprocessRequest,
    PreprocessResponse,
)
from app.services import dataset_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["Datasets"])

# Max upload size: 200 MB
_MAX_UPLOAD_BYTES = 200 * 1024 * 1024


# ── Catalog ───────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=DatasetCatalogResponse,
    summary="Dataset catalog",
    description=(
        "Returns all supported datasets with metadata, license information, "
        "and availability status (whether files exist on disk)."
    ),
)
async def list_datasets() -> DatasetCatalogResponse:
    data = await dataset_service.get_catalog()
    return DatasetCatalogResponse(**data)


# ── Statistics ────────────────────────────────────────────────────────────────

@router.get(
    "/{name}/stats",
    response_model=DatasetStats,
    summary="Dataset split statistics",
    description=(
        "Returns statistics computed by the preprocessing pipeline: "
        "row counts, class distribution, text length, deduplication counts, "
        "and cross-split leakage check results."
    ),
)
async def dataset_stats(name: str) -> DatasetStats:
    try:
        data = await dataset_service.get_dataset_stats(name)
        return DatasetStats(**data)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get(
    "/{name}/analytics",
    summary="Full dataset analytics",
    description=(
        "Computes word frequency (all / fake / real), article length distribution, "
        "class distribution, missing value report, duplicate report, and quality score. "
        "All values are computed from actual dataset files."
    ),
)
async def dataset_analytics(
    name:  str,
    split: str = Query("train", pattern="^(train|val|test)$"),
) -> dict:
    try:
        return await dataset_service.get_analytics(name, split=split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Analytics failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics computation failed: {exc}",
        )


# ── Sample rows ───────────────────────────────────────────────────────────────

@router.get(
    "/{name}/samples",
    summary="Sample dataset rows",
    description="Returns a random sample of rows from a processed split.",
)
async def dataset_samples(
    name:   str,
    split:  str           = Query("train", pattern="^(train|val|test)$"),
    n:      int           = Query(10, ge=1, le=100),
    label:  Optional[int] = Query(None, description="0=FAKE, 1=REAL, omit for all"),
) -> dict:
    try:
        return await dataset_service.get_samples(name, split=split, n=n, label_filter=label)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


# ── Audit log ─────────────────────────────────────────────────────────────────

@router.get(
    "/{name}/audit",
    summary="Preprocessing audit log",
    description=(
        "Returns the preprocessing audit log recording every step performed "
        "on the dataset, including rows dropped at each stage."
    ),
)
async def dataset_audit(name: str) -> dict:
    data = await dataset_service.get_audit_log()
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No audit log found. Run the preprocessing pipeline first: "
                "python scripts/prepare_dataset.py --files ..."
            ),
        )
    return data


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a dataset file",
    description=(
        "Upload a CSV, TSV, JSON, or JSONL dataset file to ml/datasets/raw/. "
        "The file is inspected but NOT processed automatically. "
        "Call POST /datasets/preprocess to run the pipeline."
    ),
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size ({_MAX_UPLOAD_BYTES // 1024 // 1024} MB).",
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    try:
        result = await dataset_service.handle_upload(file.filename, content)
        return DatasetUploadResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error("Upload failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {exc}",
        )


# ── Preprocessing trigger ─────────────────────────────────────────────────────

@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
    summary="Run preprocessing pipeline",
    description=(
        "Trigger the full preprocessing pipeline on specified raw files. "
        "Runs synchronously for small datasets. "
        "For ISOT (~45k rows) expect ~30–60 seconds."
    ),
)
async def preprocess_dataset(request: PreprocessRequest) -> PreprocessResponse:
    result = await dataset_service.trigger_preprocessing(request.model_dump())
    return PreprocessResponse(**result)


# ── Export ────────────────────────────────────────────────────────────────────

@router.get(
    "/{name}/export/{split}",
    summary="Export processed split as CSV",
    description=(
        "Download a processed split (train/val/test) as a CSV file. "
        "Contains cleaned_text and label columns only — not raw article text.\n\n"
        "IMPORTANT: This exports PREPROCESSED data only. "
        "Raw dataset text is never exported to respect dataset copyright."
    ),
)
async def export_split(
    name:  str,
    split: str,
) -> Response:
    if split not in ("train", "val", "test"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="split must be 'train', 'val', or 'test'.",
        )
    try:
        csv_bytes = await dataset_service.get_export_csv(split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{name}_{split}.csv"',
            "X-Export-Note": (
                "Contains preprocessed (cleaned) text only. "
                "Not raw article text. Research use only."
            ),
        },
    )
