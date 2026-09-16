"""Wiring for the ingestion services."""

from __future__ import annotations

from blob_store.seaweedfs import DocumentsBucket, ParsedBucket
from ingestion.config import Settings
from ingestion.removal import RemovalService
from ingestion.repository import DocumentRepository
from ingestion.service import IngestService


def build_service(settings: Settings) -> IngestService:
    """Wires the ingest service and its collaborators.

    Args:
        settings: The loaded ingestion settings.

    Returns:
        The service.
    """
    return IngestService(
        repository=DocumentRepository(),
        store=DocumentsBucket(),
        max_file_size_bytes=settings.max_file_size_bytes,
        allowed_media_types=settings.allowed_mime_types,
        pipeline_version=settings.pipeline_version,
    )


def build_removal() -> RemovalService:
    """Wires the removal service and its collaborators.

    Returns:
        The service.
    """
    return RemovalService(
        repository=DocumentRepository(),
        store=DocumentsBucket(),
        parsed=ParsedBucket(),
    )
