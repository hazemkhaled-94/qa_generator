"""Wiring for the chunking service."""

from __future__ import annotations

from blob_store.seaweedfs import ParsedBucket
from preprocessing.chunking.config import Settings
from preprocessing.chunking.passages import PassageBuilder
from preprocessing.chunking.repository import ChunkQueue
from preprocessing.chunking.service import ChunkingService


def build_service(settings: Settings) -> ChunkingService:
    """Wires the service and its collaborators.

    Args:
        settings: The loaded chunking settings.

    Returns:
        The service. Building it loads the embedding model's tokenizer.
    """
    return ChunkingService(
        repository=ChunkQueue(),
        parsed=ParsedBucket(),
        builder=PassageBuilder(
            embedding_model=settings.embedding_model,
            max_tokens=settings.max_tokens,
            merge_peers=settings.merge_peers,
        ),
    )
