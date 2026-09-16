"""Wiring for the extraction service."""

from __future__ import annotations

from extraction.config import Settings
from extraction.extractors import (
    BridgeExtractor,
    DigestExtractor,
    ExtractorRegistry,
    LlmExtractor,
    TableExtractor,
)
from extraction.repository import PassageQueue
from extraction.service import ExtractionService
from extraction.validation import FactChecker
from llm.client import Client
from llm.config import Settings as ModelSettings


def build_service(
    model: ModelSettings, settings: Settings | None = None
) -> ExtractionService:
    """Wires the per-passage service and its collaborators.

    Args:
        model: Which model to call, where, and how patiently.
        settings: What extraction writes. Read from the environment when not
            given.

    Returns:
        The service a worker drains the passage queue with.
    """
    settings = settings or Settings.load()
    client = Client(model)
    digests = settings.digests
    return ExtractionService(
        repository=PassageQueue(lease=model.lease),
        extractors=ExtractorRegistry(
            extractors=(TableExtractor(),), default=LlmExtractor(client)
        ),
        checker=FactChecker(settings.digest_share),
        digest=DigestExtractor(client, digests) if digests else None,
    )


def build_bridge(model: ModelSettings) -> BridgeExtractor:
    """Builds the reader of a group of passages.

    Args:
        model: Which model to call, where, and how patiently.

    Returns:
        The extractor the bridge pass calls once per group.
    """
    return BridgeExtractor(Client(model))
