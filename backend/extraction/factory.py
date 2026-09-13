"""Wiring for the extraction service."""

from __future__ import annotations

from extraction.extractors import ExtractorRegistry, LlmExtractor, TableExtractor
from extraction.repository import PassageQueue
from extraction.service import ExtractionService
from extraction.validation import FactChecker
from llm.client import Client
from llm.config import Settings


def build_service(settings: Settings) -> ExtractionService:
    """Wires the service and its collaborators.

    Takes the served model's settings directly: everything else this stage
    reads is in configs/env/backend.env and reaches it through the objects
    below, so there is no extraction configuration of its own.
    """
    return ExtractionService(
        repository=PassageQueue(lease=settings.lease),
        extractors=ExtractorRegistry(
            extractors=(TableExtractor(),),
            default=LlmExtractor(Client(settings)),
        ),
        checker=FactChecker(),
    )
