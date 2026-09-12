"""Wiring for the extraction service."""

from __future__ import annotations

from extraction.config import Settings
from extraction.extractors import ExtractorRegistry, LlmExtractor, TableExtractor
from extraction.repository import PassageQueue
from extraction.service import ExtractionService
from extraction.validation import FactChecker


def build_service(settings: Settings) -> ExtractionService:
    """Wires the service and its collaborators."""
    return ExtractionService(
        repository=PassageQueue(lease=settings.lease),
        extractors=ExtractorRegistry(
            extractors=(TableExtractor(),),
            default=LlmExtractor(
                model=settings.model,
                base_url=settings.model_base_url,
                temperature=settings.temperature,
                timeout=settings.timeout_seconds,
                max_attempts=settings.max_attempts,
                structured_mode=settings.structured_mode,
            ),
        ),
        checker=FactChecker(),
    )
