"""Shared behaviour for every format pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from preprocessing.parsing.models import Conversion, SourceDocument


class UnsupportedFormat(Exception):
    """Raised when no pipeline handles a document's media type."""


class ConversionFailed(Exception):
    """Raised when a pipeline cannot produce a document model."""


class Pipeline(ABC):
    """Converts one family of document formats into a Docling document.

    Every pipeline takes the same :class:`SourceDocument` and returns the
    same :class:`Conversion`, so the service dispatches without learning
    which format it holds. Anything format-specific - the converter's input
    stream name, whether `scanned` means anything - belongs to the pipeline
    that owns the format.
    """

    #: Media types this pipeline handles. A ClassVar rather than an abstract
    #: property, because a subclass would answer a property with a ClassVar
    #: anyway - which shadows the descriptor rather than implementing it.
    media_types: ClassVar[tuple[str, ...]]

    @abstractmethod
    def convert(self, source: SourceDocument) -> Conversion:
        """Converts one document.

        Raises:
            ConversionFailed: If the document cannot be converted.
        """
