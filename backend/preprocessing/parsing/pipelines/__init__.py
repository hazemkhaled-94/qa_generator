"""One pipeline per document format, dispatched by media type.

Adding a format is a module here and an entry in the registry.
"""

from __future__ import annotations

from preprocessing.parsing.pipelines.base import (
    ConversionFailed,
    Pipeline,
    UnsupportedFormat,
)
from preprocessing.parsing.pipelines.pdf import PdfPipeline

__all__ = [
    "ConversionFailed",
    "PdfPipeline",
    "Pipeline",
    "PipelineRegistry",
    "UnsupportedFormat",
]


class PipelineRegistry:
    """Chooses the pipeline for a document's media type."""

    def __init__(self, pipelines: tuple[Pipeline, ...]) -> None:
        """Initialises the registry.

        Args:
            pipelines: The pipelines to dispatch between.

        Raises:
            ValueError: If two pipelines claim the same media type.
        """
        self._by_media_type: dict[str, Pipeline] = {}
        for pipeline in pipelines:
            for media_type in pipeline.media_types:
                if media_type in self._by_media_type:
                    raise ValueError(f"two pipelines claim {media_type}")
                self._by_media_type[media_type] = pipeline

    def for_media_type(self, media_type: str) -> Pipeline:
        """Returns the pipeline that handles a media type.

        Args:
            media_type: The type recorded at ingest.

        Returns:
            The pipeline for that type.

        Raises:
            UnsupportedFormat: If no pipeline handles the type.
        """
        pipeline = self._by_media_type.get(media_type)
        if pipeline is None:
            raise UnsupportedFormat(
                f"no pipeline handles {media_type}; "
                f"known: {', '.join(sorted(self._by_media_type))}"
            )
        return pipeline

    @property
    def media_types(self) -> tuple[str, ...]:
        """Every media type the registry can dispatch, sorted."""
        return tuple(sorted(self._by_media_type))
