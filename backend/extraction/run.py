"""Command-line entry point for the extraction service.

Every flag is the same operation as the route beside it under /extraction.
Run with --help for the list.
"""

from __future__ import annotations

import logging
import sys

from extraction.config import Settings
from extraction.factory import build_service
from extraction.repository import PassageQueue
from stages.cli import queue_main

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Runs the extraction command line."""
    settings = Settings.load()

    def build():
        """Builds the service, naming the model it will call."""
        log.info("extracting with %s at %s", settings.model, settings.model_base_url)
        return build_service(settings)

    return queue_main(
        name="extraction",
        module="extraction.run",
        log_level=settings.log_level,
        repository=lambda: PassageQueue(lease=settings.lease),
        build_service=build,
        argv=sys.argv[1:] if argv is None else argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
