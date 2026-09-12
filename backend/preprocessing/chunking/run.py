"""Command-line entry point for the chunking service.

Every flag is the same operation as the route beside it under /chunking.
Run with --help for the list.
"""

from __future__ import annotations

import sys

from preprocessing.chunking.config import Settings
from preprocessing.chunking.factory import build_service
from preprocessing.chunking.repository import ChunkQueue
from stages.cli import queue_main


def main(argv: list[str] | None = None) -> int:
    """Runs the chunking command line."""
    settings = Settings.load()
    return queue_main(
        name="chunking",
        module="preprocessing.chunking.run",
        log_level=settings.log_level,
        repository=ChunkQueue,
        build_service=lambda: build_service(settings),
        argv=sys.argv[1:] if argv is None else argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
