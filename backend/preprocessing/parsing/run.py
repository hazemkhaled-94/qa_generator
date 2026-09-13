"""Command-line entry point for the parsing service.

Every flag is the same operation as the route beside it under /parsing.
Run with --help for the list.
"""

from __future__ import annotations

import sys

from preprocessing.parsing.config import Settings
from preprocessing.parsing.factory import build_service
from preprocessing.parsing.repository import ParseQueue
from stages.cli import queue_main


def main(argv: list[str] | None = None) -> int:
    """Runs the parsing command line."""
    settings = Settings.load()
    return queue_main(
        name="parsing",
        module="preprocessing.parsing.run",
        repository=ParseQueue,
        build_service=lambda: build_service(settings),
        argv=sys.argv[1:] if argv is None else argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
