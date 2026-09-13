"""Command-line entry point for the chunking service.

Every flag is the same operation as the route beside it under /chunking.
Run with --help for the list.
"""

from __future__ import annotations

import sys

from preprocessing.chunking.config import Settings
from preprocessing.chunking.factory import build_service
from preprocessing.chunking.repository import ChunkQueue, PassageCatalog
from preprocessing.chunking.service import revocabulary
from stages.cli import queue_main

#: This stage's own operation. Re-reads the vocabulary of passages already
#: stored, in place, so a change to how vocabulary is read reaches the topic
#: model without a re-chunk deleting every passage and its facts.
_EXTRA = {
    "revocabulary": (
        "read every stored passage's vocabulary again, in place",
        "read again",
        lambda within: revocabulary(PassageCatalog(), within),
    )
}


def main(argv: list[str] | None = None) -> int:
    """Runs the chunking command line."""
    settings = Settings.load()
    return queue_main(
        name="chunking",
        module="preprocessing.chunking.run",
        repository=ChunkQueue,
        build_service=lambda: build_service(settings),
        argv=sys.argv[1:] if argv is None else argv,
        extra=_EXTRA,
    )


if __name__ == "__main__":
    raise SystemExit(main())
