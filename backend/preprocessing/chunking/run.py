"""Command-line entry point for the chunking service.

Every flag is the same operation as the route beside it under /chunking. Run
with --help for the list.
"""

from __future__ import annotations

import sys

from preprocessing.chunking.config import Settings
from preprocessing.chunking.factory import build_service
from preprocessing.chunking.repository import ChunkQueue, PassageCatalog
from preprocessing.chunking.service import revocabulary
from settings.store import resolved
from stages.cli import queue_main

#: This stage's own operation, beyond the shared queue verbs: re-reads the
#: language and vocabulary of passages already stored, in place.
_EXTRA = {
    "revocabulary": (
        "read every stored passage's language and vocabulary again",
        "read again",
        lambda within: revocabulary(PassageCatalog(), within),
    )
}


def main(argv: list[str] | None = None) -> int:
    """Runs the chunking command line.

    Args:
        argv: Arguments to parse, defaulting to the process's own.

    Returns:
        The process exit code.
    """
    return queue_main(
        name="chunking",
        module="preprocessing.chunking.run",
        repository=ChunkQueue,
        # Read when the service is built rather than captured here, so a
        # watching worker answers to a value written since it started.
        # Building it loads the tokenizer EMBEDDING_MODEL names.
        build_service=lambda: build_service(Settings.load(resolved())),
        argv=sys.argv[1:] if argv is None else argv,
        extra=_EXTRA,
    )


if __name__ == "__main__":
    raise SystemExit(main())
