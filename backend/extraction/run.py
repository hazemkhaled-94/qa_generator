"""Command-line entry point for the extraction service.

Every flag is the same operation as the route beside it under /extraction.
Run with --help for the list.
"""

from __future__ import annotations

import logging
import sys

from extraction.factory import build_service
from extraction.repository import FactCatalog, PassageQueue
from extraction.service import revalidate
from llm.config import Settings
from stages.cli import queue_main

log = logging.getLogger(__name__)

#: This stage's own operation. The model is not called: it re-reads what the
#: checks read off facts already stored, which is how a change to the checks
#: reaches facts extracted before it.
_EXTRA = {
    "revalidate": (
        "judge every stored fact again, without calling the model",
        "judged again",
        lambda within: revalidate(FactCatalog(), within),
    )
}


def main(argv: list[str] | None = None) -> int:
    """Runs the extraction command line."""
    settings = Settings.load()

    def build():
        """Builds the service, naming the model it will call."""
        log.info("extracting with %s at %s", settings.model, settings.base_url)
        return build_service(settings)

    return queue_main(
        name="extraction",
        module="extraction.run",
        repository=lambda: PassageQueue(lease=settings.lease),
        build_service=build,
        argv=sys.argv[1:] if argv is None else argv,
        extra=_EXTRA,
    )


if __name__ == "__main__":
    raise SystemExit(main())
