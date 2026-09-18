"""Command-line entry point for the extraction service.

Every queue flag is the same operation as the route beside it under
/extraction. `--revalidate` and `--bridge` are this stage's own. Run with
--help for the list.
"""

from __future__ import annotations

import logging
import sys

from extraction.config import Settings
from extraction.factory import build_bridge, build_service
from extraction.repository import FactCatalog, PassageQueue
from extraction.service import bridge, recap, revalidate
from extraction.validation import FactChecker
from llm.config import Settings as ModelSettings
from settings.store import resolved
from stages.cli import queue_main

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Runs the extraction command line.

    Args:
        argv: The arguments to parse, or None to read them from sys.argv.

    Returns:
        The process exit code.
    """

    def configured() -> tuple[ModelSettings, Settings]:
        """Reads the settings as they stand, stored values included.

        Called rather than captured, so every operation below answers to a
        value written since this process started. A worker that read them
        once would answer to the file it booted with for as long as it ran.
        """
        source = resolved()
        return ModelSettings.load(source), Settings.load(source)

    def build():
        """Builds the service, naming the model it will call."""
        model, settings = configured()
        # The effective model, not the shared one: EXTRACTION_MODEL is what
        # this stage will actually call, and a line naming the other is a
        # line that sends somebody looking in the wrong place.
        calling = model.overridden(settings.model)
        log.info("extracting with %s at %s", calling.model, calling.base_url)
        return build_service(model, settings)

    def run_bridge(within) -> int:
        """Reads every topic's passage groups for the claims they share."""
        model, settings = configured()
        return bridge(
            FactCatalog(),
            build_bridge(model, settings),
            FactChecker(settings.digest_share),
            settings.bridges_per_topic,
            settings.bridge_passages,
            within,
        )

    #: This stage's own operations. `revalidate` calls no model: it re-reads
    #: what the checks read off facts already stored, which is how a change to
    #: the checks reaches facts extracted before it.
    extra = {
        "revalidate": (
            "judge every stored fact again, without calling the model",
            "judged again",
            lambda within: revalidate(FactCatalog(), within),
        ),
        "bridge": (
            "read every topic's passage groups for the claims they share",
            "bridged",
            run_bridge,
        ),
        "recap": (
            "refuse the atomic facts over the cap, without calling the model",
            "refused",
            lambda within: recap(FactCatalog(), configured()[1].atomic_cap, within),
        ),
    }

    return queue_main(
        name="extraction",
        module="extraction.run",
        repository=lambda: PassageQueue(lease=configured()[0].lease),
        build_service=build,
        argv=sys.argv[1:] if argv is None else argv,
        extra=extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
