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
from settings.store import snapshot
from stages.cli import queue_main

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Runs the extraction command line.

    Args:
        argv: The arguments to parse, or None to read them from sys.argv.

    Returns:
        The process exit code.
    """

    def configured() -> tuple[ModelSettings, Settings, str]:
        """Reads the settings as they stand, and what that configuration is.

        Called rather than captured, so every operation below answers to a
        value written since this process started. One read for both, so the
        settings and the version recorded cannot come from two moments.
        """
        source, version = snapshot()
        return ModelSettings.load(source), Settings.load(source), version

    def build():
        """Builds the service, naming the model it will call."""
        model, settings, version = configured()
        # The effective model, which EXTRACTION_MODEL may have replaced.
        calling = model.overridden(settings.model)
        log.info("extracting with %s at %s", calling.model, calling.base_url)
        return build_service(model, settings, version)

    def run_bridge(within) -> int:
        """Reads every topic's passage groups for the claims they share."""
        model, settings, version = configured()
        return bridge(
            FactCatalog(version=version),
            build_bridge(model, settings),
            FactChecker(settings.digest_share),
            settings.bridges_per_topic,
            settings.bridge_passages,
            within,
        )

    def run_revalidate(within) -> int:
        """Judges every stored fact again, under the settings as they stand."""
        _, settings, version = configured()
        return revalidate(
            FactCatalog(version=version),
            FactChecker(settings.digest_share),
            within,
        )

    def run_recap(within) -> int:
        """Refuses the atomic facts already stored above the cap."""
        _, settings, version = configured()
        return recap(FactCatalog(version=version), settings.atomic_cap, within)

    #: This stage's own operations. `revalidate` calls no model: it re-reads
    #: what the checks read off facts already stored, which is how a change to
    #: the checks reaches facts extracted before it.
    extra = {
        "revalidate": (
            "judge every stored fact again, without calling the model",
            "judged again",
            run_revalidate,
        ),
        "bridge": (
            "read every topic's passage groups for the claims they share",
            "bridged",
            run_bridge,
        ),
        "recap": (
            "refuse the atomic facts over the cap, without calling the model",
            "refused",
            run_recap,
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
