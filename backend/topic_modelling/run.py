"""Command-line entry point for the topic modelling service.

Every flag is the same operation as the route beside it under /topics.
`--discover` only queues a fit; pass it with no other flag to queue and drain
in one go. Run with --help for the list.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import telemetry
from blob_store.seaweedfs import ExportBucket
from database.qa_generator import engine
from settings import decimal
from settings.store import snapshot
from stages import watch
from stages.cli import parser, reloading
from topic_modelling.config import Settings
from topic_modelling.factory import build_service
from topic_modelling.repository import TopicCatalog, TopicQueue

log = logging.getLogger(__name__)

#: This stage's actions. No --start or --rerun: a refit and a first fit are
#: the same request.
_ACTIONS = {
    "status": "report the fit queue and the topics held",
    "discover": "queue a fit over the whole corpus",
    "visualise": "write each language's pyLDAvis page to a file",
    "stop": "withdraw a queued fit",
    "retry": "return a failed fit to the queue",
    "delete": "delete every topic and membership",
}

#: Where --visualise writes, one file per modelled language.
_DRAWN = Path("topics")


def main(argv: list[str] | None = None) -> int:
    """Runs the topic modelling command line.

    Args:
        argv: The arguments, defaulting to the process's own.

    Returns:
        The exit status.
    """
    args = parser("topic_modelling.run", _ACTIONS).parse_args(
        sys.argv[1:] if argv is None else argv
    )

    def build():
        """Builds the service from the settings as they stand.

        Called rather than captured: a watching worker rebuilds when a
        setting changes, so a fit queued after one answers to the new value.

        One read for the settings and the version, so the parameters a fit
        used and the version recorded on its topics cannot disagree.
        """
        source, version = snapshot()
        return build_service(Settings.load(source), version)

    telemetry.configure("topic_modelling")
    telemetry.trace_engine(engine())

    if args.status:
        log.info("topics and the fit queue: %s", TopicQueue().counts())
        return 0

    if args.visualise:
        return _visualise()

    if args.delete:
        removed = TopicCatalog().delete_all()
        # The figures go with the topics, as they do on the route. `delete_all`
        # reports the languages whose figure is now orphaned and this read that
        # list and dropped it, so the command left two pyLDAvis pages in the
        # export bucket describing topics that no longer existed.
        bucket = ExportBucket()
        figures = sum(
            bucket.remove(bucket.topic_visualisation_key(language))
            for language in removed.languages
        )
        log.info(
            "deleted %d topic(s), %d membership(s) and %d figure(s); %d label(s) lost",
            removed.topics,
            removed.memberships,
            figures,
            removed.labels,
        )
        return 0

    if args.stop:
        log.info("withdrew %d queued fit(s)", TopicQueue().stop())
        return 0

    if args.retry:
        log.info("returned %d fit(s) to the queue", TopicQueue().retry())
        return 0

    service = build()

    if args.discover:
        log.info("queued topic fit %d", service.request())

    if not args.watch:
        service.drain()
        return 0

    watch(reloading("topic_modelling", build), decimal("WORKER_POLL_SECONDS"))
    return 0


def _visualise() -> int:
    """Writes every stored visualisation to `_DRAWN`, one file per language.

    Returns:
        1 if no topics are stored or no language had a figure, else 0.
    """
    export = ExportBucket()
    languages = [one.language for one in TopicCatalog().fit_state().languages]
    if not languages:
        log.error("no topics are stored, so there is nothing to draw")
        return 1

    _DRAWN.mkdir(parents=True, exist_ok=True)
    missing = 0
    for language in languages:
        drawn = export.find(export.topic_visualisation_key(language))
        if drawn is None:
            log.warning(
                "no %s visualisation; it is drawn by a fit, so run --discover",
                language,
            )
            missing += 1
            continue
        written = _DRAWN / f"{language}.html"
        written.write_bytes(drawn)
        log.info("wrote %s (%d bytes)", written, len(drawn))
    return 1 if missing == len(languages) else 0


if __name__ == "__main__":
    raise SystemExit(main())
