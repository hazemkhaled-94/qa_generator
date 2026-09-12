"""Command-line entry point for the topic modelling service.

Every flag is the same operation as the route beside it under /topics.
`--discover` only queues a fit; pass it with no other flag to queue and drain
in one go. Run with --help for the list.
"""

from __future__ import annotations

import logging
import sys

import telemetry
from database.qa_generator import engine
from settings import decimal
from stages import watch
from stages.cli import parser
from topic_modelling.config import Settings
from topic_modelling.factory import build_service
from topic_modelling.repository import TopicCatalog, TopicQueue

log = logging.getLogger(__name__)

#: This stage's actions. No --start or --rerun: asking is what creates the
#: row, so a refit and a first fit are the same request.
_ACTIONS = {
    "status": "report the fit queue and the topics held",
    "discover": "queue a fit over the whole corpus",
    "stop": "withdraw a queued fit",
    "retry": "return a failed fit to the queue",
    "delete": "delete every topic and membership",
}


def main(argv: list[str] | None = None) -> int:
    """Runs the topic modelling command line."""
    args = parser("topic_modelling.run", _ACTIONS).parse_args(
        sys.argv[1:] if argv is None else argv
    )

    settings = Settings.load()
    telemetry.configure("topic_modelling", settings.log_level)
    telemetry.trace_engine(engine())

    if args.status:
        log.info("topics and the fit queue: %s", TopicQueue().counts())
        return 0

    if args.delete:
        removed = TopicCatalog().delete_all()
        log.info(
            "deleted %d topic(s) and %d membership(s); %d label(s) lost",
            removed.topics,
            removed.memberships,
            removed.labels,
        )
        return 0

    if args.stop:
        log.info("withdrew %d queued fit(s)", TopicQueue().stop())
        return 0

    if args.retry:
        log.info("returned %d fit(s) to the queue", TopicQueue().retry())
        return 0

    service = build_service(settings)

    if args.discover:
        log.info("queued topic fit %d", service.request())

    if not args.watch:
        service.drain()
        return 0

    watch(service.drain, decimal("WORKER_POLL_SECONDS"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
