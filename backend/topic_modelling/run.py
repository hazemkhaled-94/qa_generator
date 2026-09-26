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
from blob_store.s3 import ArchiveBucket, ExportBucket, ModelsBucket
from database.qa_generator import engine
from llm.check import before_work
from settings import decimal
from settings.runs import run_id
from settings.store import snapshot
from stages import watch
from stages.cli import parser, reloading
from stages.prompts import record
from topic_modelling import prompts
from topic_modelling.config import Settings
from topic_modelling.factory import build_service
from topic_modelling.repository import TopicCatalog, TopicQueue
from topic_modelling.space import load

log = logging.getLogger(__name__)

#: This stage's actions. No --start or --rerun: a refit and a first fit are
#: the same request.
_ACTIONS = {
    "status": "report the fit queue and the topics held",
    "discover": "queue a fit over the whole corpus",
    "visualise": "write each language's pyLDAvis page to a file",
    "render": "redraw each language's page from its stored model",
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
    built = parser("topic_modelling.run", _ACTIONS)
    built.add_argument(
        "--language",
        metavar="CODE",
        help="with --render, redraw only this language. One model is kept "
        "per language and the next fit replaces it, so this chooses which "
        "LANGUAGE to redraw and not which fit to redraw from.",
    )
    args = built.parse_args(sys.argv[1:] if argv is None else argv)

    def build():
        """Builds the service from the settings as they stand.

        Called rather than captured: a watching worker rebuilds when a
        setting changes. One read for the settings and the version.
        """
        source, version = snapshot()
        return build_service(Settings.load(source), version)

    # With the run, as `stages/cli.queue_main` does for the other four
    # stages. Without it this stage's spans went to Phoenix's `default`
    # project, so a pass over the pipeline was four projects and a heap.
    telemetry.configure("topic_modelling", run=run_id())
    telemetry.trace_engine(engine())

    if args.status:
        log.info("topics and the fit queue: %s", TopicQueue().counts())
        return 0

    if args.visualise:
        return _visualise()

    if args.render:
        return _render(args.language)

    if args.delete:
        removed = TopicCatalog().delete_all()
        # The figures go with the topics, as they do on the route. `delete_all`
        # reports the languages whose figure is now orphaned and this read that
        # list and dropped it, so the command left two pyLDAvis pages in the
        # export bucket describing topics that no longer existed.
        bucket, archive = ExportBucket(), ArchiveBucket()
        models = ModelsBucket()
        figures = sum(
            archive.take(bucket.name, bucket.topic_visualisation_key(language))
            for language in removed.languages
        )
        # The model goes with the figure. A factorisation of topics that no
        # longer exist describes nothing, and leaving it is the same orphan
        # the figures were before this read `removed.languages`.
        kept = sum(
            archive.take(models.name, models.topic_model_key(language))
            for language in removed.languages
        )
        log.info(
            "archived %d topic(s), %d membership(s), %d figure(s) and %d "
            "model(s); %d label(s) lost",
            removed.topics,
            removed.memberships,
            figures,
            kept,
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

    # As `stages/cli.queue_main` does for the other four. Only when a model
    # is configured: this stage names topics with one where it has one, and
    # falls back to their terms where it does not, so an absent model is a
    # choice and an unanswering one is a fit that would fail on every label.
    if (naming := Settings.load(snapshot()[0]).model) is not None:
        before_work(naming)
        # Here rather than at the top, for the same reason `before_work` is:
        # a deployment with no model never sends this prompt, and a row
        # saying it did would be a record of a call nothing made.
        record(prompts.SERVICE, prompts.catalogue())

    if not args.watch:
        service.drain()
        return 0

    watch(reloading("topic_modelling", build), decimal("WORKER_POLL_SECONDS"))
    return 0


def _render(only: str | None = None) -> int:
    """Redraws each language's figure from its stored model.

    The fit draws one as it goes, and this is how a figure comes back
    without one: `export` holds a view and `models` holds what the view is
    of, so a lost or corrupted page costs a render rather than a re-fit of
    the corpus.

    **The figure it writes replaces the one already there.** The key is one
    per language, and the render is deterministic - the same stored model
    through the same `prepare` gives the same page - so rewriting it is
    idempotent rather than lossy.

    **Which model is not a choice, and cannot be.** One model is kept per
    language because the database holds one fit's topics: `replace` drops
    every row and writes the new fit's. A figure drawn from an older model
    would number its topics off rows that are gone, which is the orphan a
    fit already avoids by taking the old figure away before drawing the
    new one. `only` therefore picks a LANGUAGE.

    Args:
        only: Redraw this language alone, or every modelled one.

    Returns:
        1 if nothing was drawn, else 0.
    """
    # pyLDAvis, imported here for the reason the service defers it: it is
    # needed by this one action and by nothing else the command line does.
    try:
        from topic_modelling.visualisation import render
    except ImportError:
        log.exception("pyLDAvis is missing, so nothing can be drawn")
        return 1

    models, export = ModelsBucket(), ExportBucket()
    languages = [one.language for one in TopicCatalog().fit_state().languages]
    if only is not None:
        languages = [one for one in languages if one == only]
        if not languages:
            log.error("no topics are stored for %s", only)
            return 1
    if not languages:
        log.error("no topics are stored, so there is nothing to draw")
        return 1

    drawn = 0
    for language in languages:
        held = models.find(models.topic_model_key(language))
        if held is None:
            log.warning(
                "no %s model is stored, so its figure cannot be redrawn. Only "
                "a fit since this bucket existed leaves one, so run --discover",
                language,
            )
            continue
        key = export.topic_visualisation_key(language)
        # Away before the new one goes in, exactly as the fit does it: a
        # language whose render raises should have no figure rather than
        # the one belonging to whatever was there before.
        export.remove(key)
        page = render(load(held), language)
        export.put(key, page, content_type=export.TOPIC_VISUALISATION_TYPE)
        drawn += 1
        log.info("redrew the %s figure from its model (%d bytes)", language, len(page))
    return 0 if drawn else 1


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
