"""Database access for the topic modelling service."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import delete, func, insert, select, update

from database.qa_generator import Fact, Passage, PassageTopic, Status, Topic
from database.qa_generator.passage_topics import DOMINANT as _DOMINANT
from database.qa_generator.repository import Repository
from stages import Columns, StageQueue
from topic_modelling.models import (
    FittedTopic,
    Fitting,
    LanguageFit,
    PassageVocabulary,
    StoredTopic,
    TopicFit,
    TopicRemoval,
)

#: A row that is a fit rather than a topic. `topics` is both the queue and the
#: result, and this is what separates the two.
_OUTSTANDING = Topic.topic_index.is_(None)

#: How many passages to hold at once while streaming the corpus.
_BATCH = 500

#: The next fit to run. Both conditions are load-bearing: status alone would
#: let a fitted topic that somehow reached `pending` be claimed as a request,
#: and running it would replace the whole table.
_NEXT_PENDING = (
    select(Topic.id)
    .where(_OUTSTANDING, Topic.status == Status.PENDING)
    .order_by(Topic.id)
    .with_for_update(skip_locked=True)
    .limit(1)
    .scalar_subquery()
)


class TopicQueue(StageQueue):
    """Reads the topic modelling queue and stores what each fit produced.

    Extends :class:`StageQueue` rather than :class:`RowQueue`: there is no row
    to start, because asking is what creates one.
    """

    columns = Columns(
        entity=Topic,
        key=Topic.id,
        status=Topic.status,
        error=Topic.error,
        claimed_at=Topic.claimed_at,
    )
    done = Status.MODELLED
    next_pending = _NEXT_PENDING

    def request(self) -> int:
        """Asks for a fit by putting a request row on the queue.

        Replaces any outstanding request rather than adding to it, so asking
        twice queues one fit and asking after a failure clears that failure.
        The topics themselves stay readable until a fit succeeds.
        """
        with self._session.begin() as session:
            session.execute(delete(Topic).where(_OUTSTANDING))
            return session.scalar(
                insert(Topic)
                .values(top_terms=[], status=Status.PENDING)
                .returning(Topic.id)
            )

    def stop(self, within: Any = None) -> int:
        """Withdraws a fit that has been asked for but not started.

        Deletes the request rather than parking it: a request row is only ever
        the asking, so one nobody is going to run is not a state worth keeping.

        `within` is the base's; this stage declares no scope to narrow to.
        """
        with self._session.begin() as session:
            return session.execute(
                delete(Topic).where(_OUTSTANDING, Topic.status == Status.PENDING)
            ).rowcount

    def claim(self) -> int | None:
        """Takes the next requested fit off the queue."""
        claimed = self._claim(Topic.id)
        return claimed.id if claimed else None

    def languages(self) -> list[str]:
        """Lists the languages the corpus holds, commonest first."""
        with self._session() as session:
            return [
                row.language
                for row in session.execute(
                    select(Passage.language, func.count().label("passages"))
                    .where(Passage.language.is_not(None))
                    .group_by(Passage.language)
                    .order_by(func.count().desc())
                ).all()
            ]

    def passages(self, language: str) -> Iterator[PassageVocabulary]:
        """Streams one language's vocabulary, in a stable order.

        Ordered by id so a seeded fit is reproducible, and read in batches so
        the worker's memory follows the batch rather than the corpus.
        """
        with self._session() as session:
            rows = session.execute(
                select(Passage.id, Passage.lemmas)
                .where(Passage.language == language)
                .order_by(Passage.id)
                .execution_options(yield_per=_BATCH)
            )
            for row in rows:
                yield PassageVocabulary(id=row.id, lemmas=list(row.lemmas or []))

    def excerpts(self, passage_ids: list[int]) -> list[str]:
        """Reads the text of a few passages, for naming the topic holding them."""
        if not passage_ids:
            return []
        with self._session() as session:
            return list(
                session.scalars(
                    select(Passage.text).where(Passage.id.in_(passage_ids))
                ).all()
            )

    def labelled_topics(self, language: str) -> list[FittedTopic]:
        """Reads the topics of one language a person has said something about."""
        with self._session() as session:
            return [
                FittedTopic(
                    topic_index=row.topic_index,
                    top_terms=list(row.top_terms or []),
                    label=row.label,
                    labelled_by=row.labelled_by,
                    include_in_coverage=row.include_in_coverage,
                )
                for row in session.execute(
                    select(
                        Topic.topic_index,
                        Topic.top_terms,
                        Topic.label,
                        Topic.labelled_by,
                        Topic.include_in_coverage,
                    ).where(
                        Topic.status == Status.MODELLED,
                        Topic.language == language,
                        Topic.label.is_not(None) | ~Topic.include_in_coverage,
                    )
                ).all()
            ]

    def replace(self, fit_id: int, fittings: list[Fitting]) -> int:
        """Replaces every topic and membership with one run's results.

        Every language in one transaction, because a corpus holding one
        language's new topics beside another's old memberships is a corpus
        whose weights point at the wrong subjects. The claimed request row
        goes with them, which is why this does not finish it: the new rows are
        written `modelled` outright.
        """
        with self._session.begin() as session:
            # Read as values, not as func.now(): these go into an executemany
            # below, where a SQL function cannot be a bound parameter.
            requested_at, fitted_at = session.execute(
                select(
                    select(Topic.requested_at)
                    .where(Topic.id == fit_id)
                    .scalar_subquery(),
                    func.now(),
                )
            ).one()
            session.execute(delete(Topic))

            rows = [
                {
                    "language": fitting.language,
                    "topic_index": topic.topic_index,
                    "top_terms": topic.top_terms,
                    "label": topic.label,
                    "labelled_by": topic.labelled_by,
                    "include_in_coverage": topic.include_in_coverage,
                    "status": Status.MODELLED,
                    "requested_at": requested_at or fitted_at,
                    "fitted_at": fitted_at,
                    "corpus_passages": fitting.passages,
                    "corpus_vocabulary": fitting.vocabulary,
                    "passages_without_topics": fitting.without_topics,
                }
                for fitting in fittings
                for topic in fitting.topics
            ]
            if not rows:
                return 0
            session.execute(insert(Topic), rows)

            # Read back rather than taken from RETURNING, whose row order for
            # an executemany is not guaranteed to match the parameters. Keyed
            # by language too: an index identifies a topic only within one.
            by_index = {
                (row.language, row.topic_index): row.id
                for row in session.execute(
                    select(Topic.language, Topic.topic_index, Topic.id).where(
                        ~_OUTSTANDING
                    )
                ).all()
            }
            memberships = [
                {
                    "passage_id": weight.passage_id,
                    "topic_id": by_index[(fitting.language, weight.topic_index)],
                    "weight": weight.weight,
                }
                for fitting in fittings
                for weight in fitting.weights
            ]
            if memberships:
                session.execute(insert(PassageTopic), memberships)
        return len(memberships)

    def counts(self) -> dict[str, int]:
        """Reports what this service owns, for the status panel."""
        with self._session() as session:
            memberships = (
                session.scalar(select(func.count()).select_from(PassageTopic)) or 0
            )
            covered = (
                session.scalar(
                    select(func.count(func.distinct(PassageTopic.passage_id)))
                )
                or 0
            )
        return {
            **self.counts_by_status(),
            "memberships": memberships,
            "passages_with_a_topic": covered,
        }


class TopicCatalog(Repository):
    """Reads the fitted topics, and records what a person decided about one."""

    def describe(
        self, topic_id: int, *, label: str | None, include_in_coverage: bool
    ) -> StoredTopic | None:
        """Records what a person decided about one topic.

        Returns None if no topic has that id: a fit request is not a topic and
        is not editable.
        """
        with self._session.begin() as session:
            changed = session.execute(
                update(Topic)
                .where(Topic.id == topic_id, ~_OUTSTANDING)
                .values(
                    label=(label or "").strip() or None,
                    labelled_by="person" if (label or "").strip() else None,
                    include_in_coverage=include_in_coverage,
                )
            ).rowcount
        if not changed:
            return None
        return next((topic for topic in self.topics() if topic.id == topic_id), None)

    def delete_all(self) -> TopicRemoval:
        """Removes every topic, and with it every membership.

        Any outstanding request goes too.
        """
        with self._session.begin() as session:
            languages = list(
                session.scalars(
                    select(Topic.language)
                    .where(~_OUTSTANDING, Topic.language.is_not(None))
                    .distinct()
                ).all()
            )
            topics = (
                session.scalar(
                    select(func.count()).select_from(Topic).where(~_OUTSTANDING)
                )
                or 0
            )
            labels = (
                session.scalar(
                    select(func.count())
                    .select_from(Topic)
                    .where(Topic.label.is_not(None))
                )
                or 0
            )
            memberships = (
                session.scalar(select(func.count()).select_from(PassageTopic)) or 0
            )
            session.execute(delete(Topic))
        return TopicRemoval(
            topics=topics,
            memberships=memberships,
            labels=labels,
            languages=languages,
        )

    def fit_state(self) -> TopicFit:
        """Reports the state of the model, one entry per language.

        Each language carries the live passage and membership counts beside
        the ones its fit recorded, which is what makes a stale model visible:
        re-chunking a document deletes its passages and the memberships go
        with them.
        """
        with self._session() as session:
            outstanding = session.execute(
                select(Topic.status, Topic.error, Topic.requested_at)
                .where(_OUTSTANDING)
                .order_by(Topic.id.desc())
                .limit(1)
            ).one_or_none()
            fitted = session.execute(
                select(
                    Topic.language,
                    func.count().label("topics"),
                    func.min(Topic.requested_at).label("requested_at"),
                    func.max(Topic.fitted_at).label("fitted_at"),
                    func.max(Topic.corpus_passages).label("corpus_passages"),
                    func.max(Topic.corpus_vocabulary).label("corpus_vocabulary"),
                    func.max(Topic.passages_without_topics).label("without_topics"),
                )
                .where(~_OUTSTANDING)
                .group_by(Topic.language)
                .order_by(Topic.language)
            ).all()
            live = dict(
                session.execute(
                    select(Passage.language, func.count())
                    .where(Passage.language.is_not(None))
                    .group_by(Passage.language)
                ).all()
            )
            held = dict(
                session.execute(
                    select(Topic.language, func.count(PassageTopic.passage_id))
                    .join(PassageTopic, PassageTopic.topic_id == Topic.id)
                    .where(~_OUTSTANDING)
                    .group_by(Topic.language)
                ).all()
            )
            unplaced = (
                session.scalar(
                    select(func.count())
                    .select_from(Passage)
                    .where(Passage.language.is_(None))
                )
                or 0
            )

        languages = [
            LanguageFit(
                language=row.language,
                topics=row.topics,
                corpus_passages=row.corpus_passages,
                corpus_vocabulary=row.corpus_vocabulary,
                passages_without_topics=row.without_topics,
                fitted_at=row.fitted_at.isoformat() if row.fitted_at else None,
                live_passages=live.get(row.language, 0),
                memberships=held.get(row.language, 0),
            )
            for row in fitted
        ]
        requested_at = (
            outstanding.requested_at
            if outstanding
            else min((row.requested_at for row in fitted), default=None)
        )
        return TopicFit(
            status=(outstanding.status if outstanding else Status.MODELLED)
            if (outstanding or fitted)
            else None,
            error=outstanding.error if outstanding else None,
            requested_at=requested_at.isoformat() if requested_at else None,
            topics=sum(row.topics for row in fitted),
            languages=languages,
            passages_without_language=unplaced,
        )

    def topics(self) -> list[StoredTopic]:
        """Reads the fitted topics with how much of the corpus each holds."""
        # Three queries, not one. Folding either of the first two into the
        # aggregate below would join a second row per passage and inflate both
        # the membership count and the mean weight.
        with self._session() as session:
            # Over the passages this topic owns rather than every passage
            # holding it: a passage belongs to several topics, so counting all
            # of them dilutes whatever separates one topic from another.
            owned = {
                row.topic_id: row
                for row in session.execute(
                    select(
                        _DOMINANT.c.topic_id,
                        func.count().label("passages"),
                        func.count()
                        .filter(Passage.block_type == "table")
                        .label("tables"),
                    )
                    .join(Passage, Passage.id == _DOMINANT.c.passage_id)
                    .group_by(_DOMINANT.c.topic_id)
                ).all()
            }
            drawn = dict(
                session.execute(
                    select(_DOMINANT.c.topic_id, func.count(Fact.id))
                    .join(
                        Fact,
                        (Fact.passage_id == _DOMINANT.c.passage_id) & Fact.validated,
                    )
                    .group_by(_DOMINANT.c.topic_id)
                ).all()
            )
            rows = session.execute(
                select(
                    Topic.id,
                    Topic.language,
                    Topic.topic_index,
                    Topic.top_terms,
                    Topic.label,
                    Topic.labelled_by,
                    Topic.include_in_coverage,
                    func.count(PassageTopic.passage_id).label("passages"),
                    func.coalesce(func.avg(PassageTopic.weight), 0.0).label(
                        "mean_weight"
                    ),
                    func.count(func.distinct(Passage.doc_sha256)).label("documents"),
                )
                .join(PassageTopic, PassageTopic.topic_id == Topic.id, isouter=True)
                .join(Passage, Passage.id == PassageTopic.passage_id, isouter=True)
                .where(~_OUTSTANDING)
                .group_by(
                    Topic.id,
                    Topic.language,
                    Topic.topic_index,
                    Topic.top_terms,
                    Topic.label,
                    Topic.labelled_by,
                    Topic.include_in_coverage,
                )
                .order_by(Topic.language, Topic.topic_index)
            ).all()
        return [
            StoredTopic(
                id=row.id,
                language=row.language,
                topic_index=row.topic_index,
                top_terms=list(row.top_terms or []),
                label=row.label,
                labelled_by=row.labelled_by,
                include_in_coverage=row.include_in_coverage,
                passages=row.passages,
                dominant_passages=owned[row.id].passages if row.id in owned else 0,
                mean_weight=float(row.mean_weight),
                documents=row.documents,
                table_passages=owned[row.id].tables if row.id in owned else 0,
                validated_facts=drawn.get(row.id, 0),
            )
            for row in rows
        ]
