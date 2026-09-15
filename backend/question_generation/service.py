"""The generation flow: claim a topic, deal its facts, write, gate, store."""

from __future__ import annotations

import logging
from typing import ClassVar

from opentelemetry.trace import Span

from database.qa_generator import QuestionRejection, QuestionStatus
from question_generation.config import Settings
from question_generation.generation import QuestionWriter
from question_generation.models import (
    CheckedQuestion,
    JudgedQuestion,
    TopicToCover,
    criteria_of,
)
from question_generation.repository import QuestionCatalog, QuestionQueue
from question_generation.selection import bridged, samples, spread
from question_generation.verification import (
    QuestionChecker,
    near_verdict,
    structural,
)
from stages import StageService
from telemetry import tracer

log = logging.getLogger(__name__)
span = tracer(__name__)

#: How many re-checked questions to write at once.
_RECHECK_BATCH = 500


def reverify(catalog: QuestionCatalog, settings: Settings, within=None) -> int:
    """Puts every stored question through the gates that need no model.

    The counterpart of `extract-revalidate`, and the same bargain: what the
    model wrote is the record of one generation and is kept, and only the
    verdict is replaced. The round trip is left out because it is the one
    gate that costs a call, and the three that are free are the three that
    catch what changes underneath a stored question - a fact re-judged and
    now rejected, a citation deleted by a re-extraction, a duplicate written
    by a later run.

    It only ever rejects. Accepting is a person's, and a re-check that
    un-rejected would quietly overturn somebody's decision on its next run.
    """
    verdicts: list[tuple[int, str]] = []
    written = 0

    def flush() -> None:
        """Writes what has piled up."""
        nonlocal written
        written += catalog.reject(verdicts)
        verdicts.clear()

    for question in catalog.judged(within):
        failed = _recheck(catalog, settings, question)
        if failed and question.status != QuestionStatus.REJECTED:
            code, reason = failed
            log.info("rejecting question %d: %s (%s)", question.id, reason, code)
            verdicts.append((question.id, code))
        if len(verdicts) >= _RECHECK_BATCH:
            flush()
    flush()
    log.info("re-checked and rejected %d question(s)", written)
    return written


def _recheck(
    catalog: QuestionCatalog, settings: Settings, question: JudgedQuestion
) -> tuple[str, str] | None:
    """Says why a stored question no longer holds, or None if it still does."""
    if not question.facts_validated:
        return (
            QuestionRejection.SOURCE_CHANGED,
            "a fact this question rests on no longer passes its own checks",
        )

    # A cross-document question that lost a citation is still a question and
    # is not deleted: the trigger takes one only when its last fact goes. It
    # is a question whose stored scopes stopped being true, which is a
    # quieter kind of wrong and so worth naming. Compared on the scopes and
    # not on the band, because two different spreads can land in one band
    # and the scopes are what a reader filters on.
    now = criteria_of(
        passages=question.passages,
        documents=question.documents,
        topics=question.topics,
        answer_chars=len(question.target_answer or "") or None,
        follows=question.follows_id is not None,
        long_answer=settings.long_answer_chars,
    )
    moved = [
        f"{name} was {was} and is now {is_now}"
        for name, was, is_now in (
            ("passage_scope", question.passage_scope, now.passage_scope),
            ("document_scope", question.document_scope, now.document_scope),
            ("topic_scope", question.topic_scope, now.topic_scope),
        )
        if was and was != is_now
    ]
    if moved:
        return QuestionRejection.SOURCE_CHANGED, "; ".join(moved)

    failed = structural(
        question_text=question.question_text,
        target_answer=question.target_answer,
        answerable=question.answerable,
        language=question.language,
        statements=question.statements,
        min_answer_chars=settings.min_answer_chars,
    )
    if failed:
        return failed

    if question.embedding is None:
        return None
    return near_verdict(
        catalog.nearest(question.embedding, before=question.id),
        answerable=question.answerable,
        threshold=settings.duplicate_cosine,
    )


class QuestionGenerationService(StageService):
    """Writes one topic's questions, one topic at a time.

    The unit of work is a topic because a question's subject is one: the
    facts it may be written from are the ones whose passage has this topic as
    its strongest, and a group spanning two of that topic's documents is what
    makes a cross-document question. A topic yielding nothing is not a
    failure, and neither is a question a gate rejected - that one is stored
    with the gate's name, because the share that fails is how the writer is
    judged.
    """

    name: ClassVar[str] = "question_generation"
    unit: ClassVar[str] = "topic"

    def __init__(
        self,
        *,
        repository: QuestionQueue,
        writer: QuestionWriter,
        checker: QuestionChecker,
        settings: Settings,
    ) -> None:
        """Initialises the service with its collaborators."""
        super().__init__(repository)
        self._repository: QuestionQueue = repository
        self._writer = writer
        self._checker = checker
        self._settings = settings

    def process_next(self) -> int | None:
        """Writes the questions for one queued topic and stores them."""
        topic = self._repository.claim()
        if topic is None:
            return None

        with span.start_as_current_span("generate_questions") as current:
            current.set_attribute("topic.id", topic.id)
            current.set_attribute("topic.label", topic.label or "")
            try:
                threads = self._topic(topic, current)
                stored = self._repository.store(topic.id, threads)
                written = [one for thread in threads for one in thread]
                accepted = sum(1 for one in written if one.accepted)
                followups = sum(1 for one in written if one.follows)
                self._done(current)
                current.set_attribute("questions.written", stored)
                current.set_attribute("questions.accepted", accepted)
                current.set_attribute("questions.followups", followups)
                log.info(
                    "topic %d: %d question(s) in %d thread(s), %d accepted "
                    "(%d rejected), %d follow-up(s)",
                    topic.id,
                    stored,
                    len(threads),
                    accepted,
                    stored - accepted,
                    followups,
                )
            except Exception as exc:
                self._fail(topic.id, f"{type(exc).__name__}: {exc}", current)
                log.exception("failed topic %d", topic.id)
        return topic.id

    def _topic(self, topic: TopicToCover, current: Span) -> list[list[CheckedQuestion]]:
        """Writes and gates every question one topic is still missing."""
        if not topic.include_in_coverage:
            # Finished with nothing written, not left `new`: somebody said
            # this topic is not a subject, which is an answer and not work
            # still to do. A queue that never empties says nothing.
            current.set_attribute("questions.skipped", "not in coverage")
            return []

        facts = self._repository.facts(topic.id)
        if not facts:
            current.set_attribute("questions.skipped", "no facts left to ask about")
            return []

        formed = bridged(
            samples(
                facts,
                wanted=self._settings.per_topic,
                size=self._settings.sample_size,
            ),
            self._repository.bridging(topic.id),
            share=self._settings.bridge_share,
            size=self._settings.sample_size,
        )
        current.set_attribute("questions.samples", len(formed))

        threads: list[list[CheckedQuestion]] = []
        accepted: list[CheckedQuestion] = []
        for index, sample in enumerate(formed):
            candidate = self._writer.write(
                sample,
                answerable=not spread(index, self._settings.unanswerable_share),
            )
            # Against what this run has accepted as well as what the database
            # holds: nothing is stored until the topic is finished, so
            # without it a topic would happily write the same question twice.
            checked = self._checker.check(candidate, accepted)
            thread = [checked]
            if checked.accepted:
                accepted.append(checked)
                if spread(index, self._settings.followup_share):
                    thread += self._followups(sample, checked, accepted)
            threads.append(thread)
        return threads

    def _followups(
        self,
        sample,
        root: CheckedQuestion,
        accepted: list[CheckedQuestion],
    ) -> list[CheckedQuestion]:
        """Writes the questions somebody would ask after this one.

        Only after an accepted, answerable root. A thread whose first turn
        has no answer has nothing to follow on from - the chatbot was
        supposed to say it did not know - and a thread whose root a gate
        refused is a conversation starting with a question nobody would ask.

        Stops at the first follow-up a gate refuses. The refused one is
        stored, because it is drop-rate evidence like any other, but nothing
        is written after it: a third turn following a second that was thrown
        out is a conversation with a hole in it.
        """
        if not root.answerable:
            return []

        turns: list[tuple[str, str | None]] = [(root.question_text, root.target_answer)]
        written: list[CheckedQuestion] = []
        for _ in range(self._settings.max_followups):
            candidate = self._writer.follow_up(sample, tuple(turns))
            checked = self._checker.check(candidate, accepted)
            written.append(checked)
            if not checked.accepted:
                break
            accepted.append(checked)
            turns.append((checked.question_text, checked.target_answer))
        return written
