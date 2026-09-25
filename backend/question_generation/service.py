"""The generation flow: claim a topic, deal its facts, write, gate, store."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import ClassVar

from opentelemetry.trace import Span

from confidence import confidence
from database.qa_generator import (
    AnswerForm,
    Difficulty,
    QuestionRejection,
    QuestionStatus,
)
from question_generation.balance import choose, composition, largest, quota_of
from question_generation.catalog import QuestionCatalog
from question_generation.checker import (
    PASSED,
    REFUSED,
    QuestionChecker,
    annotation,
)
from question_generation.config import Settings
from question_generation.gates import names_parties, near_verdict, structural
from question_generation.generation import QuestionWriter
from question_generation.models import (
    Candidate,
    CheckedQuestion,
    FactGroup,
    JudgedQuestion,
    SourceFact,
    TopicToCover,
    criteria_of,
)
from question_generation.planning import Plan, plans
from question_generation.queue import QuestionQueue
from question_generation.runs import ACCEPTED
from question_generation.selection import Deal, spread
from question_generation.types import PROMPT_VERSION, SPECS, spec
from settings.runs import run_id
from stages import StageService
from telemetry import tracer, working
from telemetry.evaluations import Evaluations, Verdict, current_ids

log = logging.getLogger(__name__)
span = tracer(__name__)

#: How many re-checked questions to write at once.
_RECHECK_BATCH = 500

#: The bands in order, for asking whether a question came out below the one
#: its plan wanted.
_RANK: dict[str, int] = {
    Difficulty.EASY: 0,
    Difficulty.MEDIUM: 1,
    Difficulty.HARD: 2,
}

#: What to tell the writer about a draft a gate refused, by the gate that
#: refused it. Only the gates a second attempt can do anything about are
#: here: each of these is the writer having written the question badly
#: rather than the material not supporting one, and saying what went wrong
#: is worth one more call. `duplicate` is deliberately absent - the same
#: sample asked again produces another near-twin of the same question.
_NOTES: dict[str, str] = {
    QuestionRejection.MALFORMED: (
        "it was not one well-formed question. Write exactly one, ending in a "
        "single question mark, with the answer in the field for it."
    ),
    QuestionRejection.COMPOUND: (
        "it asked two things joined by 'and', so a chatbot answering half of "
        "it would be neither right nor wrong. Ask ONE thing. If the facts "
        "have no single question between them, ask about one fact alone and "
        "cite only that one."
    ),
    QuestionRejection.WRONG_FORM: (
        "the answer was not the shape this kind of question asks for. Read "
        "the answer rule again and write an answer of that shape."
    ),
    QuestionRejection.ANSWER_TOO_SHORT: (
        "the answer was too short to score anything against. Ask a question "
        "whose answer the material states in full."
    ),
    QuestionRejection.ANSWER_TOO_LONG: (
        "the answer ran far past the length this kind of question allows. "
        "Ask something narrower, with a shorter answer."
    ),
    QuestionRejection.LEAKS_SOURCE: (
        "it said WHERE the answer is - it named or quoted a document, a "
        "section or a heading. Name the subject instead: the thing, the "
        "party, the duty, the period."
    ),
    QuestionRejection.UNANCHORED: (
        "it named nothing a person searching would know to type. Name the "
        "subject in the question itself, so somebody who has not read the "
        "passage could have asked it."
    ),
    QuestionRejection.NOT_RECOVERABLE: (
        "a second reader could not get your answer back out of the passages. "
        "Ask about something the facts state plainly, and write the answer "
        "in the words of the material."
    ),
    QuestionRejection.OFF_TOPIC: (
        "it was about something this material never mentions, so any chatbot "
        "declines it and declining it proves nothing. Stay on the subject of "
        "the fact and move one detail of it out of reach."
    ),
    QuestionRejection.ANSWERABLE_AFTER_ALL: (
        "the passages answered it, and this question is supposed to have no "
        "answer. Move further out of reach: change the party, the period or "
        "the category to one the material does not cover."
    ),
    QuestionRejection.ANSWERABLE_ELSEWHERE: (
        "another part of the corpus answers it, and this question is "
        "supposed to have no answer anywhere. Ask for a detail of this "
        "subject that no document would carry."
    ),
}

#: Told to the writer when a question was accepted but came out narrower
#: than the band its slot asked for. Not a rejection: the draft stands if
#: the second attempt is worse.
_WIDEN = (
    "it was accepted, but it used facts from only one passage where this "
    "slot needs a question that genuinely requires both. Ask ONE question "
    "that cannot be answered without facts from each passage, and cite them "
    "all. Still one question about one thing - do not weld two together."
)


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
        # What this pass has decided to reject and not yet written. Held out
        # of the duplicate probe, which reads the stored status and would
        # otherwise answer differently either side of a flush.
        pending = [question_id for question_id, _ in verdicts]
        failed = _recheck(catalog, settings, question, pending)
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
    catalog: QuestionCatalog,
    settings: Settings,
    question: JudgedQuestion,
    pending: Sequence[int] = (),
) -> tuple[str, str] | None:
    """Says why a stored question no longer holds, or None if it still does.

    `pending` is what this pass has already rejected and not yet written,
    which the duplicate probe must not offer back as an accepted twin.
    """
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

    failed, _ = structural(
        question_text=question.question_text,
        target_answer=question.target_answer,
        answerable=question.answerable,
        language=question.language,
        statements=question.statements,
        # The form the row was written under, not the ones its type will
        # take today: a type whose answer forms changed under a stored
        # question would otherwise re-judge every question written before
        # the change, and a re-check that moves a verdict for that reason
        # is measuring the settings rather than the row.
        forms=(question.answer_form or spec(question.question_type).form,),
        bounds=settings.answer_chars,
    )
    if failed:
        return failed

    if question.embedding is None:
        return None
    return near_verdict(
        catalog.nearest(question.embedding, before=question.id, excluding=pending),
        answerable=question.answerable,
        threshold=settings.duplicate_cosine,
    )


def balance(catalog: QuestionCatalog, settings: Settings, within=None) -> int:
    """Draws a balanced release out of everything the run accepted.

    Args:
        catalog: Where the accepted questions are read and the draw written.
        settings: The shares the release is held to.
        within: A condition narrowing which questions may be drawn, or None.

    Returns:
        How many questions were drawn into the release.
    """
    pool = catalog.releasable(within)
    if not pool:
        log.warning("no accepted question carries both a difficulty and a type")
        return 0

    bands = _supplied(pool, settings.release_difficulty, "band", "difficulty")
    types = _supplied(pool, settings.type_mix, "kind", "question_type")
    if not bands or not types:
        log.warning("nothing accepted can fill any quota; no release was drawn")
        catalog.release([])
        return 0

    if settings.release_size:
        release = choose(
            pool,
            quota_of(
                settings.release_size,
                bands,
                types,
                settings.release_unanswerable,
            ),
        )
    else:
        release = largest(pool, bands, types, settings.release_unanswerable)

    drawn, written = catalog.release(release.ids)
    _report(pool, release, drawn, written)
    return written


def _supplied(pool, weights, what: str, attribute: str) -> dict[str, int]:
    """The weights narrowed to the buckets this pool can actually supply.

    A quota nothing can fill makes the whole draw empty rather than smaller:
    the choosing needs every bucket at zero together, so one bucket with no
    questions in it refuses every size, down to one. Over 29 accepted
    questions spread across eleven kinds that is exactly what happened -
    most kinds had nothing, and the release came out empty.

    Dropping the bucket is the lesser wrong, and it is said out loud. What
    the release then holds is reported off the rows either way, so a missing
    kind or a missing band shows up in the composition rather than being
    promised and quietly not delivered.
    """
    present = {getattr(row, attribute) for row in pool}
    kept = {
        name: weight
        for name, weight in weights.items()
        if weight > 0 and name in present
    }
    dropped = [
        name for name, weight in weights.items() if weight > 0 and name not in present
    ]
    if dropped:
        log.warning(
            "no accepted question is of %s %s, so the release has no quota for "
            "%s. Generate more, or accept a set without %s.",
            what,
            ", ".join(sorted(dropped)),
            "them" if len(dropped) > 1 else "it",
            "them" if len(dropped) > 1 else "it",
        )
    return kept


def _report(pool, release, drawn, written: int) -> None:
    """Logs what the draw came out as, and what it could not fill."""
    log.info(
        "release %s: %d of %d accepted question(s), a yield of %.0f%%",
        drawn,
        written,
        len(pool),
        100.0 * written / len(pool),
    )
    for name, counted in composition(pool, release.ids).items():
        log.info(
            "  %-11s %s",
            name,
            ", ".join(
                f"{key} {value} ({100.0 * value / max(written, 1):.0f}%)"
                for key, value in sorted(counted.items())
            ),
        )
    if release.short:
        log.warning(
            "the pool could not fill: %s. Generate more, or lower the share "
            "that is short.",
            ", ".join(
                f"{name} short by {count}" for name, count in release.short.items()
            ),
        )


def again(checked: CheckedQuestion, plan: Plan) -> str | None:
    """What to tell the writer about this draft, or None to keep it as it is.

    Two reasons to ask a second time, and they are not the same kind of
    reason.

    A gate refused it, and it is a gate the writer could have satisfied.
    Those are most of the rejections that are not about the material: over
    one corpus, 21 compound questions, 11 malformed ones and 43 in all were
    thrown away for how the question was written rather than for what it
    was about.

    Or it was accepted and came out below the band its slot asked for. That
    is the other half of the difficulty problem: the writer is offered two
    passages, takes the escape hatch its own rules give it, cites one fact,
    and the question bands easy. Planned-medium collapsed to easy 83 times
    and planned-hard 30 times, which is why the accepted set came out 96.5%
    easy however the mix was set. This one only ever asks - the accepted
    draft stands if the second attempt turns out worse.
    """
    if checked.rejected_reason in _NOTES:
        return _NOTES[checked.rejected_reason]
    if (
        checked.accepted
        and plan.answerable
        and plan.spans
        and _RANK[checked.criteria.difficulty] < _RANK[plan.band]
    ):
        return _WIDEN
    return None


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
        # Held for the life of the service rather than per topic: the
        # client is one HTTP connection and the batch spans topics, so a
        # topic writing nine questions does not cost a call of its own.
        self._evaluations = Evaluations()

    def process_next(self) -> int | None:
        """Writes the questions for one queued topic and stores them."""
        topic = self._repository.claim()
        if topic is None:
            return None

        with working(
            span,
            "generate_questions",
            {
                "stage": self.name,
                "topic.id": topic.id,
                "topic.label": topic.label or "",
            },
        ) as current:
            try:
                threads = self._topic(topic, current)
                stored = self._repository.store(topic.id, threads)
                written = [one for thread in threads for one in thread]
                accepted = sum(1 for one in written if one.accepted)
                followups = sum(1 for one in written if one.follows)
                self._done(current)
                # Whatever is under the batch size, now that the topic is
                # finished. Without it the last topic of every run is the
                # one missing from Phoenix.
                self._evaluations.flush()
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
                    extra={
                        "questions.written": stored,
                        "questions.threads": len(threads),
                        "questions.accepted": accepted,
                        "questions.rejected": stored - accepted,
                        "questions.followups": followups,
                    },
                )
            except Exception as exc:
                self._fail(topic.id, f"{type(exc).__name__}: {exc}", current)
                log.exception("failed topic %d", topic.id)
        return topic.id

    def _topic(self, topic: TopicToCover, current: Span) -> list[list[CheckedQuestion]]:
        """Writes and gates every question one topic is still missing.

        The plan comes first and the material second. Each slot says what kind
        of question to write and how wide a sample that needs; the deal
        supplies the widest sample the topic can actually offer, and the
        question is written to whatever came back. A topic sitting in one
        document has no cross-document question in it, and asking for one is
        not a reason to write nothing about that topic.
        """
        if not topic.include_in_coverage:
            # Finished with nothing written, not left `new`: somebody said
            # this topic is not a subject, which is an answer and not work
            # still to do. A queue that never empties says nothing.
            current.set_attribute("questions.skipped", "not in coverage")
            return []

        facts = self._subjects(self._repository.facts(topic.id))
        if not facts:
            current.set_attribute("questions.skipped", "no facts left to ask about")
            return []

        deal = Deal(
            facts,
            self._subjects(self._repository.bridging(topic.id)),
            wanted=self._settings.per_topic,
            size=self._settings.sample_size,
            rounds=self._settings.samples_per_passage,
            floor=self._settings.meets_floor,
        )
        planned = plans(
            wanted=self._settings.per_topic,
            types=self._settings.type_mix,
            bands=self._settings.difficulty_mix,
            unanswerable_share=self._settings.unanswerable_share,
        )

        threads: list[list[CheckedQuestion]] = []
        accepted: list[CheckedQuestion] = []
        written = 0
        followed = 0
        #: Accepted roots so far, which is what the follow-up share is a
        #: share OF. Counted here rather than taken from `accepted`, which
        #: a follow-up joins as soon as it is kept.
        roots = 0
        for plan in planned:
            # A reading is written from a passage condensed, a value from
            # the claim carrying the number. Which facts are offered first
            # follows from that, and it is what the weld responds to.
            sample = deal.sample(
                plan.shape, condensed=plan.spec.form != AnswerForm.VALUE
            )
            if sample is None:
                # The topic ran out of passages before the plan ran out of
                # slots. Everything it had has been asked about once.
                break
            written += 1
            drafts = self._attempt(sample, plan, accepted)
            # Every draft but the kept one goes in on its own, not as a turn
            # of the thread: a thread links each row to the one before it,
            # and a discarded attempt is drop-rate evidence rather than a
            # question somebody asked next.
            threads.extend([one] for one in drafts[:-1])
            checked = drafts[-1]
            thread = [checked]
            if checked.accepted:
                accepted.append(checked)
                # Spread over the accepted roots rather than over the plan's
                # slots. Over the slots the share was picked before the
                # question was written, so what landed was this share TIMES
                # the acceptance rate: 0.5 gave 448 threads from 1,209
                # accepted roots, which is 37%. A setting whose value moves
                # with an unrelated rate cannot be reasoned about, and both
                # the catalogue and the README already said it meant this.
                if spread(roots, self._settings.followup_share):
                    thread += self._followups(
                        deal.widen(sample),
                        plan,
                        checked,
                        accepted,
                        followed,
                    )
                    followed += 1
                roots += 1
            threads.append(thread)
        current.set_attribute("questions.samples", written)
        return threads

    def _subjects(self, facts: list[SourceFact]) -> list[SourceFact]:
        """The facts whose passage is about a subject rather than about people.

        The credits page, which the repetition reading cannot see: every
        document thanks DIFFERENT people, so the section is identical in
        shape and different in words. `names_parties` reads what it is made
        of instead.

        Here and not in the queue, because it needs a tagger and a
        catalogue the api holds must not import the module that loads one.
        Read once per distinct passage rather than once per fact: a passage
        carries five or six facts and the reading is the same for all of
        them.
        """
        if self._settings.party_density <= 0:
            return facts
        seen: dict[int, bool] = {}
        kept = []
        for fact in facts:
            anchor = fact.anchor
            if anchor.id not in seen:
                seen[anchor.id] = names_parties(
                    anchor.text, anchor.language, self._settings.party_density
                )
            if not seen[anchor.id]:
                kept.append(fact)
        return kept

    def _checked(
        self, candidate: Candidate, accepted: list[CheckedQuestion]
    ) -> CheckedQuestion:
        """Puts one candidate through the gates, inside a span of its own.

        The verdict was in two places and neither could be queried: a column
        in Postgres, and a line in the log. Postgres answers how many of
        each gate fired - `question_generation.runs` is that query - and it
        cannot answer what a gate COST, because the calls, the tokens and
        the latency are in the spans.

        A span per question is what joins the two. The model calls this
        check makes are its children, so a run's `leaks_source` rejections
        carry the price of the calls they wasted, and Phoenix groups on the
        attributes below without anything being logged twice.

        Cheap: a question costs one to three model calls, each already a
        span, so this adds a third of what is there rather than doubling it.

        The verdict is recorded TWICE and the two are not redundant. An
        attribute is what a span is filtered and grouped by; an annotation
        is what Phoenix shows in its Evaluations view, with a label, a
        score and the reason, comparable across two projects without a
        query. Phoenix does not know an arbitrary attribute is a
        judgement, and the annotation cannot be filtered on in the trace
        view, so each does the half the other cannot.
        """
        with span.start_as_current_span("check_question") as current:
            checked = self._checker.check(candidate, accepted)
            current.set_attribute("run.id", run_id())
            # The gate that stopped it, or `accepted`. One attribute rather
            # than a boolean and a nullable code, because what every query
            # here groups on is "what became of it".
            current.set_attribute("question.gate", checked.rejected_reason or ACCEPTED)
            # And every gate that READ it, in order. The line above says
            # what stopped a question and can never say what it got past,
            # because the checker returns on the first failure - so a
            # question refused at the round trip and one refused at the
            # first rule were indistinguishable in how far they got.
            current.set_attribute("question.gates_ran", list(checked.gates_ran))
            current.set_attribute("question.accepted", checked.accepted)
            # What the measuring gates read, so a trace can be filtered to
            # the questions that only just survived one. Per gate as well as
            # the aggregate: a Grafana panel charts one of these and a
            # person reading a trace wants the other.
            for one in checked.readings:
                current.set_attribute(f"question.score.{one.gate}", one.value)
            if (margin := confidence(checked.readings)) is not None:
                current.set_attribute("question.confidence", margin)
            current.set_attribute("question.answerable", checked.answerable)
            current.set_attribute("question.language", checked.language)
            current.set_attribute("question.follows", checked.follows)
            for name, value in (
                ("question.type", checked.question_type),
                ("question.form", checked.answer_form),
                ("question.difficulty", checked.criteria.difficulty),
                ("question.planned_difficulty", checked.planned_difficulty),
                ("question.cognitive_level", checked.cognitive_level),
            ):
                if value:
                    current.set_attribute(name, value)

            # On the ROW as well, which is the only durable end of the
            # link: the span is written now and aged out by Phoenix's own
            # retention, and the question is append-only and never
            # hard-deleted. Without them nothing joins the two - a span
            # carries no question id, because the question has none yet
            # when this runs.
            trace_id, span_id = current_ids()
            checked = replace(checked, trace_id=trace_id, span_id=span_id)

            # And as an annotation, which is the Evaluations view. CODE and
            # not LLM: a gate is a rule reading a parse, and the phrasing
            # judgements that are a model's opinion are the ones posted as
            # LLM. The score is 1 or 0, so a project's mean over this
            # annotation IS its acceptance rate.
            if span_id:
                self._evaluations.record(*self._verdicts(span_id, checked))
            return checked

    def _verdicts(self, span_id: str, checked: CheckedQuestion) -> list[Verdict]:
        """One verdict for the question, and one for each gate that read it.

        The summary says what became of it. The per-gate ones are what
        make the Evaluations view a table of the pipeline rather than one
        column: each gate is its own annotation, so its mean over a
        project IS its pass rate, and two runs compare gate by gate
        without a query.

        Only the gates that RAN. A question refused at `structural` never
        reached the round trip, and scoring it there as passed or failed
        would both be untrue - the absence is the fact. Which is also why
        the count is not fixed: a gate's mean is over the questions that
        reached it.

        The last gate that ran is the one that refused it, if anything
        did. That is what `check` guarantees by returning on the first
        failure, and it is the only reason a code can be turned back into
        a gate.
        """
        about = {
            "run_id": run_id(),
            "language": checked.language,
            "answerable": checked.answerable,
            "question_type": checked.question_type or "",
            "prompt_version": PROMPT_VERSION,
        }
        verdicts = [
            Verdict(
                span_id=span_id,
                name="gate",
                label=checked.rejected_reason or ACCEPTED,
                score=float(checked.accepted),
                explanation=self._why(checked),
                metadata=about | {"gates_ran": ",".join(checked.gates_ran)},
            )
        ]
        read = {one.gate: one for one in checked.readings}
        for position, gate in enumerate(checked.gates_ran, 1):
            stopped = not checked.accepted and gate == checked.gates_ran[-1]
            # Beside the pass or refuse, never instead of it: a Phoenix mean
            # over one of these annotations is the gate's pass rate, and
            # scoring the margin here would silently redefine every chart
            # already drawn on one.
            measured = read.get(gate)
            verdicts.append(
                Verdict(
                    span_id=span_id,
                    name=annotation(gate),
                    label=REFUSED if stopped else PASSED,
                    score=0.0 if stopped else 1.0,
                    # Only where it refused. A gate that let a question
                    # through has no reason to give, and the question's
                    # own text under every gate it passed is six copies
                    # of one string.
                    explanation=self._why(checked) if stopped else "",
                    metadata=about
                    | {"position": position}
                    | (measured.recorded() if measured else {}),
                )
            )
        # One more, so the Evaluations view can sort a project by what only
        # just survived. Absent where nothing measured anything, rather
        # than scored zero - see `confidence.py`.
        if (margin := confidence(checked.readings)) is not None:
            verdicts.append(
                Verdict(
                    span_id=span_id,
                    name="confidence",
                    label=checked.rejected_reason or ACCEPTED,
                    score=margin,
                    explanation="",
                    metadata=about,
                )
            )
        return verdicts

    @staticmethod
    def _why(checked: CheckedQuestion) -> str:
        """What to show a reader beside the verdict.

        The question itself for an accepted one - there is no reason to
        give, and a blank explanation column is worse than the thing it
        was reached about. The gate's own wording is not carried on
        `CheckedQuestion`: it is logged where it is decided and stored as
        a code, and re-deriving it here would be a second copy of every
        gate's sentence.
        """
        if checked.accepted:
            return checked.question_text
        return f"{checked.rejected_reason}: {checked.question_text}"

    def _attempt(
        self, sample: FactGroup, plan: Plan, accepted: list[CheckedQuestion]
    ) -> list[CheckedQuestion]:
        """Writes one question, and writes it again when a gate can be answered.

        Returns every draft with the one to keep last. All of them are
        stored: a draft a gate refused is the same drop-rate evidence as any
        other refusal, and quietly dropping it would make the writer look
        better than it is while hiding what the retry cost.

        A retry can only improve the outcome. If the second draft is refused
        where the first was not, the first is what is kept - which is what
        makes it safe to ask again about a question that was merely narrower
        than its slot wanted.
        """
        # Checked against what this run has accepted as well as what the
        # database holds: nothing is stored until the topic is finished, so
        # without it a topic would happily write the same question twice.
        drafts = [self._checked(self._writer.write(sample, plan), accepted)]
        for _ in range(self._settings.retries):
            note = again(drafts[-1], plan)
            if note is None:
                break
            log.info("asking again for %r: %s", drafts[-1].question_text, note)
            # Numbered as it is written, not as it is stored: the best draft
            # is moved to the end below, so a row's position in this list
            # stops saying which attempt produced it the moment it moves.
            drafts.append(
                replace(
                    self._checked(self._writer.write(sample, plan, note), accepted),
                    attempt=len(drafts) + 1,
                )
            )
        best = max(range(len(drafts)), key=lambda one: (drafts[one].accepted, one))
        drafts.append(drafts.pop(best))
        return drafts

    def _followups(
        self,
        sample: FactGroup,
        plan: Plan,
        root: CheckedQuestion,
        accepted: list[CheckedQuestion],
        followed: int = 0,
    ) -> list[CheckedQuestion]:
        """Writes the questions somebody would ask after this one.

        Each turn takes the next type in QUESTIONS_FOLLOWUP_TYPES, so a thread
        moves from a value to the circumstances it applies in to the reason
        behind it rather than asking the same kind of thing three times.

        The cycle starts where this thread sits among the followed ones, not
        at zero. Starting every thread at the first type means a run never
        reaches past QUESTIONS_MAX_FOLLOWUPS of them: two turns over three
        configured types left `comparison` unwritten across 3,119 questions,
        and every follow-up in the set came out `condition` or `reason`.

        Counted over the threads that were followed rather than over the plan,
        because QUESTIONS_FOLLOWUP_SHARE picks by position too: a share of 0.5
        takes every odd slot, so offsetting by the slot number would hold
        `index % 2` at 1 forever and rotate nothing.

        Only after an accepted, answerable root. A thread whose first turn
        has no answer has nothing to follow on from - the chatbot was
        supposed to say it did not know - and a thread whose root a gate
        refused is a conversation starting with a question nobody would ask.

        Stops at the first follow-up a gate refuses. The refused one is
        stored, because it is drop-rate evidence like any other, but nothing
        is written after it: a third turn following a second that was thrown
        out is a conversation with a hole in it.
        """
        if not root.answerable or not self._settings.followup_types:
            return []

        turns: list[tuple[str, str | None]] = [(root.question_text, root.target_answer)]
        written: list[CheckedQuestion] = []
        # What the two thread gates read. The root's facts stay the root's
        # all the way down - a third turn re-asking the first is the defect
        # either way - while the passages are the turn before this one's,
        # because a thread is allowed to walk from one passage to the next.
        cited = root.fact_ids
        previous = root.passage_ids
        for turn in range(self._settings.max_followups):
            names = self._settings.followup_types
            candidate = self._writer.follow_up(
                sample,
                tuple(turns),
                Plan(
                    spec=SPECS[names[(followed + turn) % len(names)]],
                    band=plan.band,
                    shape=plan.shape,
                    answerable=True,
                ),
                root_facts=cited,
                parent_passages=previous,
            )
            checked = self._checked(candidate, accepted)
            written.append(checked)
            if not checked.accepted:
                break
            accepted.append(checked)
            turns.append((checked.question_text, checked.target_answer))
            previous = checked.passage_ids
        return written
