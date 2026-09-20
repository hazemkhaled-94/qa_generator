"""The order the gates are applied in, and the verdict that comes out.

Cheapest first, because each one that fires saves the cost of those behind
it:

  malformed       structural, free
  answer_too_short  free: nothing worth scoring against
  answer_too_long   free: not the form of answer that was asked for
  wrong_form        free: a value carrying a verb, an explanation with none
  leaks_source    free where the title is quoted, the verifier's otherwise
  off_topic       free: an unanswerable question about nothing the material
                  mentions, which any chatbot declines
  duplicate       one index probe against the questions already accepted
  answerable      the same probe, when an unanswerable question has a twin
                  the corpus does answer
  compound        it asks two things, so half an answer is neither right
                  nor wrong
  unanchored      a question nobody could have asked without the passage
  recoverable     the answer is not in the evidence the question cites
  answerable_elsewhere  one corpus-wide passage probe and a second call, on
                  the unanswerable share only: a passage it does not cite
                  answers it. Cosine over passages.embedding where the corpus
                  has been embedded, shared lemmas where it has not

The free ones are in `gates`, which imports no client; the calls are the
`verifier`'s. Recoverability is the one no similarity measure makes: it asks
whether the answer can be got back out of the passages, which is what a
question is for, and a paraphrase, a decomposition and a resolved pronoun all
survive it where a threshold would not.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from database.qa_generator import (
    Derivation,
    QuestionRejection,
    QuestionStatus,
)
from nlp.analysis import content, pointing
from nlp.embedding import Embedder, cosine
from question_generation.gates import (
    BOUNDS,
    NAMES_SOMETHING,
    OFF_TOPIC_OVERLAP,
    OVERLAP,
    adds_up,
    agrees,
    anchored,
    asserted,
    cites_source,
    near_verdict,
    on_topic,
    structural,
    subject,
)
from question_generation.models import (
    LONG_ANSWER_CHARS,
    Candidate,
    CheckedQuestion,
    Neighbour,
)
from question_generation.verifier import Verifier

log = logging.getLogger(__name__)


class QuestionChecker:
    """Puts one candidate through every gate, cheapest first."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        verifier: Verifier,
        nearest,
        threshold: float,
        judge_phrasing: bool = True,
        bounds: Mapping[str, tuple[int, int]] | None = None,
        overlap: float = OVERLAP,
        long_answer_chars: int = LONG_ANSWER_CHARS,
        elsewhere=None,
        elsewhere_passages: int = 0,
        off_topic_overlap: float = OFF_TOPIC_OVERLAP,
        entail: bool = True,
        phrasing=None,
    ) -> None:
        """Initialises the checker with its collaborators.

        `nearest` is a callable rather than the catalogue itself: what this
        needs of the database is one probe, and taking the whole repository
        for it would let a gate read anything.

        `judge_phrasing` is whether the verifier's opinion of the question
        may reject one. Those gates are the only ones here that are an
        opinion rather than a measurement, and an opinion needs an
        independent holder: a writer marking its own work rejected `According
        to the ECB and NCAs, who conducts the due diligence check?` for
        naming nothing. The factory turns it off when QUESTIONS_VERIFIER_MODEL
        is unset, and the verdict is logged either way. Recoverability stays
        on regardless, because that one is checkable against the passage.

        `elsewhere` is the corpus-wide passage probe the unanswerable gate
        reads, taking the question's lemmas, its language, the passages it
        already cites and a limit. A callable and not the catalogue, for the
        same reason `nearest` is one. None, or a limit of 0, leaves an
        unanswerable question judged against its own passages alone.
        """
        self._embedder = embedder
        self._verifier = verifier
        self._nearest = nearest
        self._judge_phrasing = judge_phrasing
        self._bounds = dict(bounds or BOUNDS)
        self._overlap = overlap
        self._long_answer_chars = long_answer_chars
        self._threshold = threshold
        self._elsewhere = elsewhere
        self._elsewhere_passages = elsewhere_passages
        self._off_topic_overlap = off_topic_overlap
        self._entail = entail
        self._phrasing_judge = phrasing

    def check(
        self, candidate: Candidate, seen: Sequence[CheckedQuestion] = ()
    ) -> CheckedQuestion:
        """Judges one candidate, returning it stored either way.

        `seen` is what this run has already accepted. Without it the dedup
        gate would compare a candidate only against what was in the database
        when the run started, and a topic would happily write the same
        question four times.

        Raises:
            ModelUnavailable: If the verifier could not be reached.
        """
        failed, form = structural(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            language=candidate.group.language,
            statements=candidate.group.statements,
            forms=candidate.spec.forms,
            bounds=self._bounds,
            titles=candidate.group.titles,
        )
        if failed:
            return self._verdict(candidate, failed, None, self._long_answer_chars, form)

        # Free, and only an unanswerable question can fail it: an answerable
        # one is about its material by construction, because the answer came
        # out of it.
        if not candidate.answerable and not on_topic(
            candidate.question_text,
            candidate.group.language,
            candidate.group.lemmas,
            self._off_topic_overlap,
        ):
            return self._verdict(
                candidate,
                (
                    QuestionRejection.OFF_TOPIC,
                    (
                        "it names almost nothing the passages it was drawn from "
                        "name, so any chatbot declines it and declining it "
                        "proves nothing"
                    ),
                ),
                None,
                self._long_answer_chars,
                form,
            )

        embedding = self._embedder.embed(candidate.question_text)
        failed = near_verdict(
            self._near(embedding, seen),
            answerable=candidate.answerable,
            threshold=self._threshold,
        )
        if failed:
            return self._verdict(
                candidate, failed, embedding, self._long_answer_chars, form
            )

        # Before the round trip, because none of it reads a passage: a
        # question naming its own source is refused without ever paying for
        # the call that would have answered it.
        failed = self._phrasing(candidate)
        if failed:
            return self._verdict(
                candidate, failed, embedding, self._long_answer_chars, form
            )

        return self._verdict(
            candidate,
            self._round_trip(candidate, form, embedding),
            embedding,
            self._long_answer_chars,
            form,
        )

    def _phrasing(self, candidate: Candidate) -> tuple[str, str] | None:
        """What the question's own wording says, before any passage is read.

        Ahead of the round trip because none of it needs the passages, so a
        question that carries its own source no longer pays for a call that
        reads them. The three judgements here were one field each on that
        call until the harness measured what sharing it cost: 7/7 in English
        against 6/12 in German. See `phrasing.py`.

        Two of the three now need no model. `cites_source` settles the
        patterns and only the residue is asked; `anchored` is the whole of
        the naming verdict, where it used to be a veto on a model's opinion
        that was wrong about `According to the ECB and NCAs, who conducts the
        due diligence check?`.

        A rule is not an opinion, so `judge_phrasing` does not gate one. It
        gates what a model said, which is what needs an independent holder.
        """
        question = candidate.question_text
        language = candidate.group.language

        # A question carrying its own source has already done the work it
        # was meant to test. Judged for a follow-up too - leaning on the
        # conversation is allowed, naming the file is not.
        cited = cites_source(question, language)
        ruled = cited is not None
        if cited is None and self._phrasing_judge is not None:
            cited = self._phrasing_judge.names_its_source(question)
        if cited:
            reason = (
                "it names the material the answer is in, so it asks a question "
                "and answers half of it" + ("" if ruled else ", the verifier says")
            )
            if ruled or self._judge_phrasing:
                return QuestionRejection.LEAKS_SOURCE, reason
            log.info("keeping %r: %s, but the writer judged itself", question, reason)

        # Never for a follow-up. `And for an urgent one?` names nothing and
        # is exactly the question a person asks second; leaning on the thread
        # is what a follow-up is for, so judging it as though it had been
        # asked cold would reject every one of them.
        if candidate.follows:
            return None

        # Still two holders, and the reason is measured in both directions.
        # The opinion alone rejected `According to the ECB and NCAs, who
        # conducts the due diligence check?` for naming nothing. The
        # measurement alone rejects `Why must a request be confirmed in
        # writing?`, because NAMES_SOMETHING was calibrated as a veto on
        # German questions, where one compound carries what two English
        # words do. Neither is a verdict on its own.
        #
        # What changed is which measurement: the parse reads what a question
        # names, where the model used to be asked to copy it out inside a
        # call about the passages. Over the labelled cases the parse is
        # right 19 times out of 19 and that field managed 16.
        if not anchored(question, language):
            named = (
                None
                if self._phrasing_judge is None
                else self._phrasing_judge.names_something(question)
            )
            if named is False:
                reason = (
                    f"it names nothing a person searching would know - the "
                    f"parse finds {subject(question, language)!r} - and it "
                    f"names no name, no number and fewer than "
                    f"{NAMES_SOMETHING} things, so it could not have been "
                    f"asked without the passage"
                )
                if self._judge_phrasing:
                    return QuestionRejection.UNANCHORED, reason
                log.info(
                    "keeping %r: %s, but the writer judged itself", question, reason
                )

        # The other way a question fails to stand on its own: not naming too
        # little, but pointing at something the asker cannot see. The
        # measurement over-fires by design - a question that sets a case up
        # and refers back to it carries a pointer and is fine - so it only
        # ever asks the question, and the verdict is the model's.
        pointers = pointing(question, language)
        if not pointers or self._phrasing_judge is None:
            return None
        if self._phrasing_judge.self_contained(question, pointers) is False:
            reason = (
                f"it points outward with {', '.join(repr(one) for one in pointers)} "
                f"and there is nothing in the question to point at, so only "
                f"somebody holding the passage could have asked it"
            )
            if self._judge_phrasing:
                return QuestionRejection.UNANCHORED, reason
            log.info("keeping %r: %s, but the writer judged itself", question, reason)
        return None

    def _round_trip(
        self, candidate: Candidate, form: str, embedding: list[float]
    ) -> tuple[str, str] | None:
        """Asks whether the passages give the answer back."""
        read = self._verifier.read(
            candidate.question_text,
            candidate.group.passages,
            candidate.thread,
        )

        if candidate.answerable:
            target = candidate.target_answer or ""
            # A derived answer is not in the passages and is not meant to
            # be. Asking recall for it is asking the verifier to contradict
            # the type's own directive - `aggregation` says the total must
            # be something the material does not write down - and it did:
            # 7 of its 12 questions went as not_recoverable, and the type
            # accepted at 8% where the rest of them averaged 40%.
            if candidate.spec.derived:
                return self._computable(candidate, target)
            if read.recovered is None:
                if self._backed(candidate, target):
                    return None
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    "the verifier could not find the answer in the cited passages",
                )
            if not agrees(
                read.recovered,
                target,
                candidate.group.language,
                form,
                self._overlap,
                candidate.question_text,
            ):
                if self._backed(candidate, target):
                    return None
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    (
                        f"the verifier recovered {read.recovered!r} where the "
                        f"target answer is {target!r}"
                    ),
                )
            return None

        if read.recovered is not None:
            return (
                QuestionRejection.ANSWERABLE_AFTER_ALL,
                f"the cited passages answer it after all, with {read.recovered!r}",
            )
        return self._corpus(candidate, embedding)

    def _computable(self, candidate: Candidate, target: str) -> tuple[str, str] | None:
        """Whether a derived answer follows from what the passages do state.

        The gate a derived type gets instead of recoverability, because the
        two ask opposite questions. Recoverability asks whether the passages
        STATE the answer; these types are defined by their not doing so, and
        the answer is right when what it needs is there and the step from
        there to here holds.

        Which step is asked depends on how the type derives. Arithmetic is
        not entailment: `Do the arithmetic` is the wrong instruction for a
        conclusion drawn from two rules, and `does this follow` is the wrong
        one for a total, which follows from anything if the reader is
        generous about addition.

        Still a model's judgement, and still the verifier's rather than the
        writer's, so nothing here marks its own work.
        """
        if not target:
            return (
                QuestionRejection.NOT_RECOVERABLE,
                "it carries no answer to check the passages against",
            )
        if candidate.spec.derived == Derivation.ARITHMETIC:
            # Done here rather than asked, when the answer is a figure. A
            # total has a right value rather than a likely one, and a model
            # asked whether two numbers come to a third is being asked to
            # agree. None is an answer this cannot read - a total in words, a
            # range, a date - and falls through to the model below.
            computed = adds_up(target, candidate.group.passages)
            if computed is True:
                return None
            if computed is False:
                return (
                    QuestionRejection.NOT_RECOVERABLE,
                    (
                        f"{target!r} is not a total of any figures the cited "
                        f"passages state, or is two years added together"
                    ),
                )
        asked = (
            self._verifier.computes
            if candidate.spec.derived == Derivation.ARITHMETIC
            else self._verifier.follows
        )
        if asked(
            candidate.question_text,
            target,
            candidate.group.passages,
            candidate.thread,
        ):
            return None
        return (
            QuestionRejection.NOT_RECOVERABLE,
            (f"{target!r} does not follow from what the cited passages state"),
        )

    def _backed(self, candidate: Candidate, target: str) -> bool:
        """Whether the entailment pass rescues an answer recall did not find.

        Two guards, both of which have to hold. Every number, date and name
        the target asserts must occur in the passages, which no opinion can
        overrule: that is what keeps `4 hours` from being confirmed against
        a passage that says 48. Then the verifier is shown the answer and
        asked the narrower question.

        It can only ever accept. A question reaching here was already being
        rejected, so a pass that says nothing leaves the verdict where it
        was and costs one call on the failures alone.
        """
        if not self._entail or not target:
            return False
        passages = candidate.group.passages
        if not asserted(target, passages, candidate.group.language):
            return False
        backed = self._verifier.supports(
            candidate.question_text, target, passages, candidate.thread
        )
        if backed:
            log.info(
                "keeping %r: recall missed it but the passages support %r",
                candidate.question_text,
                target,
            )
        return backed

    def _corpus(
        self, candidate: Candidate, embedding: list[float]
    ) -> tuple[str, str] | None:
        """Asks whether the rest of the corpus answers what its passages do not.

        The one claim in this dataset that is about the whole corpus rather
        than about the passages a question cites. Every other gate is right
        to read the citation alone - what is being measured is the dataset
        and not a retriever - but "there is no answer to this anywhere" is
        not a property of two passages, and a question wrongly carrying it
        marks a correct chatbot wrong, which is the one failure the gates
        exist to stop.

        Costs one probe and one call, on the unanswerable share only.
        """
        if self._elsewhere is None or self._elsewhere_passages <= 0:
            return None
        language = candidate.group.language
        passages = self._elsewhere(
            sorted(content(candidate.question_text, language)),
            language,
            [passage.id for passage in candidate.group.resting],
            self._elsewhere_passages,
            embedding,
        )
        if not passages:
            return None
        read = self._verifier.read(candidate.question_text, passages, candidate.thread)
        if read.recovered is None:
            return None
        return (
            QuestionRejection.ANSWERABLE_ELSEWHERE,
            (
                f"a passage it does not cite answers it, with "
                f"{read.recovered!r}, so the corpus is not silent about this"
            ),
        )

    def _near(
        self, embedding: list[float], seen: Sequence[CheckedQuestion]
    ) -> Neighbour | None:
        """The nearest accepted question, stored or written earlier in this run."""
        near = self._nearest(embedding)
        for other in seen:
            if other.embedding is None:
                continue
            score = cosine(embedding, other.embedding)
            if near is None or score > near.similarity:
                near = Neighbour(other.question_text, other.answerable, score)
        return near

    @staticmethod
    def _verdict(
        candidate: Candidate,
        failed: tuple[str, str] | None,
        embedding: list[float] | None,
        long_answer: int = LONG_ANSWER_CHARS,
        form: str | None = None,
    ) -> CheckedQuestion:
        """Assembles one judged question, kept whichever way it went.

        `form` is the one the answer turned out to fit, of those its type
        will take. Stored rather than the asked-for one, because it is what
        the gates read it against and a row saying otherwise could not be
        re-checked to the same verdict.
        """
        code, reason = failed if failed else (None, "")
        if code:
            log.info("rejected %r: %s (%s)", candidate.question_text, reason, code)
        return CheckedQuestion(
            question_text=candidate.question_text,
            target_answer=candidate.target_answer,
            answerable=candidate.answerable,
            criteria=candidate.group.criteria(
                len(candidate.target_answer) if candidate.target_answer else None,
                follows=candidate.follows,
                long_answer=long_answer,
            ),
            language=candidate.group.language,
            status=QuestionStatus.REJECTED if code else QuestionStatus.ACCEPTED,
            rejected_reason=code,
            fact_ids=tuple(fact.id for fact in candidate.group.facts),
            embedding=embedding,
            thread_position=candidate.thread_position,
            question_type=candidate.spec.name,
            cognitive_level=candidate.spec.level,
            # An unanswerable question has no answer, so it has no form: the
            # column says what shape the answer takes and there is none.
            answer_form=(form or candidate.spec.form) if candidate.answerable else None,
            planned_difficulty=candidate.planned_difficulty,
        )
