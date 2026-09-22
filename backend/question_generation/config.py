"""Configuration for the question generation service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

from database.qa_generator import AnswerForm, Difficulty
from question_generation.queue import ASKABLE
from question_generation.types import SPECS
from settings import Source, csv, decimal, integer, mapping, optional, required


def _weights(
    name: str,
    allowed: Mapping[str, object] | tuple[str, ...],
    source: Source = None,
) -> dict[str, int]:
    """Reads a `name:weight` mix, refusing a name nothing answers to.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If a name is not one of `allowed`, or a weight is not a
            whole number, or every weight is zero.
    """
    read = {}
    for key, value in mapping(name, source).items():
        if key not in allowed:
            raise ValueError(
                f"{name} names {key!r}, which is not one of "
                f"{', '.join(sorted(allowed))}"
            )
        try:
            read[key] = int(value)
        except ValueError:
            raise ValueError(
                f"{name} gives {key!r} the weight {value!r}, which is not a "
                f"whole number"
            ) from None
    if not any(weight > 0 for weight in read.values()):
        raise ValueError(f"{name} gives every entry a weight of 0, so nothing is asked")
    return read


def _bounds(name: str, source: Source = None) -> dict[str, tuple[int, int]]:
    """Reads the `form:min:max` length bounds of each answer form.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If a form is unknown, a bound is not a whole number, a
            floor is above its ceiling, or a form is missing.
    """
    read = {}
    for form, value in mapping(name, source).items():
        if form not in tuple(AnswerForm):
            raise ValueError(
                f"{name} names the answer form {form!r}, which is not one of "
                f"{', '.join(AnswerForm)}"
            )
        low, _, high = value.partition(":")
        try:
            read[form] = (int(low), int(high))
        except ValueError:
            raise ValueError(
                f"{name} gives {form!r} the bounds {value!r}; it takes "
                f"form:min:max, as value:1:80"
            ) from None
        # An inverted pair parses and then refuses every answer of the form,
        # as `_span` refuses one for the same reason.
        if read[form][0] > read[form][1]:
            raise ValueError(
                f"{name} gives {form!r} a floor of {read[form][0]} above its "
                f"ceiling of {read[form][1]}, so no answer of that form could "
                f"pass"
            )
    missing = [form for form in AnswerForm if form not in read]
    if missing:
        raise ValueError(f"{name} says nothing about {', '.join(missing)}")
    return read


def _span(name: str, source: Source = None) -> tuple[int, int]:
    """Reads one `min:max` pair of character bounds.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If either bound is not a whole number, or the floor is
            above the ceiling.
    """
    low, _, high = required(name, source).partition(":")
    try:
        span = (int(low), int(high))
    except ValueError:
        raise ValueError(
            f"{name} gives the bounds {low}:{high}; it takes min:max, as 150:900"
        ) from None
    if span[0] > span[1]:
        raise ValueError(f"{name} gives a floor of {span[0]} above its ceiling")
    return span


@dataclass(frozen=True)
class Settings:
    """What to write per topic, of what kinds, and what to hold it to.

    Which model writes them is not here: that is `llm.config.Settings`, which
    every stage that calls a model shares. Only the verifier's name is here,
    because it is the one model choice this stage makes on its own.
    """

    per_topic: int
    sample_size: int
    #: How many times one passage may be offered, each time with facts no
    #: earlier sample took. The ceiling on it; what really stops a passage
    #: is running out of facts nothing has been written from.
    samples_per_passage: int
    #: Which kinds of fact a question may be written from. Extraction reads a
    #: passage four ways and each shape seeds a different question: an atomic
    #: claim a factoid, a summary a definition, an outline an enumeration,
    #: and a bridge a cross-document one.
    fact_kinds: tuple[str, ...]
    #: Which question types are written, and in what proportion. A type with a
    #: weight of 0, or absent, is never written.
    type_mix: dict[str, int]
    #: Which difficulty bands the plan aims for, in what proportion. A request
    #: for a shape of sample, not a verdict: `difficulty` is still read off
    #: what the question turned out to cite.
    difficulty_mix: dict[str, int]
    #: The types a follow-up may take, cycled in order down a thread.
    followup_types: tuple[str, ...]
    unanswerable_share: float
    followup_share: float
    max_followups: int
    #: How many more times a question may be written when a gate refused the
    #: draft, or when it came out below the band its slot asked for. 0 keeps
    #: the first draft whatever it is.
    retries: int
    #: Shortest and longest target answer per form.
    answer_chars: dict[str, tuple[int, int]]
    #: Shortest and longest explanation, whatever form the target took. One
    #: pair rather than one per form, because the explanation is the same
    #: job every time: the target is what changes shape, not the reading of
    #: it. A floor at all because a one-line explanation is the target said
    #: twice, which is the failure this column was added to fix.
    explanation_chars: tuple[int, int]
    #: How much of a list or an explanation has to come back for the verifier
    #: to be agreeing with it.
    answer_overlap: float
    #: How much of what the verifier recovered the target has to account for
    #: before the target is read as an incomplete key. The same recovery as
    #: `answer_overlap`, read the other way round. 0 turns the reading off.
    answer_coverage: float
    #: How much of a passage may be the names of people before it is read as
    #: a credits page and never asked about. 0 turns the reading off.
    party_density: float
    #: How much a candidate fact must have in common with the head of its
    #: sample before a second passage is offered at all. 0 offers one
    #: whatever it holds, which is what every run so far has done.
    #:
    #: Off by default, and the sweep is why. Over 1,853 accepted
    #: multi-passage questions, a floor of 0.84 would have narrowed 37% of
    #: them while only 46% of those read as welded, and the weld rate fell
    #: from 34.8% to 24.3%. There is no knee: the score of the pair a writer
    #: actually used barely separates the welds from the sound questions.
    #: The mechanism is here to be tuned against a corpus, not because a
    #: threshold is known.
    meets_floor: float
    #: How much of what an unanswerable question is about has to occur in the
    #: material it was drawn from. Below it the question is off topic: any
    #: chatbot declines one about something the corpus never mentions.
    off_topic_overlap: float
    #: How many passages the corpus-wide probe shows the verifier before an
    #: unanswerable question is accepted. 0 judges one against its own
    #: passages alone, as every other gate does.
    elsewhere_passages: int
    long_answer_chars: int
    duplicate_cosine: float
    #: How alike a passage must be to one in ANOTHER document before it is
    #: read as boilerplate and never asked about. 0 turns the reading off.
    #:
    #: A corpus repeats its own furniture: a copyright notice, a table of
    #: contents, an accreditation clause, a revision table. Each is real
    #: text that a fact can be extracted from and a question can be written
    #: about, and no gate refuses the result, because nothing is wrong with
    #: it except that nobody wants to know. What separates it from subject
    #: matter is that it appears in document after document nearly
    #: unchanged, which is a measurement rather than a word list: no
    #: language, no domain and no section name is assumed.
    #:
    #: A corpus of one document has no other document to repeat into, so
    #: nothing is excluded, which is the right answer rather than a gap.
    boilerplate_cosine: float
    #: How many questions a balanced release holds. 0 draws the largest one
    #: the accepted pool can fill without missing a quota.
    release_size: int
    #: The most of a release that may be questions with no answer. A ceiling
    #: and not a target, unlike the two mixes: too few tests a little less
    #: than it could, too many tests mostly whether a chatbot can say no.
    release_unanswerable: float
    #: How a release spreads over the difficulty bands. The whole set, so
    #: the unanswerable questions are counted in it - they are always easy.
    #: Not the same thing as difficulty_mix, which is what the planner aims
    #: the answerable ones at.
    release_difficulty: dict[str, int]
    embedding_model: str
    max_tokens: int
    #: The model that writes a question, or None for the shared one.
    model: str | None
    verifier_model: str | None
    #: The model asked what a question's wording amounts to where a rule has
    #: not settled it, or None to ask the verifier's.
    phrasing_model: str | None = None
    #: A local NLI encoder for the entailment pass, or None to ask the
    #: verifier. Not an LLM_MODEL: it is loaded in the worker like the
    #: embedding model is.
    entailment_model: str | None = None
    entailment_threshold: float = 0.5
    entailment_overlap: float = 0.3
    #: A local extractive QA encoder for the recall half, or None to ask the
    #: verifier for every question. Not an LLM_MODEL either.
    answer_model: str | None = None
    answer_confidence: float = 0.9
    #: The longest pair either encoder reads, in tokens.
    encoder_max_tokens: int = 512

    def lease(self, call_seconds: float) -> timedelta:
        """How long one topic may go unfinished before a run sweeps it.

        Derived from the work rather than declared. A topic is not a passage:
        it costs `per_topic` candidates and each one is two model calls, a
        writer's and a verifier's, so a lease sized for one call fails a
        worker that is only halfway through its first topic.

        A thread is more than one question, so `max_followups` is in it too:
        every root may be followed twice, and each follow-up is another pair
        of calls.

        Three calls per candidate, not two: the writer's, the verifier's,
        and the one a candidate may cost on top - the entailment pass when
        recall came back empty, or the corpus probe on an unanswerable
        question. And every candidate may be written `retries` times more.

        This is a worst case on a worst case - every call taking its full
        timeout on every attempt, on every question of the topic - so the
        figure is days rather than hours and it grows with
        QUESTIONS_PER_TOPIC. That is the wrong direction for the one thing
        the lease is for: a worker killed mid-topic leaves that row
        unclaimable until it runs out, and nothing but time moves it. The
        remedy meanwhile is `make questions-retry`, which returns a stuck
        topic to the queue without waiting.
        """
        return timedelta(
            seconds=call_seconds
            * self.per_topic
            * (1 + self.max_followups)
            * (1 + self.retries)
            * 3
        )

    @classmethod
    def load(cls, source: Source = None) -> Settings:
        """Reads settings from the environment, or from an override.

        Args:
            source: Where to read them, or None for the process environment.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number, or a mix names
                a type, a band or an answer form that does not exist.
        """
        followups = csv("QUESTIONS_FOLLOWUP_TYPES", source)
        unknown = [name for name in followups if name not in SPECS]
        if unknown:
            raise ValueError(
                f"QUESTIONS_FOLLOWUP_TYPES names {', '.join(unknown)}, which is "
                f"not among {', '.join(sorted(SPECS))}"
            )
        kinds = csv("QUESTIONS_FACT_KINDS", source)
        unusable = [kind for kind in kinds if kind not in ASKABLE]
        if unusable:
            raise ValueError(
                f"QUESTIONS_FACT_KINDS names {', '.join(unusable)}, which this "
                f"stage cannot write a question from. It takes "
                f"{', '.join(ASKABLE)}; see the note beside ASKABLE for what "
                f"each other kind would need first."
            )
        return cls(
            per_topic=integer("QUESTIONS_PER_TOPIC", source),
            sample_size=integer("QUESTIONS_FACT_SAMPLE", source),
            samples_per_passage=integer("QUESTIONS_SAMPLES_PER_PASSAGE", source),
            fact_kinds=kinds,
            type_mix=_weights("QUESTIONS_TYPE_MIX", SPECS, source),
            difficulty_mix=_weights(
                "QUESTIONS_DIFFICULTY_MIX", tuple(Difficulty), source
            ),
            followup_types=followups,
            unanswerable_share=decimal("QUESTIONS_UNANSWERABLE_SHARE", source),
            followup_share=decimal("QUESTIONS_FOLLOWUP_SHARE", source),
            max_followups=integer("QUESTIONS_MAX_FOLLOWUPS", source),
            retries=integer("QUESTIONS_RETRIES", source),
            answer_chars=_bounds("QUESTIONS_ANSWER_CHARS", source),
            explanation_chars=_span("QUESTIONS_EXPLANATION_CHARS", source),
            answer_overlap=decimal("QUESTIONS_ANSWER_OVERLAP", source),
            answer_coverage=decimal("QUESTIONS_ANSWER_COVERAGE", source),
            party_density=decimal("QUESTIONS_PARTY_DENSITY", source),
            meets_floor=decimal("QUESTIONS_MEETS_FLOOR", source),
            off_topic_overlap=decimal("QUESTIONS_OFF_TOPIC_OVERLAP", source),
            elsewhere_passages=integer("QUESTIONS_ELSEWHERE_PASSAGES", source),
            long_answer_chars=integer("QUESTIONS_LONG_ANSWER_CHARS", source),
            duplicate_cosine=decimal("QUESTIONS_DUPLICATE_COSINE", source),
            boilerplate_cosine=decimal("QUESTIONS_BOILERPLATE_COSINE", source),
            release_size=integer("QUESTIONS_RELEASE_SIZE", source),
            release_unanswerable=decimal("QUESTIONS_RELEASE_UNANSWERABLE", source),
            release_difficulty=_weights(
                "QUESTIONS_RELEASE_DIFFICULTY", tuple(Difficulty), source
            ),
            # The one embedding model, as chunking reads it: a question
            # embedded by a model other than the one a passage was sized by
            # measures distance in a space the corpus was never put in.
            embedding_model=required("EMBEDDING_MODEL", source),
            max_tokens=integer("EMBEDDING_MAX_TOKENS", source),
            # Absent means the model LLM_MODEL names.
            model=optional("QUESTIONS_MODEL", source),
            # Absent means the writer verifies its own questions, which is
            # worth knowing about rather than guessing at, so the service
            # warns rather than failing.
            verifier_model=optional("QUESTIONS_VERIFIER_MODEL", source),
            phrasing_model=optional("QUESTIONS_PHRASING_MODEL", source),
            entailment_model=optional("NLI_MODEL", source),
            entailment_threshold=decimal("NLI_ENTAILMENT_THRESHOLD", source),
            entailment_overlap=decimal("QUESTIONS_ENTAILMENT_OVERLAP", source),
            encoder_max_tokens=integer("ENCODER_MAX_TOKENS", source),
            answer_model=optional("QA_MODEL", source),
            answer_confidence=decimal("QA_ANSWER_CONFIDENCE", source),
        )
