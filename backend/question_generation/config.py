"""Configuration for the question generation service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

from database.qa_generator import AnswerForm, Difficulty
from question_generation.types import SPECS
from settings import csv, decimal, integer, mapping, optional, required


def _weights(
    name: str, allowed: Mapping[str, object] | tuple[str, ...]
) -> dict[str, int]:
    """Reads a `name:weight` mix, refusing a name nothing answers to.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If a name is not one of `allowed`, or a weight is not a
            whole number, or every weight is zero.
    """
    read = {}
    for key, value in mapping(name).items():
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


def _bounds(name: str) -> dict[str, tuple[int, int]]:
    """Reads the `form:min:max` length bounds of each answer form.

    Raises:
        KeyError: If the setting is unset or empty.
        ValueError: If a form is unknown, a bound is not a whole number, or a
            form is missing.
    """
    read = {}
    for form, value in mapping(name).items():
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
    missing = [form for form in AnswerForm if form not in read]
    if missing:
        raise ValueError(f"{name} says nothing about {', '.join(missing)}")
    return read


@dataclass(frozen=True)
class Settings:
    """What to write per topic, of what kinds, and what to hold it to.

    Which model writes them is not here: that is `llm.config.Settings`, which
    every stage that calls a model shares. Only the verifier's name is here,
    because it is the one model choice this stage makes on its own.
    """

    per_topic: int
    sample_size: int
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
    #: Shortest and longest target answer per form.
    answer_chars: dict[str, tuple[int, int]]
    #: How much of a list or an explanation has to come back for the verifier
    #: to be agreeing with it.
    answer_overlap: float
    long_answer_chars: int
    duplicate_cosine: float
    embedding_model: str
    max_tokens: int
    verifier_model: str | None

    def lease(self, call_seconds: float) -> timedelta:
        """How long one topic may go unfinished before a run sweeps it.

        Derived from the work rather than declared. A topic is not a passage:
        it costs `per_topic` candidates and each one is two model calls, a
        writer's and a verifier's, so a lease sized for one call fails a
        worker that is only halfway through its first topic.

        A thread is more than one question, so `max_followups` is in it too:
        every root may be followed twice, and each follow-up is another pair
        of calls.

        Two calls per candidate and not a round number above it. This is
        already the worst case - every call taking its full timeout on every
        attempt - and padding a worst case is what makes a lease long enough
        to matter: a worker killed mid-topic leaves that row unclaimable
        until the lease runs out, and nothing but time moves it.
        """
        return timedelta(
            seconds=call_seconds * self.per_topic * (1 + self.max_followups) * 2
        )

    @classmethod
    def load(cls) -> Settings:
        """Reads settings from the environment.

        Raises:
            KeyError: If any required setting is missing.
            ValueError: If a numeric setting is not a number, or a mix names
                a type, a band or an answer form that does not exist.
        """
        followups = csv("QUESTIONS_FOLLOWUP_TYPES")
        unknown = [name for name in followups if name not in SPECS]
        if unknown:
            raise ValueError(
                f"QUESTIONS_FOLLOWUP_TYPES names {', '.join(unknown)}, which is "
                f"not among {', '.join(sorted(SPECS))}"
            )
        return cls(
            per_topic=integer("QUESTIONS_PER_TOPIC"),
            sample_size=integer("QUESTIONS_FACT_SAMPLE"),
            type_mix=_weights("QUESTIONS_TYPE_MIX", SPECS),
            difficulty_mix=_weights("QUESTIONS_DIFFICULTY_MIX", tuple(Difficulty)),
            followup_types=followups,
            unanswerable_share=decimal("QUESTIONS_UNANSWERABLE_SHARE"),
            followup_share=decimal("QUESTIONS_FOLLOWUP_SHARE"),
            max_followups=integer("QUESTIONS_MAX_FOLLOWUPS"),
            answer_chars=_bounds("QUESTIONS_ANSWER_CHARS"),
            answer_overlap=decimal("QUESTIONS_ANSWER_OVERLAP"),
            long_answer_chars=integer("QUESTIONS_LONG_ANSWER_CHARS"),
            duplicate_cosine=decimal("QUESTIONS_DUPLICATE_COSINE"),
            # The one embedding model, as chunking reads it: a question
            # embedded by a model other than the one a passage was sized by
            # measures distance in a space the corpus was never put in.
            embedding_model=required("EMBEDDING_MODEL"),
            max_tokens=integer("EMBEDDING_MAX_TOKENS"),
            # Absent means the writer verifies its own questions, which is
            # worth knowing about rather than guessing at, so the service
            # warns rather than failing.
            verifier_model=optional("QUESTIONS_VERIFIER_MODEL"),
        )
