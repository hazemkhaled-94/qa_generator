"""How well whatever answers the three phrasing judgements answers them.

`names_its_source`, `self_contained` and `subject` are judgements about a
question alone - no passage decides any of them. They ride along in the
verifier's reading call today, and until this module existed nothing measured
them: the only ground truth was the model that answered, so a change to the
prompt or to the model moved the numbers with nothing to say which way.

That is the point of this file, and it is deliberately here before anything
is swapped. A judgement with no ground truth cannot be moved off a model
safely, because "the small model disagrees with the large one" and "the small
model is wrong" are the same observation until somebody has labelled the
cases.

Scored one judgement at a time. A single accuracy over a call answering three
things hides the one that got worse, and these three fail differently:
`names_its_source` confuses a source with a party, `self_contained` confuses a
pointing word with a reference, and `subject` is a span two correct readers
disagree about the edges of.

Printed rather than asserted, like every other eval module, with one
exception: a judgement no better than answering the same way every time is a
judgement that is not working, and that is a wiring failure rather than a
matter of taste.

    make test-eval

Skipped without a model, which is every CI run.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import pytest

from evaluation.cases import PHRASING

pytestmark = [pytest.mark.eval, pytest.mark.nlp]

#: A passage for the reading half of the call, which these cases are not
#: about. The judgements are made on the question; something has to be in
#: front of the model for it to answer at all.
_NOTHING = "Dieser Abschnitt behandelt nichts, was die Frage beantwortet."


@dataclass
class Score:
    """How one judgement did over the whole set."""

    name: str
    right: int = 0
    total: int = 0
    #: Cases it got wrong, as (case name, expected, answered).
    missed: list[tuple[str, object, object]] | None = None

    def record(self, case: str, expected, answered) -> None:
        """Counts one case."""
        self.total += 1
        if expected == answered:
            self.right += 1
        else:
            self.missed = (self.missed or []) + [(case, expected, answered)]

    @property
    def accuracy(self) -> float:
        """The share it got right."""
        return self.right / self.total if self.total else 0.0

    def report(self) -> str:
        """One line, and every case it got wrong under it."""
        lines = [f"  {self.name:20} {self.right:2}/{self.total:2}  {self.accuracy:.1%}"]
        for case, expected, answered in self.missed or ():
            lines.append(f"      {case}: expected {expected!r}, got {answered!r}")
        return "\n".join(lines)


def majority(judgement: str) -> float:
    """What answering every case the same way would score.

    The floor a judgement has to clear to be doing anything. Four of nineteen
    cases are not self-contained, so a judge that says True every time scores
    0.79 on that one and has measured nothing.
    """
    counts = Counter(case[judgement] for case in PHRASING)
    return max(counts.values()) / len(PHRASING)


@pytest.fixture(scope="module")
def verifier():
    """The real verifier against the served model, or a skip."""
    if not os.environ.get("LLM_MODEL"):
        pytest.skip("LLM_MODEL is unset; no model to measure")

    from dataclasses import replace

    from llm.client import Client, ModelUnavailable
    from llm.config import Settings
    from question_generation.config import Settings as QuestionSettings
    from question_generation.verifier import Verifier

    settings = Settings.load()
    questions = QuestionSettings.load()
    built = Verifier(
        Client(
            replace(settings, model=questions.verifier_model)
            if questions.verifier_model
            else settings
        )
    )
    try:
        built.read("Is this on?", ["This is on."])
    except ModelUnavailable as exc:
        pytest.skip(f"the verifier is not answering: {exc}")
    return built


@pytest.fixture(scope="module")
def scored(verifier):
    """Every case put to the verifier once, scored per judgement."""
    source = Score("names_its_source")
    contained = Score("self_contained")
    subject = Score("subject")

    for case in PHRASING:
        reading = verifier.read(case["question"], [_NOTHING])
        source.record(case["name"], case["names_its_source"], reading.names_its_source)
        contained.record(case["name"], case["self_contained"], reading.self_contained)
        # Containment, not equality: `holds` is a word the copied span must
        # carry, and "" is a question that names nothing.
        wanted = case["holds"]
        got = reading.subject
        subject.record(
            case["name"],
            bool(wanted),
            bool(got) and wanted.casefold() in got.casefold(),
        )
    return source, contained, subject


@pytest.mark.parametrize("judgement", ["names_its_source", "self_contained"])
def test_a_judgement_beats_answering_the_same_way_every_time(scored, judgement) -> None:
    """A judge that ignores its input scores the majority class."""
    found = next(one for one in scored if one.name == judgement)
    floor = majority(judgement)

    print(f"\n{found.report()}\n      (answering the same way every time: {floor:.1%})")

    assert found.accuracy > floor, (
        f"{judgement} scored {found.accuracy:.1%} against a floor of {floor:.1%}: "
        f"it is not reading the question"
    )


def test_every_judgement_is_scored_on_its_own(scored) -> None:
    """The table a swap is compared against. Printed, never a gate."""
    print("\nphrasing judgements, one line each:")
    for one in scored:
        print(one.report())

    assert all(one.total == len(PHRASING) for one in scored), (
        "a judgement was not put every case"
    )


def test_the_german_cases_are_judged_as_well_as_the_english(scored, verifier) -> None:
    """A model that reads German worse is the risk a swap carries.

    Printed rather than asserted: twelve cases against seven is not a
    difference anybody should gate on. It is here because the corpus is
    German and an English-first judge would pass every test above.
    """
    by_language: dict[str, list[bool]] = {"de": [], "en": []}
    for case in PHRASING:
        reading = verifier.read(case["question"], [_NOTHING])
        by_language[case["language"]].append(
            reading.names_its_source == case["names_its_source"]
            and reading.self_contained == case["self_contained"]
        )

    for language, results in by_language.items():
        share = sum(results) / len(results) if results else 0.0
        print(
            f"\n{language}: both judgements right on {sum(results)}/{len(results)} = {share:.1%}"
        )

    assert by_language["de"], "no German case was scored"
