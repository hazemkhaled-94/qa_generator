"""What the writer asks the model, and what it makes of the answer.

The model is stood in for. What is covered here is everything around the
call: which prompt each kind of question uses, what reaches the model, and
the shape that comes back.
"""

from __future__ import annotations

from factories import group, source

from question_generation.generation import QuestionWriter


class StubModel:
    """A served model that answers from a script and records the asks."""

    model = "ollama/stub"
    temperature = 0.0

    def __init__(self, **fields) -> None:
        """Initialises the model with the fields it answers with."""
        self._fields = fields
        self.asked: list[tuple[str, str]] = []

    def answer(self, *, system: str, user: str, shape):
        """Records the call and answers in the shape asked for."""
        self.asked.append((system, user))
        return shape(
            **{k: v for k, v in self._fields.items() if k in shape.model_fields}
        )

    @property
    def system(self) -> str:
        """The system prompt of the last call."""
        return self.asked[-1][0]

    @property
    def user(self) -> str:
        """The user message of the last call."""
        return self.asked[-1][1]


def test_an_answerable_question_carries_the_answer_it_is_scored_against() -> None:
    """The target answer is the whole point of an answerable question."""
    model = StubModel(question="What does the device weigh?", answer="4 kg")

    written = QuestionWriter(model).write(group(), answerable=True)

    assert written.question_text == "What does the device weigh?"
    assert written.target_answer == "4 kg"
    assert written.answerable is True


def test_an_unanswerable_question_never_carries_a_target_answer() -> None:
    """The questions table refuses one, and it would be a lie either way."""
    model = StubModel(question="What does the device weigh on Mars?", answer="4 kg")

    written = QuestionWriter(model).write(group(), answerable=False)

    assert written.answerable is False
    assert written.target_answer is None


def test_the_two_kinds_are_written_from_different_prompts() -> None:
    """A model told to write an answerable question writes one."""
    model = StubModel(question="Q?", answer="A")
    writer = QuestionWriter(model)

    writer.write(group(), answerable=True)
    asking = model.system
    writer.write(group(), answerable=False)

    assert asking != model.system
    assert "does NOT answer" in model.system


def test_a_perturbation_is_taken_from_one_fact_even_out_of_a_larger_group() -> None:
    """Moving a claim out of reach is a change to one claim.

    A reader given two facts would have two ways to notice, which makes the
    question easier than it is meant to be.
    """
    model = StubModel(question="Q?", answer="A")
    pair = group(source(1, document="a"), source(2, document="b"))

    written = QuestionWriter(model).write(pair, answerable=False)

    assert len(written.group.facts) == 1
    assert "[2]" not in model.user


def test_every_fact_of_a_group_reaches_the_model_numbered() -> None:
    """The model is asked for one question needing all of them."""
    model = StubModel(question="Q?", answer="A")
    pair = group(
        source(1, statement="The device weighs 4 kg."),
        source(2, statement="The device runs for 12 hours."),
    )

    QuestionWriter(model).write(pair, answerable=True)

    assert "[1] The device weighs 4 kg." in model.user
    assert "[2] The device runs for 12 hours." in model.user


def test_the_model_is_shown_the_passage_each_fact_came_from() -> None:
    """Without it there is nothing to write a question out of but the fact.

    A single atomic statement is one triple, so the only question available
    is that statement with one part replaced by a question word. This test
    asserted the opposite when the stage shipped, and locked the defect in.
    """
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(group(source(1, statement="A claim.")), answerable=True)

    assert "It ships from Hamburg." in model.user, "the passage was withheld"
    assert "PASSAGE" in model.user, "and it was not labelled as context"


def test_the_heading_a_passage_sits_under_reaches_the_model() -> None:
    """Most of what says whose rule or which year a question is about."""
    model = StubModel(question="Q?", answer="A")
    under = group(source(1, section_path="MaRisk > BTO 1.2 > Kreditentscheidung"))

    QuestionWriter(model).write(under, answerable=True)

    assert "MaRisk > BTO 1.2 > Kreditentscheidung" in model.user


def test_the_facts_are_marked_as_what_the_answer_must_come_from() -> None:
    """The passage is for phrasing; the facts are what may be asked about.

    A model shown both unlabelled asks about whatever it finds in the
    passage, and the answer stops resting on a checked claim.
    """
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(group(), answerable=True)

    facts_at = model.user.index("FACTS")
    passage_at = model.user.index("PASSAGE")
    assert facts_at < passage_at, "the facts must be stated before the context"


def test_the_question_cites_only_the_facts_the_model_said_it_used() -> None:
    """A sample is an offer; the citation is what the question needs.

    Recording the whole sample gave a question written from one fact a
    `cross_document` difficulty earned by a fact it never used.
    """
    model = StubModel(question="Q?", answer="A", facts=[2])
    sample = group(
        source(1, document="a", passage_id=1),
        source(2, document="b", passage_id=2),
    )

    written = QuestionWriter(model).write(sample, answerable=True)

    assert [one.id for one in written.group.facts] == [2]
    assert written.group.difficulty == "single_passage"


def test_naming_no_facts_falls_back_to_the_first() -> None:
    """The answer still has to rest on something.

    A candidate with no citation is one the orphan trigger never reaches, so
    it would sit in the table forever with nothing to trace it to.
    """
    model = StubModel(question="Q?", answer="A", facts=[])
    sample = group(source(1, document="a"), source(2, document="b"))

    written = QuestionWriter(model).write(sample, answerable=True)

    assert [one.id for one in written.group.facts] == [1]


def test_a_fact_number_the_sample_does_not_have_is_dropped() -> None:
    """A model counting past the list must not index into another topic."""
    model = StubModel(question="Q?", answer="A", facts=[1, 7, 0, -3])
    sample = group(source(1), source(2))

    written = QuestionWriter(model).write(sample, answerable=True)

    assert [one.id for one in written.group.facts] == [1]


def test_a_blank_answer_is_no_answer_rather_than_an_empty_one() -> None:
    """Which the malformed gate then rejects, and counts."""
    model = StubModel(question="What does the device weigh?", answer="   ")

    written = QuestionWriter(model).write(group(), answerable=True)

    assert written.target_answer is None


def test_the_question_is_kept_even_when_it_is_unusable() -> None:
    """A dropped candidate is a rejection nobody can count."""
    model = StubModel(question="   ", answer="")

    written = QuestionWriter(model).write(group(), answerable=True)

    assert written.question_text == ""
