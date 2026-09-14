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

    def __init__(self, **fields: str) -> None:
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


def test_the_model_never_sees_the_passage_a_fact_came_from() -> None:
    """A fact already stands on its own; a passage would be a summary task."""
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(group(source(1, statement="A claim.")), answerable=True)

    assert "It ships from Hamburg." not in model.user


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
