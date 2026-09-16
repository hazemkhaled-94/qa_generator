"""What the writer asks the model, and what it makes of the answer.

The model is stood in for. What is covered here is everything around the
call: which prompt each kind of question uses, what reaches the model, and
the shape that comes back.
"""

from __future__ import annotations

from factories import group, plan, source

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

    written = QuestionWriter(model).write(group(), plan())

    assert written.question_text == "What does the device weigh?"
    assert written.target_answer == "4 kg"
    assert written.answerable is True


def test_an_unanswerable_question_never_carries_a_target_answer() -> None:
    """The questions table refuses one, and it would be a lie either way."""
    model = StubModel(question="What does the device weigh on Mars?", answer="4 kg")

    written = QuestionWriter(model).write(group(), plan(answerable=False))

    assert written.answerable is False
    assert written.target_answer is None


def test_the_two_kinds_are_written_from_different_prompts() -> None:
    """A model told to write an answerable question writes one."""
    model = StubModel(question="Q?", answer="A")
    writer = QuestionWriter(model)

    writer.write(group(), plan())
    asking = model.system
    writer.write(group(), plan(answerable=False))

    assert asking != model.system
    assert "does NOT answer" in model.system


def test_a_perturbation_is_taken_from_one_fact_even_out_of_a_larger_group() -> None:
    """Moving a claim out of reach is a change to one claim.

    A reader given two facts would have two ways to notice, which makes the
    question easier than it is meant to be.
    """
    model = StubModel(question="Q?", answer="A")
    pair = group(source(1, document="a"), source(2, document="b"))

    written = QuestionWriter(model).write(pair, plan(answerable=False))

    assert len(written.group.facts) == 1
    assert "[2]" not in model.user


def test_every_fact_of_a_group_reaches_the_model_numbered() -> None:
    """The model is asked for one question needing all of them."""
    model = StubModel(question="Q?", answer="A")
    pair = group(
        source(1, statement="The device weighs 4 kg."),
        source(2, statement="The device runs for 12 hours."),
    )

    QuestionWriter(model).write(pair, plan())

    assert "[1] The device weighs 4 kg." in model.user
    assert "[2] The device runs for 12 hours." in model.user


def test_the_model_is_shown_the_passage_each_fact_came_from() -> None:
    """Without it there is nothing to write a question out of but the fact.

    A single atomic statement is one triple, so the only question available
    is that statement with one part replaced by a question word. This test
    asserted the opposite when the stage shipped, and locked the defect in.
    """
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(group(source(1, statement="A claim.")), plan())

    assert "It ships from Hamburg." in model.user, "the passage was withheld"
    assert "PASSAGE" in model.user, "and it was not labelled as context"


def test_the_heading_a_passage_sits_under_reaches_the_model() -> None:
    """Most of what says whose rule or which year a question is about."""
    model = StubModel(question="Q?", answer="A")
    under = group(source(1, section_path="MaRisk > BTO 1.2 > Kreditentscheidung"))

    QuestionWriter(model).write(under, plan())

    assert "MaRisk > BTO 1.2 > Kreditentscheidung" in model.user


def test_the_facts_are_marked_as_what_the_answer_must_come_from() -> None:
    """The passage is for phrasing; the facts are what may be asked about.

    A model shown both unlabelled asks about whatever it finds in the
    passage, and the answer stops resting on a checked claim.
    """
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(group(), plan())

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

    written = QuestionWriter(model).write(sample, plan())

    assert [one.id for one in written.group.facts] == [2]
    assert written.group.criteria().passage_scope == "single_passage"


def test_naming_no_facts_falls_back_to_the_first() -> None:
    """The answer still has to rest on something.

    A candidate with no citation is one the orphan trigger never reaches, so
    it would sit in the table forever with nothing to trace it to.
    """
    model = StubModel(question="Q?", answer="A", facts=[])
    sample = group(source(1, document="a"), source(2, document="b"))

    written = QuestionWriter(model).write(sample, plan())

    assert [one.id for one in written.group.facts] == [1]


def test_a_fact_number_the_sample_does_not_have_is_dropped() -> None:
    """A model counting past the list must not index into another topic."""
    model = StubModel(question="Q?", answer="A", facts=[1, 7, 0, -3])
    sample = group(source(1), source(2))

    written = QuestionWriter(model).write(sample, plan())

    assert [one.id for one in written.group.facts] == [1]


def test_a_blank_answer_is_no_answer_rather_than_an_empty_one() -> None:
    """Which the malformed gate then rejects, and counts."""
    model = StubModel(question="What does the device weigh?", answer="   ")

    written = QuestionWriter(model).write(group(), plan())

    assert written.target_answer is None


def test_the_question_is_kept_even_when_it_is_unusable() -> None:
    """A dropped candidate is a rejection nobody can count."""
    model = StubModel(question="   ", answer="")

    written = QuestionWriter(model).write(group(), plan())

    assert written.question_text == ""


# ── The kind of question the plan asked for ───────────────────────────────


def test_each_kind_is_written_from_its_own_prompt() -> None:
    """One prompt for every kind wrote one kind of question.

    The prompt that shipped demanded a short noun phrase with no verb in it,
    so a why, a how and a what-happens-if were all unwritable whatever the
    material said.
    """
    model = StubModel(question="Q?", answer="A")
    writer = QuestionWriter(model)

    writer.write(group(), plan(question_type="factoid"))
    factoid = model.system
    writer.write(group(), plan(question_type="reason"))
    reason = model.system

    assert factoid != reason
    assert "why" in reason
    assert "SHORT NOUN PHRASE" in factoid
    assert "SHORT NOUN PHRASE" not in reason


def test_a_question_carries_the_kind_and_the_band_it_was_planned_as() -> None:
    """What was asked for is stored beside what came out."""
    model = StubModel(question="Q?", answer="A")

    written = QuestionWriter(model).write(
        group(), plan(question_type="reason", band="medium")
    )

    assert written.spec.name == "reason"
    assert written.spec.form == "explanation"
    assert written.planned_difficulty == "medium"


def test_a_wide_sample_tells_the_writer_to_use_both_passages() -> None:
    """Otherwise it answers the first and the spread is a fiction.

    The row would carry a cross-document label earned by a fact the question
    never used, which is the defect the citation reporting exists for.
    """
    model = StubModel(question="Q?", answer="A")
    writer = QuestionWriter(model)

    writer.write(group(), plan(shape="single"))
    narrow = model.system
    writer.write(group(), plan(shape="cross"))

    assert "MORE THAN ONE" not in narrow
    assert "MORE THAN ONE" in model.system


def test_a_perturbation_is_told_what_kind_of_question_to_ask() -> None:
    """An unanswerable question is of a kind too, or it tests nothing."""
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(
        group(), plan(question_type="procedure", answerable=False)
    )

    assert "does NOT answer" in model.system
    assert "how something is done" in model.system


def test_a_follow_up_takes_its_own_kind() -> None:
    """A conversation asking the same kind three times is one question."""
    model = StubModel(question="And why?", answer="because it was rebuilt")

    written = QuestionWriter(model).follow_up(
        group(), (("How heavy is it?", "4 kg"),), plan(question_type="reason")
    )

    assert written.spec.name == "reason"
    assert "why" in model.system
    assert "CONVERSATION so far" in model.user


def test_the_writer_is_told_not_to_pad_a_question_with_hedges() -> None:
    """Nobody types `specific` into a search box.

    Measured over 24 real questions: six used `spezifisch` and two `konkret`.
    A third of the set read like a form rather than like somebody asking.
    """
    model = StubModel(question="Q?", answer="A")

    QuestionWriter(model).write(group(), plan())

    assert "PADDING" in model.system


def test_the_unanswerable_prompt_carries_the_same_reading_rules() -> None:
    """A rule in one prompt and not the other is how the two disagree.

    Measured: every question padded with `spezifische Kriterien` was an
    unanswerable one, because the perturbation was the only prompt that had
    never been told not to pad. It had its own copy of some rules and none of
    the rest.
    """
    model = StubModel(question="Q?", answer="A")
    writer = QuestionWriter(model)

    writer.write(group(), plan())
    answerable = model.system
    writer.write(group(), plan(answerable=False))

    for rule in ("PADDING", "NEVER SAY WHERE THE ANSWER IS", "ONE question"):
        assert rule in answerable, rule
        assert rule in model.system, f"{rule} is missing from the perturbation"
