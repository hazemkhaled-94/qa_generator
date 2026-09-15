"""Invariants, over inputs nobody thought to write down.

The example-based tests say what happens to the text this corpus holds.
These say what must hold for any text at all, which is what a document
nobody has seen yet will be.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import pytest
from factories import chunked
from hypothesis import given, settings
from hypothesis import strategies as st

from preprocessing.chunking.passages import NoPassages

#: The setting the parser properties read and write.
NAME = "QA_PROPERTY_SETTING"

#: Anything a settings file can actually carry. The environment holds
#: UTF-8 bytes, so os.environ refuses a NUL or a lone surrogate before the
#: reader ever sees one.
SETTING = st.text(
    alphabet=st.characters(exclude_characters="\x00", exclude_categories=("Cs",)),
    max_size=20,
)

#: Prose-shaped text: printable, and not only whitespace.
PROSE = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x2FF, categories=None),
    min_size=1,
    max_size=200,
)


@given(chunks=st.lists(PROSE, min_size=1, max_size=8))
@settings(max_examples=100, deadline=None)
def test_chunking_stores_every_chunk_that_holds_anything(chunks) -> None:
    """Nothing is discarded on size, and a blank chunk is not a passage."""
    expected = [text.strip() for text in chunks if text.strip()]
    if not expected:
        with pytest.raises(NoPassages):
            chunked(chunks)
        return

    stored = chunked(chunks).passages

    assert [one.text for one in stored] == expected


@given(chunks=st.lists(PROSE, min_size=1, max_size=8))
@settings(max_examples=100, deadline=None)
def test_ordinals_run_from_one_without_a_gap(chunks) -> None:
    """A reader counts on the first being 1 and on there being no hole."""
    try:
        stored = chunked(chunks).passages
    except NoPassages:
        return

    assert [one.ordinal for one in stored] == list(range(1, len(stored) + 1))


@given(chunks=st.lists(PROSE, min_size=1, max_size=8), budget=st.integers(1, 20))
@settings(max_examples=100, deadline=None)
def test_the_oversized_count_never_exceeds_the_passages_stored(chunks, budget) -> None:
    """A tripwire, not a filter: it counts what was kept."""
    try:
        result = chunked(chunks, max_tokens=budget)
    except NoPassages:
        return

    assert 0 <= result.oversized <= len(result.passages)


@given(text=PROSE)
@settings(max_examples=200, deadline=None)
def test_repairing_text_twice_changes_nothing_the_second_time(text) -> None:
    """The repair is a normalisation, so it has to be idempotent."""
    from preprocessing.parsing.pipelines.pdf import _mend

    once = _mend(text)

    assert _mend(once) == once


@given(text=PROSE)
@settings(max_examples=200, deadline=None)
def test_repairing_text_leaves_no_control_character(text) -> None:
    """Newline and tab are the two the passage text is allowed to keep."""
    from preprocessing.parsing.pipelines.pdf import _mend

    mended = _mend(text)

    assert not [
        char
        for char in mended
        if ord(char) < 32 and char not in "\n\t" or ord(char) == 0x7F
    ]


@given(text=PROSE)
@settings(max_examples=200, deadline=None)
def test_repairing_text_leaves_only_ordinary_spaces(text) -> None:
    """A model asked to copy a span back types U+0020 for all of them."""
    from preprocessing.parsing.pipelines.pdf import _mend

    mended = _mend(text)

    assert not [char for char in mended if char.isspace() and char not in " \n\t"], (
        repr(mended)
    )


@given(typed=st.text(min_size=1, max_size=40))
@settings(max_examples=200, deadline=None)
def test_a_search_never_reaches_the_database_as_a_wildcard(typed) -> None:
    """Whatever a person types is characters, not a pattern."""
    from sqlalchemy import select
    from sqlalchemy.dialects import postgresql

    from database.qa_generator.documents import Document
    from database.qa_generator.repository import matching

    statement = (
        select(Document.sha256)
        .where(matching(typed, Document.title))
        .compile(dialect=postgresql.dialect())
    )

    assert "ESCAPE" in str(statement)
    escaped = next(
        value for value in statement.params.values() if isinstance(value, str)
    )
    assert "%" not in escaped.replace("/%", "")
    assert "_" not in escaped.replace("/_", "")


#: A fact id, a document and a passage, which is all grouping reads.
FACTS = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=10_000),
        st.sampled_from("abcd"),
        st.integers(min_value=1, max_value=20),
    ),
    min_size=0,
    max_size=24,
    unique_by=lambda one: one[0],
)


def _sources(rows):
    """Turns the generated triples into the facts selection reads."""
    from factories import source

    return [
        source(fact_id, document=document, passage_id=passage)
        for fact_id, document, passage in rows
    ]


#: Every shape a plan may ask a deal for.
SHAPES = st.sampled_from(["single", "cross", "bridge"])


def _dealt(rows, wanted, size, shape):
    """Every sample one deal produces, for the shape asked for."""
    from question_generation.selection import Deal

    deal = Deal(_sources(rows), wanted=wanted, size=size)
    formed = []
    while (one := deal.sample(shape)) is not None:
        formed.append(one)
    return formed


@given(
    rows=FACTS,
    wanted=st.integers(1, 12),
    size=st.integers(1, 4),
    shape=SHAPES,
)
@settings(max_examples=200, deadline=None)
def test_a_fact_is_never_written_about_twice_in_one_deal(
    rows, wanted, size, shape
) -> None:
    """A fact in two samples is the same claim asked about twice."""
    used = [fact.id for one in _dealt(rows, wanted, size, shape) for fact in one.facts]

    assert len(used) == len(set(used))


@given(
    rows=FACTS,
    wanted=st.integers(1, 12),
    size=st.integers(1, 4),
    shape=SHAPES,
)
@settings(max_examples=200, deadline=None)
def test_no_sample_is_empty_or_over_the_cap(rows, wanted, size, shape) -> None:
    """An empty sample has no language and no difficulty to read off it.

    The cap is the whole offer and not a per-passage allowance: it used to
    flush a group and then add the next passage entire, so a passage of
    thirty-seven facts was offered whole and nothing fitted beside it.
    """
    for one in _dealt(rows, wanted, size, shape):
        assert one.facts, "an empty sample was offered"
        assert len(one.facts) <= max(size, len({f.passage_id for f in one.facts}))


@given(
    rows=FACTS,
    wanted=st.integers(1, 12),
    size=st.integers(1, 4),
    shape=SHAPES,
)
@settings(max_examples=200, deadline=None)
def test_a_sample_never_spans_more_passages_than_its_shape_asked_for(
    rows, wanted, size, shape
) -> None:
    """One passage for `single`, at most two for anything wider.

    A shape a topic cannot supply falls back to a narrower one, so this is a
    ceiling and never a floor.
    """
    for one in _dealt(rows, wanted, size, shape):
        reached = len({fact.passage_id for fact in one.facts})
        assert reached == 1 if shape == "single" else reached <= 2


@given(
    rows=FACTS,
    wanted=st.integers(1, 12),
    size=st.integers(1, 4),
    shape=SHAPES,
)
@settings(max_examples=200, deadline=None)
def test_every_scope_always_matches_the_spread_it_is_read_from(
    rows, wanted, size, shape
) -> None:
    """The property the re-check relies on to notice evidence that moved."""
    from question_generation.models import criteria_of

    for one in _dealt(rows, wanted, size, shape):
        assert one.criteria() == criteria_of(
            passages=len({fact.passage_id for fact in one.facts}),
            documents=len({fact.doc_sha256 for fact in one.facts}),
            topics=len({fact.topic_id for fact in one.facts if fact.topic_id}),
            answer_chars=None,
        )


@given(
    passages=st.integers(1, 5),
    documents=st.integers(1, 5),
    topics=st.integers(1, 5),
    answer=st.integers(0, 200),
    follows=st.booleans(),
)
@settings(max_examples=300, deadline=None)
def test_the_difficulty_band_never_disagrees_with_its_own_score(
    passages, documents, topics, answer, follows
) -> None:
    """The band is the total, and nothing else may move it.

    A weighting would be an opinion, and the point of deriving difficulty
    rather than judging it is that nobody has to hold one.
    """
    from question_generation.models import criteria_of

    read = criteria_of(
        passages=passages,
        documents=documents,
        topics=topics,
        answer_chars=answer,
        follows=follows,
    )
    from question_generation.models import band

    assert read.difficulty == band(read.score)


@given(
    passages=st.integers(1, 5),
    documents=st.integers(1, 5),
    topics=st.integers(1, 5),
)
@settings(max_examples=200, deadline=None)
def test_a_wider_scope_is_never_easier(passages, documents, topics) -> None:
    """Every criterion points the same way, so the band is monotonic."""
    from question_generation.models import criteria_of

    order = {"easy": 0, "medium": 1, "hard": 2}
    narrow = criteria_of(
        passages=1, documents=1, topics=1, answer_chars=0, follows=False
    )
    wide = criteria_of(
        passages=passages,
        documents=documents,
        topics=topics,
        answer_chars=0,
        follows=False,
    )

    assert order[wide.difficulty] >= order[narrow.difficulty]


#: A mix of types, as QUESTIONS_TYPE_MIX carries one.
MIX = st.dictionaries(
    st.sampled_from(["factoid", "reason", "comparison", "procedure", "condition"]),
    st.integers(0, 6),
    min_size=1,
    max_size=5,
).filter(lambda one: any(weight > 0 for weight in one.values()))


@given(mix=MIX, slots=st.integers(0, 60))
@settings(max_examples=300, deadline=None)
def test_a_kind_with_no_weight_is_never_written(mix, slots) -> None:
    """Which is the switch for choosing what a run produces.

    Everything else here is a proportion; this one is absolute, and a type
    leaking into a run that turned it off is a set nobody asked for.
    """
    from question_generation.planning import allocate

    chosen = allocate(mix, slots)

    assert all(mix.get(name, 0) > 0 for name in chosen)


@given(mix=MIX, slots=st.integers(0, 60))
@settings(max_examples=300, deadline=None)
def test_the_slots_allocated_are_the_slots_asked_for(mix, slots) -> None:
    """A mix decides the proportions, never how many questions there are."""
    from question_generation.planning import allocate

    assert len(allocate(mix, slots)) == slots


@given(mix=MIX, slots=st.integers(1, 60))
@settings(max_examples=300, deadline=None)
def test_a_heavier_weight_never_takes_fewer_slots_than_a_lighter_one(
    mix, slots
) -> None:
    """The one thing a mix has to mean.

    Not that every count is its exact quota: highest averages is quota-free
    and a very uneven mix can run a slot past it. What it must never do is
    invert the order somebody wrote down.
    """
    from collections import Counter

    from question_generation.planning import allocate

    counts = Counter(allocate(mix, slots))

    for name, weight in mix.items():
        for other, lighter in mix.items():
            if weight > lighter > 0:
                assert counts[name] >= counts[other]


@given(
    wanted=st.integers(0, 40),
    mix=MIX,
    share=st.floats(0.0, 1.0),
)
@settings(max_examples=200, deadline=None)
def test_a_plan_never_asks_for_a_shape_its_type_cannot_use(wanted, mix, share) -> None:
    """A comparison drawn from one passage is a question about one thing."""
    from question_generation.planning import plans

    for one in plans(
        wanted=wanted,
        types=mix,
        bands={"easy": 1, "medium": 1, "hard": 1},
        unanswerable_share=share,
    ):
        if one.spec.spans and one.answerable:
            assert one.shape != "single"
        if not one.answerable:
            assert one.shape == "single"


@given(count=st.integers(1, 40), share=st.floats(0.0, 1.0))
@settings(max_examples=200, deadline=None)
def test_the_unanswerable_share_is_never_more_than_it_was_asked_for(
    count, share
) -> None:
    """A run must not quietly become mostly questions with no answer."""
    from question_generation.selection import spread

    over = sum(spread(index, share) for index in range(count))

    assert 0 <= over <= count
    assert over == int(count * share) or over == int(count * share) + 1


@contextmanager
def given_setting(value: str):
    """Sets the setting under test, and takes it back out afterwards.

    A context manager rather than monkeypatch: a function-scoped fixture is
    not reset between the inputs `@given` generates, so one example would
    leave its value behind for the next.
    """
    os.environ[NAME] = value
    try:
        yield
    finally:
        os.environ.pop(NAME, None)


@given(value=st.integers(min_value=-(10**9), max_value=10**9))
@settings(max_examples=200, deadline=None)
def test_a_whole_number_setting_round_trips(value) -> None:
    """Whatever int() accepts, read back as the same number."""
    from settings import integer

    with given_setting(str(value)):
        assert integer(NAME) == value


@given(value=SETTING)
@settings(max_examples=200, deadline=None)
def test_a_setting_that_is_not_a_number_is_always_refused(value) -> None:
    """Never a silent zero, whatever the string is."""
    from settings import integer

    with given_setting(value):
        try:
            expected = int(value.strip())
        except ValueError:
            with pytest.raises((ValueError, KeyError), match=NAME):
                integer(NAME)
            return
        assert integer(NAME) == expected


@given(
    names=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6),
        min_size=1,
        max_size=6,
    )
)
@settings(max_examples=100, deadline=None)
def test_a_list_setting_keeps_every_name_and_invents_none(names) -> None:
    """The separator is what splits it, and nothing else."""
    from settings import csv

    with given_setting(",".join(names)):
        assert csv(NAME) == tuple(names)


@given(passage_id=st.integers(min_value=1, max_value=2**31 - 1))
@settings(max_examples=100, deadline=None)
def test_a_narrowing_coerces_any_id_the_column_can_hold(passage_id) -> None:
    """Comparing an integer column against a string is an error at the database."""
    from extraction.repository import PassageQueue

    narrowed = PassageQueue().narrow("passage", str(passage_id))

    assert narrowed.right.value == passage_id
