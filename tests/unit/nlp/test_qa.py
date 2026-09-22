"""The span decode, which no library does for us any more.

`pipeline("question-answering")` is gone from transformers 5, so the null
score, the start/end pairing, the offsets back to characters and the sliding
window are written out in `nlp.qa`. That is arithmetic over logits, and
arithmetic over logits fails quietly: a wrong pair returns a plausible
sentence rather than an error.

No weights are loaded. The tokenizer and the model are stood in for, so what
is under test is the decode and not a checkpoint - which is the half that is
ours. The reader against real weights is `tests/eval/`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from nlp.qa import Extractive, Span, _best_of


@dataclass
class Answered:
    """What a question-answering head returns."""

    start_logits: object
    end_logits: object


class Encoding(dict):
    """What a fast tokenizer returns, with the two methods `_span` uses."""

    def __init__(self, sequences: list[list[int | None]], offsets: list[list]) -> None:
        """Holds one entry per window."""
        super().__init__(input_ids=[[0] * len(one) for one in sequences])
        self["offset_mapping"] = offsets
        self._sequences = sequences

    def sequence_ids(self, index: int) -> list[int | None]:
        """Which side of the pair each token of one window came from."""
        return self._sequences[index]


class Tokenizer:
    """Splits on whitespace and reports real offsets into the context.

    One window per context, which is what a passage shorter than the model's
    own length produces.
    """

    def __init__(self, windows: int = 1) -> None:
        """Repeats the same context over this many windows."""
        self._windows = windows

    def __call__(self, question: str, context: str, **ignored) -> Encoding:
        """Encodes one pair."""
        del ignored
        asked = question.split()
        words = [(one.start(), one.end()) for one in re.finditer(r"\S+", context)]
        sequence = [None, *([0] * len(asked)), None, *([1] * len(words))]
        offsets = [(0, 0)] * (len(asked) + 2) + [list(one) for one in words]
        return Encoding([sequence] * self._windows, [offsets] * self._windows)


class Model:
    """A head whose argmax a test places where it wants it."""

    def __init__(self, starts: list[list[float]], ends: list[list[float]]) -> None:
        """Holds one row of logits per window."""
        import torch

        self._answered = Answered(torch.tensor(starts), torch.tensor(ends))

    def __call__(self, **encoded) -> Answered:
        """The logits, whatever it was handed."""
        del encoded
        return self._answered


QUESTION = "how heavy"
CONTEXT = "The compact model weighs 4 kg in total."

#: Where each word of CONTEXT sits once the question and its separators are
#: in front of it: two for [CLS] and the question, one for [SEP].
_OFFSET = len(QUESTION.split()) + 2


def at(word: int) -> int:
    """The token position of one word of CONTEXT."""
    return _OFFSET + word


def logits(**placed: float) -> list[float]:
    """One window's logits, flat at zero except where a test puts a peak."""
    row = [0.0] * (_OFFSET + len(CONTEXT.split()))
    for name, value in placed.items():
        row[int(name.removeprefix("w"))] = value
    return row


def reader(starts: list[float], ends: list[float], windows: int = 1) -> Extractive:
    """A reader whose weights are these logits.

    `cached_property` keeps its value in the instance dictionary, so this is
    what a loaded model looks like without one being loaded.
    """
    built = Extractive("a/checkpoint-that-does-not-exist")
    built.__dict__["_loaded"] = (
        Tokenizer(windows),
        Model([starts] * windows, [ends] * windows),
    )
    return built


# ── The pair, and the text it covers ──────────────────────────────────────


def test_the_span_is_exactly_the_characters_the_offsets_cover() -> None:
    """A span a reader is shown has to be the passage's own words."""
    found = reader(
        logits(**{f"w{at(4)}": 9.0}),
        logits(**{f"w{at(5)}": 9.0}),
    ).answer(QUESTION, [CONTEXT], confidence=0.5)

    assert found is not None
    assert found.text == "4 kg"
    assert found.text in CONTEXT


def test_a_head_that_would_rather_say_nothing_says_nothing() -> None:
    """The null span at position 0 is what SQuAD 2.0 trained it to give."""
    null = {"w0": 20.0}
    found = reader(
        logits(**null, **{f"w{at(4)}": 1.0}),
        logits(**null, **{f"w{at(5)}": 1.0}),
    ).answer(QUESTION, [CONTEXT], confidence=0.0)

    assert found is None


def test_the_score_is_measured_against_answering_nothing() -> None:
    """Two numbers, so it reads as "how sure against declining"."""
    found = reader(
        logits(**{f"w{at(4)}": 4.0}),
        logits(**{f"w{at(5)}": 4.0}),
    ).answer(QUESTION, [CONTEXT], confidence=0.0)

    assert found is not None
    # The best pair scores 8 against a null of 0, so softmax([0, 8])[1].
    assert found.score == pytest.approx(0.99966, abs=1e-4)
    assert 0.0 <= found.score <= 1.0


def test_a_pair_running_backwards_is_not_an_answer() -> None:
    """An end before its start covers nothing, however high it scores.

    The best of each is `kg` starting and `4` ending, which scores 18 and
    names an empty slice. The best pair that runs forwards scores 10.
    """
    found = reader(
        logits(**{f"w{at(5)}": 9.0, f"w{at(0)}": 1.0}),
        logits(**{f"w{at(4)}": 9.0, f"w{at(1)}": 1.0}),
    ).answer(QUESTION, [CONTEXT], confidence=0.0)

    assert found is not None
    assert found.text == "The compact model weighs 4"


def test_an_answer_longer_than_the_cap_is_not_taken() -> None:
    """Without one the best pair is most of the passage."""
    from nlp.qa import _MAX_ANSWER_TOKENS

    wide = "word " * (_MAX_ANSWER_TOKENS + 10)
    tokens = len(wide.split())
    starts = [0.0] * (_OFFSET + tokens)
    ends = [0.0] * (_OFFSET + tokens)
    starts[_OFFSET] = 9.0
    ends[_OFFSET + tokens - 1] = 9.0

    built = Extractive("a/checkpoint")
    built.__dict__["_loaded"] = (Tokenizer(), Model([starts], [ends]))
    found = built.answer(QUESTION, [wide.strip()], confidence=0.0)

    assert found is not None
    assert len(found.text.split()) <= _MAX_ANSWER_TOKENS


def test_no_span_is_taken_from_the_question_itself() -> None:
    """Only `sequence_ids` 1 is the passage; 0 is what was asked."""
    found = reader(
        logits(w1=20.0, **{f"w{at(4)}": 1.0}),
        logits(w2=20.0, **{f"w{at(5)}": 1.0}),
    ).answer(QUESTION, [CONTEXT], confidence=0.0)

    assert found is not None
    assert found.text == "4 kg", "a question token was paired"


def test_a_token_covering_no_characters_is_not_a_span() -> None:
    """Padding on the passage side carries the offset pair (0, 0).

    Paired it names an empty slice, which is a confident answer of nothing.
    """
    built = Extractive("a/checkpoint")
    words = CONTEXT.split()
    sequence = [None, 0, 0, None, *([1] * (len(words) + 1))]
    offsets = [(0, 0)] * 4 + [
        list(one)
        for one in [(m.start(), m.end()) for m in re.finditer(r"\S+", CONTEXT)]
    ]
    # One more token on the passage side, covering nothing.
    offsets.append((0, 0))
    padded = len(sequence) - 1
    starts = [0.0] * len(sequence)
    ends = [0.0] * len(sequence)
    starts[padded] = ends[padded] = 20.0
    starts[4], ends[5] = 1.0, 1.0
    built.__dict__["_loaded"] = (
        lambda *args, **kwargs: Encoding([sequence], [offsets]),
        Model([starts], [ends]),
    )

    found = built.answer(QUESTION, [CONTEXT], confidence=0.0)

    assert found is not None
    assert found.text == "The compact"


def test_the_model_names_itself_for_the_row_it_read() -> None:
    """Recorded beside what it read."""
    assert Extractive("a/checkpoint").model == "a/checkpoint"


def test_a_window_carrying_none_of_the_passage_is_skipped() -> None:
    """A window that is all question and padding has nothing to offer."""
    built = Extractive("a/checkpoint")
    tokenizer = Tokenizer()
    encoding = Encoding([[None, 0, 0, None]], [[(0, 0)] * 4])
    built.__dict__["_loaded"] = (
        lambda *args, **kwargs: encoding,
        Model([[0.0] * 4], [[0.0] * 4]),
    )
    del tokenizer

    assert built.answer(QUESTION, [CONTEXT], confidence=0.0) is None


# ── Over several passages ─────────────────────────────────────────────────


def test_the_best_passage_wins_rather_than_the_first() -> None:
    """An answer is in the material when SOME passage holds it."""
    built = Extractive("a/checkpoint")
    scores = iter([0.6, 0.95, 0.7])
    built._span = lambda question, context: ("found", next(scores))  # type: ignore[method-assign]

    found = built.answer(QUESTION, ["a", "b", "c"], confidence=0.5)

    assert found is not None
    assert found.passage == 1
    assert found.score == pytest.approx(0.95)


def test_a_span_below_the_confidence_falls_through() -> None:
    """Everything else is answered by something that can read."""
    built = Extractive("a/checkpoint")
    built._span = lambda question, context: ("found", 0.4)  # type: ignore[method-assign]

    assert built.answer(QUESTION, ["a"], confidence=0.9) is None


def test_a_blank_passage_costs_no_forward_pass() -> None:
    """Nothing to read, so nothing is loaded to read it."""
    asked = []
    built = Extractive("a/checkpoint-that-does-not-exist")
    built._span = lambda question, context: asked.append(context)  # type: ignore[method-assign]

    assert built.answer(QUESTION, ["", "   ", "\n"], confidence=0.0) is None
    assert asked == []


def test_no_passages_at_all_answers_nothing() -> None:
    """The gate calling this may have nothing to offer it."""
    assert (
        Extractive("a/checkpoint-that-does-not-exist").answer(QUESTION, [], 0.5) is None
    )


# ── The shortlist the pairing runs over ───────────────────────────────────


def test_only_positions_inside_the_passage_are_shortlisted() -> None:
    """`_best_of` is what keeps the question out of the answer."""
    assert _best_of([9.0, 1.0, 5.0, 3.0], inside=[1, 2, 3]) == [2, 3, 1]


def test_the_shortlist_is_capped() -> None:
    """The decode is O(k squared) per window."""
    from nlp.qa import _TOP_K

    logits = list(range(100))

    assert len(_best_of(logits, inside=list(range(100)))) == _TOP_K


def test_a_span_carries_where_it_was_found() -> None:
    """The passage's position, so a caller knows which one answered."""
    assert Span(text="4 kg", score=0.9, passage=2).passage == 2
