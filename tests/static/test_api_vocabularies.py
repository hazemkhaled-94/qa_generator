"""The filters the API offers against the values the database holds.

A route spells its choices as a `Literal` so the OpenAPI document lists them
and the frontend's pickers cannot drift from what the backend will accept.
That is two copies of one vocabulary, and the copy in the route is the one
nothing else would notice going stale: adding a question type reaches the
enum, the CHECK constraint and the writer, and a listing that cannot filter
to the new kind still answers every request it is given.

Which is what happened. `implication` and `application` were written,
stored and constrained for a day before the route's list knew about them.
"""

from __future__ import annotations

from typing import Literal, get_args

import pytest

from database.qa_generator import (
    AnswerForm,
    CognitiveLevel,
    Difficulty,
    DocumentScope,
    PassageScope,
    QuestionType,
    TopicScope,
)

#: Each route alias beside the enum it has to agree with. The route is
#: imported lazily inside the test so this module stays cheap to collect.
PINNED = {
    "QuestionType": QuestionType,
    "CognitiveLevel": CognitiveLevel,
    "AnswerForm": AnswerForm,
    "PassageScope": PassageScope,
    "DocumentScope": DocumentScope,
    "TopicScope": TopicScope,
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_a_filter_offers_exactly_what_the_column_holds(name: str) -> None:
    """Neither more than the database accepts nor less than it stores."""
    from api.routes import questions

    offered = set(get_args(getattr(questions, name)))
    held = {str(one) for one in PINNED[name]}

    assert offered == held, (
        f"{name}: the route offers {sorted(offered - held)} that no column "
        f"holds, and is missing {sorted(held - offered)} that one does"
    )


def test_the_band_filter_offers_exactly_the_three_bands() -> None:
    """Spelled `Band` rather than after its enum, so it is checked by name."""
    from api.routes import questions

    assert set(get_args(questions.Band)) == {str(one) for one in Difficulty}


def test_every_pinned_alias_is_a_literal() -> None:
    """A free string would pass the check above by accident."""
    from api.routes import questions

    for name in (*PINNED, "Band"):
        assert get_args(getattr(questions, name)), f"{name} is not a Literal"
        assert Literal[get_args(getattr(questions, name))]
