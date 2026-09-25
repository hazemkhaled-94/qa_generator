"""Every free-text query parameter is bounded before it reaches the driver.

PostgreSQL text cannot hold a NUL byte and psycopg refuses one rather than
truncating, so an unbounded string parameter is a 500 waiting for anybody
who sends `%00`. `api/params.py` exists for that, and the routes added
after it are the ones that have to remember to reach for it: `/prompts`
took three bare strings and answered `?service=%00` with a server error.

Read off the route signatures rather than off a built application, so this
needs no container.
"""

from __future__ import annotations

import inspect

import pytest

from api.routes import prompts

#: The routes whose free-text parameters this checks, as (handler, names).
#: Only the ones a caller types into: an enumerated `Literal` refuses
#: anything off its list already, and an id is bounded by `RowId`.
FREE_TEXT = [(prompts.read, ("service", "version", "name"))]

CASES = [(handler, name) for handler, names in FREE_TEXT for name in names]


def bounds(handler, name: str) -> dict[str, object]:
    """The constraints one parameter of one route handler was declared with.

    Read out of `metadata`, which is where a `Query` keeps them: pydantic
    holds each as its own annotated-types object rather than as an
    attribute, and reading by attribute name survives it renaming the
    classes.
    """
    default = inspect.signature(handler).parameters[name].default
    found: dict[str, object] = {}
    for one in getattr(default, "metadata", ()):
        for attribute in ("max_length", "pattern"):
            value = getattr(one, attribute, None)
            if value is not None:
                found[attribute] = value
    return found


@pytest.mark.parametrize(("handler", "name"), CASES, ids=[one[1] for one in CASES])
def test_a_free_text_parameter_has_a_ceiling(handler, name) -> None:
    """A value past it is one no row could hold anyway."""
    assert bounds(handler, name).get("max_length")


@pytest.mark.parametrize(("handler", "name"), CASES, ids=[one[1] for one in CASES])
def test_a_free_text_parameter_refuses_a_nul(handler, name) -> None:
    """Refused at the edge, which is cheaper than teaching every query."""
    pattern = bounds(handler, name).get("pattern")

    assert isinstance(pattern, str) and "\\x00" in pattern
