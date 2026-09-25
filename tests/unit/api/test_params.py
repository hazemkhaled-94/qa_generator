"""Every free-text query parameter is bounded before it reaches the driver.

PostgreSQL text cannot hold a NUL byte and psycopg refuses one rather than
truncating, so an unbounded string parameter is a 500 waiting for anybody
who sends `%00`. `api/params.py` exists for that, and the routes added
after it are the ones that have to remember to reach for it: `/prompts`
took three bare strings and answered `?service=%00` with a server error.

**Nothing here imports a route at module scope, and that is load-bearing.**
`api.routes.prompts` cannot be imported without `api/routes/__init__.py`,
which imports every router, which imports `api.dependencies` - and that
builds the engine and hands `StatusService` a sessionmaker at import time.
Both are `lru_cache`d and the sessionmaker is captured, so an import during
COLLECTION binds them to the placeholder DATABASE_URL in tests/conftest.py,
before the container fixture can clear either. A deselected test is still
collected, so a module-scope import here failed 183 integration tests in a
run that never executed this file. The import goes inside the test, where
`-m integration` never reaches it.
"""

from __future__ import annotations

import inspect
from importlib import import_module

import pytest

#: The routes whose free-text parameters this checks, by `module.function`
#: under `api.routes`, and the parameters of each. Only the ones a caller
#: types into: an enumerated `Literal` refuses anything off its list
#: already, and an id is bounded by `RowId`.
ROUTES = {"prompts.read": ("service", "version", "name")}

CASES = [(route, name) for route, names in ROUTES.items() for name in names]


def bounds(route: str, name: str) -> dict[str, object]:
    """The constraints one parameter of one route handler was declared with.

    Read out of `metadata`, which is where a `Query` keeps them: pydantic
    holds each as its own annotated-types object rather than as an
    attribute, and reading by attribute name survives it renaming the
    classes.
    """
    module, _, function = route.partition(".")
    handler = getattr(import_module(f"api.routes.{module}"), function)
    default = inspect.signature(handler).parameters[name].default
    found: dict[str, object] = {}
    for one in getattr(default, "metadata", ()):
        for attribute in ("max_length", "pattern"):
            value = getattr(one, attribute, None)
            if value is not None:
                found[attribute] = value
    return found


@pytest.mark.parametrize(("route", "name"), CASES, ids=[one[1] for one in CASES])
def test_a_free_text_parameter_has_a_ceiling(route, name) -> None:
    """A value past it is one no row could hold anyway."""
    assert bounds(route, name).get("max_length")


@pytest.mark.parametrize(("route", "name"), CASES, ids=[one[1] for one in CASES])
def test_a_free_text_parameter_refuses_a_nul(route, name) -> None:
    """Refused at the edge, which is cheaper than teaching every query."""
    pattern = bounds(route, name).get("pattern")

    assert isinstance(pattern, str) and "\\x00" in pattern
