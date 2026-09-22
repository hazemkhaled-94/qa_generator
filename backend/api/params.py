"""The bounds every route puts on what a caller may send.

Declared once and annotated onto the parameters, so the refusal happens in
FastAPI's validation - which is already documented as 422 on every route -
rather than in psycopg, which has no status code and answers 500.

Both bounds here were found by `tests/contract/test_openapi_conformance.py`
generating calls from the published document. Neither is hypothetical: each
is a request an unauthenticated caller can send today that reaches
PostgreSQL and comes back as a server error.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Path, Query

#: The largest value a PostgreSQL `bigint` holds. Every id column in this
#: schema is one, so an id above this is not a row that is missing - it is a
#: value the column cannot be compared against, and the driver raises
#: `NumericValueOutOfRange` before any row is read.
#:
#: Unbounded, `GET /passages/9223372036854775808` answered 500. Bounded, it
#: is a 422 naming the parameter, which is the answer a caller can act on
#: and the one the document already promises.
BIGINT_MAX = 2**63 - 1

#: A row id, as a path parameter. `ge=1` because every id here is a
#: generated identity column and starts at 1, so 0 and negatives name
#: nothing either.
RowId = Annotated[int, Path(ge=1, le=BIGINT_MAX)]

#: The same id as a QUERY parameter, where it narrows a listing rather than
#: naming the row being fetched. Separate because FastAPI reads the location
#: off the annotation, and a `Path` here would make it a required segment.
TopicId = Annotated[int | None, Query(ge=1, le=BIGINT_MAX)]

#: Text a caller may search with. The length cap was already on each of
#: these; the pattern is the new half.
#:
#: PostgreSQL text cannot hold a NUL byte, and psycopg refuses one rather
#: than truncating - so `?q=%00` reached the driver and answered 500. A
#: caller has no legitimate use for one in a search term, and refusing it
#: at the edge is cheaper than teaching every query to strip it.
SEARCH_MAX = 200
_NO_NUL = r"^[^\x00]*$"

SearchText = Annotated[
    str | None,
    Query(max_length=SEARCH_MAX, pattern=_NO_NUL),
]

#: A document filter, which carried the same NUL defect as the search terms
#: because it was a bare string.
#:
#: Deliberately NOT held to `^[0-9a-f]{64}$`, though that is what a caller
#: sends. A digest that is not one currently selects no rows and answers an
#: empty page, which is a working call; refusing it would be a second
#: behaviour change riding along with a crash fix, and this one only has to
#: stop the crash.
DocumentFilter = Annotated[
    str | None,
    Query(max_length=SEARCH_MAX, pattern=_NO_NUL),
]

#: One page of rows. The ceiling was already here; `offset` had a floor and
#: no ceiling, so `?offset=9223372036854775808` went to the database as a
#: bigint and came back 500 exactly as an over-large id did.
#:
#: The ceiling is the same BIGINT_MAX rather than something smaller and
#: tidier: an offset past the end of a table is a legitimate request that
#: answers an empty page, and picking an arbitrary smaller bound would
#: refuse a caller paging through a corpus larger than the bound.
PAGE_MAX = 500

Limit = Annotated[int, Query(ge=1, le=PAGE_MAX)]
Offset = Annotated[int, Query(ge=0, le=BIGINT_MAX)]
