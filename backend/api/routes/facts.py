"""Read routes over the facts the extraction service produced."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Query

from api.dependencies import fact_catalog
from extraction.models import FactQuality, StoredFact

router = APIRouter(tags=["facts"])

#: The columns `q` may look in, and the methods `method` accepts. Literals
#: rather than free strings, so the OpenAPI document lists them and the
#: frontend's pickers cannot drift from what the backend will accept.
SearchField = Literal["statement", "evidence", "both"]
Method = Literal["llm", "deterministic"]


@dataclass(frozen=True)
class FactPage:
    """One page of facts, and the total behind it."""

    total: int
    facts: list[StoredFact]


@router.get("/facts")
def facts(
    document: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    field: SearchField = "both",
    method: Method | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> FactPage:
    """Lists stored facts, including those that failed a check."""
    total, rows = fact_catalog.page(document, limit, offset, q, method, field)
    return FactPage(total=total, facts=rows)


@router.get("/facts/quality")
def quality(
    document: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    field: SearchField = "both",
    method: Method | None = None,
) -> FactQuality:
    """Reports how well extraction is doing, under the same filter.

    The numbers that say whether the facts are facts: how many passed every
    check, why the rest did not, and how a statement compares to the sentences
    it cites in length and in how many claims it carries.
    """
    return fact_catalog.quality(document, q, method, field)
