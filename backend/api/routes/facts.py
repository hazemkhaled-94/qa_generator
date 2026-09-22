"""Read routes over the facts the extraction service produced."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter

from api.dependencies import fact_catalog
from api.params import DocumentFilter, Limit, Offset, RowId, SearchText
from extraction.models import FactQuality, StoredFact

router = APIRouter(tags=["facts"])

#: The columns `q` may look in, the methods `method` accepts and the kinds
#: `kind` accepts. Literals rather than free strings, so the OpenAPI document
#: lists them and the frontend's pickers cannot drift from what the backend
#: will accept.
SearchField = Literal["statement", "evidence", "both"]
Method = Literal["llm", "deterministic"]
Kind = Literal["atomic", "summary", "outline", "bridge"]


@dataclass(frozen=True)
class FactPage:
    """One page of facts, and the total behind it."""

    total: int
    facts: list[StoredFact]


@dataclass(frozen=True)
class FactSources:
    """The passages one fact rests on, as their ids alone."""

    fact: int
    passages: list[int]


@router.get("/facts")
def facts(
    document: DocumentFilter = None,
    q: SearchText = None,
    field: SearchField = "both",
    method: Method | None = None,
    kind: Kind | None = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> FactPage:
    """Lists stored facts, including those that failed a check."""
    total, rows = fact_catalog.page(document, limit, offset, q, method, field, kind)
    return FactPage(total=total, facts=rows)


@router.get("/facts/quality")
def quality(
    document: DocumentFilter = None,
    q: SearchText = None,
    field: SearchField = "both",
    method: Method | None = None,
    kind: Kind | None = None,
) -> FactQuality:
    """Reports how well extraction is doing, under the same filter.

    The numbers that say whether the facts are facts: how many passed every
    check, why the rest did not, and how a statement compares to the sentences
    it cites in length and in how many claims it carries.
    """
    return fact_catalog.quality(document, q, method, field, kind)


@router.get("/facts/{fact_id}/passages")
def passages(fact_id: RowId) -> FactSources:
    """Lists the passages one fact rests on, in the order the model saw them.

    The same passages /facts carries on every row, addressable on their own.
    """
    return FactSources(fact=fact_id, passages=fact_catalog.passages_of(fact_id))
