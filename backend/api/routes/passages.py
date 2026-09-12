"""Read routes over the passages the chunking service produced."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Query

from api.dependencies import passage_catalog
from api.errors import ApiError, ErrorBody
from preprocessing.chunking.models import PassageDetail, StoredPassage

router = APIRouter(tags=["passages"])

#: The columns `q` may look in. A Literal rather than a free string, so the
#: OpenAPI document lists them and the frontend's picker cannot drift from
#: what the backend will accept.
SearchField = Literal["text", "section", "both"]


@dataclass(frozen=True)
class PassagePage:
    """One page of passages, and the total behind it."""

    total: int
    passages: list[StoredPassage]


@router.get("/passages")
def passages(
    document: str | None = None,
    q: str | None = Query(default=None, max_length=200),
    field: SearchField = "text",
    block_type: str | None = Query(default=None, max_length=60),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PassagePage:
    """Lists stored passages."""
    total, rows = passage_catalog.page(document, limit, offset, q, block_type, field)
    return PassagePage(total=total, passages=rows)


# Ahead of /passages/{passage_id}: FastAPI matches in declaration order, and a
# literal path declared after a parameterised one is never reached.
@router.get("/passages/types")
def passage_types() -> list[str]:
    """Lists the block types the corpus actually contains."""
    return passage_catalog.block_types()


@router.get("/passages/{passage_id}", responses={404: {"model": ErrorBody}})
def passage(passage_id: int) -> PassageDetail:
    """Reads one passage in full, its cell grids and sentences included.

    The listing carries counts rather than the grids, because a page of fifty
    passages would otherwise ship every cell of every table in them.
    """
    found = passage_catalog.passage(passage_id)
    if found is None:
        raise ApiError(404, "unknown_passage", "no passage with that id")
    return found
