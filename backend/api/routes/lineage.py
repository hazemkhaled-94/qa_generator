"""The route that answers how one artefact was produced."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter

from api import lineage
from api.errors import ApiError, ErrorBody

router = APIRouter(prefix="/lineage", tags=["lineage"])

#: What may be traced. A literal rather than a free string, so the OpenAPI
#: document lists them and a caller cannot ask for a kind nothing holds.
Kind = Literal["document", "passage", "fact", "topic", "question"]


@router.get("/{kind}/{artifact_id}", responses={404: {"model": ErrorBody}})
def of(kind: Kind, artifact_id: str) -> lineage.Lineage:
    """Every artefact one artefact was produced from, in pipeline order."""
    found = lineage.of(kind, artifact_id)
    if found is None:
        raise ApiError(404, f"unknown_{kind}", f"no {kind} with that id")
    return found
