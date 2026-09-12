"""The shape every refusal from this API takes.

Each deliberate refusal answers with a stable `code` beside its message, so a
caller can branch on the code rather than on English. The codes are the
ingest outcomes and validation failures named elsewhere in the project.

Two things are left alone: FastAPI's own 422 for a malformed query keeps its
standard shape, and a stage's own failures are recorded against the row and
read back through its status rather than answered here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class ErrorBody:
    """What a refused request is answered with."""

    code: str
    detail: str


class ApiError(Exception):
    """A refusal answered as an :class:`ErrorBody` at a chosen status.

    Raised instead of `HTTPException` so the body carries a code.
    """

    def __init__(self, status: int, code: str, detail: str) -> None:
        """Initialises the refusal."""
        super().__init__(detail)
        self.status = status
        self.body = ErrorBody(code=code, detail=detail)


def install(app: FastAPI) -> None:
    """Teaches an application to answer :class:`ApiError` with its body."""

    @app.exception_handler(ApiError)
    async def _handle(request: Request, exc: ApiError) -> JSONResponse:
        """Renders one refusal."""
        return JSONResponse(status_code=exc.status, content=asdict(exc.body))
