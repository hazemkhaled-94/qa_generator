"""The shape every refusal from this API takes.

Each deliberate refusal answers with a stable `code` beside its message, so a
caller can branch on the code rather than on English. The codes are the
ingest outcomes and validation failures named elsewhere in the project.

Two things are left alone: FastAPI's own 422 for a malformed query keeps its
standard shape, and a stage's own failures are recorded against the row and
read back through its status rather than answered here.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)


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
        """Renders one refusal, and records it.

        One line here rather than one at each `raise`: every deliberate
        refusal passes through, so the code a caller branched on is also
        the code a dashboard can count. At warning, not error - a 404 for
        an unknown id is the API working.
        """
        log.warning(
            "%s %s refused: %s",
            request.method,
            request.url.path,
            exc.body.code,
            extra={
                "http.request.method": request.method,
                "url.path": request.url.path,
                "http.response.status_code": exc.status,
                "error.code": exc.body.code,
            },
        )
        return JSONResponse(status_code=exc.status, content=asdict(exc.body))
