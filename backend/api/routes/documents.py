"""Routes served by the ingestion service."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Query, Request, Response, UploadFile

from api.dependencies import ingest_service, ingesting, removal_service
from api.errors import ApiError, ErrorBody
from database.qa_generator import Outcome
from ingestion.models import (
    DocumentName,
    IngestResult,
    Removal,
    StoredDocument,
    UploadedFile,
)

#: Declared on the routes that can raise them, so a refusal appears in the
#: OpenAPI document.
_REFUSALS = {code: {"model": ErrorBody} for code in (400, 404, 413, 415)}

#: Ingest outcomes that are refusals. Anything absent answers 200; a
#: duplicate is an answer, not an error.
_ERROR_STATUS: dict[str, int] = {
    Outcome.TOO_LARGE: 413,
    Outcome.UNSUPPORTED_TYPE: 415,
}

#: What a digest looks like. Checked before a path parameter becomes part of
#: an object key.
_DIGEST = re.compile(r"[0-9a-f]{64}")

#: The states `parse_status` may be narrowed to. A Literal rather than a free
#: string, so the OpenAPI document lists them and the frontend's picker cannot
#: drift from what the backend accepts.
ParseStatus = Literal["new", "pending", "in_progress", "parsed", "failed"]

router = APIRouter(tags=["documents"])


@dataclass(frozen=True)
class DocumentPage:
    """One page of documents."""

    total: int
    documents: list[StoredDocument]


@router.get("/documents")
def list_documents(
    q: str | None = Query(default=None, max_length=200),
    parse_status: ParseStatus | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentPage:
    """Lists stored documents and where each has reached in the pipeline."""
    total, rows = ingest_service.documents(q, limit, offset, parse_status)
    return DocumentPage(total=total, documents=rows)


@router.get("/documents/names")
def document_names() -> list[DocumentName]:
    """Lists every document by name, for a picker to offer.

    Separate from `/documents` because the three pages that draw that picker
    need no passage counts, and aggregating them on every poll is the most
    expensive query the frontend makes.
    """
    return ingest_service.document_names()


@router.get(
    "/documents/{sha256}/file",
    responses={code: _REFUSALS[code] for code in (400, 404)},
)
def document_file(sha256: str) -> Response:
    """Serves one stored document, exactly as it was uploaded.

    Raises:
        ApiError: 400 `invalid_digest` if it is not a digest, 404
            `unknown_document` if no document has it.
    """
    if not _DIGEST.fullmatch(sha256):
        raise ApiError(400, "invalid_digest", "not a SHA-256 digest")

    found = ingest_service.stored_file(sha256)
    if found is None:
        raise ApiError(404, "unknown_document", "no document with that digest")

    data, media_type = found
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{sha256[:12]}.pdf"'},
    )


@router.delete(
    "/documents/{sha256}", responses={code: _REFUSALS[code] for code in (400, 404)}
)
def delete_document(sha256: str) -> Removal:
    """Deletes a document, its file, its converted form and its derived rows.

    Irreversible. The upload record in ingest_events survives.

    Raises:
        ApiError: 400 `invalid_digest` if it is not a digest, 404
            `unknown_document` if no document has it.
    """
    return _removed(sha256, removal_service.delete)


@router.delete(
    "/documents/{sha256}/derived",
    responses={code: _REFUSALS[code] for code in (400, 404)},
)
def delete_derived(sha256: str) -> Removal:
    """Deletes what the pipeline built from a document, keeping the document.

    The file and the converted form stay and chunking returns to `new`, so
    the next run rebuilds the passages and facts without re-parsing once
    somebody starts it.

    Raises:
        ApiError: 400 `invalid_digest` if it is not a digest, 404
            `unknown_document` if no document has it.
    """
    return _removed(sha256, removal_service.delete_derived)


def _removed(sha256: str, remove: Callable[[str], Removal | None]) -> Removal:
    """Validates a digest and reports what the given deletion removed.

    Raises:
        ApiError: 400 if the digest is not one, 404 if nothing has it.
    """
    if not _DIGEST.fullmatch(sha256):
        raise ApiError(400, "invalid_digest", "not a SHA-256 digest")
    removed = remove(sha256)
    if removed is None:
        raise ApiError(404, "unknown_document", "no document with that digest")
    return removed


@router.post("/documents", responses={code: _REFUSALS[code] for code in (413, 415)})
async def add_document(request: Request, file: UploadFile) -> IngestResult:
    """Accepts one uploaded document, synchronously.

    Everything after this is batch work the workers pick up by watching
    documents.parse_status.

    Raises:
        ApiError: 413 `too_large` if the file is over the size limit, 415
            `unsupported_type` if it is not a readable document of an
            accepted type. Each code is the ingest outcome itself.
    """
    filename = file.filename or "unnamed"
    # Per upload: ingestion has no worker to pick a changed limit up.
    service = ingesting()

    # Checked before the body is touched. Content-Length covers the whole
    # multipart body, so it is an upper bound rather than the file's size.
    declared = _declared_size(request)
    if declared is not None and declared > service.max_file_size_bytes:
        return _respond(service.refuse_oversized(filename, declared))

    # ponytail: a request declaring no length is still read in full and
    # measured afterwards. Every client that posts a file sets Content-Length;
    # the upgrade path for one that does not is a bounded chunked read, or a
    # body limit on a reverse proxy in front of this.
    upload = UploadedFile(filename=filename, data=await file.read())
    return _respond(service.ingest(upload))


def _declared_size(request: Request) -> int | None:
    """Reads the size the request declares for its body."""
    declared = request.headers.get("content-length")
    return int(declared) if declared is not None and declared.isdigit() else None


def _respond(result: IngestResult) -> IngestResult:
    """Passes an accepted outcome through, or raises the matching error.

    Raises:
        ApiError: If the outcome was a refusal. The outcome doubles as the
            code.
    """
    if result.outcome in _ERROR_STATUS:
        raise ApiError(
            _ERROR_STATUS[result.outcome], result.outcome, result.detail or ""
        )
    return result
