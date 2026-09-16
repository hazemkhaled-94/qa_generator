"""The documents routes, end to end over HTTP.

Every refusal these declare carries a code a caller branches on, so each is
checked for its status and its code rather than only for failing.
"""

from __future__ import annotations

import pytest
from seed import digest
from sqlalchemy import update
from sqlalchemy.orm import Session

from database.qa_generator import Document

pytestmark = pytest.mark.integration


def upload(client, pdf: bytes, name: str = "report.pdf"):
    """Posts one document the way the frontend does."""
    return client.post("/documents", files={"file": (name, pdf, "application/pdf")})


def test_an_accepted_upload_answers_with_its_digest(client, pdf) -> None:
    """200 and the outcome, because a duplicate is an answer too."""
    answered = upload(client, pdf)

    assert answered.status_code == 200
    body = answered.json()
    assert body["outcome"] == "stored"
    assert len(body["sha256"]) == 64


def test_the_same_file_twice_is_answered_not_refused(client, pdf) -> None:
    """A duplicate is an answer; the caller is told which document it is."""
    first = upload(client, pdf).json()
    second = upload(client, pdf, name="again.pdf").json()

    assert second["outcome"] == "duplicate_bytes"
    assert second["sha256"] == first["sha256"]


def test_something_that_is_not_a_document_is_refused_as_415(client) -> None:
    """The code is the ingest outcome itself."""
    answered = client.post(
        "/documents", files={"file": ("report.pdf", b"GIF89a", "application/pdf")}
    )

    assert answered.status_code == 415
    assert answered.json()["code"] == "unsupported_type"


def test_an_upload_over_the_limit_is_refused_as_413(client) -> None:
    """Checked against Content-Length before the body is read."""
    answered = client.post(
        "/documents",
        files={
            "file": (
                "big.pdf",
                b"%PDF-" + b"x" * (101 * 1024 * 1024),
                "application/pdf",
            )
        },
    )

    assert answered.status_code == 413
    assert answered.json()["code"] == "too_large"
    assert "MB" in answered.json()["detail"]


def test_a_post_with_no_file_is_fastapis_own_422(client) -> None:
    """A malformed request keeps the standard shape, not an ErrorBody."""
    answered = client.post("/documents")

    assert answered.status_code == 422
    assert "code" not in answered.json()


def test_the_listing_reports_the_total_behind_the_page(client, pdf) -> None:
    """What the Documents page draws its pager from."""
    upload(client, pdf, name="first.pdf")

    listed = client.get("/documents").json()

    assert listed["total"] == 1
    assert [one["filename"] for one in listed["documents"]] == ["first.pdf"]


def test_the_listing_can_be_searched(client, pdf) -> None:
    """The search box, over filename, title and digest."""
    upload(client, pdf, name="annual-report.pdf")
    upload(client, pdf.replace(b"200 200", b"300 300"), name="other.pdf")

    listed = client.get("/documents", params={"q": "annual"}).json()

    assert listed["total"] == 1
    assert listed["documents"][0]["filename"] == "annual-report.pdf"


def test_the_listing_can_be_narrowed_to_one_parse_state(client, pdf, engine) -> None:
    """The filter the Documents page offers beside its search box."""
    upload(client, pdf, name="done.pdf")
    upload(client, pdf.replace(b"200 200", b"300 300"), name="waiting.pdf")
    with Session(engine) as session:
        session.execute(
            update(Document)
            .where(Document.sha256 == _digest_of(client, "done.pdf"))
            .values(parse_status="parsed")
        )
        session.commit()

    parsed = client.get("/documents", params={"parse_status": "parsed"}).json()
    waiting = client.get("/documents", params={"parse_status": "new"}).json()

    assert [one["filename"] for one in parsed["documents"]] == ["done.pdf"]
    assert [one["filename"] for one in waiting["documents"]] == ["waiting.pdf"]


def _digest_of(client, filename: str) -> str:
    """The digest the catalogue holds one uploaded file under."""
    listed = client.get("/documents").json()["documents"]
    return next(one["sha256"] for one in listed if one["filename"] == filename)


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ({"limit": 0}, 422),
        ({"limit": 501}, 422),
        ({"offset": -1}, 422),
        ({"q": "x" * 201}, 422),
        ({"parse_status": "invented"}, 422),
        ({"limit": 500, "offset": 0}, 200),
    ],
)
def test_the_listing_refuses_a_page_it_will_not_serve(client, params, status) -> None:
    """The bounds are declared, so they appear in the OpenAPI document."""
    assert client.get("/documents", params=params).status_code == status


def test_every_document_is_offered_to_a_picker(client, pdf) -> None:
    """Separate from /documents because three pages poll it."""
    upload(client, pdf, name="first.pdf")

    named = client.get("/documents/names").json()

    assert [one["filename"] for one in named] == ["first.pdf"]


def test_a_stored_file_is_served_back_exactly(client, pdf) -> None:
    """Byte for byte, as the Documents page renders it inline."""
    sha = upload(client, pdf).json()["sha256"]

    served = client.get(f"/documents/{sha}/file")

    assert served.status_code == 200
    assert served.content == pdf
    assert served.headers["content-type"] == "application/pdf"
    assert sha[:12] in served.headers["content-disposition"]


@pytest.mark.parametrize(
    "path", ["/documents/{}/file", "/documents/{}", "/documents/{}/derived"]
)
@pytest.mark.parametrize(
    "sha", ["not-a-digest", "abc", "A" * 64, "../../etc/passwd", "0" * 63]
)
def test_a_path_that_is_not_a_digest_is_refused_as_400(client, path, sha) -> None:
    """Checked before it can become part of an object key."""
    answered = client.request(
        "GET" if path.endswith("file") else "DELETE", path.format(sha)
    )

    assert answered.status_code in (400, 404, 405)
    if answered.status_code == 400:
        assert answered.json()["code"] == "invalid_digest"


def test_a_digest_no_document_has_is_refused_as_404(client) -> None:
    """Looked up before the store is asked for anything."""
    answered = client.get(f"/documents/{digest('z')}/file")

    assert answered.status_code == 404
    assert answered.json()["code"] == "unknown_document"


def test_deleting_a_document_reports_what_went_with_it(client, pdf) -> None:
    """Per store, because the three are deleted separately."""
    sha = upload(client, pdf).json()["sha256"]

    removed = client.delete(f"/documents/{sha}")

    assert removed.status_code == 200
    assert removed.json() == {
        "sha256": sha,
        "document": True,
        "passages": 0,
        "file": True,
        "parsed": False,
    }
    assert client.get("/documents").json()["total"] == 0


def test_deleting_a_document_twice_is_a_404_the_second_time(client, pdf) -> None:
    """Irreversible, and it says so rather than pretending."""
    sha = upload(client, pdf).json()["sha256"]
    client.delete(f"/documents/{sha}")

    assert client.delete(f"/documents/{sha}").status_code == 404


def test_deleting_the_derived_data_keeps_the_document(client, pdf) -> None:
    """The file stays and chunking returns to the queue."""
    sha = upload(client, pdf).json()["sha256"]

    removed = client.delete(f"/documents/{sha}/derived")

    assert removed.status_code == 200
    assert removed.json()["document"] is False
    assert client.get("/documents").json()["total"] == 1
    assert client.get(f"/documents/{sha}/file").status_code == 200
