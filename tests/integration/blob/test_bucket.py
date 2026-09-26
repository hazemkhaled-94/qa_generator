"""Storing and reading objects, against a real S3 gateway."""

from __future__ import annotations

from urllib.parse import unquote

import pytest
from seed import digest

pytestmark = pytest.mark.integration

PDF = b"%PDF-1.7\nnot really a document\n"


@pytest.fixture
def documents(buckets):
    """The documents bucket, pointed at the gateway."""
    from blob_store.s3 import DocumentsBucket

    return DocumentsBucket()


def stored_metadata(bucket, key: str) -> dict[str, str]:
    """Reads back what `put` attached, as S3 holds it."""
    from blob_store.s3 import s3_client

    head = s3_client().head_object(Bucket=bucket.name, Key=key)
    return {name: unquote(value) for name, value in head["Metadata"].items()}


def test_an_object_comes_back_exactly_as_it_went_in(documents) -> None:
    """Byte for byte: the stored file is the source of truth."""
    key = documents.key_for(digest(), "application/pdf")

    documents.put(key, PDF, content_type="application/pdf")

    assert documents.get(key) == PDF


def test_an_empty_object_is_stored_and_read_back(documents) -> None:
    """Zero bytes is a value, not an absence."""
    key = documents.key_for(digest("b"), "application/pdf")

    documents.put(key, b"", content_type="application/pdf")

    assert documents.get(key) == b""


def test_the_metadata_survives_the_round_trip(documents) -> None:
    """Enough for the bucket alone to rebuild the database."""
    key = documents.key_for(digest(), "application/pdf")
    written = {
        "sha256": digest(),
        "original-filename": "report.pdf",
        "pipeline-version": "0.0.1",
    }

    documents.put(key, PDF, content_type="application/pdf", metadata=written)

    assert stored_metadata(documents, key) == written


def test_a_metadata_value_needing_escaping_survives(documents) -> None:
    """A filename is a person's, so it carries spaces and accents."""
    key = documents.key_for(digest(), "application/pdf")

    documents.put(
        key,
        PDF,
        content_type="application/pdf",
        metadata={"original-filename": "Jahresbericht 2026 – Übersicht.pdf"},
    )

    assert stored_metadata(documents, key) == {
        "original-filename": "Jahresbericht 2026 – Übersicht.pdf"
    }


@pytest.mark.parametrize("name", ["not a token", "a:b", "", "über"])
def test_a_metadata_key_s3_cannot_carry_is_refused(documents, name) -> None:
    """It travels as a header suffix, so it is a token or nothing."""
    key = documents.key_for(digest(), "application/pdf")

    with pytest.raises(ValueError, match="invalid S3 metadata key"):
        documents.put(key, PDF, content_type="application/pdf", metadata={name: "x"})


def test_finding_an_object_that_is_there_reads_it(documents) -> None:
    """`find` is `get` for a caller that expects to miss."""
    key = documents.key_for(digest(), "application/pdf")
    documents.put(key, PDF, content_type="application/pdf")

    assert documents.find(key) == PDF


def test_finding_an_object_that_is_not_there_answers_nothing(documents) -> None:
    """Holding nothing is an answer, not an error."""
    assert documents.find(documents.key_for(digest("z"), "application/pdf")) is None


def test_getting_an_object_that_is_not_there_raises(documents) -> None:
    """A caller that cannot proceed without it is told."""
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError):
        documents.get(documents.key_for(digest("z"), "application/pdf"))


def test_removing_an_object_takes_it_out_and_says_so(documents) -> None:
    """What deleting a document does to the file."""
    key = documents.key_for(digest(), "application/pdf")
    documents.put(key, PDF, content_type="application/pdf")

    assert documents.remove(key) is True
    assert documents.find(key) is None


def test_removing_something_already_gone_says_so(documents) -> None:
    """S3's own delete is silent about a key that was never there."""
    assert documents.remove(documents.key_for(digest("y"), "application/pdf")) is False


def test_rewriting_a_content_addressed_key_is_harmless(documents) -> None:
    """The same digest always holds the same bytes."""
    key = documents.key_for(digest(), "application/pdf")

    documents.put(key, PDF, content_type="application/pdf")
    documents.put(key, PDF, content_type="application/pdf")

    assert documents.get(key) == PDF
    assert documents.count() == 1


def test_the_bucket_counts_what_it_holds(documents) -> None:
    """What the status panel reports for the object store."""
    for seed in "abc":
        documents.put(
            documents.key_for(digest(seed), "application/pdf"),
            PDF,
            content_type="application/pdf",
        )

    assert documents.count() == 3


def test_the_count_is_answered_from_a_cache(documents) -> None:
    """The status panel polls it; the staleness is the documented cost."""
    assert documents.count() == 0

    documents.put(
        documents.key_for(digest(), "application/pdf"),
        PDF,
        content_type="application/pdf",
    )

    assert documents.count() == 0, "the cached total should still be the old one"


def test_the_fanout_key_is_where_the_object_lands(documents) -> None:
    """The directories are derived from the digest, never stored."""
    from blob_store.s3 import s3_client

    sha = digest("c")
    key = documents.key_for(sha, "application/pdf")
    documents.put(key, PDF, content_type="application/pdf")

    listed = s3_client().list_objects_v2(Bucket="documents")["Contents"]
    assert [one["Key"] for one in listed] == [f"{sha[:2]}/{sha[2:4]}/{sha}.pdf"]


def test_each_bucket_holds_its_own_objects(buckets) -> None:
    """A collection is the unit of storage policy; they are separate."""
    from blob_store.s3 import DocumentsBucket, ParsedBucket

    source, parsed = DocumentsBucket(), ParsedBucket()
    key = source.key_for(digest(), "application/pdf")
    source.put(key, PDF, content_type="application/pdf")

    assert source.find(key) == PDF
    assert parsed.find(key) is None


def test_the_gateway_lists_every_bucket_the_pipeline_uses(buckets) -> None:
    """What the health check reads."""
    from blob_store.s3 import bucket_names

    assert {"documents", "parsed", "export"} <= set(bucket_names())
