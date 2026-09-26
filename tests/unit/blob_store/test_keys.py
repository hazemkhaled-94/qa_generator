"""The object key a stored document is found under.

Content-addressed, with a 2x2-hex fanout: the filer keys filemeta on
(dirhash, name), so one flat directory becomes a write hotspot.
"""

from __future__ import annotations

import pytest

from blob_store.s3.bucket import Bucket
from blob_store.s3.documents import DocumentsBucket

SHA = "abcd" + "e" * 60


def test_the_key_fans_out_on_the_first_two_byte_pairs() -> None:
    """Two levels of two hex characters, then the digest itself."""
    assert Bucket.fanout_key(SHA, "pdf") == f"ab/cd/{SHA}.pdf"


def test_the_fanout_matches_the_start_of_the_digest() -> None:
    """The directories are derived, never stored."""
    key = Bucket.fanout_key(SHA, "json")
    first, second, name = key.split("/")
    assert SHA.startswith(first + second)
    assert name == f"{SHA}.json"


def test_a_document_takes_the_extension_its_media_type_implies() -> None:
    """A second format needs a pipeline and nothing here."""
    assert DocumentsBucket.key_for(SHA, "application/pdf") == f"ab/cd/{SHA}.pdf"


def test_a_media_type_with_no_known_extension_is_refused() -> None:
    """A key nothing could find again is worse than an error."""
    with pytest.raises(ValueError, match="no file extension is known"):
        DocumentsBucket.key_for(SHA, "application/x-nothing-of-the-sort")


def test_a_bucket_that_does_not_name_itself_is_refused() -> None:
    """Every method addresses self.name, so a missing one fails inside boto3."""
    with pytest.raises(TypeError, match="does not declare name"):

        class Unnamed(Bucket):
            """A bucket with no name."""
