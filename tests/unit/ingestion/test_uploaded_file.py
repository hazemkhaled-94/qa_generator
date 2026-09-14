"""What is derived from an upload's bytes, and from nothing else.

The trust boundary: neither the declared content type nor the filename
extension is evidence of anything.
"""

from __future__ import annotations

import hashlib

from ingestion.models import DETECTABLE, UploadedFile

PDF = b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n"


def test_the_type_is_decided_by_the_leading_bytes() -> None:
    """A PDF is a PDF whatever it is called."""
    assert UploadedFile("report.txt", PDF).media_type == "application/pdf"
    assert UploadedFile("no-extension-at-all", PDF).media_type == "application/pdf"


def test_a_filename_claiming_pdf_proves_nothing() -> None:
    """The extension is not evidence."""
    assert UploadedFile("invoice.pdf", b"GIF89a...").media_type is None
    assert UploadedFile("invoice.pdf", b"<html></html>").media_type is None


def test_an_unrecognised_or_empty_upload_has_no_type() -> None:
    """Unrecognised is None, not a guess."""
    assert UploadedFile("empty.pdf", b"").media_type is None
    assert UploadedFile("short.pdf", b"%PD").media_type is None


def test_the_magic_prefix_must_lead() -> None:
    """A signature further in is not a signature."""
    assert UploadedFile("padded.pdf", b"\x00\x00%PDF-1.7").media_type is None


def test_every_detectable_type_is_one_the_allowlist_may_name() -> None:
    """ALLOWED_MIME_TYPES is checked against this at start-up."""
    assert DETECTABLE == frozenset({"application/pdf"})


def test_the_digest_is_of_the_raw_bytes() -> None:
    """The document's identity, and its object key."""
    assert UploadedFile("a.pdf", PDF).sha256 == hashlib.sha256(PDF).hexdigest()
    assert len(UploadedFile("a.pdf", PDF).sha256) == 64


def test_the_same_bytes_under_another_name_are_the_same_document() -> None:
    """The filename is not part of the identity."""
    assert UploadedFile("a.pdf", PDF).sha256 == UploadedFile("b.pdf", PDF).sha256


def test_a_changed_byte_is_a_different_document() -> None:
    """Duplicate detection is byte-exact."""
    assert UploadedFile("a.pdf", PDF).sha256 != UploadedFile("a.pdf", PDF + b" ").sha256


def test_the_size_is_the_length_of_the_bytes() -> None:
    """What the size limit is checked against."""
    assert UploadedFile("a.pdf", PDF).size_bytes == len(PDF)
    assert UploadedFile("a.pdf", b"").size_bytes == 0


def test_nothing_derived_can_disagree_with_the_content() -> None:
    """Size, digest and type are read off the bytes, never stored beside them."""
    upload = UploadedFile("a.pdf", PDF)
    assert set(upload.__dataclass_fields__) == {"filename", "data"}
