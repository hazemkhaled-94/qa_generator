"""Reading the expiry off a token the provider is about to reject.

`AZURE_OPENAI_AD_TOKEN` is the one Entra ID credential that goes stale on
its own: it is minted by hand and lasts about an hour. An expired one is
taken in preference to a working credential, so every call is refused with
"missing, invalid, audience is incorrect, or have expired" — four causes in
one sentence, and which of them it was is the whole question.

`make doctor` answers it by reading the `exp` claim before it calls
anything. The signature is not verified and must not be: this is reading a
date out of a string, not authenticating anybody.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import pytest

from llm.check import expired


def token(claims: dict | None = None, *, body: bytes | None = None) -> str:
    """A JWT-shaped string carrying `claims`. Never signed; nothing reads that."""
    raw = body if body is not None else json.dumps(claims).encode()
    payload = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"header.{payload}.signature"


def test_an_expired_token_reports_when_it_expired() -> None:
    """The date is what makes the message actionable rather than a guess."""
    when = datetime.now(UTC) - timedelta(days=4)
    assert expired(token({"exp": when.timestamp()})) is not None


def test_a_live_token_is_not_reported() -> None:
    """A credential that still works is not a finding."""
    when = datetime.now(UTC) + timedelta(hours=1)
    assert expired(token({"exp": when.timestamp()})) is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("unset", ""),
        ("not a jwt", "garbage"),
        ("two segments", "header.payload"),
        ("payload is not base64", "header.!!!.signature"),
        ("payload is not json", token(body=b"not json")),
        ("no exp claim", token({})),
        ("exp is not a number", token({"exp": "soon"})),
    ],
)
def test_anything_unreadable_is_not_a_finding(name: str, value: str) -> None:
    """A token this cannot read is left to the provider to refuse.

    The check exists to explain one specific failure. Guessing at a string
    it does not understand would turn a working deployment into a refusal
    from the doctor, which is the opposite of the point.
    """
    assert expired(value) is None, name
