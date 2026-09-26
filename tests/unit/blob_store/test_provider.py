"""What the S3 client is built with, per provider.

Asserted on the arguments rather than on the built client, because the
logic under test is the mapping from env to those arguments: empty means
the SDK's own default, and absent is a missing configuration.
"""

from __future__ import annotations

import pytest

from blob_store.s3 import client as client_module

#: The names that must be set, whatever they are set to.
REQUIRED = ("S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY")

#: The names that need not be.
OPTIONAL = ("S3_REGION", "S3_ADDRESSING_STYLE")


@pytest.fixture
def built(monkeypatch):
    """Builds the client with a given environment, capturing the arguments."""
    captured: dict = {}

    def capture(_service: str, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(client_module.boto3, "client", capture)

    def build(**environment: str) -> dict:
        for name in REQUIRED + OPTIONAL:
            monkeypatch.delenv(name, raising=False)
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        client_module.s3_client.cache_clear()
        client_module.s3_client()
        return captured

    yield build
    client_module.s3_client.cache_clear()


SEAWEEDFS = {
    "S3_ENDPOINT": "http://seaweedfs-s3:8333",
    "S3_ACCESS_KEY": "key",
    "S3_SECRET_KEY": "secret",
}

AWS = {"S3_ENDPOINT": "", "S3_ACCESS_KEY": "", "S3_SECRET_KEY": ""}


def test_a_self_hosted_gateway_is_addressed_by_path_in_a_local_region(built) -> None:
    """The defaults are SeaweedFS's, so an existing .env keeps working."""
    made = built(**SEAWEEDFS)
    assert made["endpoint_url"] == "http://seaweedfs-s3:8333"
    assert made["aws_access_key_id"] == "key"
    assert made["region_name"] == "local"
    assert made["config"].s3["addressing_style"] == "path"


def test_an_empty_endpoint_lets_the_sdk_resolve_its_own(built) -> None:
    """AWS answers on an address boto3 derives from the region."""
    assert built(**AWS, S3_REGION="eu-central-1")["endpoint_url"] is None


def test_an_empty_credential_lets_the_sdk_find_one(built) -> None:
    """An instance role is a credential nothing in .env can name."""
    made = built(**AWS, S3_REGION="eu-central-1")
    assert made["aws_access_key_id"] is None
    assert made["aws_secret_access_key"] is None


def test_the_region_and_the_addressing_style_come_from_the_environment(
    built,
) -> None:
    """The two that differ between providers holding the same API."""
    made = built(**AWS, S3_REGION="eu-central-1", S3_ADDRESSING_STYLE="auto")
    assert made["region_name"] == "eu-central-1"
    assert made["config"].s3["addressing_style"] == "auto"


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_name_that_is_not_set_at_all_is_refused(built, missing: str) -> None:
    """Absent is a missing env file.

    Falling back to a default would address a real account with it.
    """
    supplied = {name: value for name, value in SEAWEEDFS.items() if name != missing}
    with pytest.raises(KeyError, match=missing):
        built(**supplied)
