"""The health panel, against the two stores it reports on.

Never raises: a component that cannot be reached is reported as unhealthy
with the reason, because a status page that 500s says nothing at all.
"""

from __future__ import annotations

import pytest

from api.status import Counter, StatusService

pytestmark = pytest.mark.integration


@pytest.fixture
def service(database, buckets):
    """The service with one counter, as the composition root builds it."""
    from ingestion.repository import DocumentRepository

    return StatusService(
        {"ingestion": Counter("Documents held.", DocumentRepository().counts)}
    )


def test_a_healthy_backend_reports_every_component(service) -> None:
    """The database, the object store and each service's own numbers."""
    snapshot = service.snapshot()

    assert set(snapshot) == {"database", "object_store", "ingestion"}
    assert all(component.ok for component in snapshot.values()), snapshot


def test_the_database_reports_the_schema_as_applied(service) -> None:
    """Every model's table is compared against the live ones."""
    assert service.snapshot()["database"].detail == "Connected, schema applied."


def test_the_object_store_names_and_measures_its_buckets(service) -> None:
    """The count per bucket is what the panel draws."""
    store = service.snapshot()["object_store"]

    assert store.ok
    assert set(store.metrics) == {"documents", "parsed", "export", "archive"}
    assert all(count == 0 for count in store.metrics.values()), store.metrics


def test_a_stored_object_shows_up_in_the_count(service, buckets) -> None:
    """What a person watches after an upload."""
    from seed import digest

    from blob_store.seaweedfs import DocumentsBucket

    bucket = DocumentsBucket()
    bucket.put(
        bucket.key_for(digest(), "application/pdf"),
        b"%PDF-1.7\n",
        content_type="application/pdf",
    )

    assert service.snapshot()["object_store"].metrics["documents"] == 1


def test_a_service_counter_is_reported_with_its_explanation(service) -> None:
    """The panel renders whatever is there without knowing the names."""
    ingestion = service.snapshot()["ingestion"]

    assert ingestion.ok
    assert ingestion.detail == "Documents held."
    assert isinstance(ingestion.metrics, dict)


def test_a_counter_that_raises_is_reported_rather_than_thrown(
    database, buckets
) -> None:
    """One broken query must not take the whole panel down."""

    def broken() -> dict[str, int]:
        """Fails the way a dropped table would."""
        raise RuntimeError("relation does not exist")

    snapshot = StatusService(
        {"ingestion": Counter("Documents held.", broken)}
    ).snapshot()

    assert snapshot["ingestion"].ok is False
    assert "relation does not exist" in snapshot["ingestion"].detail
    assert snapshot["database"].ok is True, "the rest of the panel still reports"


def test_an_unreachable_object_store_is_reported_rather_than_thrown(
    service, monkeypatch
) -> None:
    """The database half of the panel still answers."""
    monkeypatch.setenv("S3_ENDPOINT", "http://127.0.0.1:1")
    from blob_store.seaweedfs import client

    client.s3_client.cache_clear()

    snapshot = service.snapshot()

    assert snapshot["object_store"].ok is False
    assert "Unreachable" in snapshot["object_store"].detail
    assert snapshot["database"].ok is True
    client.s3_client.cache_clear()


def test_a_missing_bucket_is_named(service, buckets) -> None:
    """A bucket nobody created is a stage that will fail on its first write."""
    from blob_store.seaweedfs import s3_client

    s3_client().delete_bucket(Bucket="export")

    store = service.snapshot()["object_store"]

    assert store.ok is False
    assert "export" in store.detail
