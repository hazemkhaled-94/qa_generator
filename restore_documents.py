"""Restore documents from the archive bucket and archived_rows table.

Run after an accidental `make wipe`, before `make archive-purge`:

    make LOADENV source && PYTHONPATH=backend poetry run python restore_documents.py

Or let the Makefile load the environment for you:

    set -a && . ./configs/env/backend.env && \
      { [ ! -f ./configs/env/provider.env ] || . ./configs/env/provider.env; } && \
      . ./.env && set +a && \
      PYTHONPATH=backend poetry run python restore_documents.py
"""

from __future__ import annotations

import mimetypes
import os
import sys

import boto3
import botocore.exceptions
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

DATABASE_URL = os.environ["DATABASE_URL"].replace(
    "postgresql+psycopg://", "postgresql://", 1
)
S3_ENDPOINT = os.environ.get("S3_ENDPOINT") or None
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_REGION = os.environ.get("S3_REGION", "local")
S3_STYLE = os.environ.get("S3_ADDRESSING_STYLE", "path")


def fanout_key(sha256: str, ext: str) -> str:
    """The key an object is stored under, fanned out by its first two bytes.

    The same shape `blob_store` writes: a flat bucket of a hundred thousand
    keys is one a listing cannot page through usefully.
    """
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}.{ext}"


def doc_key(sha256: str, media_type: str) -> str:
    """Where one document's own file lives.

    Raises:
        ValueError: If the media type names no extension, which is a row
            the archive cannot be read back into.
    """
    ext = mimetypes.guess_extension(media_type)
    if ext is None:
        raise ValueError(f"unknown extension for {media_type!r}")
    return fanout_key(sha256, ext.lstrip("."))


def copy_back(
    s3: object, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str
) -> bool:
    """Copies one archived object back, and says whether it was there.

    Returns:
        Whether the source existed. A miss is not an error here: the parsed
        JSON is rebuildable and the caller decides which absences matter.

    Raises:
        botocore.exceptions.ClientError: On anything but a missing key.
    """
    try:
        s3.copy_object(
            Bucket=dst_bucket,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
        )
        return True
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def main() -> None:
    """Puts every archived document back where the pipeline reads it."""
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        config=boto3.session.Config(s3={"addressing_style": S3_STYLE}),
    )

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        doc_rows = conn.execute(
            "SELECT payload FROM archived_rows WHERE table_name = 'documents' ORDER BY archived_at"
        ).fetchall()

        if not doc_rows:
            print("No archived documents found. Nothing to restore.")
            sys.exit(0)

        print(f"Found {len(doc_rows)} archived document(s).\n")
        errors = 0

        for r in doc_rows:
            p = r["payload"]
            sha256 = p["sha256"]
            media_type = p["media_type"]
            title = (p.get("title") or sha256)[:70]
            print(f"  {title}")

            # Restore raw file
            dk = doc_key(sha256, media_type)
            ok = copy_back(s3, "archive", f"documents/{dk}", "documents", dk)
            print(f"    raw file  {'ok' if ok else 'NOT FOUND in archive'}")
            if not ok:
                errors += 1

            # Restore parsed JSON (optional — missing means re-parse)
            pk = fanout_key(sha256, "json")
            parsed_ok = copy_back(s3, "archive", f"parsed/{pk}", "parsed", pk)
            print(
                f"    parsed    {'ok' if parsed_ok else 'not in archive (will re-parse)'}"
            )

            # Restore DB row
            try:
                conn.execute(
                    """
                    INSERT INTO documents
                    SELECT * FROM jsonb_populate_record(null::documents, %s)
                    ON CONFLICT (sha256) DO NOTHING
                    """,
                    (Jsonb(p),),
                )
                print("    db row    ok")
            except Exception as exc:  # noqa: BLE001 - one bad row is not the rest
                print(f"    db row    FAILED: {exc}")
                errors += 1

        # Restore ingest_events so upload history reappears in the UI
        event_rows = conn.execute(
            "SELECT payload FROM archived_rows WHERE table_name = 'ingest_events' ORDER BY archived_at"
        ).fetchall()
        if event_rows:
            restored_events = 0
            for r in event_rows:
                try:
                    conn.execute(
                        """
                        INSERT INTO ingest_events
                        SELECT * FROM jsonb_populate_record(null::ingest_events, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (Jsonb(r["payload"]),),
                    )
                    restored_events += 1
                except Exception as exc:  # noqa: BLE001 - as above
                    print(f"  ingest_events insert failed: {exc}")
            conn.commit()
            print(f"\n  {restored_events} upload event(s) restored.")
        else:
            conn.commit()

    if errors:
        print(f"\nFinished with {errors} error(s). Check output above.")
        sys.exit(1)

    print("\nAll documents restored. Now re-run the pipeline from parse:")
    print("  make parse-start && make parse")
    print("  make chunk-start && make chunk")
    print("  make extract-start && make extract")
    print("  make topics-discover && make topics")
    print("  make questions-start && make questions")


if __name__ == "__main__":
    main()
