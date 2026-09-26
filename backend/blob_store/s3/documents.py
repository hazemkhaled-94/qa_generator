"""The documents bucket."""

from __future__ import annotations

import mimetypes
from typing import ClassVar

from blob_store.s3.bucket import Bucket


class DocumentsBucket(Bucket):
    """Source documents, exactly as uploaded, and never expiring.

    The system's source of truth. Objects are content-addressed, so a key
    always holds the same bytes and rewriting one is harmless.
    """

    name: ClassVar[str] = "documents"

    @classmethod
    def key_for(cls, sha256: str, media_type: str) -> str:
        """Builds the key a document is stored under.

        The extension comes from the media type, so a second format needs a
        pipeline and nothing here.

        Raises:
            ValueError: If no file extension is known for the media type,
                which would produce a key nothing could find again.
        """
        extension = mimetypes.guess_extension(media_type)
        if extension is None:
            raise ValueError(f"no file extension is known for {media_type}")
        return cls.fanout_key(sha256, extension.lstrip("."))
