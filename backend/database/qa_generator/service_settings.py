"""The service_settings table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Identity, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from database.qa_generator.base import Base


class ServiceSetting(Base):
    """One setting a deployment changed, overriding what the environment says.

    A row is an override and never a default: the files still supply every
    setting, and deleting a row returns one to what the file says. There is
    no column for a default.

    The value is text, as the environment hands it over, so `settings.env`
    reads a stored setting and an environment one with the same reader.

    An empty value overrides a setting to absent, and is stored only for the
    settings whose absence means something.

    There is no version column: what version the settings are at is a digest
    of the rows, in `settings.store.version`.
    """

    __tablename__ = "service_settings"
    __table_args__ = (
        UniqueConstraint("name", name="service_settings_name_unique"),
        {
            "comment": "Settings changed through the API, the CLI or a page, one row "
            "per setting. An override over what configs/env/backend.env and .env say, "
            "never a default: deleting a row returns the setting to the file. Workers "
            "read this when they claim a row, so a change reaches them without a "
            "restart."
        },
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    service: Mapped[str] = mapped_column(
        Text,
        index=True,
        comment="Which service configures it: ingestion, parsing, chunking, "
        "extraction, topics, questions or platform. Denormalised from "
        "settings.catalog so one service's settings are one query.",
    )
    name: Mapped[str] = mapped_column(
        Text,
        comment="The variable, spelled as the environment spells it. Unique: two "
        "rows for one setting would make which value wins depend on row order.",
    )
    value: Mapped[str] = mapped_column(
        Text,
        comment="The value, as text, exactly as the environment would hand it over. "
        "Empty only for a setting whose absence means something, which is how one is "
        "overridden to absent; the store refuses an empty value for any other.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="When this setting was last written.",
    )
