"""The service_settings table."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Identity, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from database.qa_generator.base import Base


class ServiceSetting(Base):
    """One setting a deployment changed, overriding what the environment says.

    A row is an override and never a default: `configs/env/backend.env` and
    `.env` still supply every setting, and a missing variable still stops a
    service at start-up naming itself. Deleting a row is what returns a
    setting to what the file says, which is why there is no column for a
    default.

    The value is text, as the environment hands it over, so the parsers in
    `settings.env` read a stored setting and an environment one the same way.
    A number is checked by the same reader either way rather than by a second
    copy of the rule that agrees with it today.

    An empty value is how a setting is overridden to absent, and is stored
    only for the settings whose absence means something. `optional` reads
    empty as absent, so a file naming a verifier model or a reasoning effort
    can be overridden back to neither. For every other setting empty is what
    `required` refuses, and the store will not write one.

    There is no version column. What version the settings are at is read off
    the rows themselves, as a digest of them - see `settings.store.version` -
    because a counter kept here would fall when a row was deleted, and a
    version that falls describes an older configuration than the one running.
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
