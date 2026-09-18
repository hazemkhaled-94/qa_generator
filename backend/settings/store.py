"""The settings a deployment changed, and the source a stage reads.

`resolved` is the environment overlaid with every stored row. Resolution
ignores the service column: names are unique across the table, and a stage
reads its own settings and the platform's.

`version` names a configuration by its content, as a digest of the
overrides. It changes when anything changes, deletions included.

Importing this reaches the database.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import delete, select

from database.qa_generator.engine import sessions
from database.qa_generator.service_settings import ServiceSetting
from settings import catalog

log = logging.getLogger(__name__)

#: How much of the digest is kept.
_VERSION_CHARS = 12

#: What `version` answers when nothing is stored.
UNCHANGED = "environment"


def version(overrides: Mapping[str, str]) -> str:
    """Names one configuration by its content.

    Sorted before hashing, so the name does not depend on row order.
    """
    if not overrides:
        return UNCHANGED
    written = json.dumps(dict(sorted(overrides.items())), separators=(",", ":"))
    return hashlib.sha256(written.encode()).hexdigest()[:_VERSION_CHARS]


def resolved() -> dict[str, str]:
    """The environment overlaid with what a deployment changed.

    Built when called rather than held, so a process reaches the database
    only when it needs a setting.
    """
    return Settings().resolved()


def current() -> str:
    """Names the configuration running now."""
    return Settings().version()


def snapshot() -> tuple[dict[str, str], str]:
    """The source to read settings from, and the name of that configuration.

    Both from one read of the table, so a row cannot be stamped with a
    configuration it was not produced under.
    """
    overrides = Settings().overrides()
    return {**os.environ, **overrides}, version(overrides)


class Settings:
    """The stored overrides, read and written one setting at a time."""

    def __init__(self) -> None:
        """Binds to the session factory."""
        self._session = sessions()

    def overrides(self) -> dict[str, str]:
        """Every setting a deployment changed, by name."""
        with self._session() as session:
            return {
                row.name: row.value
                for row in session.scalars(select(ServiceSetting)).all()
            }

    def changed_at(self) -> dict[str, datetime]:
        """When each stored setting was last written, by name."""
        with self._session() as session:
            return {
                row.name: row.updated_at
                for row in session.scalars(select(ServiceSetting)).all()
            }

    def resolved(self) -> dict[str, str]:
        """The environment, overlaid with what a deployment changed.

        A copy, not a view: it must not change while a row is worked.
        """
        return {**os.environ, **self.overrides()}

    def version(self) -> str:
        """Names the configuration running now."""
        return version(self.overrides())

    def write(self, name: str, value: str) -> str:
        """Stores one setting, and returns the version it produced.

        Checks only what the catalogue says of the setting. What a value has
        to parse as is the stage's own `Settings.load`, which a caller runs
        first.

        Raises:
            KeyError: If nothing is configurable under that name.
            ValueError: If the deployment owns the setting, or the value is
                empty and its absence means nothing.
        """
        setting = catalog.writable(name)
        value = value.strip()
        if not value and not setting.optional:
            raise ValueError(
                f"{name} must carry a value. Empty is how a setting whose "
                f"absence means something is turned off, and {name} is not "
                f"one of those; delete the override to return it to what "
                f"configs/env/backend.env says."
            )

        with self._session.begin() as session:
            stored = session.scalar(
                select(ServiceSetting).where(ServiceSetting.name == name)
            )
            if stored is None:
                session.add(
                    ServiceSetting(service=setting.service, name=name, value=value)
                )
            else:
                stored.value = value
                # Rewritten in case the setting moved service.
                stored.service = setting.service

        log.info("%s set to %r", name, value)
        return self.version()

    def clear(self, name: str) -> bool:
        """Deletes one override, saying whether there was one.

        Returns the setting to whatever the environment says. There is no
        stored copy of a default to restore.

        Raises:
            KeyError: If nothing is configurable under that name.
            ValueError: If the deployment owns the setting.
        """
        catalog.writable(name)

        with self._session.begin() as session:
            removed = session.execute(
                delete(ServiceSetting).where(ServiceSetting.name == name)
            ).rowcount

        if not removed:
            return False
        log.info("%s returned to the environment", name)
        return True
