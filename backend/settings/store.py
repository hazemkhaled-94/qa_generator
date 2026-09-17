"""The settings a deployment changed, and the source a stage reads.

The environment is what a container starts from and the store is what a
person changed since. `resolved` is the two of them together, which is what
every `Settings.load` is handed: the environment overlaid with the stored
rows, so a setting nobody has touched reads exactly as the file says it.

Names are unique across the whole table, so resolving takes every row rather
than one service's. A stage reads its own settings and the platform's - a
worker calls a model and embeds with a tokenizer - and scoping the overlay to
one service is how extraction would come to miss a change to LLM_MODEL.

`version` is a digest of the overrides rather than a counter. A counter is
the obvious thing and it is wrong here: the largest revision a set of rows
carries falls when a row is deleted, so returning a setting to the file would
hand out a version describing an older configuration than the one running. A
digest changes when anything changes, deletions included, and two callers
computing it from the same rows agree without having to coordinate. It is an
identity rather than an order, which is what both readers of it want: a row
recording what produced it, and a write refusing to land on a change it never
saw.

Importing this reaches the database. Nothing that has to answer before a
connection exists may use it: the pool sizes and the log level are read from
the environment where they are used, which is why they are not configurable.
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

#: How much of the digest is kept. Long enough that two configurations will
#: not collide in one deployment's lifetime, short enough to read out loud
#: and to sit in a column beside a fact.
_VERSION_CHARS = 12

#: What `version` answers when nothing is stored, so the environment on its
#: own has a name a row can record.
UNCHANGED = "environment"


def version(overrides: Mapping[str, str]) -> str:
    """Names one configuration by its content.

    Sorted and JSON-encoded before hashing, so the same overrides give the
    same name whatever order the rows came back in.
    """
    if not overrides:
        return UNCHANGED
    written = json.dumps(dict(sorted(overrides.items())), separators=(",", ":"))
    return hashlib.sha256(written.encode()).hexdigest()[:_VERSION_CHARS]


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

        What every `Settings.load` is handed. A copy rather than a view: the
        result is read many times while a row is worked and must not change
        underneath it.
        """
        return {**os.environ, **self.overrides()}

    def version(self) -> str:
        """Names the configuration running now."""
        return version(self.overrides())

    def write(self, name: str, value: str) -> str:
        """Stores one setting, and returns the version it produced.

        Validates nothing about the value beyond what the catalogue says of
        the setting itself. What a value has to parse as is the stage's own
        `Settings.load`, which a caller runs over the resolved source before
        calling this: a value stored and then found unparseable would stop
        the stage rather than the request.

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
                # Written every time: a setting that moved service between
                # releases would otherwise keep answering to the old page.
                stored.service = setting.service

        log.info("%s set to %r", name, value)
        return self.version()

    def clear(self, name: str) -> bool:
        """Deletes one override, saying whether there was one.

        What returns a setting to whatever the environment says, which is the
        only way back: there is no stored copy of a default to restore.

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
