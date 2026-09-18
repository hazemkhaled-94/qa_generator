"""A setting can be changed without a restart.

Tuning values were read once, out of the process environment, when a service
started. `service_settings` is where a change goes instead: one row per
setting, the value as text exactly as the environment would hand it over, so
`settings.env` reads a stored setting and an environment one with the same
reader.

A row is an override and never a default. The environment still supplies
every setting, and deleting a row returns one to what the file says. An
empty table means the files are being obeyed exactly.

Names are unique across the table rather than per service. The `service`
column says which page configures a setting; resolution ignores it, because
a stage reads its own settings and the platform's.

There is no version column: what configuration is running is a digest of the
rows, in settings.store.

Additive and empty, so the workers running against this database when it is
applied keep reading the environment until they are restarted.

Revision: b8f24e07d3a1
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b8f24e07d3a1"
down_revision: str | None = "7a3e15c9b204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Creates the table a changed setting is stored in."""
    op.create_table(
        "service_settings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column(
            "service",
            sa.Text(),
            nullable=False,
            comment="Which service configures it: ingestion, parsing, chunking, "
            "extraction, topics, questions or platform. Denormalised from "
            "settings.catalog so one service's settings are one query.",
        ),
        sa.Column(
            "name",
            sa.Text(),
            nullable=False,
            comment="The variable, spelled as the environment spells it. Unique: "
            "two rows for one setting would make which value wins depend on row "
            "order.",
        ),
        sa.Column(
            "value",
            sa.Text(),
            nullable=False,
            comment="The value, as text, exactly as the environment would hand it "
            "over. Empty only for a setting whose absence means something, which "
            "is how one is overridden to absent; the store refuses an empty value "
            "for any other.",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            comment="When this setting was last written.",
        ),
        comment="Settings changed through the API, the CLI or a page, one row per "
        "setting. An override over what configs/env/backend.env and .env say, "
        "never a default: deleting a row returns the setting to the file. Workers "
        "read this when they claim a row, so a change reaches them without a "
        "restart.",
    )
    # One row per setting, or row order would decide the value.
    op.create_unique_constraint(
        "service_settings_name_unique", "service_settings", ["name"]
    )
    # A page asks for one service's settings and nothing else.
    op.create_index("ix_service_settings_service", "service_settings", ["service"])


def downgrade() -> None:
    """Drops the table, and with it every setting anybody changed.

    No setting is lost: the environment carries all of them, so every
    service returns to what the files say.
    """
    op.drop_index("ix_service_settings_service", table_name="service_settings")
    op.drop_constraint(
        "service_settings_name_unique", "service_settings", type_="unique"
    )
    op.drop_table("service_settings")
