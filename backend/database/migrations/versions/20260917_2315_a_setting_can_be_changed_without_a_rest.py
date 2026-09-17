"""A setting can be changed without a restart.

Every tuning value was read once, out of the process environment, when a
service started. Changing one meant editing `configs/env/backend.env` and
recreating six containers, which is why nothing in the UI could configure
anything: the API cannot write that file and must not restart a worker.

`service_settings` is where a change goes instead. One row per setting, the
value as text exactly as the environment would hand it over, so the parsers
in `settings.env` read a stored setting and an environment one the same way
and a number is checked by the same reader either way.

A row is an override and never a default. The environment still supplies
every setting and a missing variable still stops a service at start-up
naming itself; deleting a row is what returns a setting to what the file
says. That is why there is no column for a default, and why the table is
empty on a deployment nobody has configured - an empty table means the
files are being obeyed exactly.

Names are unique across the table rather than per service. The `service`
column says which page configures a setting, and resolution ignores it: a
worker overlays every row, because extraction reads its own settings and the
platform's, and scoping the overlay by service is how it would come to miss
a change to LLM_MODEL.

There is no version column. What configuration is running is read off the
rows as a digest of them, because a counter kept here would fall when a row
was deleted and a version that falls describes an older configuration than
the one running.

Additive and empty, so the workers running against this database when it is
applied keep working: they read the environment until they are restarted, and
an empty table resolves to exactly that.

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
    # One row per setting. Without this, which value a stage reads would
    # depend on which row came back first.
    op.create_unique_constraint(
        "service_settings_name_unique", "service_settings", ["name"]
    )
    # A page asks for one service's settings and nothing else.
    op.create_index("ix_service_settings_service", "service_settings", ["service"])


def downgrade() -> None:
    """Drops the table, and with it every setting anybody changed.

    The settings themselves are not lost: the environment still carries all
    of them, and dropping the overrides returns every service to what
    `configs/env/backend.env` and `.env` say.
    """
    op.drop_index("ix_service_settings_service", table_name="service_settings")
    op.drop_constraint(
        "service_settings_name_unique", "service_settings", type_="unique"
    )
    op.drop_table("service_settings")
