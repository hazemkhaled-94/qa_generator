"""The shape of the migration history.

Read through alembic's own script directory, so this sees what `alembic
upgrade` would. No database: whether the schema the revisions build matches
the models is a question for a live one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "backend/database/migrations/versions"

_REVISION = re.compile(r"^revision: str = [\"']([^\"']+)[\"']", re.MULTILINE)


@pytest.fixture(scope="module")
def scripts() -> ScriptDirectory:
    """The revisions, as alembic reads them."""
    return ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))


def test_there_is_exactly_one_head(scripts: ScriptDirectory) -> None:
    """Two heads mean two people wrote a revision against the same parent."""
    heads = scripts.get_heads()

    assert len(heads) == 1, f"{len(heads)} heads: {heads}. Merge them."


def test_there_is_exactly_one_base(scripts: ScriptDirectory) -> None:
    """One history, starting once."""
    bases = scripts.get_bases()

    assert len(bases) == 1, f"{len(bases)} bases: {bases}"


def test_every_revision_file_is_reachable_from_the_head(
    scripts: ScriptDirectory,
) -> None:
    """A revision off the chain is one `upgrade head` never runs."""
    reachable = {revision.revision for revision in scripts.walk_revisions()}
    on_disk = {
        match.group(1)
        for path in VERSIONS.glob("*.py")
        if (match := _REVISION.search(path.read_text()))
    }

    assert on_disk, f"no revisions found in {VERSIONS}"
    assert on_disk == reachable, f"off the chain: {sorted(on_disk - reachable)}"


def test_every_parent_named_exists(scripts: ScriptDirectory) -> None:
    """A down_revision pointing at nothing breaks the whole upgrade."""
    reachable = {revision.revision for revision in scripts.walk_revisions()}

    for revision in scripts.walk_revisions():
        named = revision.down_revision
        parents = (named,) if isinstance(named, str) else tuple(named or ())
        missing = set(parents) - reachable
        assert not missing, f"{revision.revision} names {missing}, which is not here"


def test_every_revision_can_be_taken_back_out(scripts: ScriptDirectory) -> None:
    """An upgrade with no downgrade is a one-way door."""
    for revision in scripts.walk_revisions():
        source = Path(revision.path).read_text()
        assert "def downgrade()" in source, revision.path
        body = source.split("def downgrade()", 1)[1]
        assert "pass" not in body.split("\n")[1:3], (
            f"{Path(revision.path).name} has an empty downgrade"
        )
