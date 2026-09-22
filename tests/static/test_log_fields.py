"""Every field a log line is labelled with is one the shipper declares.

The other direction from `test_dashboards.py`, which checks the fields a
panel reads. This checks the fields the code writes, and it is the one that
was missing: `document.media_type` and `topic_fit.language` were bound for
weeks and named nowhere in `setup.template.append_fields`, so Elasticsearch
would have mapped each as `text` on the first line carrying it - analysed,
no doc values, and invisible to anything that groups.

Read as files, so this needs no container and no shipped line.

Only dict literals are seen. A field bound through a variable - `about` in
`llm/client.py` - is invisible here, the same blind spot
`test_settings_documented.py` has.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FILEBEAT = ROOT / "configs" / "filebeat" / "filebeat.yml"

#: Where the code that writes a log line lives.
SOURCES = ("backend", "frontend", "telemetry", "orchestration", "review", "evaluation")

#: ECS names the code binds that Filebeat's own template already maps, so
#: they are correct without being declared beside the project's own.
ECS = frozenset(
    {
        "error.code",
        "url.path",
        "http.request.method",
        "http.response.status_code",
    }
)


def declared() -> set[str]:
    """Every field name the shipper's template maps, ECS included."""
    shipper = yaml.safe_load(FILEBEAT.read_text())
    appended = {one["name"] for one in shipper["setup.template.append_fields"]}
    return appended | ECS


def _labelling(call: ast.Call):
    """The expressions one call labels its lines with, if it labels any."""
    name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
    if name == "bind" and call.args:
        yield call.args[0]
    if name == "working" and len(call.args) > 2:
        yield call.args[2]
    for keyword in call.keywords:
        if keyword.arg == "extra":
            yield keyword.value


def bound() -> dict[str, str]:
    """Every field name the code labels a line with, and where."""
    found: dict[str, str] = {}
    for source in SOURCES:
        for path in (ROOT / source).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for labelled in _labelling(node):
                    for inner in ast.walk(labelled):
                        if not isinstance(inner, ast.Dict):
                            continue
                        for key in inner.keys:
                            if isinstance(key, ast.Constant) and isinstance(
                                key.value, str
                            ):
                                found.setdefault(key.value, str(path.relative_to(ROOT)))
    return found


def test_the_scan_finds_the_fields_it_is_meant_to_guard() -> None:
    """A move of the sources would otherwise pass silently."""
    assert "stage" in bound(), "no bound field was read"
    assert len(declared()) > len(ECS), "no appended field was read"


def test_every_field_a_line_is_labelled_with_is_declared() -> None:
    """An undeclared field arrives as `text`, and nothing can group on it."""
    known = declared()
    missing = {name: where for name, where in bound().items() if name not in known}

    assert not missing, (
        "bound and declared nowhere: "
        + ", ".join(f"{name} ({where})" for name, where in sorted(missing.items()))
        + ". Add it to setup.template.append_fields in "
        "configs/filebeat/filebeat.yml, or Elasticsearch maps it as `text`."
    )
