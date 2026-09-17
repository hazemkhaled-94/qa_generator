"""The provisioned dashboards, against what is provisioned to serve them.

A Grafana dashboard fails quietly. A panel naming a datasource that is not
there renders an error inside the panel and the dashboard still loads; a
panel grouping on a field Elasticsearch mapped as `text` renders empty,
because `text` is analysed and has no doc values. Both look like "no data
yet", which on a pipeline that has been idle is also what no data looks
like.

Read as files rather than by asking Grafana, so this needs no container.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROVISIONING = ROOT / "configs" / "grafana" / "provisioning"
FILEBEAT = ROOT / "configs" / "filebeat" / "filebeat.yml"

#: ECS field names the dashboards use that the application does not declare.
#: Filebeat's own template maps these, which is the reason the log formatter
#: writes ECS names in the first place.
ECS = frozenset(
    {
        "@timestamp",
        "message",
        "log.level",
        "log.logger",
        "service.name",
        "host.name",
        "process.pid",
        "process.thread.name",
        "log.origin.file.name",
        "log.origin.file.line",
        "log.origin.function",
        "error.code",
        "error.type",
        "error.message",
        "error.stack_trace",
        "trace.id",
        "span.id",
        "url.path",
        "http.request.method",
        "http.response.status_code",
    }
)


def dashboards() -> dict[str, dict]:
    """Every provisioned dashboard, by file name."""
    return {
        path.name: json.loads(path.read_text())
        for path in sorted((PROVISIONING / "dashboards").glob("*.json"))
    }


def datasource_uids() -> set[str]:
    """Every datasource uid provisioning declares."""
    found = set()
    for path in (PROVISIONING / "datasources").glob("*.yml"):
        declared = yaml.safe_load(path.read_text())
        found.update(one["uid"] for one in declared.get("datasources", []))
    return found


def declared_fields() -> set[str]:
    """Every field the shipper's template maps, beside the ECS ones."""
    shipper = yaml.safe_load(FILEBEAT.read_text())
    appended = {one["name"] for one in shipper.get("setup.template.append_fields", [])}
    return appended | ECS


DASHBOARDS = dashboards()


def test_the_scan_finds_the_dashboards_it_is_meant_to_guard() -> None:
    """A move of the provisioning directory would otherwise pass silently."""
    assert len(DASHBOARDS) >= 3, sorted(DASHBOARDS)
    assert datasource_uids(), "no provisioned datasource was read"
    assert len(declared_fields()) > len(ECS), "no appended field was read"


def test_every_dashboard_has_its_own_uid() -> None:
    """Two sharing one means the second replaces the first on provisioning."""
    uids = [one["uid"] for one in DASHBOARDS.values()]

    assert len(uids) == len(set(uids)), sorted(uids)


@pytest.mark.parametrize("name", DASHBOARDS)
def test_every_panel_names_a_datasource_that_is_provisioned(name: str) -> None:
    """A uid nothing declares renders the panel as an error, not as empty."""
    declared = datasource_uids()
    wanted = {
        source["uid"]
        for source in _sources(DASHBOARDS[name])
        if isinstance(source, dict) and "uid" in source
    }
    missing = {one for one in wanted if one not in declared and not one.startswith("$")}

    assert not missing, (
        f"{name} reads {', '.join(sorted(missing))}, which "
        f"configs/grafana/provisioning/datasources/ does not declare"
    )


@pytest.mark.parametrize("name", DASHBOARDS)
def test_every_field_a_log_panel_groups_on_is_mapped(name: str) -> None:
    """Grouping on a `text` field returns nothing and says nothing.

    Only the fields a query aggregates on, which are the ones that need doc
    values. What a panel merely displays can be anything.
    """
    declared = declared_fields()
    used = {
        field
        for target in _targets(DASHBOARDS[name])
        if _is_elasticsearch(target)
        for field in _aggregated(target)
    }
    missing = sorted(used - declared)

    assert not missing, (
        f"{name} aggregates on {', '.join(missing)}, which is neither ECS nor "
        f"declared in setup.template.append_fields in configs/filebeat/"
        f"filebeat.yml. Elasticsearch will map it as `text`, and the panel "
        f"will render empty."
    )


def _sources(dashboard: dict):
    """Every datasource reference in a dashboard, panel and target alike."""
    for panel in dashboard.get("panels", []):
        if "datasource" in panel:
            yield panel["datasource"]
        for target in panel.get("targets", []):
            if "datasource" in target:
                yield target["datasource"]
    for variable in dashboard.get("templating", {}).get("list", []):
        if "datasource" in variable:
            yield variable["datasource"]


def _targets(dashboard: dict):
    """Every query in a dashboard."""
    for panel in dashboard.get("panels", []):
        yield from panel.get("targets", [])


def _is_elasticsearch(target: dict) -> bool:
    """Whether one query goes to the log store."""
    return target.get("datasource", {}).get("type") == "elasticsearch"


def _aggregated(target: dict) -> set[str]:
    """The fields one Elasticsearch query buckets or measures on."""
    found = set()
    for bucket in target.get("bucketAggs", []):
        if bucket.get("type") == "terms" and bucket.get("field"):
            found.add(bucket["field"])
    for metric in target.get("metrics", []):
        if metric.get("field"):
            found.add(metric["field"])
    return found
