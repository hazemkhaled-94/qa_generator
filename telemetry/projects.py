"""What Phoenix is holding, and taking back the projects holding no work.

A project is created by the first span filed under it and never by
anything else, so one appears whenever a process exports and disappears
never. That was survivable while a project meant a run somebody asked
for. It stopped being survivable when `run_id` minted a uuid per PROCESS:
a `--watch` worker is a process per restart, a credential that has expired
restarts it every minute, and the deployment this was written against
reached **five hundred projects**. Sampling a hundred and twenty of them
found a hundred and nineteen holding exactly one span - the `completion`
of a single failed preflight call.

Two things came out of that, and this is the second:

- `telemetry/pipeline.py` no longer gives an unnamed run a project of its
  own, and `stages/cli.py` no longer installs the exporter until the
  preflight has passed. Together those stop the count growing.
- This takes back what the old arrangement left behind.

**What counts as holding no work.** Not "empty" - none of those five
hundred was empty, which is why the first version of this found nothing
to do. A project holds work when something other than a model call
happened in it: an `assess`, an `extract`, a `parse`. A project whose
every span is a `completion` is a process that proved the model answers
and then died, and the call it proved it with is in the logs.

Runs on the host, like `evaluation/` and `review/`: no image carries a
reason to delete a project.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

#: How many projects one page of the listing asks for.
PAGE = 500

#: How many spans are read per project to decide whether it holds work.
#: One would do for the common case - a leftover holds exactly one - and
#: this is enough that a project whose first spans happen to be model
#: calls is still read correctly.
SAMPLE = 50

#: The span a model call produces, whoever made it. A project holding
#: nothing else is a preflight that never became a run.
CALL = "completion"

#: Never deleted, whatever it holds. Phoenix files a span carrying no
#: project name here, so it is where anything misconfigured ends up - and
#: a tool that swept it would take the evidence with it.
KEEP = frozenset({"default"})


@dataclass(frozen=True)
class Project:
    """One Phoenix project, and whether anything happened in it."""

    id: str
    name: str
    spans: int
    #: Whether any span is something other than a model call.
    work: bool
    #: Whether every span was read, rather than the first `SAMPLE` of them.
    whole: bool

    @property
    def spent(self) -> bool:
        """Whether this project is a leftover rather than a run.

        Nothing but model calls, AND few enough of them that every span was
        actually read. A project that filled the sample is one this has
        only seen the start of, and the first fifty spans of a real run
        are model calls too - `5-topic_modelling` reads exactly that way
        and holds a corpus of labelling.

        So a project this cannot see the end of is kept. The cost is a
        leftover nobody sweeps; the cost the other way is somebody's data.
        """
        return not self.work and self.whole


#: How long one Phoenix call may take. A prune walks every project and
#: deletes some of them, and the server is doing real work for each.
_TIMEOUT = 60.0


class _Phoenix(requests.Session):
    """A session that answers relative paths and times every call out.

    `requests` has neither a base URL nor a default timeout, and both are
    what the call sites here rely on. The same shape `orchestration/client`
    and `frontend/lib/backend/base` build by hand; a Session subclass
    because this one is used as a context manager and passed around as a
    client rather than wrapped in a class of its own.
    """

    def __init__(self, base: str) -> None:
        """Points the session at one Phoenix."""
        super().__init__()
        self._base = base.rstrip("/")

    def request(self, method, url, *args, **kwargs):
        """Sends one call, resolving the path and defaulting the timeout."""
        kwargs.setdefault("timeout", _TIMEOUT)
        return super().request(method, f"{self._base}{url}", *args, **kwargs)


def _client() -> _Phoenix:
    """An HTTP client pointed at Phoenix, with the bearer it wants.

    requests rather than httpx, which is what every other first-party
    caller in this repository speaks. httpx arrives transitively through
    the model stack and was never declared here.
    """
    base = os.environ.get("PHOENIX_BASE_URL")
    if not base:
        raise SystemExit(
            "PHOENIX_BASE_URL is unset, so there is nothing to ask. It is in "
            ".env; see .env.example."
        )
    key = os.environ.get("PHOENIX_API_KEY") or os.environ.get("PHOENIX_ADMIN_SECRET")
    client = _Phoenix(base)
    if key:
        client.headers["Authorization"] = f"Bearer {key}"
    return client


def listed(client, path: str = "/v1/projects") -> Iterator[dict]:
    """Everything Phoenix holds at one path, following the cursor to the end.

    **Paginated, and the first version of this was not.** Phoenix answers
    with a page and a `next_cursor`, and a tool that reads one page reports
    a number that is really "the page size" and prunes a slice of the
    backlog. Against a deployment holding more than five hundred, that read
    as 500 projects, pruned 497, and then reported 500 again from the next
    page - which looks exactly like a delete that silently failed.
    """
    cursor = None
    while True:
        params = {"limit": PAGE}
        if cursor:
            params["cursor"] = cursor
        answered = client.get(path, params=params)
        answered.raise_for_status()
        page = answered.json()
        yield from page.get("data", [])
        cursor = page.get("next_cursor")
        if not cursor:
            return


def held(client) -> list[Project]:
    """Every project Phoenix holds, and whether work happened in each.

    One request per project on top of the listing, which is why this is a
    command somebody runs rather than something a page polls.
    """
    found = []
    for one in listed(client):
        spans = (
            client.get(f"/v1/projects/{one['id']}/spans", params={"limit": SAMPLE})
            .json()
            .get("data", [])
            or []
        )
        found.append(
            Project(
                id=one["id"],
                name=one["name"],
                spans=len(spans),
                work=any(span.get("name") != CALL for span in spans),
                whole=len(spans) < SAMPLE,
            )
        )
    return found


def spent(projects: list[Project]) -> list[Project]:
    """The projects worth taking back, `default` never among them."""
    return [one for one in projects if one.spent and one.name not in KEEP]


def report(projects: list[Project]) -> list[str]:
    """What Phoenix is holding, as lines to print."""
    taking = spent(projects)
    return [
        f"{len(projects)} project(s) in Phoenix",
        (
            f"{len(taking)} of them hold no work - every span is a {CALL}, "
            f"which is a preflight that never became a run"
        ),
        (
            f"{len(projects) - len(taking)} hold something, or are too big "
            f"to read in one page and are left alone"
        ),
        "",
        *(f"  no work: {one.name} ({one.spans} span(s))" for one in taking[:25]),
        *([f"  ... and {len(taking) - 25} more"] if len(taking) > 25 else []),
    ]


def prune(client, projects: list[Project], everything: bool = False) -> int:
    """Deletes projects, and says how many went.

    Args:
        client: The Phoenix client.
        projects: Everything Phoenix holds.
        everything: Take all of them rather than only the spent ones. What
            `make wipe` passes: a corpus that is gone has no traces worth
            keeping, and the spent-only rule would leave every project that
            recorded a real run - which after a wipe is all of them.

    Returns:
        How many were deleted.
    """
    taking = projects if everything else spent(projects)
    if not taking:
        return 0
    log.warning(
        "deleting %d project(s) holding %d span(s) between them%s",
        len(taking),
        sum(one.spans for one in taking),
        "" if everything else ", all of them model calls no run followed",
    )
    deleted = 0
    for one in taking:
        answered = client.delete(f"/v1/projects/{one.id}")
        if answered.ok:
            deleted += 1
        else:
            log.warning("could not delete %s: %s", one.name, answered.status_code)
    return deleted


def purge_datasets(client) -> int:
    """Deletes every dataset, and says how many went.

    A different resource from a project and not reached by deleting one:
    `/v1/projects` holds the traces a run wrote, `/v1/datasets` holds the
    golden cases `make eval-upload` puts there and the experiments scored
    against them, which Phoenix keeps under the dataset.

    Its own flag rather than part of `--purge`, because `make wipe
    KEEP=golden` spares these and takes the projects anyway. The cases are
    re-uploadable from `evaluation/`, which is what makes either choice
    reasonable.

    Returns:
        How many datasets were deleted.
    """
    taking = list(listed(client, "/v1/datasets"))
    if not taking:
        return 0
    log.warning("deleting %d dataset(s) and their experiments", len(taking))
    deleted = 0
    for one in taking:
        answered = client.delete(f"/v1/datasets/{one['id']}")
        if answered.ok:
            deleted += 1
        else:
            log.warning(
                "could not delete dataset %s: %s",
                one.get("name", one["id"]),
                answered.status_code,
            )
    return deleted


def parser() -> argparse.ArgumentParser:
    """Builds the argument parser."""
    built = argparse.ArgumentParser(
        prog="python -m telemetry.projects",
        description="What Phoenix is holding, and taking back the projects "
        "holding no work.",
    )
    group = built.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="report what Phoenix holds, and change nothing",
    )
    group.add_argument(
        "--prune",
        action="store_true",
        help="delete the projects in which nothing but a model call ever "
        "happened. Irreversible, and it says what it is taking first.",
    )
    group.add_argument(
        "--purge",
        action="store_true",
        help="delete EVERY project, whatever it holds. What `make wipe` "
        "runs: traces of a corpus that no longer exists. Irreversible.",
    )
    group.add_argument(
        "--purge-datasets",
        action="store_true",
        dest="purge_datasets",
        help="delete every DATASET and the experiments under it - the "
        "golden cases `make eval-upload` puts there. A different resource "
        "from a project, so `--purge` does not reach them. Irreversible, "
        "and re-uploadable from evaluation/.",
    )
    return built


def main(argv: list[str] | None = None) -> int:
    """Runs the command line.

    Returns:
        The process exit code.
    """
    import telemetry

    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    telemetry.configure("telemetry-projects")

    with _client() as client:
        # Only when a project is being read or taken: `held` costs one
        # request per project, and the dataset flag touches none of them.
        projects = [] if args.purge_datasets else held(client)
        for line in report(projects) if not args.purge_datasets else ():
            log.info("%s", line)
        if args.prune or args.purge:
            log.info("deleted %d project(s)", prune(client, projects, args.purge))
        if args.purge_datasets:
            log.info("deleted %d dataset(s)", purge_datasets(client))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
