"""The document and the server, checked against each other by generated calls.

`test_openapi.py` beside this reads the document: which paths exist, which
the frontend calls, which refusals each route declares. Nothing there sends
a request, so nothing there can find the case where the document and the
server disagree - a route that answers 500 on a query parameter it was never
given, or one that returns a body its own schema does not describe.

That is what schemathesis is: it reads the OpenAPI document, generates calls
against every operation in it, and checks each answer back against what the
document promised. The cases are derived from the schema, so a route added
with a new parameter is covered by having been published rather than by
somebody writing a case for it.

**Read-only operations only, and that is deliberate.** The generator would
happily send `DELETE /documents/{sha256}` and `POST /questions/rerun`
against the container, which is allowed to be destructive - it is a
throwaway database - but a mutating call that succeeds leaves the next
generated call reading different state, so a failure would not reproduce.
The undeclared-500 this exists to find lives on the read paths, which are
the ones taking a filter, an enum and a page number off the query string.

`--experimental=...` is not needed: everything used here is stable in 4.x.
"""

from __future__ import annotations

import pytest
import schemathesis
from hypothesis import HealthCheck, settings
from schemathesis.specs.openapi.checks import (
    content_type_conformance,
    response_schema_conformance,
    status_code_conformance,
)

pytestmark = pytest.mark.integration

#: The verbs a generated call may use. See the module docstring.
READ_ONLY = {"GET"}

#: The four checks that measure the document against the server, and no
#: others. Each asks "did the answer match what was published".
#:
#: `unsupported_method` is deliberately absent, and its absence is a finding
#: rather than a convenience. It sends a verb the path does not declare and
#: wants 405 back; here `POST /questions/{action}` is a real route, so a
#: POST to `/questions/quality` genuinely matches it and is refused 422 for
#: an action that is not one. That is the router working. Leaving the check
#: on made 30 of 33 operations fail on it and hid everything else.
#:
#: `negative_data_rejection` and `positive_data_acceptance` are opinions
#: about which status a malformed query deserves; `ignored_auth`,
#: `use_after_free` and `ensure_resource_availability` need an auth scheme
#: and a stateful link graph, and this API publishes neither.
CONFORMANCE = [
    # A 5xx is a defect whatever the document says.
    schemathesis.checks.not_a_server_error,
    status_code_conformance,
    response_schema_conformance,
    content_type_conformance,
]

#: How many cases each operation gets. The default is 100, which over ~60
#: read operations is a six-minute job on a layer that already starts two
#: containers. Twenty finds the shapes that matter - a missing enum member,
#: an integer where a string was promised - and keeps the layer usable.
EXAMPLES = 20


@pytest.fixture(scope="session")
def api_schema(application):
    """The published document, loaded straight off the ASGI application.

    No network and no port: schemathesis calls the app in this process, the
    same way TestClient does, so this needs nothing compose has to be
    running for.

    """
    return schemathesis.openapi.from_asgi("/openapi.json", application)


#: Narrowed to the read verbs, so a mutating operation is never generated
#: and the count this layer reports is the count it made.
#:
#: The filter goes on the LAZY schema and not inside the fixture. Both
#: objects have `include`, and filtering the one the fixture returns is
#: silently ignored - `parametrize` collects its operations from the lazy
#: wrapper. That looked like it worked, and the give-away was PATCH
#: operations in the failure list of a run restricted to GET.
#: Read routes whose generated calls are not worth what they report.
#:
#: `/pipeline/runs` answers 503 `no_orchestrator` when DAGSTER_URL is unset,
#: which is every test run: nothing here starts a Dagster. That status is
#: DECLARED on the route and the frontend branches on it, so the document
#: and the server agree and `status_code_conformance` is content. Only
#: `not_a_server_error` objects, and it objects to the shape of the number
#: rather than to anything being wrong - which is the same reason
#: `unsupported_method` is off above.
#:
#: What the route does with a reachable orchestrator is covered by
#: `tests/unit/api/test_orchestrator.py` and `tests/integration/api/
#: test_pipeline_routes.py`, both of which can stand one in.
UNGENERATED = ("/pipeline/runs",)

schema = (
    schemathesis.pytest.from_fixture("api_schema")
    .include(method=sorted(READ_ONLY))
    .exclude(path_regex="^(" + "|".join(UNGENERATED) + ")$")
)


@schema.parametrize()
@settings(
    max_examples=EXAMPLES,
    deadline=None,
    # The containers are session-scoped and shared, so the first call of a
    # run pays for a connection the rest do not. That is slow data setup
    # rather than a slow test.
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_a_read_route_answers_what_its_schema_promises(case, database, buckets) -> None:
    """Every generated call, checked against the document it came from.

    `database` and `buckets` put the call against an EMPTY corpus, which is
    the state every read route has to answer correctly in. A listing with
    nothing to list, a detail route for an id that is not there and a
    quality report over no rows are exactly where an unguarded `[0]` or a
    division by a count of zero lives.
    """
    case.call_and_validate(checks=CONFORMANCE)
