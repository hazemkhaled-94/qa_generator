"""The backend's HTTP surface, and the only address the frontend holds.

Assembles the application. Wiring is in dependencies; endpoints are in
routes/.
"""

from __future__ import annotations

from fastapi import FastAPI

import telemetry
from api import errors
from api.routes import ROUTERS

app = FastAPI(title="qa_generator API")

# Attached to the instance, after it exists, so a caller's trace continues
# here rather than a new one beginning.
telemetry.trace_app(app)

errors.install(app)

for router in ROUTERS:
    app.include_router(router)
