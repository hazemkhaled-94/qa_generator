"""The backend's HTTP surface, and the only address the frontend holds.

Assembles the application. Wiring is in dependencies; endpoints are in
routes/.
"""

from __future__ import annotations

from fastapi import FastAPI

from api import errors
from api.routes import ROUTERS

app = FastAPI(title="qa_generator API")

errors.install(app)

for router in ROUTERS:
    app.include_router(router)
