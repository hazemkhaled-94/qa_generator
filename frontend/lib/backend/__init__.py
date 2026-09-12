"""Backend clients, one module per group of calls.

Each is an :class:`Endpoint` subclass holding only the calls its pages make.
All of them share one session, and therefore one connection pool.

Only the session is cached: a client is two attribute assignments around it,
and caching those meant an edited client class went on serving an instance
built from the old one.
"""

import requests
import streamlit as st

from lib import config
from lib.backend.base import Endpoint
from lib.backend.catalog import CatalogApi
from lib.backend.health import HealthApi
from lib.backend.upload import UploadApi

__all__ = [
    "CatalogApi",
    "Endpoint",
    "HealthApi",
    "UploadApi",
    "catalog_api",
    "health_api",
    "upload_api",
]


@st.cache_resource
def _session() -> requests.Session:
    """Returns the shared HTTP session, built once per server process.

    Cached because Streamlit re-runs the whole script on every interaction,
    which would otherwise rebuild the connection pool on every click.
    """
    return requests.Session()


def upload_api() -> UploadApi:
    """Returns the Upload page's client."""
    return UploadApi(config.BACKEND_URL, _session())


def health_api() -> HealthApi:
    """Returns the System health page's client."""
    return HealthApi(config.BACKEND_URL, _session())


def catalog_api() -> CatalogApi:
    """Returns the client the Documents, Passages and Facts pages share."""
    return CatalogApi(config.BACKEND_URL, _session())
