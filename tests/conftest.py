"""Hermetic defaults for API and browser-contract test collection.

Tests must never append demo projects, audit events, uploads, or issued reports to
the developer database. CI can opt into dedicated paths with the
``BUILI_TEST_DATABASE_URL`` and ``BUILI_TEST_STORAGE_ROOT`` variables.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="buili-tests-"))
os.environ["BUILI_DATABASE_URL"] = os.environ.get(
    "BUILI_TEST_DATABASE_URL", f"sqlite:///{_TEST_ROOT / 'buili-test.db'}"
)
os.environ["BUILI_STORAGE_ROOT"] = os.environ.get(
    "BUILI_TEST_STORAGE_ROOT", str(_TEST_ROOT / "storage")
)
os.environ["BUILI_PUBLIC_BASE_URL"] = "http://testserver"
os.environ["BUILI_CORS_ORIGINS"] = "http://testserver"
os.environ["BUILI_AUTH_REQUIRED"] = "false"
os.environ["BUILI_SECURE_COOKIES"] = "false"
os.environ["BUILI_AUTH_SECRET"] = "buili-test-only-session-secret-000000000000"
os.environ["BUILI_PILOT_PASSWORD"] = "BuiliTestOnly!2026"


@pytest.fixture(scope="session", autouse=True)
def _remove_test_storage() -> Iterator[None]:
    yield
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
