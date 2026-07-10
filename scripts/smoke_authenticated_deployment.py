#!/usr/bin/env python3
"""Non-destructive smoke check for an authenticated Buili web deployment.

Credentials are accepted only through environment variables so a pilot password
does not land in shell history or process arguments.
"""

from __future__ import annotations

import json
import os
import sys
from http.cookiejar import CookieJar
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor, Request, build_opener


BASE_URL = os.environ.get("BUILI_SMOKE_URL", "").rstrip("/") + "/"
EMAIL = os.environ.get("BUILI_E2E_EMAIL", "")
PASSWORD = os.environ.get("BUILI_E2E_PASSWORD", "")


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    if not BASE_URL.startswith(("http://", "https://")):
        fail("set BUILI_SMOKE_URL to the deployed web origin")
    if not EMAIL or not PASSWORD:
        fail("set BUILI_E2E_EMAIL and BUILI_E2E_PASSWORD in the environment")

    cookies = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookies))

    def call(
        path: str,
        *,
        method: str = "GET",
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object | None]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            urljoin(BASE_URL, path.lstrip("/")),
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
                **(headers or {}),
            },
        )
        try:
            response = opener.open(request, timeout=30)
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
        except HTTPError as exc:
            raw = exc.read()
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = raw.decode(errors="replace")
            return exc.code, parsed

    status, health = call("/api/readyz")
    if status != 200 or not isinstance(health, dict) or health.get("status") != "ready":
        fail(f"proxy health check failed with HTTP {status}")

    status, _ = call(
        "/api/v1/projects",
        headers={"X-Buili-Actor": "forged", "X-Buili-Role": "admin"},
    )
    if status != 401:
        fail(f"unauthenticated forged identity was not rejected (HTTP {status})")

    status, identity = call(
        "/api/v1/auth/login",
        method="POST",
        payload={"email": EMAIL, "password": PASSWORD, "remember_me": False},
    )
    if status != 200 or not isinstance(identity, dict):
        fail(f"login failed with HTTP {status}")
    session_cookie = next((cookie for cookie in cookies if cookie.name == "buili_session"), None)
    if not session_cookie or not session_cookie.has_nonstandard_attr("HttpOnly"):
        fail("login did not issue an HttpOnly buili_session cookie")

    status, projects = call("/api/v1/projects")
    if status != 200 or not isinstance(projects, list) or len(projects) != 1:
        fail("pilot identity must be scoped to exactly one project")
    project_id = projects[0].get("project_id")
    if not project_id:
        fail("pilot project has no project_id")

    status, documents = call(f"/api/v1/projects/{project_id}/documents")
    current_plans = (
        [item for item in documents if item.get("type") == "plan" and item.get("is_current")]
        if status == 200 and isinstance(documents, list)
        else []
    )
    if len(current_plans) != 1:
        fail("pilot project must expose exactly one authoritative current plan")

    status, _ = call("/api/v1/auth/logout", method="POST")
    if status != 204:
        fail(f"logout failed with HTTP {status}")
    status, _ = call("/api/v1/projects")
    if status != 401:
        fail("logout did not revoke project access")

    print("Authenticated deployment smoke check passed: proxy, login, scope, plan, logout.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, URLError) as exc:
        print(f"Smoke check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
