"""HTTP acceptance checks for the first-party pilot login boundary."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from services.api.buili.auth import COOKIE_NAME
from services.api.buili.config import get_settings
from services.api.buili.database import SessionLocal
from services.api.buili.main import app as api_app
from services.api.buili.models import AuditEvent, Issue, Organization, Project


PILOT_EMAIL = "jordan.davis@northstarbuild.example"
TEST_PASSWORD = "BuiliTestOnly!2026"


def _login(client: TestClient) -> dict:
    response = client.post(
        "/v1/auth/login",
        json={"email": PILOT_EMAIL, "password": TEST_PASSWORD, "remember_me": False},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_bad_login_is_generic_and_does_not_enumerate_accounts() -> None:
    with TestClient(api_app) as client:
        malformed = client.post(
            "/v1/auth/login",
            json={"email": "not-an-email", "password": "irrelevant"},
        )
        unknown = client.post(
            "/v1/auth/login",
            json={"email": f"missing-{uuid4().hex}@example.com", "password": "irrelevant"},
        )

    assert malformed.status_code == unknown.status_code == 401
    assert malformed.json() == unknown.json() == {"detail": "invalid email or password"}


def test_session_cookie_is_httponly_scoped_and_revoked_on_logout() -> None:
    foreign_org = Organization(name=f"Unrelated tenant {uuid4().hex[:8]}")
    with SessionLocal() as session:
        session.add(foreign_org)
        session.flush()
        foreign_project = Project(
            org_id=foreign_org.org_id,
            name="Cross-tenant sentinel",
            address="QA only",
        )
        session.add(foreign_project)
        session.flush()
        foreign_issue = Issue(
            project_id=foreign_project.project_id,
            type="quality_observation",
            title="Cross-tenant issue sentinel",
        )
        session.add(foreign_issue)
        session.commit()
        foreign_project_id = foreign_project.project_id
        foreign_issue_id = foreign_issue.issue_id

    with TestClient(api_app) as client:
        login = client.post(
            "/v1/auth/login",
            json={"email": PILOT_EMAIL, "password": TEST_PASSWORD},
        )
        assert login.status_code == 200, login.text
        set_cookie = login.headers["set-cookie"].lower()
        assert f"{COOKIE_NAME}=" in set_cookie
        assert "httponly" in set_cookie
        assert "samesite=lax" in set_cookie
        assert "path=/" in set_cookie

        raw_cookie = client.cookies.get(COOKIE_NAME)
        assert raw_cookie and raw_cookie.startswith("ses_") and "." in raw_cookie
        session_id = raw_cookie.split(".", 1)[0]

        me = client.get("/v1/auth/me")
        assert me.status_code == 200, me.text
        identity = me.json()
        assert identity["user"]["email"] == PILOT_EMAIL
        assert identity["user"]["role"] == "project_manager"
        assert identity["user"]["organization"]["name"] == "Northstar Builders"
        assert foreign_project_id not in {
            project["project_id"] for project in identity["projects"]
        }

        projects = client.get("/v1/projects")
        assert projects.status_code == 200, projects.text
        assert foreign_project_id not in {
            project["project_id"] for project in projects.json()
        }
        hidden_issue = client.get(f"/v1/issues/{foreign_issue_id}")
        assert hidden_issue.status_code == 404

        logout = client.post("/v1/auth/logout")
        assert logout.status_code == 204, logout.text
        assert client.get("/v1/auth/me").status_code == 401

    # Cookie deletion alone is not enough: replaying the old bearer must fail
    # because the server-side session row was revoked.
    with TestClient(api_app) as replay:
        replay.cookies.set(COOKIE_NAME, raw_cookie)
        assert replay.get("/v1/auth/me").status_code == 401

    with SessionLocal() as session:
        actions = set(
            session.scalars(
                select(AuditEvent.action).where(AuditEvent.entity_id == session_id)
            ).all()
        )
    assert {"USER_LOGGED_IN", "USER_LOGGED_OUT"} <= actions


def test_production_mode_ignores_forged_browser_identity_headers() -> None:
    settings = get_settings()
    previous = settings.auth_required
    settings.auth_required = True
    try:
        with TestClient(api_app) as client:
            forged = client.get(
                "/v1/projects",
                headers={"X-Buili-Actor": "forged", "X-Buili-Role": "admin"},
            )
            assert forged.status_code == 401

            identity = _login(client)
            assert identity["user"]["role"] == "project_manager"
            assert client.get("/v1/projects").status_code == 200
    finally:
        settings.auth_required = previous
