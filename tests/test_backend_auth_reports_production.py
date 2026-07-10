from __future__ import annotations

import base64
import hashlib
from uuid import uuid4
from urllib.parse import urlsplit

import fitz
from fastapi.testclient import TestClient
from sqlalchemy import select

from services.api.buili.auth import ensure_pilot_identity
from services.api.buili.config import get_settings
from services.api.buili.database import SessionLocal
from services.api.buili.main import app
from services.api.buili.models import AuditEvent, Membership, Project

PILOT_PROJECT = "Cooper Residence — Electrical Rough-In Verification"


def _pilot_project(client: TestClient) -> dict:
    projects = client.get("/v1/projects").json()
    return next(project for project in projects if project["name"] == PILOT_PROJECT)


def _pilot_issues(client: TestClient, project_id: str) -> tuple[dict, dict]:
    issues = client.get(f"/v1/projects/{project_id}/issues").json()
    primary = next(
        item
        for item in issues
        if "Note 3" in str((item.get("requirement") or {}).get("citation") or "")
    )
    other = next(item for item in issues if item["issue_id"] != primary["issue_id"])
    return primary, other


def _approve(client: TestClient, issue_id: str, reviewer: str = "Jordan Davis") -> dict:
    response = client.post(
        f"/v1/issues/{issue_id}/reviews",
        json={
            "decision": "approve",
            "reviewer": reviewer,
            "reason": "Current source and location-confirmed field evidence verified.",
        },
        headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": reviewer},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_plan_revision(
    client: TestClient,
    project_id: str,
    *,
    revision: str,
    body: bytes,
) -> str:
    presign = client.post(
        "/v1/uploads/presign",
        json={
            "project_id": project_id,
            "filename": "revision-switch-plan.txt",
            "mime": "text/plain",
            "size": len(body),
            "kind": "document",
        },
    )
    assert presign.status_code == 200, presign.text
    upload = client.post(
        urlsplit(presign.json()["upload_url"]).path,
        files={"file": ("revision-switch-plan.txt", body, "text/plain")},
    )
    assert upload.status_code == 200, upload.text
    complete = client.post(
        urlsplit(presign.json()["complete_url"]).path,
        json={"document_type": "plan", "revision": revision},
    )
    assert complete.status_code == 200, complete.text
    return complete.json()["document_id"]


def test_pilot_source_chain_media_and_type_specific_selected_reports() -> None:
    with TestClient(app) as client:
        project = _pilot_project(client)
        project_id = project["project_id"]
        assert project["address"] == "305 W. Shangri-La, Toquerville, UT 84774"

        documents = client.get(f"/v1/projects/{project_id}/documents").json()
        assert len(documents) == 1
        document = documents[0]
        assert document["filename"] == "Cooper-Residence-E1.1-Electrical.pdf"
        assert document["revision"] == "Final"
        assert document["metadata_json"]["sheet_number"] == "E1.1"
        assert document["issue_date"] == "2021-01-11"
        assert document["is_current"] is True

        media = client.get(f"/v1/projects/{project_id}/media").json()
        by_name = {item["filename"]: item for item in media}
        assert {
            "garage-east-wall-context.png",
            "receptacle-rough-in-detail.png",
            "box-elevation-measurement.png",
            "foreman-voice-note.mp3",
        }.issubset(by_name)
        for filename in ("garage-east-wall-context.png", "foreman-voice-note.mp3"):
            response = client.get(urlsplit(by_name[filename]["download_url"]).path)
            assert response.status_code == 200
            assert response.content
            assert response.headers["cache-control"] == "private, no-store"
            assert response.headers["content-disposition"].startswith("inline")
        assert by_name["foreman-voice-note.mp3"]["mime"] == "audio/mpeg"
        assert "garage" in by_name["foreman-voice-note.mp3"]["metadata_json"]["transcript"].lower()

        issues = client.get(f"/v1/projects/{project_id}/issues").json()
        issue = next(item for item in issues if item["title"].startswith("Garage GFCI"))
        assert issue["room"] == "Main Floor · Garage · East wall near entry door"
        assert issue["requirement"]["source"] == "E1.1"
        assert issue["requirement"]["revision"] == "Final"
        assert "Note 3" in issue["requirement"]["citation"]
        assert len([item for item in issue["evidence"] if item["evidence_type"] == "field_evidence"]) == 4

        rfi = client.post(
            f"/v1/projects/{project_id}/reports",
            json={"report_type": "rfi", "format": "pdf", "issue_ids": [issue["issue_id"]]},
        )
        assert rfi.status_code == 200, rfi.text
        assert rfi.json()["issue_ids"] == [issue["issue_id"]]
        pdf = client.get(urlsplit(rfi.json()["download_url"]).path)
        assert pdf.status_code == 200
        with fitz.open(stream=pdf.content, filetype="pdf") as document_pdf:
            text = "\n".join(page.get_text() for page in document_pdf)
        for expected in (
            "Report status: DRAFT",
            "Exact location:",
            "Current contract source:",
            "Source citation:",
            "Existing field condition:",
            "Ambiguity / difference:",
            "Question requiring response:",
            "Evidence / attachment manifest",
            "Mike Torres",
            "SHA-256",
        ):
            assert expected in text

        punch = client.post(
            f"/v1/projects/{project_id}/reports",
            json={"report_type": "punch", "format": "pdf", "issue_ids": [issue["issue_id"]]},
        )
        assert punch.status_code == 200, punch.text
        punch_pdf = client.get(urlsplit(punch.json()["download_url"]).path)
        with fitz.open(stream=punch_pdf.content, filetype="pdf") as document_pdf:
            punch_text = "\n".join(page.get_text() for page in document_pdf)
        for expected in (
            "Defect / observed condition:",
            "Required completed condition:",
            "Responsible trade: Delta Electrical",
            "Due date: 2026-07-11",
            "Before evidence:",
            "After evidence:",
        ):
            assert expected in punch_text


def test_managed_pilot_password_rotation_revokes_sessions_and_is_audited() -> None:
    settings = get_settings()
    original = settings.pilot_password
    old_password = "RotationOld!2026"
    new_password = "RotationNew!2026"
    try:
        with SessionLocal() as session:
            project = session.scalar(select(Project).where(Project.name == PILOT_PROJECT))
            assert project
            settings.pilot_password = old_password
            ensure_pilot_identity(session, project)

        with TestClient(app) as client:
            logged_in = client.post(
                "/v1/auth/login",
                json={"email": settings.pilot_email, "password": old_password},
            )
            assert logged_in.status_code == 200
            assert client.get("/v1/auth/me").status_code == 200

            with SessionLocal() as session:
                project = session.scalar(select(Project).where(Project.name == PILOT_PROJECT))
                assert project
                settings.pilot_password = new_password
                ensure_pilot_identity(session, project)
            assert client.get("/v1/auth/me").status_code == 401
            assert client.post(
                "/v1/auth/login",
                json={"email": settings.pilot_email, "password": old_password},
            ).status_code == 401
            assert client.post(
                "/v1/auth/login",
                json={"email": settings.pilot_email, "password": new_password},
            ).status_code == 200

        with SessionLocal() as session:
            event = session.scalar(
                select(AuditEvent)
                .where(AuditEvent.action == "PILOT_PASSWORD_ROTATED")
                .order_by(AuditEvent.created_at.desc())
            )
            assert event
            assert event.metadata_json["sessions_revoked"] >= 1
    finally:
        settings.pilot_password = original
        with SessionLocal() as session:
            project = session.scalar(select(Project).where(Project.name == PILOT_PROJECT))
            if project:
                ensure_pilot_identity(session, project)


def test_explicit_report_scope_requires_every_selected_issue_and_issues_approved_package() -> None:
    with TestClient(app) as client:
        project = _pilot_project(client)
        issues = client.get(f"/v1/projects/{project['project_id']}/issues").json()
        primary = next(item for item in issues if item["title"].startswith("Garage GFCI"))
        other = next(item for item in issues if item["issue_id"] != primary["issue_id"])
        strict = client.post(
            f"/v1/projects/{project['project_id']}/reports",
            json={
                "report_type": "rfi",
                "format": "pdf",
                "issue_ids": [primary["issue_id"], other["issue_id"]],
            },
        )
        assert strict.status_code == 200

        review = client.post(
            f"/v1/issues/{primary['issue_id']}/reviews",
            json={
                "decision": "approve",
                "reviewer": "Jordan Davis",
                "reason": "Current E1.1 Note 3 and all four field records verified.",
            },
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert review.status_code == 200, review.text
        blocked = client.post(
            f"/v1/reports/{strict.json()['report_id']}/export",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert blocked.status_code == 409
        assert other["issue_id"] in str(blocked.json())

        selected = client.post(
            f"/v1/projects/{project['project_id']}/reports",
            json={
                "report_type": "rfi",
                "format": "pdf",
                "issue_ids": [primary["issue_id"]],
            },
        )
        assert selected.status_code == 200
        assert selected.json()["can_issue"] is True
        issued = client.post(
            f"/v1/reports/{selected.json()['report_id']}/export",
            json={"recipients": ["electrical.engineer@example.test"]},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert issued.status_code == 200, issued.text
        assert issued.json()["status"] == "issued"
        assert issued.json()["reviewer"] == "Jordan Davis"


def test_issued_body_is_rebuilt_from_current_approved_scope() -> None:
    with TestClient(app) as client:
        project = _pilot_project(client)
        primary, other = _pilot_issues(client, project["project_id"])
        unapproved_title = f"UNAPPROVED-{uuid4().hex}-MUST-NOT-ISSUE"
        pre_draft_title = f"PRE-DRAFT-{uuid4().hex}"
        current_title = f"POST-DRAFT-{uuid4().hex}-CURRENT"
        for issue_id, title in (
            (other["issue_id"], unapproved_title),
            (primary["issue_id"], pre_draft_title),
        ):
            patched = client.patch(
                f"/v1/issues/{issue_id}",
                json={"title": title},
                headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
            )
            assert patched.status_code == 200, patched.text
        draft = client.post(
            f"/v1/projects/{project['project_id']}/reports",
            json={"report_type": "rfi", "format": "pdf"},
        )
        assert draft.status_code == 200, draft.text
        edited = client.patch(
            f"/v1/issues/{primary['issue_id']}",
            json={"title": current_title},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert edited.status_code == 200, edited.text
        _approve(client, primary["issue_id"])
        issued = client.post(
            f"/v1/reports/{draft.json()['report_id']}/export",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert issued.status_code == 200, issued.text
        artifact = client.get(urlsplit(issued.json()["download_url"]).path)
        with fitz.open(stream=artifact.content, filetype="pdf") as document_pdf:
            issued_text = "\n".join(page.get_text() for page in document_pdf)
        assert current_title in issued_text
        assert pre_draft_title not in issued_text
        assert unapproved_title not in issued_text


def test_evidence_changes_and_links_invalidate_every_affected_approval() -> None:
    with TestClient(app) as client:
        project = _pilot_project(client)
        primary, other = _pilot_issues(client, project["project_id"])
        evidence_id = next(
            item["ref_id"]
            for item in primary["evidence"]
            if item["evidence_type"] == "field_evidence"
        )
        linked = client.post(
            f"/v1/evidence/{evidence_id}/link",
            json={"issue_id": other["issue_id"], "relevance": "supports"},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert linked.status_code == 200, linked.text
        _approve(client, primary["issue_id"])
        _approve(client, other["issue_id"])
        versions_before = {
            issue_id: client.get(f"/v1/issues/{issue_id}").json()["issue_version"]
            for issue_id in (primary["issue_id"], other["issue_id"])
        }
        patched = client.patch(
            f"/v1/evidence/{evidence_id}",
            json={"metadata": {"field_note": "Foreman corrected capture description."}},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert patched.status_code == 200, patched.text
        for issue_id in (primary["issue_id"], other["issue_id"]):
            detail = client.get(f"/v1/issues/{issue_id}").json()
            assert detail["review_status"] == "stale_evidence_review"
            assert detail["issue_version"] > versions_before[issue_id]

        _approve(client, primary["issue_id"])
        content = b"new-location-confirmed-field-photo"
        digest = hashlib.sha256(content).hexdigest()
        synced = client.post(
            "/v1/evidence/sync",
            json={
                "project_id": project["project_id"],
                "client_capture_id": f"security-link-{uuid4().hex}",
                "media_type": "photo",
                "filename": "new-evidence.jpg",
                "mime": "image/jpeg",
                "content_base64": base64.b64encode(content).decode(),
                "sha256": digest,
                "location": {"room": "Garage", "wall": "East wall"},
                "sufficiency": "sufficient",
            },
            headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "Mike Torres"},
        )
        assert synced.status_code == 200, synced.text
        new_link = client.post(
            f"/v1/evidence/{synced.json()['evidence_id']}/link",
            json={"issue_id": primary["issue_id"], "relevance": "supports"},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert new_link.status_code == 200, new_link.text
        assert (
            client.get(f"/v1/issues/{primary['issue_id']}").json()["review_status"]
            == "stale_evidence_review"
        )
    with SessionLocal() as session:
        invalidations = list(
            session.scalars(
                select(AuditEvent).where(
                    AuditEvent.action == "ISSUE_APPROVAL_INVALIDATED",
                    AuditEvent.entity_id.in_([primary["issue_id"], other["issue_id"]]),
                )
            ).all()
        )
        assert {event.entity_id for event in invalidations} == {
            primary["issue_id"],
            other["issue_id"],
        }


def test_authenticated_principal_controls_reviewer_and_session_role() -> None:
    settings = get_settings()
    previous_required = settings.auth_required
    settings.auth_required = True
    try:
        with TestClient(app) as client:
            login = client.post(
                "/v1/auth/login",
                json={"email": settings.pilot_email, "password": settings.pilot_password},
            )
            assert login.status_code == 200, login.text
            project = _pilot_project(client)
            primary, _ = _pilot_issues(client, project["project_id"])
            review = _approve(client, primary["issue_id"], reviewer="forged-reviewer")
            assert review["reviewer"] == settings.pilot_email
            draft = client.post(
                f"/v1/projects/{project['project_id']}/reports",
                json={
                    "report_type": "rfi",
                    "format": "pdf",
                    "issue_ids": [primary["issue_id"]],
                },
            )
            issued = client.post(f"/v1/reports/{draft.json()['report_id']}/export", json={})
            assert issued.status_code == 200, issued.text
            assert issued.json()["reviewer"] == settings.pilot_email

            with SessionLocal() as session:
                membership = session.scalar(
                    select(Membership).where(Membership.user_id == login.json()["user"]["user_id"])
                )
                assert membership
                membership.role = "field_user"
                session.commit()
            assert client.get("/v1/auth/me").json()["user"]["role"] == "field_user"
            denied = client.post(
                f"/v1/issues/{primary['issue_id']}/reviews",
                json={"decision": "approve", "reviewer": "forged-reviewer"},
            )
            assert denied.status_code == 403
            with SessionLocal() as session:
                membership = session.scalar(
                    select(Membership).where(Membership.user_id == login.json()["user"]["user_id"])
                )
                assert membership
                session.delete(membership)
                session.commit()
            assert client.get("/v1/auth/me").status_code == 401
    finally:
        settings.auth_required = previous_required
        with SessionLocal() as session:
            project = session.scalar(select(Project).where(Project.name == PILOT_PROJECT))
            assert project
            ensure_pilot_identity(session, project)


def test_revision_switch_regenerates_spatial_graph_and_never_reuses_stale_glb() -> None:
    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={"name": f"Spatial revision {uuid4().hex}", "address": "QA"},
        ).json()
        project_id = project["project_id"]
        doc_a = _upload_plan_revision(
            client,
            project_id,
            revision="A",
            body=b"E1.1 REV A Electrical Plan\nGarage requires one GFCI outlet.\n",
        )
        client.post(f"/v1/projects/{project_id}/analyze", json={})
        doc_b = _upload_plan_revision(
            client,
            project_id,
            revision="B",
            body=b"E1.1 REV B Electrical Plan\nGarage requires two GFCI outlets.\n",
        )
        analysis = client.post(f"/v1/projects/{project_id}/analyze", json={})
        assert analysis.status_code == 200, analysis.text
        pre_graph = client.get(f"/v1/projects/{project_id}/spatial/plan-graph").json()
        assert pre_graph["source_doc_id"] == doc_a
        pre_asset = client.post(
            f"/v1/projects/{project_id}/spatial/design-3d",
            json={"plan_graph_id": pre_graph["id"], "force": False},
        ).json()

        activated = client.post(
            f"/v1/documents/{doc_b}/activate",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "Jordan Davis"},
        )
        assert activated.status_code == 200, activated.text
        assert activated.json()["regenerated_spatial"]["plan_graph_id"]
        graph = client.get(f"/v1/projects/{project_id}/spatial/plan-graph").json()
        assert graph["id"] != pre_graph["id"]
        assert graph["source_doc_id"] == doc_b
        provenance = graph["graph_json"]["provenance"]
        documents = client.get(f"/v1/projects/{project_id}/documents").json()
        current = next(document for document in documents if document["is_current"])
        assert current["doc_id"] == doc_b
        assert current["revision"] == "B"
        assert provenance["source_hash"] == current["hash"]
        assert provenance["source_revision"] == "B"
        assert all(
            source.get("doc_id") == doc_b
            for source in graph["graph_json"]["sources"]
            if source.get("doc_id")
        )
        assert client.get(f"/v1/spatial-assets/{pre_asset['id']}").status_code == 404
        asset = client.post(
            f"/v1/projects/{project_id}/spatial/design-3d",
            json={"plan_graph_id": graph["id"], "force": False},
        ).json()
        assert asset["id"] != pre_asset["id"]
        assert asset["metadata_json"]["source_doc_id"] == doc_b
        assert asset["metadata_json"]["source_hash"] == current["hash"]
