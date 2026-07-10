"""Release contracts traced to Buili Product Specification sections 7, 9, 10, 16 and 19.

These tests intentionally exercise the HTTP boundary.  A green UI control is not
evidence that a review, revision, or chain-of-custody invariant is enforced.
"""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import fitz
from fastapi.testclient import TestClient
from PIL import Image

from services.api.buili.main import app as api_app


def _client() -> TestClient:
    return TestClient(api_app)


def _create_project(client: TestClient, label: str) -> dict:
    response = client.post(
        "/v1/projects",
        json={
            "name": f"Spec acceptance {label} {uuid4().hex[:8]}",
            "address": "QA",
            "project_type": "tenant_improvement",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload(
    client: TestClient,
    project_id: str,
    *,
    filename: str,
    body: bytes,
    revision: str,
    kind: str = "document",
    mime: str = "text/plain",
) -> tuple[dict, dict, dict]:
    presign_response = client.post(
        "/v1/uploads/presign",
        json={
            "project_id": project_id,
            "filename": filename,
            "mime": mime,
            "size": len(body),
            "kind": kind,
        },
    )
    assert presign_response.status_code == 200, presign_response.text
    presign = presign_response.json()
    upload_response = client.post(
        urlsplit(presign["upload_url"]).path,
        files={"file": (Path(filename).name, body, mime)},
    )
    assert upload_response.status_code == 200, upload_response.text
    complete_response = client.post(
        urlsplit(presign["complete_url"]).path,
        json={"document_type": "plan", "revision": revision},
    )
    assert complete_response.status_code == 200, complete_response.text
    return presign, upload_response.json(), complete_response.json()


def _analyzed_issue(client: TestClient, label: str) -> tuple[dict, dict]:
    project = _create_project(client, label)
    body = (
        b"E-204 CURRENT CONSTRUCTION DRAWING\n"
        b"Room 204 north wall requires a weatherproof GFCI receptacle.\n"
        b"Field photo observation requires PM verification before issue.\n"
    )
    _, _, completed = _upload(
        client,
        project["project_id"],
        filename="E-204.txt",
        body=body,
        revision="3",
    )
    activated = client.post(
        f"/v1/documents/{completed['document_id']}/activate",
        json={},
        headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
    )
    assert activated.status_code == 200, activated.text
    job_response = client.post(f"/v1/projects/{project['project_id']}/analyze", json={})
    assert job_response.status_code == 200, job_response.text
    job = job_response.json()
    refreshed = client.get(f"/v1/jobs/{job['job_id']}")
    assert refreshed.status_code == 200
    assert refreshed.json()["state"] == "review_ready"
    issues_response = client.get(f"/v1/projects/{project['project_id']}/issues")
    assert issues_response.status_code == 200
    issues = issues_response.json()
    assert issues
    return project, issues[0]


def _jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), color=(82, 104, 91)).save(output, format="JPEG")
    return output.getvalue()


def _link_complete_field_evidence(client: TestClient, project: dict, issue: dict) -> dict:
    original = _jpeg_bytes()
    digest = hashlib.sha256(original).hexdigest()
    capture_id = f"capture-{uuid4().hex}"
    synced = client.post(
        f"/v1/projects/{project['project_id']}/evidence/sync",
        json={
            "project_id": project["project_id"],
            "client_capture_id": capture_id,
            "media_type": "photo",
            "filename": "room-204-wide-angle.jpg",
            "mime": "image/jpeg",
            "content_base64": base64.b64encode(original).decode("ascii"),
            "sha256": digest,
            "hash": digest,
            "captured_at": datetime.now(UTC).isoformat(),
            "author": "qa-field-user",
            "observation": "Wide-angle location-confirmed context photograph.",
            "location": {"floor": "Level 02", "room": issue["room"], "method": "manual"},
            "location_method": "manual",
        },
        headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "qa-field-user"},
    )
    assert synced.status_code in {200, 201}, synced.text
    evidence = synced.json()
    linked = client.post(
        f"/v1/evidence/{evidence['evidence_id']}/link",
        json={
            "issue_id": issue["issue_id"],
            "relevance": "supports",
            "annotation": "Location and condition are visible.",
        },
        headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "qa-field-user"},
    )
    assert linked.status_code == 200, linked.text
    return evidence


def _event_action(event: dict) -> str:
    return str(event.get("action") or event.get("event_type") or "").upper()


def test_upload_filename_is_normalized_and_original_hash_is_preserved() -> None:
    """Spec 7.2/16: raw source names cannot escape storage and bytes retain SHA-256."""

    with _client() as client:
        project = _create_project(client, "upload-integrity")
        body = b"A-101 revision A immutable source bytes\n"
        presign, uploaded, completed = _upload(
            client,
            project["project_id"],
            filename="../../A-101.txt",
            body=body,
            revision="A",
        )

        assert ".." not in presign["r2_key"]
        assert presign["r2_key"].endswith("_A-101.txt")
        assert f"/project/{project['project_id']}/raw/" in presign["r2_key"]
        assert uploaded["sha256"] == hashlib.sha256(body).hexdigest()

        documents = client.get(f"/v1/projects/{project['project_id']}/documents").json()
        document = next(item for item in documents if item["doc_id"] == completed["document_id"])
        assert document["filename"] == "A-101.txt"
        assert document["hash"] == hashlib.sha256(body).hexdigest()


def test_upload_size_mismatch_is_rejected_and_cannot_be_completed() -> None:
    """A truncated offline/network upload must never become evidence or a source."""

    with _client() as client:
        project = _create_project(client, "truncated-upload")
        body = b"partial source"
        presign_response = client.post(
            "/v1/uploads/presign",
            json={
                "project_id": project["project_id"],
                "filename": "A-102.txt",
                "mime": "text/plain",
                "size": len(body) + 3,
                "kind": "document",
            },
        )
        assert presign_response.status_code == 200
        presign = presign_response.json()

        upload = client.post(
            urlsplit(presign["upload_url"]).path,
            files={"file": ("A-102.txt", body, "text/plain")},
        )
        assert upload.status_code == 400
        complete = client.post(
            urlsplit(presign["complete_url"]).path,
            json={"document_type": "plan", "revision": "A"},
        )
        assert complete.status_code == 409


def test_project_resources_return_not_found_for_unknown_project() -> None:
    """Spec 16: project scope must be validated consistently, including empty lists."""

    unknown = f"prj_missing_{uuid4().hex}"
    with _client() as client:
        for path in (
            f"/v1/projects/{unknown}/documents",
            f"/v1/projects/{unknown}/media",
            f"/v1/projects/{unknown}/observations",
            f"/v1/projects/{unknown}/issues",
            f"/v1/projects/{unknown}/rag/search?q=current",
            f"/v1/projects/{unknown}/audit-events",
            f"/v1/projects/{unknown}/reports",
        ):
            response = client.get(path)
            assert response.status_code == 404, f"{path}: {response.status_code} {response.text}"


def test_report_download_rejects_encoded_path_traversal() -> None:
    """Spec 16: a report URL may never read objects outside the report root."""

    with _client() as client:
        for path in (
            "/v1/reports/%2e%2e%2f%2e%2e%2f%2e%2e%2fREADME.md",
            "/v1/reports/..%2F..%2F..%2F.env.example",
        ):
            response = client.get(path)
            assert response.status_code in {400, 404}, response.text[:200]
            assert b"Buili" not in response.content


def test_human_review_gate_blocks_direct_official_transition_and_export() -> None:
    """Spec 1.3/9.4/19.2: no official action exists without an approval record."""

    with _client() as client:
        project, issue = _analyzed_issue(client, "human-gate")

        direct_issue = client.patch(
            f"/v1/issues/{issue['issue_id']}",
            json={"status": "issued"},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert direct_issue.status_code in {409, 422}

        draft = client.post(
            f"/v1/projects/{project['project_id']}/reports",
            json={"report_type": "rfi", "format": "pdf"},
        )
        assert draft.status_code == 200, draft.text
        blocked_export = client.post(
            f"/v1/reports/{draft.json()['report_id']}/export",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert blocked_export.status_code == 409

        premature_review = client.post(
            f"/v1/issues/{issue['issue_id']}/reviews",
            json={
                "decision": "approve",
                "reviewer": "qa-pm",
                "reason": "Attempted before evidence was complete.",
            },
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert premature_review.status_code == 409

        _link_complete_field_evidence(client, project, issue)
        review = client.post(
            f"/v1/issues/{issue['issue_id']}/reviews",
            json={
                "decision": "approve",
                "reviewer": "qa-pm",
                "reason": "Current source and field evidence verified.",
            },
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert review.status_code == 200, review.text
        review_payload = review.json()
        assert review_payload["decision"] == "approve"
        assert review_payload["reviewer"] == "qa-pm"
        assert review_payload["reason"]
        assert review_payload.get("timestamp") or review_payload.get("created_at")

        exported = client.post(
            f"/v1/reports/{draft.json()['report_id']}/export",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert exported.status_code == 200, exported.text
        assert exported.json()["status"] == "issued"


def test_review_decisions_are_role_guarded_structured_and_auditable() -> None:
    """Spec 9.4/16.1: decision reason and before/after survive independently of UI."""

    with _client() as client:
        project, issue = _analyzed_issue(client, "review-audit")
        endpoint = f"/v1/issues/{issue['issue_id']}/reviews"
        denied = client.post(
            endpoint,
            json={
                "decision": "reject",
                "reviewer": "field-user",
                "reason": "Wrong source.",
                "reason_code": "wrong_source",
            },
            headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "field-user"},
        )
        assert denied.status_code == 403

        requested = client.post(
            endpoint,
            json={
                "decision": "request_evidence",
                "reviewer": "qa-pm",
                "reason": "Add a wide-angle context photo.",
                "reason_code": "insufficient_evidence",
            },
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert requested.status_code == 200, requested.text
        assert requested.json()["decision"] == "request_evidence"

        events_response = client.get(f"/v1/projects/{project['project_id']}/audit-events")
        assert events_response.status_code == 200
        events = events_response.json()
        review_events = [event for event in events if _event_action(event) == "REVIEW_DECIDED"]
        assert review_events
        event = review_events[-1]
        assert event["actor"] == "qa-pm"
        assert event["timestamp"]
        assert event["before"] is not None
        assert event["after"] is not None
        assert "request_evidence" in str(event["after"])


def test_revision_activation_preserves_history_and_flags_affected_issue() -> None:
    """Spec 7.2: activation supersedes, never replaces, an issue's cited source."""

    with _client() as client:
        project, issue = _analyzed_issue(client, "revision-integrity")
        old_documents = client.get(f"/v1/projects/{project['project_id']}/documents").json()
        old_document = next(item for item in old_documents if item["filename"] == "E-204.txt")

        _, _, completed = _upload(
            client,
            project["project_id"],
            filename="E-204.txt",
            body=(
                b"E-204 CURRENT CONSTRUCTION DRAWING REVISION 4\n"
                b"Room 204 north wall receptacle note revised after addendum.\n"
            ),
            revision="4",
        )
        new_document_id = completed["document_id"]
        activated = client.post(
            f"/v1/documents/{new_document_id}/activate",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert activated.status_code == 200, activated.text
        activation = activated.json()
        assert activation["current_document_id"] == new_document_id
        assert old_document["doc_id"] in activation["superseded_document_ids"]
        assert issue["issue_id"] in activation["affected_issue_ids"]

        documents = client.get(f"/v1/projects/{project['project_id']}/documents").json()
        assert {old_document["doc_id"], new_document_id} <= {item["doc_id"] for item in documents}
        old_after = next(item for item in documents if item["doc_id"] == old_document["doc_id"])
        new_after = next(item for item in documents if item["doc_id"] == new_document_id)
        assert old_after["hash"] == old_document["hash"]
        assert old_after["revision"] == "3"
        assert new_after["revision"] == "4"
        assert old_after.get("is_current") is False
        assert new_after.get("is_current") is True

        affected_issue = next(
            item
            for item in client.get(f"/v1/projects/{project['project_id']}/issues").json()
            if item["issue_id"] == issue["issue_id"]
        )
        risk_flags = affected_issue.get("risk_flags") or affected_issue.get("requirement", {}).get(
            "risk_flags", []
        )
        assert affected_issue.get("source_status") == "stale" or "stale_source" in risk_flags

        events = client.get(f"/v1/projects/{project['project_id']}/audit-events").json()
        revision_events = [
            event for event in events if _event_action(event) == "REVISION_ACTIVATED"
        ]
        assert revision_events
        assert old_document["doc_id"] in str(revision_events[-1]["before"])
        assert new_document_id in str(revision_events[-1]["after"])


def test_offline_capture_sync_is_idempotent_and_hash_checked() -> None:
    """Spec 8.2/19.2: retrying a local capture cannot lose or duplicate evidence."""

    with _client() as client:
        project = _create_project(client, "offline-sync")
        original = b"offline-field-photo-original-bytes"
        digest = hashlib.sha256(original).hexdigest()
        capture_id = f"capture-{uuid4().hex}"
        payload = {
            "project_id": project["project_id"],
            "client_capture_id": capture_id,
            "filename": "room-204-context.jpg",
            "mime": "image/jpeg",
            "content_base64": base64.b64encode(original).decode("ascii"),
            "sha256": digest,
            "hash": digest,
            "captured_at": datetime.now(UTC).isoformat(),
            "author": "qa-field-user",
            "observation": "Wide-angle context before rough-in closeout.",
            "location": {"floor": "Level 02", "room": "204", "method": "manual"},
        }
        endpoint = f"/v1/projects/{project['project_id']}/evidence/sync"

        first = client.post(
            endpoint,
            json=payload,
            headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "qa-field-user"},
        )
        assert first.status_code in {200, 201}, first.text
        first_payload = first.json()
        evidence_id = first_payload.get("evidence_id") or first_payload.get("media_id")
        assert evidence_id
        assert first_payload["sha256"] == digest

        replay = client.post(
            endpoint,
            json=payload,
            headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "qa-field-user"},
        )
        assert replay.status_code == 200, replay.text
        replay_payload = replay.json()
        assert (replay_payload.get("evidence_id") or replay_payload.get("media_id")) == evidence_id
        assert (
            replay_payload.get("deduplicated") is True
            or replay_payload.get("idempotent_replay") is True
        )

        changed = {
            **payload,
            "content_base64": base64.b64encode(original + b"changed").decode("ascii"),
        }
        conflict = client.post(
            endpoint,
            json=changed,
            headers={"X-Buili-Role": "field_user", "X-Buili-Actor": "qa-field-user"},
        )
        assert conflict.status_code == 409

        media = client.get(f"/v1/projects/{project['project_id']}/media").json()
        matching = [
            item
            for item in media
            if item.get("metadata_json", {}).get("client_capture_id") == capture_id
        ]
        assert len(matching) == 1
        assert matching[0]["hash"] == digest
        assert matching[0]["metadata_json"]["location"]["room"] == "204"


def test_issued_report_version_contains_review_source_snapshot_and_checksum() -> None:
    """Spec 10.3/19.2: an issued PDF is source-backed, review-backed and immutable."""

    with _client() as client:
        project, issue = _analyzed_issue(client, "report-version")
        _link_complete_field_evidence(client, project, issue)
        review = client.post(
            f"/v1/issues/{issue['issue_id']}/reviews",
            json={
                "decision": "approve",
                "reviewer": "qa-pm",
                "reason": "Package is ready to issue.",
            },
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert review.status_code == 200, review.text
        draft = client.post(
            f"/v1/projects/{project['project_id']}/reports",
            json={"report_type": "rfi", "format": "pdf"},
        )
        assert draft.status_code == 200, draft.text
        issued = client.post(
            f"/v1/reports/{draft.json()['report_id']}/export",
            json={},
            headers={"X-Buili-Role": "project_manager", "X-Buili-Actor": "qa-pm"},
        )
        assert issued.status_code == 200, issued.text

        listing_response = client.get(f"/v1/projects/{project['project_id']}/reports")
        assert listing_response.status_code == 200
        reports = listing_response.json()
        report = next(item for item in reports if item["report_id"] == draft.json()["report_id"])
        assert report["status"] == "issued"
        versions_response = client.get(f"/v1/reports/{report['report_id']}/versions")
        assert versions_response.status_code == 200
        version = next(item for item in versions_response.json() if item["status"] == "issued")
        assert int(version["version"]) >= 1
        assert version["reviewer"] == "qa-pm"
        assert version["checksum"]
        assert len(version["checksum"]) == 64
        assert version["source_snapshot"]

        download_url = version.get("download_url") or issued.json().get("download_url")
        download = client.get(urlsplit(download_url).path)
        assert download.status_code == 200
        assert hashlib.sha256(download.content).hexdigest() == version["checksum"]
        with fitz.open(stream=download.content, filetype="pdf") as document:
            text = "\n".join(page.get_text() for page in document)
        assert issue["issue_id"] in text
        assert "qa-pm" in text
        assert "Version" in text
        assert str(issue["requirement"].get("source", "")).strip() in text
