from fastapi.testclient import TestClient
from urllib.parse import urlsplit

from services.api.buili.main import app as api_app
from services.model_gateway.buili_model_gateway.main import app as model_app


def test_api_healthz() -> None:
    client = TestClient(api_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "buili-api"


def test_model_gateway_forces_gpu_7() -> None:
    client = TestClient(model_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["cuda_visible_devices"] == "7"


def test_document_upload_analysis_roundtrip() -> None:
    client = TestClient(api_app)
    project = client.post(
        "/v1/projects",
        json={"name": "Test Roundtrip", "address": "QA", "project_type": "tenant_improvement"},
    ).json()
    body = b"E-101\nConference Room 203 north wall requires two duplex outlets.\n"
    presign = client.post(
        "/v1/uploads/presign",
        json={
            "project_id": project["project_id"],
            "filename": "roundtrip-plan.txt",
            "mime": "text/plain",
            "size": len(body),
            "kind": "document",
        },
    )
    assert presign.status_code == 200
    upload_path = urlsplit(presign.json()["upload_url"]).path
    upload = client.post(upload_path, files={"file": ("roundtrip-plan.txt", body, "text/plain")})
    assert upload.status_code == 200
    complete_path = urlsplit(presign.json()["complete_url"]).path
    complete = client.post(complete_path, json={"document_type": "plan", "revision": "A"})
    assert complete.status_code == 200

    job = client.post(f"/v1/projects/{project['project_id']}/analyze", json={}).json()
    assert job["state"] in {"queued", "review_ready"}
    refreshed = client.get(f"/v1/jobs/{job['job_id']}").json()
    assert refreshed["state"] == "review_ready"
    issues = client.get(f"/v1/projects/{project['project_id']}/issues").json()
    assert issues
    serialized = str(issues)
    assert "stub" not in serialized
    assert issues[0]["evidence"]


def test_upload_rejects_unsupported_document_type() -> None:
    client = TestClient(api_app)
    project = client.post(
        "/v1/projects",
        json={"name": "Test Reject", "address": "QA", "project_type": "tenant_improvement"},
    ).json()
    response = client.post(
        "/v1/uploads/presign",
        json={
            "project_id": project["project_id"],
            "filename": "script.exe",
            "mime": "application/octet-stream",
            "size": 16,
            "kind": "document",
        },
    )
    assert response.status_code == 415
