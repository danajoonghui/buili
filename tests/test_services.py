from urllib.parse import urlsplit

from fastapi.testclient import TestClient

from services.api.buili.gpu import assert_gpu_7, force_gpu_7, gpu_policy
from services.api.buili.main import app as api_app
from services.model_gateway.buili_model_gateway.main import app as model_app


def _create_analyzed_project(client: TestClient, name: str) -> dict:
    project = client.post(
        "/v1/projects",
        json={"name": name, "address": "QA", "project_type": "tenant_improvement"},
    ).json()
    body = (
        b"E1.1 Electrical Plans\n"
        b"AFCI protection is required for outlets in living and sleeping areas.\n"
        b"GFCI/WP exterior receptacles and smoke detector placement require field verification.\n"
    )
    presign = client.post(
        "/v1/uploads/presign",
        json={
            "project_id": project["project_id"],
            "filename": "e11-plan.txt",
            "mime": "text/plain",
            "size": len(body),
            "kind": "document",
        },
    )
    assert presign.status_code == 200
    upload = client.post(
        urlsplit(presign.json()["upload_url"]).path,
        files={"file": ("e11-plan.txt", body, "text/plain")},
    )
    assert upload.status_code == 200
    complete = client.post(
        urlsplit(presign.json()["complete_url"]).path,
        json={"document_type": "plan", "revision": "E1.1"},
    )
    assert complete.status_code == 200
    job = client.post(f"/v1/projects/{project['project_id']}/analyze", json={}).json()
    refreshed = client.get(f"/v1/jobs/{job['job_id']}").json()
    assert refreshed["state"] == "review_ready"
    return project


def test_api_healthz() -> None:
    client = TestClient(api_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "buili-api"


def test_api_root_serves_buili_app() -> None:
    client = TestClient(api_app)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Buili" in response.text


def test_api_prefix_alias_supports_single_render_domain() -> None:
    client = TestClient(api_app)
    response = client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json()["service"] == "buili-api"


def test_model_gateway_forces_gpu_7() -> None:
    client = TestClient(model_app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["cuda_visible_devices"] == "7"


def test_gpu_policy_forces_only_device_7(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert force_gpu_7() == "7"
    assert_gpu_7()
    assert gpu_policy()["cuda_visible_devices"] == "7"


def test_training_status_reports_completed_ai_stack() -> None:
    client = TestClient(api_app)
    response = client.get("/v1/training/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_training_progress_percent"] == 100
    assert payload["gpu_policy"]["cuda_visible_devices"] == "7"
    assert {item["key"] for item in payload["technologies"]} == {
        "pdf_rag",
        "plan_symbol",
        "media_recognition",
        "mismatch_candidates",
        "reports",
    }
    assert all(item["training_progress_percent"] == 100 for item in payload["technologies"])
    assert all(item["sha256"] and item["rows"] > 0 for item in payload["datasets"])


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


def test_rag_returns_answer_and_citations() -> None:
    client = TestClient(api_app)
    project = _create_analyzed_project(client, "RAG Roundtrip")
    response = client.get(
        f"/v1/projects/{project['project_id']}/rag/search",
        params={"q": "AFCI GFCI smoke detector outlet electrical plan"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["citations"]
    assert payload["returned_context"]
    assert payload["retrieval"]["rerank_top_k"] == 8


def test_report_types_generate_downloadable_artifacts() -> None:
    client = TestClient(api_app)
    project = _create_analyzed_project(client, "Report Roundtrip")
    requested = [
        ("punch", "csv", b"issue_id"),
        ("rfi", "md", b"Buili RFI Draft Package"),
        ("co_evidence", "xlsx", b"PK"),
    ]
    for report_type, fmt, marker in requested:
        response = client.post(
            f"/v1/projects/{project['project_id']}/reports",
            json={"report_type": report_type, "format": fmt},
        )
        assert response.status_code == 200
        report = response.json()
        download = client.get(urlsplit(report["download_url"]).path)
        assert download.status_code == 200
        assert marker in download.content[:4096]
