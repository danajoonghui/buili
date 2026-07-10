"""Deployment contracts for the authenticated Render pilot.

These checks keep credentials server-only and make the web health probe exercise
the same-origin trust boundary used by real browser requests.
"""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _blueprint() -> dict:
    return yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))


def _service(name: str) -> dict:
    return next(service for service in _blueprint()["services"] if service["name"] == name)


def _env(service: dict) -> dict[str, dict]:
    return {item["key"]: item for item in service.get("envVars", [])}


def test_render_api_fails_closed_with_server_only_session_secrets() -> None:
    api = _service("buili-api")
    env = _env(api)

    assert str(env["BUILI_AUTH_REQUIRED"]["value"]).lower() == "true"
    assert str(env["BUILI_SECURE_COOKIES"]["value"]).lower() == "true"
    assert env["BUILI_AUTH_SECRET"] == {
        "key": "BUILI_AUTH_SECRET",
        "generateValue": True,
    }
    assert env["BUILI_PILOT_PASSWORD"] == {
        "key": "BUILI_PILOT_PASSWORD",
        "sync": False,
    }
    assert str(env["BUILI_PILOT_SEED_ENABLED"]["value"]).lower() == "true"
    assert env["BUILI_PILOT_EMAIL"]["value"] == "jordan.davis@northstarbuild.example"
    assert env["BUILI_PILOT_NAME"]["value"] == "Jordan Davis"
    assert env["BUILI_PILOT_ORG_NAME"]["value"] == "Northstar Builders"
    assert (
        env["BUILI_PILOT_PROJECT_NAME"]["value"]
        == "Cooper Residence — Electrical Rough-In Verification"
    )
    assert api["healthCheckPath"] == "/readyz"
    assert api["disk"]["mountPath"] == "/var/data"
    assert env["BUILI_DATABASE_URL"]["fromDatabase"]["name"] == "buili-postgres"
    assert env["BUILI_CORS_ORIGINS"]["value"].startswith("https://")
    assert "*" not in env["BUILI_CORS_ORIGINS"]["value"]


def test_render_web_health_probe_crosses_the_private_api_proxy() -> None:
    web = _service("buili-web")
    env = _env(web)

    assert web["healthCheckPath"] == "/api/readyz"
    assert env["NEXT_PUBLIC_API_URL"]["value"] == "/api"
    assert env["BUILI_INTERNAL_API_URL"]["fromService"]["name"] == "buili-api"
    assert all(
        not key.startswith("NEXT_PUBLIC_")
        or not any(fragment in key for fragment in ("PASSWORD", "SECRET", "TOKEN"))
        for key in env
    )


def test_example_environment_contains_no_pilot_password() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "BUILI_AUTH_REQUIRED=false" in example
    assert "BUILI_SECURE_COOKIES=false" in example
    assert "BUILI_PILOT_PASSWORD=\n" in example
    assert "NEXT_PUBLIC_BUILI_PILOT_PASSWORD" not in example


def test_render_build_keeps_seed_source_and_excludes_training_outputs() -> None:
    source = ROOT / "data" / "sources" / "utah-e11-electrical-plans.pdf"
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert source.is_file() and source.stat().st_size > 1_000_000
    assert source.read_bytes()[:4] == b"%PDF"
    assert "data/sources/utah-e11-electrical-plans.pdf" not in ignore
    assert "/runs/" in ignore
    assert "/yolo*.pt" in ignore
