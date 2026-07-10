from functools import lru_cache
import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BUILI_", env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: str = "sqlite:///./data/buili.db"
    storage_root: Path = Path("./data/storage")
    public_base_url: str = "http://localhost:8000"
    model_gateway_url: str = "http://localhost:8100"
    app_name: str = "Buili"
    cors_origins: str = "*"
    max_upload_bytes: int = 250 * 1024 * 1024
    redis_url: str = "redis://localhost:6379/0"
    spatial_enabled: bool = True
    spatial_alignment_min_confidence: float = 0.5
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    # Local/demo clients may omit identity headers.  Production deployments behind
    # a trusted OIDC/SAML proxy should set this to true and have the proxy inject
    # X-Buili-Actor and X-Buili-Role after stripping any client-supplied values.
    require_auth_headers: bool = False
    # First-party session authentication.  Render/production must enable this and
    # provide a random secret of at least 32 characters.
    auth_required: bool = False
    auth_secret: str = ""
    secure_cookies: bool = False
    session_hours: int = 12
    remember_session_days: int = 30
    pilot_seed_enabled: bool = True
    pilot_email: str = "jordan.davis@northstarbuild.example"
    pilot_password: str = ""
    pilot_name: str = "Jordan Davis"
    pilot_org_name: str = "Northstar Builders"
    pilot_project_name: str = "Cooper Residence — Electrical Rough-In Verification"

    @property
    def session_secret(self) -> str:
        # Development remains zero-config while production fails closed in the
        # lifespan validator before accepting authenticated traffic.
        return self.auth_secret or "buili-local-session-secret-not-for-production"

    @property
    def cors_allow_origins(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    try:
        settings.storage_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback = Path("/tmp/buili/storage")
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "BUILI_STORAGE_ROOT=%s is not writable; using ephemeral fallback %s",
            settings.storage_root,
            fallback,
        )
        settings.storage_root = fallback
    return settings
