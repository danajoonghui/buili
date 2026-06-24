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
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""

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
