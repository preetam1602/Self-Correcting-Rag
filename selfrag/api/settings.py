"""API-only settings (auth and uploads), read from the same .env as the core Settings."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from selfrag.config.settings import PROJECT_ROOT


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    admin_api_key: str | None = None        # X-API-Key header -> admin
    jwt_secret: str | None = None           # Bearer token with a tenant_id claim -> user
    jwt_algorithm: str = "HS256"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    max_upload_mb: int = 50
