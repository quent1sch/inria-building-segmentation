"""
config.py

Centralised configuration via pydantic-settings.
Every environment-specific value is read from env vars with sensible
local defaults — no code changes needed to move from local to Azure.

Local (zero config needed):
    STORAGE_BACKEND=local          (default)
    DATABASE_URL=sqlite+aiosqlite:///jobs.db   (default)
    WORKER_MODE=thread             (default)

Azure production (only these change):
    STORAGE_BACKEND=azure
    AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
    AZURE_STORAGE_CONTAINER=segmentation-results
    DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname
    WORKER_MODE=queue
    AZURE_QUEUE_CONNECTION_STRING=...
    AZURE_QUEUE_NAME=inference-jobs

Auth:
    DEFAULT_USER_ID=local # single-user local default
    API_KEY=              # optional — if set, require X-API-Key header
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ───────────────────────────────────────────────────────────────
    app_name:    str = "Inria Building Segmentation API"
    app_version: str = "2.0.0"
    debug:       bool = False

    # ── Auth ──────────────────────────────────────────────────────────────
    # Single-user local default. In production, extract from JWT/API key.
    default_user_id: str = "local"
    # If set, all requests must include X-API-Key: <value>
    api_key: str = ""

    # ── Model ─────────────────────────────────────────────────────────────
    checkpoint_path: str  = "checkpoints/best_model.pth"
    config_path:     str  = "configs/config.yaml"

    # ── Storage ───────────────────────────────────────────────────────────
    storage_backend: Literal["local", "azure"] = "local"

    # Local storage
    storage_local_root: str = "data/results"

    # Azure Blob Storage
    # Connection string from Azure Portal -> Storage Account -> Access keys
    azure_storage_connection_string: str = ""
    azure_storage_container:         str = "segmentation-results"

    # ── Database ──────────────────────────────────────────────────────────
    # SQLite for local, PostgreSQL for cloud
    database_url: str = "sqlite+aiosqlite:///jobs.db"

    # ── Worker ────────────────────────────────────────────────────────────
    worker_mode: Literal["thread", "queue"] = "thread"
    # Thread pool size (thread mode only)
    worker_threads: int = 2

    # Azure Queue Storage (queue mode only)
    azure_queue_connection_string: str = ""
    azure_queue_name:              str = "inference-jobs"
    # How long a worker has to process a job before it becomes visible again
    queue_visibility_timeout_s: int = 300

    # ── Result caching ────────────────────────────────────────────────────
    cache_enabled:   bool = True
    # Signed URL expiry for Azure Blob (seconds)
    result_url_expiry_s: int = 3600

    # ── Upload limits ─────────────────────────────────────────────────────
    # Max upload size for multipart endpoint (bytes). Default 300MB.
    max_upload_bytes: int = 300 * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """
    Return cached Settings instance.
    lru_cache: computed once per process. env vars read at startup only.
    """
    return Settings()