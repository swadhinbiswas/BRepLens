"""Service configuration (environment-driven, no secrets in the repo)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BREPLENS_", env_file=".env", extra="ignore")

    data_dir: Path = ROOT / "data" / "service"
    database_url: str = ""
    max_upload_mb: int = 25
    queue_backend: str = "sqlite"  # sqlite | rq
    redis_url: str = "redis://localhost:6379/0"
    model_path: Path = ROOT / "models" / "gnn_synthetic.pt"
    similarity_index: Path = ROOT / "demo" / "library.json"
    default_mode: str = "hybrid"
    model_device: str = "cpu"
    worker_poll_seconds: float = 1.0
    log_level: str = "INFO"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'breplens.db').as_posix()}"

    def ensure_dirs(self) -> None:
        (self.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "results").mkdir(parents=True, exist_ok=True)
