from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCALLAB_", env_file=(".env", "../.env"), extra="ignore")

    data_dir: Path = Path("./data")
    database_url: str = "sqlite:///./data/locallab.db"
    api_host: str = "127.0.0.1"
    api_port: int = 8321
    cors_origins: list[str] = ["http://localhost:3210", "http://127.0.0.1:3210"]
    log_level: str = "INFO"
    log_json: bool = False
    hf_cache_dir: Path = Path("~/.cache/huggingface/hub")
    ollama_url: str = "http://127.0.0.1:11434"
    # Hosts experiments may send inference traffic to (guards against SSRF via the API).
    allowed_runtime_hosts: list[str] = ["127.0.0.1", "localhost", "::1"]
    docker_bin: str = "docker"
    vllm_image: str = "vllm/vllm-openai:latest"
    managed_startup_timeout_s: float = 600.0
    worker_poll_s: float = 1.0
    worker_stale_s: float = 30.0

    @property
    def instance_id_path(self) -> Path:
        return self.data_dir / "instance_id"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"


@lru_cache
def get_settings() -> Settings:
    return Settings()
