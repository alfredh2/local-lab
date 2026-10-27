from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from locallab.config import get_settings

Profile = Literal["short_chat", "long_prefill", "long_decode", "concurrent", "custom"]

# Presets fill the form; the stored config is always fully explicit.
PRESETS: dict[str, dict] = {
    "short_chat": {"input_tokens": 128, "output_tokens": 256, "num_requests": 32, "concurrency": 1, "warmup_requests": 4},
    "long_prefill": {"input_tokens": 4096, "output_tokens": 32, "num_requests": 16, "concurrency": 1, "warmup_requests": 2},
    "long_decode": {"input_tokens": 128, "output_tokens": 1024, "num_requests": 8, "concurrency": 1, "warmup_requests": 2},
    "concurrent": {"input_tokens": 256, "output_tokens": 128, "num_requests": 64, "concurrency": 16, "warmup_requests": 16},
}


class InferenceBenchmarkConfig(BaseModel):
    runtime: Literal["vllm"] = "vllm"
    server_url: str = Field("http://127.0.0.1:8100", description="OpenAI-compatible vLLM server (must be an allowed host)")
    model: str = Field(..., min_length=1, description="Served model name as reported by /v1/models")
    registry_model_id: int | None = Field(None, description="Optional LocalLab registry model this server is running")
    profile: Profile = "custom"
    prompt_mode: Literal["token_ids", "text"] = Field(
        "token_ids", description="token_ids: exact input length (random ids, server-verified). text: approximate length")
    input_tokens: int = Field(128, ge=1, le=262_144)
    output_tokens: int = Field(128, ge=1, le=32_768)
    num_requests: int = Field(32, ge=1, le=5000, description="Measured requests per iteration")
    concurrency: int = Field(1, ge=1, le=512, description="Closed-loop in-flight requests")
    warmup_requests: int = Field(4, ge=0, le=200)
    iterations: int = Field(1, ge=1, le=20, description="Repeat the measured phase this many times")
    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int | None = Field(None, ge=1)
    seed: int = 0
    request_timeout_s: float = Field(300, ge=5, le=3600)
    abort_after_failures: int = Field(10, ge=1, le=1000, description="Abort after this many consecutive failed requests")

    @field_validator("server_url")
    @classmethod
    def _allowed_host(cls, v: str) -> str:
        u = urlparse(v)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError("server_url must be an http(s) URL")
        allowed = get_settings().allowed_runtime_hosts
        if u.hostname not in allowed:
            raise ValueError(f"host {u.hostname!r} is not in LOCALLAB_ALLOWED_RUNTIME_HOSTS {allowed}")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _warmup_sane(self) -> "InferenceBenchmarkConfig":
        if self.profile != "custom" and self.profile not in PRESETS:
            raise ValueError(f"unknown profile {self.profile}")
        return self
