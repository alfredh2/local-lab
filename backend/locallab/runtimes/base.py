"""Runtime adapter interface.

Adapters talk to an inference runtime and hide protocol details from experiments.
Today only externally managed servers exist; a managed adapter (start/stop the
server process) can later add `launch()`/`shutdown()` without changing `generate`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class RuntimeError_(Exception):
    """Runtime is unreachable or returned something unusable (message is user-facing)."""


@dataclass
class ModelInfo:
    id: str
    root: str | None = None
    max_model_len: int | None = None


@dataclass
class ServerInfo:
    runtime: str
    base_url: str
    version: str | None
    models: list[ModelInfo]
    kv_cache_tokens: int | None = None  # num_gpu_blocks * block_size, if the server exposes it
    extra: dict[str, Any] = field(default_factory=dict)  # raw, runtime-reported configuration


@dataclass
class GenerationRequest:
    model: str
    prompt: str | list[int]  # ignored when `messages` is set
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None
    force_length: bool = True  # generate exactly max_tokens (ignore EOS)
    timeout_s: float = 300.0
    messages: list[dict[str, str]] | None = None  # chat request: the server applies the model's chat template
    chat_template_kwargs: dict[str, Any] | None = None


@dataclass
class RequestOutcome:
    """Raw timings use time.perf_counter() seconds; derived metrics are computed elsewhere."""

    ok: bool
    t_start: float
    t_end: float
    t_first_token: float | None = None
    chunk_times: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    token_source: str = "server"  # "server" (usage object) | "estimated" (counted stream chunks)
    finish_reason: str | None = None
    error_type: str | None = None  # timeout|connection|connection_lost|oom|bad_request|server_error|incomplete_stream|cancelled
    error_message: str | None = None
    http_status: int | None = None
    text: str = ""  # generated text (evaluation scoring)


class RuntimeAdapter(ABC):
    name: str

    @abstractmethod
    async def probe(self) -> ServerInfo:
        """Return server identity/limits. Raise RuntimeError_ if unreachable."""

    @abstractmethod
    async def generate(self, req: GenerationRequest) -> RequestOutcome:
        """Stream one completion and time it. Failures are returned, not raised; cancellation propagates."""

    @abstractmethod
    async def aclose(self) -> None: ...
