"""Optimization matrix configuration, expansion and static feasibility.

The matrix is the cross product of the axes that require a *server restart* (model, context length,
weight quantization, KV-cache dtype). Everything that does not (concurrency, input/output lengths, which
evals) is measured inside one server launch.
"""
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse  # noqa: F401  (kept for parity with SweepConfig validators)

from pydantic import BaseModel, Field, field_validator, model_validator

from locallab.evals.registry import SUITES
from locallab.runtimes.vllm_compat import ManagedVllmConfig

RETRIEVAL_HEADROOM = 256  # prompt + answer must fit with this much slack inside max_model_len
MAX_TOTAL_PERF_REQUESTS = 60_000


class IoPair(BaseModel):
    input_tokens: int = Field(128, ge=1, le=262_144)
    output_tokens: int = Field(128, ge=1, le=32_768)


def _unique(v: list, what: str) -> list:
    out = list(dict.fromkeys(v))
    if not out:
        raise ValueError(f"{what} must not be empty")
    return out


class MatrixConfig(BaseModel):
    # --- axes that need a server restart ---
    registry_model_ids: list[int] = Field(..., description="Registry (Hugging Face cache) models to compare")
    max_model_lens: list[int] = Field(default_factory=lambda: [8192])
    quantizations: list[Literal["none", "awq", "gptq", "fp8"]] = Field(default_factory=lambda: ["none"])
    kv_cache_dtypes: list[Literal["auto", "fp8"]] = Field(default_factory=lambda: ["auto"])
    # --- fixed runtime settings shared by all configurations ---
    gpu_memory_utilization: float = Field(0.30, ge=0.05, le=0.95)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_num_seqs: int = Field(64, ge=1, le=1024)
    enforce_eager: bool = False
    gpu_index: int = Field(0, ge=0, le=15)
    startup_timeout_s: float = Field(600, ge=5, le=3600)
    allow_contention: bool = False
    # --- synthetic performance, measured inside each server launch ---
    perf_enabled: bool = True
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 4, 16])
    io_pairs: list[IoPair] = Field(default_factory=lambda: [IoPair()])
    requests_per_level: int = Field(32, ge=1, le=2000)
    min_requests_per_slot: int = Field(2, ge=1, le=100)
    warmup_requests: int = Field(4, ge=0, le=100)
    iterations: int = Field(2, ge=1, le=10)
    # --- quality evaluation on real tasks ---
    evals: list[Literal["structured_json", "coding_python", "retrieval_needle"]] = Field(default_factory=lambda: list(SUITES))
    eval_split: Literal["heldout", "dev"] = Field("heldout", description="Compare configurations on held-out cases; dev is for iterating on prompts")
    eval_concurrency: int = Field(8, ge=1, le=64)
    eval_timeout_s: float = Field(180, ge=5, le=1800)
    enable_thinking: bool = False
    retrieval_lengths: list[int] = Field(default_factory=lambda: [2048, 4096])
    retrieval_positions: list[float] = Field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    retrieval_repeats: int = Field(2, ge=1, le=10)
    # --- limits ---
    max_configurations: int = Field(12, ge=1, le=64)
    max_runtime_s: int = Field(3600, ge=10, le=172_800, description="No new configuration starts after this much wall time")
    config_timeout_s: int = Field(1800, ge=10, le=86_400)
    temperature: float = 0.0
    seed: int = 0
    request_timeout_s: float = Field(300, ge=5, le=3600)
    abort_after_failures: int = Field(10, ge=1, le=1000)

    @field_validator("registry_model_ids", "max_model_lens", "quantizations", "kv_cache_dtypes", "retrieval_lengths", "retrieval_positions")
    @classmethod
    def _dedupe(cls, v: list) -> list:
        return _unique(v, "axis")

    @field_validator("evals")
    @classmethod
    def _evals(cls, v: list) -> list:
        return list(dict.fromkeys(v))  # may be empty when only performance is measured

    @field_validator("concurrency_levels")
    @classmethod
    def _levels(cls, v: list[int]) -> list[int]:
        v = sorted(set(v))
        if not v or len(v) > 8 or any(not 1 <= c <= 512 for c in v):
            raise ValueError("concurrency_levels needs 1-8 values between 1 and 512")
        return v

    @field_validator("io_pairs")
    @classmethod
    def _io(cls, v: list[IoPair]) -> list[IoPair]:
        seen, out = set(), []
        for p in v:
            k = (p.input_tokens, p.output_tokens)
            if k not in seen:
                seen.add(k)
                out.append(p)
        if not out or len(out) > 4:
            raise ValueError("io_pairs needs 1-4 distinct pairs")
        return out

    @field_validator("retrieval_positions")
    @classmethod
    def _pos(cls, v: list[float]) -> list[float]:
        if any(not 0 <= p <= 1 for p in v):
            raise ValueError("retrieval_positions must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _shape(self) -> "MatrixConfig":
        if not self.perf_enabled and not self.evals:
            raise ValueError("enable performance measurement and/or at least one eval suite")
        n = len(expand(self))
        if n > self.max_configurations:
            raise ValueError(f"the matrix expands to {n} configurations, above max_configurations={self.max_configurations}; "
                             "narrow the axes or raise the limit")
        total = sum(self.level_requests(c) * self.iterations + self.level_warmup(c) for c in self.concurrency_levels) * len(self.io_pairs) * n
        if self.perf_enabled and total > MAX_TOTAL_PERF_REQUESTS:
            raise ValueError(f"performance workload exceeds {MAX_TOTAL_PERF_REQUESTS} requests in total")
        return self

    def level_requests(self, c: int) -> int:
        return min(2000, max(self.requests_per_level, c * self.min_requests_per_slot))

    def level_warmup(self, c: int) -> int:
        return min(100, max(self.warmup_requests, min(c, 64)))

    def runtime_for(self, sc: "ServerConfig") -> ManagedVllmConfig:
        return ManagedVllmConfig(max_model_len=sc.max_model_len, gpu_memory_utilization=self.gpu_memory_utilization, dtype=self.dtype,
                                 quantization=sc.quantization, kv_cache_dtype=sc.kv_cache_dtype, max_num_seqs=self.max_num_seqs,
                                 enforce_eager=self.enforce_eager, gpu_index=self.gpu_index, startup_timeout_s=self.startup_timeout_s,
                                 allow_contention=self.allow_contention)


@dataclass
class ServerConfig:
    """One server launch: the loading-time configuration."""

    index: int
    model_id: int
    max_model_len: int
    quantization: str
    kv_cache_dtype: str
    model_name: str = ""
    revision: str = ""
    key: str = ""
    notes: list[str] = field(default_factory=list)  # cell-level skips decided from max_model_len

    @property
    def label(self) -> str:
        parts = [self.model_name or f"model {self.model_id}", f"ctx {self.max_model_len}"]
        if self.quantization != "none":
            parts.append(f"w:{self.quantization}")
        if self.kv_cache_dtype != "auto":
            parts.append(f"kv:{self.kv_cache_dtype}")
        return " · ".join(parts)

    def loading_config(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "model_name": self.model_name, "revision": self.revision, "max_model_len": self.max_model_len,
                "quantization": self.quantization, "kv_cache_dtype": self.kv_cache_dtype}


def config_key(model_id: int, revision: str, max_model_len: int, quantization: str, kv_cache_dtype: str) -> str:
    blob = json.dumps([model_id, revision, max_model_len, quantization, kv_cache_dtype])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def expand(cfg: MatrixConfig) -> list[ServerConfig]:
    """Cross product of the restart-requiring axes, model-major so one model's launches are contiguous
    (its weights stay in the OS page cache), duplicates removed."""
    out: list[ServerConfig] = []
    seen: set[tuple] = set()
    for m, ctx, q, kv in itertools.product(cfg.registry_model_ids, cfg.max_model_lens, cfg.quantizations, cfg.kv_cache_dtypes):
        k = (m, ctx, q, kv)
        if k in seen:
            continue
        seen.add(k)
        out.append(ServerConfig(index=len(out), model_id=m, max_model_len=ctx, quantization=q, kv_cache_dtype=kv))
    return out


def cell_plan(cfg: MatrixConfig, sc: ServerConfig) -> dict[str, Any]:
    """Which perf cells and eval lengths fit this server's context length (the rest are skipped, with reasons)."""
    notes: list[str] = []
    io_ok = []
    for p in cfg.io_pairs:
        if p.input_tokens + p.output_tokens <= sc.max_model_len:
            io_ok.append(p)
        else:
            notes.append(f"performance cell {p.input_tokens}/{p.output_tokens} skipped: exceeds max_model_len {sc.max_model_len}")
    lengths = [L for L in cfg.retrieval_lengths if L + RETRIEVAL_HEADROOM <= sc.max_model_len]
    skipped = [L for L in cfg.retrieval_lengths if L not in lengths]
    if skipped:
        notes.append(f"retrieval lengths {skipped} skipped: need prompt + {RETRIEVAL_HEADROOM} tokens <= max_model_len {sc.max_model_len}")
    evals = [e for e in cfg.evals if not (e == "retrieval_needle" and not lengths)]
    if "retrieval_needle" in cfg.evals and not lengths:
        notes.append("retrieval suite skipped: no configured length fits this context")
    if cfg.perf_enabled and not io_ok:
        notes.append("no performance cell fits this context")
    return {"io_pairs": io_ok, "retrieval_lengths": lengths, "evals": evals, "notes": notes}
