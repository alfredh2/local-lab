"""Feasibility checks shared by API validation and job start."""
import logging
from dataclasses import dataclass, field

from sqlalchemy import select

from locallab.db import session_scope
from locallab.experiments.inference.config import InferenceBenchmarkConfig
from locallab.models import ModelRecord
from locallab.runtimes.base import RuntimeError_, ServerInfo
from locallab.runtimes.vllm_openai import VllmOpenAIAdapter

log = logging.getLogger(__name__)


class PreflightError(ValueError):
    """The requested workload cannot run as configured (message is user-facing)."""


@dataclass
class Preflight:
    info: ServerInfo
    served_model: dict
    registry_model: dict | None
    warnings: list[str] = field(default_factory=list)


async def run_preflight(cfg: InferenceBenchmarkConfig, adapter: VllmOpenAIAdapter | None = None) -> Preflight:
    own = adapter is None
    adapter = adapter or VllmOpenAIAdapter(cfg.server_url)
    try:
        try:
            info = await adapter.probe()
        except RuntimeError_ as e:
            raise PreflightError(str(e)) from e
    finally:
        if own:
            await adapter.aclose()
    served = next((m for m in info.models if m.id == cfg.model), None)
    if served is None:
        raise PreflightError(f"Model {cfg.model!r} is not served at {cfg.server_url}. Served: {[m.id for m in info.models]}")
    warnings: list[str] = []
    need = cfg.input_tokens + cfg.output_tokens
    if served.max_model_len is not None and need > served.max_model_len:
        raise PreflightError(
            f"input_tokens + output_tokens = {need} exceeds the server's max_model_len = {served.max_model_len}")
    if served.max_model_len is None:
        warnings.append("Server did not report max_model_len; length feasibility could not be checked.")
    eff = min(cfg.concurrency, cfg.num_requests)
    if info.kv_cache_tokens is not None and eff * need > info.kv_cache_tokens:
        warnings.append(
            f"{eff} concurrent requests x {need} tokens = {eff * need} exceeds the server KV cache "
            f"({info.kv_cache_tokens} tokens); vLLM will queue or preempt requests, so measured latency includes that.")
    if info.kv_cache_tokens is None:
        warnings.append("KV cache capacity unknown (no /metrics); queueing/preemption cannot be predicted.")
    if cfg.prompt_mode == "text":
        warnings.append("text prompt mode: input length is approximate; the server-reported count is recorded per request.")
    if cfg.concurrency > cfg.num_requests:
        warnings.append(f"concurrency {cfg.concurrency} > num_requests {cfg.num_requests}; effective concurrency is {eff}.")
    reg = None
    if cfg.registry_model_id is not None:
        with session_scope() as s:
            rec = s.scalars(select(ModelRecord).where(ModelRecord.id == cfg.registry_model_id)).first()
            if rec is None:
                raise PreflightError(f"registry_model_id {cfg.registry_model_id} not found")
            reg = {"id": rec.id, "name": rec.name, "source": rec.source, "ref": rec.ref, "revision": rec.revision,
                   "format": rec.format, "quantization": rec.quantization, "size_bytes": rec.size_bytes}
        if served.root and rec.name.split("/")[-1].lower() not in served.root.lower():
            warnings.append(f"Registry model {rec.name!r} does not obviously match the server's model root {served.root!r}.")
    return Preflight(info, {"id": served.id, "root": served.root, "max_model_len": served.max_model_len}, reg, warnings)
