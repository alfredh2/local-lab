"""Managed vLLM serving: launch an owned, labelled container, wait for readiness, always clean up."""
import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import httpx
from sqlalchemy import select

from locallab.config import Settings
from locallab.db import session_scope
from locallab.models import Job, JobStatus, ManagedServer, ModelRecord, utcnow
from locallab.runtimes.base import ModelInfo, RuntimeError_, ServerInfo
from locallab.runtimes.docker import (
    CUDA_COMPAT_FIX_LD_LIBRARY_PATH, ContainerSpec, DockerAPI, DockerError, NotOwnedError, OwnedDocker, new_container_name,
)
from locallab.runtimes.vllm_compat import (
    ManagedVllmConfig, checkpoint_info, classify_launch_failure, supported_architectures, validate_runtime,
)
from locallab.runtimes.vllm_openai import VllmOpenAIAdapter
from locallab.services.gpu_preflight import GpuPreflight, check_gpu

log = logging.getLogger(__name__)
HF_MOUNT = "/root/.cache/huggingface/hub"
LOG_TAIL_CHARS = 20_000


class ManagedServerError(RuntimeError):
    """Launch/readiness/liveness failure with a machine-readable kind."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass
class ManagedPlan:
    config: ManagedVllmConfig
    model: dict[str, Any]  # registry snapshot incl. revision
    repo_id: str
    served_name: str
    snapshot: Path
    image: str
    gpu: GpuPreflight
    estimates: dict[str, Any]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def command(self) -> list[str]:
        c = self.config
        cmd = [self.repo_id, "--revision", self.model["revision"], "--served-model-name", self.served_name,
               "--max-model-len", str(c.max_model_len), "--gpu-memory-utilization", f"{c.gpu_memory_utilization:g}",
               "--dtype", c.dtype, "--max-num-seqs", str(c.max_num_seqs), "--port", "8000"]
        if c.quantization != "none":
            cmd += ["--quantization", c.quantization]
        if c.kv_cache_dtype != "auto":
            cmd += ["--kv-cache-dtype", c.kv_cache_dtype]
        if c.enforce_eager:
            cmd.append("--enforce-eager")
        return cmd


def build_plan(cfg: ManagedVllmConfig, registry_model_id: int | None, settings: Settings, docker: DockerAPI,
               *, allow_run_oneoff: bool, exclude_pids: set[int] | None = None) -> ManagedPlan:
    """Validate everything knowable before launching. Never starts a server container."""
    errors: list[str] = []
    warnings: list[str] = []
    image = cfg.image or settings.vllm_image
    model: dict[str, Any] = {}
    ck: dict[str, Any] | None = None
    snapshot = Path(".")
    if registry_model_id is None:
        errors.append("registry_model_id is required for a managed server.")
    else:
        with session_scope() as s:
            rec = s.get(ModelRecord, registry_model_id)
            if rec is None:
                errors.append(f"registry model {registry_model_id} not found")
            else:
                model = {"id": rec.id, "name": rec.name, "source": rec.source, "ref": rec.ref, "revision": rec.revision,
                         "format": rec.format, "quantization": rec.quantization, "size_bytes": rec.size_bytes}
    if model:
        snapshot = Path(model["ref"])
        hf_root = settings.hf_cache_dir.expanduser().resolve()
        if model["source"] != "hf_cache":
            errors.append(f"Only Hugging Face cache models can be served by vLLM (this one is {model['source']}).")
        elif not snapshot.resolve().is_relative_to(hf_root):
            errors.append("Model snapshot is outside the configured Hugging Face cache and cannot be mounted.")
        elif not model["revision"]:
            errors.append("Model has no revision; rescan the registry.")
        else:
            try:
                ck = checkpoint_info(snapshot)
            except (OSError, ValueError) as e:
                errors.append(f"Cannot read the model checkpoint config: {e}")
            if ck is not None and ck["weights_mb"] <= 0:
                errors.append("Model has no safetensors weights in the cache (incomplete download?).")
                ck = None
    img = docker.image_inspect(image)
    if img is None:
        errors.append(f"Docker image {image!r} is not available locally (LocalLab does not pull images).")
    gpu = check_gpu(cfg.gpu_index, cfg.gpu_memory_utilization, cfg.allow_contention, exclude_pids)
    errors += gpu.errors
    warnings += gpu.warnings
    est: dict[str, Any] = {}
    if ck is not None:
        archs = None
        if img is not None:
            try:
                archs = supported_architectures(docker, image, img.id, settings, allow_run_oneoff)
            except (DockerError, ValueError) as e:
                warnings.append(f"Could not read vLLM's supported architectures ({e}); relying on the launch result.")
        e2, w2, est = validate_runtime(cfg, ck, gpu.compute_capability, gpu.total_mb, archs)
        errors += e2
        warnings += w2
    served = (model.get("name", "model").split("/")[-1]).lower()
    return ManagedPlan(config=cfg, model=model, repo_id=model.get("name", ""), served_name=served, snapshot=snapshot,
                       image=image, gpu=gpu, estimates=est, errors=errors, warnings=warnings)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ManagedVllmServer:
    """One launch. `shutdown` is idempotent and safe to call from any failure path."""

    def __init__(self, plan: ManagedPlan, docker: OwnedDocker, settings: Settings, job_id: int, experiment_id: int,
                 log_fn: Callable[[str, str], None] = lambda lvl, msg: None) -> None:
        self.plan, self.docker, self.settings = plan, docker, settings
        self.job_id, self.experiment_id, self.log = job_id, experiment_id, log_fn
        self.row_id: int | None = None
        self.container_id: str | None = None
        self.container_name = new_container_name(job_id)
        self.host_port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.host_port}"
        self._lock = threading.Lock()
        self._closed = False
        self.startup_s: float | None = None
        self.final_logs: str = ""

    # --- lifecycle -------------------------------------------------------------------------
    def launch(self) -> None:
        img = self.docker.docker.image_inspect(self.plan.image)
        spec = ContainerSpec(
            name=self.container_name, image=self.plan.image, labels=self.docker.labels_for(self.job_id, self.experiment_id),
            env={"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
                 "LD_LIBRARY_PATH": CUDA_COMPAT_FIX_LD_LIBRARY_PATH},
            mounts=[(str(self.settings.hf_cache_dir.expanduser().resolve()), HF_MOUNT, True)],
            host_port=self.host_port, container_port=8000, command=self.plan.command(), gpu_index=self.plan.config.gpu_index)
        with session_scope() as s:
            row = ManagedServer(job_id=self.job_id, container_name=self.container_name, instance_id=self.docker.instance_id,
                                status="starting", image_ref=self.plan.image, image_id=img.id if img else None,
                                image_digest=(img.repo_digests[0] if img and img.repo_digests else None),
                                host_port=self.host_port, config=self.plan.config.model_dump(), model=self.plan.model,
                                command=spec.command)
            s.add(row)
            s.flush()
            self.row_id = row.id
        self.log("INFO", f"Launching {self.container_name} ({self.plan.image}) on 127.0.0.1:{self.host_port}")
        try:
            self.container_id = self.docker.docker.create_and_start(spec)
        except DockerError as e:
            self._finish_row("failed", f"docker run failed: {e}")
            raise ManagedServerError("docker_failed", str(e)) from e
        with session_scope() as s:
            s.get(ManagedServer, self.row_id).container_id = self.container_id  # type: ignore[union-attr]

    def wait_ready(self, should_cancel: Callable[[], bool], poll_s: float = 2.0) -> ServerInfo:
        """Block until /v1/models answers. Raises Cancelled-like via should_cancel, or ManagedServerError."""
        assert self.container_id
        deadline = time.monotonic() + self.plan.config.startup_timeout_s
        t0 = time.monotonic()
        last_log = 0.0
        while True:
            if should_cancel():
                raise ManagedServerError("cancelled", "cancelled during server startup")
            info = self.docker.docker.inspect(self.container_id)
            if info is None or not info.running:
                logs = self.docker.docker.logs(self.container_id, tail=400) if info is not None else ""
                kind, msg = classify_launch_failure(logs, bool(info and info.oom_killed))
                self.final_logs = logs
                self._record_exit(info)
                raise ManagedServerError(kind, f"{msg} (container {'gone' if info is None else f'exited {info.exit_code}'})")
            try:
                server = self._probe()
                self.startup_s = time.monotonic() - t0
                with session_scope() as s:
                    row = s.get(ManagedServer, self.row_id)
                    row.status, row.ready_at, row.startup_s = "ready", utcnow(), self.startup_s  # type: ignore[union-attr]
                self.log("INFO", f"Server ready after {self.startup_s:.1f}s")
                return server
            except RuntimeError_:
                pass
            if time.monotonic() - last_log > 30:
                last_log = time.monotonic()
                self.log("INFO", f"Waiting for server… {last_log - t0:.0f}s elapsed")
            if time.monotonic() > deadline:
                raise ManagedServerError("startup_timeout", f"server not ready after {self.plan.config.startup_timeout_s:.0f}s")
            time.sleep(poll_s)

    def _probe(self) -> ServerInfo:
        try:
            r = httpx.get(f"{self.base_url}/v1/models", timeout=3.0)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError_(str(e)) from e
        return ServerInfo(runtime="vllm", base_url=self.base_url, version=None,
                          models=[ModelInfo(m["id"], m.get("root"), m.get("max_model_len")) for m in r.json().get("data", [])])

    def check_alive(self) -> str | None:
        """None if healthy, else a description of what happened to the container (for the watchdog)."""
        if self._closed or not self.container_id:
            return None
        try:
            info = self.docker.docker.inspect(self.container_id)
        except DockerError as e:
            return f"cannot inspect server container: {e}"
        if info is None:
            return "server container disappeared"
        if not info.running:
            logs = self.docker.docker.logs(self.container_id, tail=400)
            kind, msg = classify_launch_failure(logs, info.oom_killed)
            self.final_logs = logs
            return f"server container exited (code {info.exit_code}, {kind}): {msg}"
        return None

    def server_pids(self) -> set[int]:
        try:
            return self.docker.docker.top_pids(self.container_id) if self.container_id and not self._closed else set()
        except DockerError:
            return set()

    def adapter(self) -> VllmOpenAIAdapter:
        return VllmOpenAIAdapter(self.base_url)

    def shutdown(self, reason: str = "job finished") -> None:
        """Idempotent. Captures logs, stops and removes the owned container, finalizes the DB row."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        status = "stopped"
        try:
            if self.container_id:
                try:
                    info = self.docker.docker.inspect(self.container_id)
                    if info is not None:
                        self.final_logs = self.docker.docker.logs(self.container_id) or self.final_logs
                        self._record_exit(info)
                except DockerError as e:
                    log.warning("could not capture final logs for %s: %s", self.container_name, e)
                outcome = self.docker.stop_and_remove(self.container_id)
                self.log("INFO", f"Server container {self.container_name} {outcome}")
        except NotOwnedError:
            status = "failed"
            reason += " (container was not owned; left untouched)"
            log.error("shutdown refused for %s", self.container_name)
        except DockerError as e:
            status = "failed"
            reason += f" (cleanup error: {e})"
            log.error("cleanup of %s failed: %s", self.container_name, e)
        self._finish_row(status, reason)

    # --- bookkeeping -----------------------------------------------------------------------
    def _record_exit(self, info: Any) -> None:
        if info is None or self.row_id is None:
            return
        with session_scope() as s:
            row = s.get(ManagedServer, self.row_id)
            if row is not None:
                row.exit_code, row.oom_killed = info.exit_code, info.oom_killed

    def _finish_row(self, status: str, reason: str) -> None:
        if self.row_id is None:
            return
        with session_scope() as s:
            row = s.get(ManagedServer, self.row_id)
            if row is None:
                return
            already_final = row.status in ("stopped", "failed", "orphan_removed")
            if not already_final or status == "failed":
                row.status = status
            row.stopped_at = row.stopped_at or utcnow()
            if not (already_final and row.stop_reason):  # keep the first terminal reason (the root cause)
                row.stop_reason = reason
            row.log_tail = (self.final_logs or "")[-LOG_TAIL_CHARS:] or row.log_tail


def reconcile_orphans(docker: OwnedDocker, settings: Settings) -> list[str]:
    """Remove owned containers whose job is finished/missing/stale. Never touches unowned containers.

    Call after recover_stale() so jobs whose worker died are already terminal.
    """
    removed: list[str] = []
    stale_cutoff = utcnow() - timedelta(seconds=settings.worker_stale_s)
    for c in docker.list_owned():
        with session_scope() as s:
            row = s.scalars(select(ManagedServer).where(ManagedServer.container_name == c.name)).first()
            job = s.get(Job, row.job_id) if row and row.job_id else None
            active = job is not None and job.status in (JobStatus.preparing.value, JobStatus.running.value)
            fresh = active and job.heartbeat_at is not None and job.heartbeat_at >= stale_cutoff
            if fresh:
                continue  # a live worker is using it
            reason = ("job missing" if job is None else f"job {job.id} is {job.status}")
            try:
                logs = docker.docker.logs(c.id, tail=400)
                docker.stop_and_remove(c.id)
            except DockerError as e:
                log.error("reconcile: could not remove %s: %s", c.name, e)
                continue
            if row is not None:
                row.status, row.stopped_at = "orphan_removed", utcnow()
                row.stop_reason = f"orphaned container removed by reconciliation ({reason})"
                row.log_tail = row.log_tail or logs[-LOG_TAIL_CHARS:]
            removed.append(c.name)
            log.warning("reconcile: removed orphaned managed container %s (%s)", c.name, reason)
    # DB rows that claim to be running but whose container no longer exists
    with session_scope() as s:
        live = {c.name for c in docker.list_owned()}
        for row in s.scalars(select(ManagedServer).where(ManagedServer.status.in_(("starting", "ready")))):
            job = s.get(Job, row.job_id) if row.job_id else None
            if row.container_name not in live and (job is None or JobStatus(job.status).terminal):
                row.status, row.stopped_at = "failed", row.stopped_at or utcnow()
                row.stop_reason = row.stop_reason or "container not found during reconciliation"
    return removed
