"""Concurrency sweep: the same workload at several concurrency levels against one server configuration.

Measurement is delegated to InferenceBenchmark (_body/_finalize) once per level, so a level's numbers are
produced by exactly the code that produces a single benchmark's numbers. This module adds the managed
server lifecycle, cross-level statistics and artifacts.
"""
import asyncio
import logging
import sys
import time
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from locallab.config import get_settings
from locallab.db import session_scope
from locallab.errors import ExpectedFailure, ServerCrashed
from locallab.experiments.base import Cancelled, Experiment, ExperimentContext, PreflightResult
from locallab.experiments.inference.config import InferenceBenchmarkConfig
from locallab.experiments.inference.experiment import InferenceBenchmark
from locallab.experiments.inference.metrics import DEFINITIONS, Record
from locallab.experiments.inference.preflight import PreflightError, run_preflight
from locallab.experiments.inference.runner import Progress
from locallab.experiments.inference.sweep_stats import THRESHOLD_NOTE, analyze, gate_dist, level_stats
from locallab.models import Job, ManagedServer
from locallab.runtimes import vllm_managed as vm
from locallab.runtimes.vllm_compat import ManagedVllmConfig
from locallab.runtimes.vllm_openai import VllmOpenAIAdapter
from locallab.services.gpu import GpuSampler, read_gpus

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1
MAX_TOTAL_REQUESTS = 50_000
STOP_AFTER_FAILED_LEVELS = 2


class SweepConfig(BaseModel):
    mode: Literal["managed", "external"] = Field("managed", description="managed: LocalLab launches and removes a vLLM container")
    # managed
    registry_model_id: int | None = Field(None, description="Registry (HF cache) model to serve; required for managed mode")
    runtime: ManagedVllmConfig = ManagedVllmConfig()
    runtime_config_id: int | None = Field(None, description="Saved runtime configuration this was based on (informational)")
    # external
    server_url: str = "http://127.0.0.1:8100"
    model: str | None = Field(None, description="Served model name (external mode)")
    # workload
    concurrency_levels: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    input_tokens: int = Field(128, ge=1, le=262_144)
    output_tokens: int = Field(128, ge=1, le=32_768)
    prompt_mode: Literal["token_ids", "text"] = "token_ids"
    requests_per_level: int = Field(32, ge=1, le=5000, description="Measured requests per iteration (raised to concurrency x min_requests_per_slot)")
    min_requests_per_slot: int = Field(4, ge=1, le=100, description="Each concurrency slot serves at least this many requests per iteration")
    warmup_requests: int = Field(4, ge=0, le=200, description="Warmup per level (at least min(concurrency, 64) are always sent)")
    iterations: int = Field(3, ge=1, le=20, description="Repeats of the measured phase per level; >= 3 enables confidence intervals")
    temperature: float = Field(0.0, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    top_k: int | None = Field(None, ge=1)
    seed: int = 0
    request_timeout_s: float = Field(300, ge=5, le=3600)
    abort_after_failures: int = Field(10, ge=1, le=1000)

    @field_validator("concurrency_levels")
    @classmethod
    def _levels(cls, v: list[int]) -> list[int]:
        if not v or len(v) > 12 or any(not 1 <= c <= 512 for c in v):
            raise ValueError("concurrency_levels needs 1-12 values, each between 1 and 512")
        return sorted(set(v))

    @field_validator("server_url")
    @classmethod
    def _host(cls, v: str) -> str:
        host = urlparse(v).hostname
        if host not in get_settings().allowed_runtime_hosts:
            raise ValueError(f"host {host!r} is not in LOCALLAB_ALLOWED_RUNTIME_HOSTS {get_settings().allowed_runtime_hosts}")
        return v.rstrip("/")

    @model_validator(mode="after")
    def _mode_fields(self) -> "SweepConfig":
        if self.mode == "managed" and self.registry_model_id is None:
            raise ValueError("registry_model_id is required in managed mode")
        if self.mode == "external" and not self.model:
            raise ValueError("model (served model name) is required in external mode")
        if sum(self.level_requests(c) * self.iterations + self.level_warmup(c) for c in self.concurrency_levels) > MAX_TOTAL_REQUESTS:
            raise ValueError(f"workload exceeds {MAX_TOTAL_REQUESTS} total requests; reduce levels, requests or iterations")
        return self

    def level_requests(self, c: int) -> int:
        return min(5000, max(self.requests_per_level, c * self.min_requests_per_slot))

    def level_warmup(self, c: int) -> int:
        return min(200, max(self.warmup_requests, min(c, 64)))

    def level_config(self, index: int, c: int, server_url: str, model: str) -> InferenceBenchmarkConfig:
        """Config for one level. The seed differs per level so levels never replay each other's prompts
        (which would let a prefix cache inflate later levels' prefill speed)."""
        return InferenceBenchmarkConfig(
            server_url=server_url, model=model, profile="custom", prompt_mode=self.prompt_mode,
            input_tokens=self.input_tokens, output_tokens=self.output_tokens, num_requests=self.level_requests(c),
            concurrency=c, warmup_requests=self.level_warmup(c), iterations=self.iterations, temperature=self.temperature,
            top_p=self.top_p, top_k=self.top_k, seed=self.seed * 1009 + index, request_timeout_s=self.request_timeout_s,
            abort_after_failures=self.abort_after_failures)


SWEEP_DEFINITIONS = {
    "scaling_efficiency": "aggregate output tok/s at concurrency c / (c x aggregate output tok/s at concurrency 1). 1.0 = perfect linear scaling.",
    "ci95": "95% confidence interval of the mean across iterations (Student t); only with >= 3 iterations.",
    "server_gpu_memory_mb": "NVML per-process GPU memory summed over the managed container's PIDs. Only attributable for managed servers.",
    "gpu_peak_memory_mb": "Device-wide, includes every process on the GPU. Not a per-process measurement.",
    "thresholds": THRESHOLD_NOTE,
}


class ConcurrencySweep(Experiment):
    type_name = "concurrency_sweep"
    title = "Concurrency sweep"
    description = ("Runs one workload at several concurrency levels (default 1, 2, 4, 8, 16). Managed mode launches a "
                   "labelled vLLM container, benchmarks it, and removes it; external mode sweeps a server you run.")
    config_model = SweepConfig

    # ---- validation (API create/validate and dry checks) ----
    def preflight(self, config: BaseModel) -> PreflightResult:
        cfg = SweepConfig.model_validate(config)
        settings = get_settings()
        warnings = self._sample_warnings(cfg)
        totals = {"levels": [{"concurrency": c, "requests": cfg.level_requests(c), "warmup": cfg.level_warmup(c)}
                             for c in cfg.concurrency_levels],
                  "total_requests": sum(cfg.level_requests(c) * cfg.iterations + cfg.level_warmup(c) for c in cfg.concurrency_levels)}
        if cfg.mode == "external":
            lc = cfg.level_config(0, max(cfg.concurrency_levels), cfg.server_url, cfg.model or "")
            try:
                pf = asyncio.run(run_preflight(lc))
            except PreflightError as e:
                raise ValueError(str(e)) from e
            warnings += pf.warnings + ["External server: if its prefix cache survives between runs, repeating the same seed can "
                                       "make prefill look faster. Change the seed or restart the server between sweeps."]
            return PreflightResult(warnings=warnings, info={"mode": "external", "server": pf.info.base_url,
                                                            "served_model": pf.served_model, **totals})
        plan = vm.build_plan(cfg.runtime, cfg.registry_model_id, settings, vm.make_docker(settings), allow_run_oneoff=False)
        if plan.errors:
            raise ValueError("; ".join(plan.errors))
        kv = plan.estimates.get("kv_tokens")
        need = cfg.input_tokens + cfg.output_tokens
        if kv:
            for c in cfg.concurrency_levels:
                if c * need > kv:
                    warnings.append(f"Concurrency {c} x {need} tokens exceeds the estimated KV capacity ({kv} tokens); that level "
                                    "will queue or preempt.")
        return PreflightResult(warnings=plan.warnings + warnings, info={
            "mode": "managed", "gpu": plan.gpu.as_dict(), "estimates": plan.estimates, "model": plan.model,
            "image": plan.image, "command": plan.command(), "served_name": plan.served_name, **totals})

    @staticmethod
    def _sample_warnings(cfg: SweepConfig) -> list[str]:
        out = []
        if cfg.iterations < 3:
            out.append("Fewer than 3 iterations: no confidence intervals or noise detection.")
        small = [c for c in cfg.concurrency_levels if cfg.level_requests(c) * cfg.iterations < 20]
        if small:
            out.append(f"Levels {small} will have fewer than 20 samples; p95/p99 will not be reported for them.")
        return out

    # ---- execution ----
    def run(self, config: BaseModel, ctx: ExperimentContext) -> dict[str, Any]:
        return asyncio.run(self._run(SweepConfig.model_validate(config), ctx))

    async def _run(self, cfg: SweepConfig, ctx: ExperimentContext) -> dict[str, Any]:
        settings = get_settings()
        bench = InferenceBenchmark()
        server: vm.ManagedVllmServer | None = None
        plan: vm.ManagedPlan | None = None
        adapter: VllmOpenAIAdapter | None = None
        sampler: GpuSampler | None = None
        per_level: list[tuple[int, list[Record], dict[str, float]]] = []
        sweep_t0 = time.perf_counter()
        monitor_exc: list[BaseException] = []
        succeeded = False
        try:
            if cfg.mode == "managed":
                docker = vm.owned_docker(settings)
                ctx.set_progress({"phase": "preflight"})
                plan = await asyncio.to_thread(vm.build_plan, cfg.runtime, cfg.registry_model_id, settings, docker.docker,
                                               allow_run_oneoff=True)
                for w in plan.warnings:
                    ctx.log("WARNING", w)
                if plan.errors:
                    raise ExpectedFailure("Preflight failed, no server was started: " + "; ".join(plan.errors))
                with session_scope() as s:
                    experiment_id = s.get(Job, ctx.job_id).experiment_id  # type: ignore[union-attr]
                server = vm.ManagedVllmServer(plan, docker, settings, ctx.job_id, experiment_id, ctx.log)
                ctx.set_progress({"phase": "launching_server"})
                await asyncio.to_thread(server.launch)
                t_wait = time.monotonic()

                def tick() -> bool:
                    ctx.set_progress({"phase": "waiting_for_server", "elapsed_s": round(time.monotonic() - t_wait, 1),
                                      "timeout_s": cfg.runtime.startup_timeout_s})
                    return ctx.should_cancel()

                try:
                    await asyncio.to_thread(server.wait_ready, tick)
                except vm.ManagedServerError as e:
                    if e.kind == "cancelled":
                        raise Cancelled() from e
                    raise
                adapter = server.adapter()
                url, served = server.base_url, plan.served_name
            else:
                adapter = VllmOpenAIAdapter(cfg.server_url, max_connections=max(cfg.concurrency_levels) + 8)
                url, served = cfg.server_url, cfg.model or ""
            info = await adapter.probe()
            if served not in [m.id for m in info.models]:
                raise ExpectedFailure(f"Server does not serve {served!r}; it serves {[m.id for m in info.models]}")
            sampler = GpuSampler(pid_provider=server.server_pids if server else None)
            sampler.start()
            baseline = list(sampler.samples)

            progress = Progress()
            levels: list[dict[str, Any]] = []
            state: dict[str, Any] = {"stopped_early": None}
            body = asyncio.create_task(self._levels(cfg, bench, ctx, adapter, url, served, plan, progress, per_level,
                                                    sampler, baseline, server, levels, state))
            monitor = asyncio.create_task(bench._monitor(ctx, progress, body, monitor_exc,
                                                         health=server.check_alive if server else None))
            try:
                await body
            except asyncio.CancelledError:
                if not monitor_exc:
                    raise
            finally:
                monitor.cancel()
            if monitor_exc:
                crash = monitor_exc[0]
                if isinstance(crash, ServerCrashed) and any(lv["status"] == "ok" for lv in levels):
                    c = progress.context["sweep"]["concurrency"]  # keep finished levels; the crash is itself a finding
                    levels.append({"concurrency": c, "status": "failed", "error": f"{crash}",
                                   "num_requests": cfg.level_requests(c)})
                    state["stopped_early"] = f"server crashed during concurrency {c}"
                    ctx.log("ERROR", f"{crash}")
                else:
                    raise crash
            result = self._assemble(cfg, levels, state["stopped_early"], plan, server, info, served, sampler)
            succeeded = True
            return result
        finally:
            if sampler:
                sampler.stop()
            if adapter:
                await adapter.aclose()
            exc = sys.exc_info()[1]
            if server:
                server.shutdown("job completed" if succeeded else f"job ended early: {type(exc).__name__ if exc else 'unknown'}")
            self._write_artifacts(ctx, per_level, sampler, sweep_t0, succeeded, server)

    async def _levels(self, cfg, bench, ctx, adapter, url, served, plan, progress, per_level, sampler, baseline, server,
                      levels: list[dict[str, Any]], state: dict[str, Any]) -> None:
        failed_in_row = 0
        for idx, c in enumerate(cfg.concurrency_levels):
            lcfg = cfg.level_config(idx, c, url, served)
            progress.context = {"sweep": {"levels": cfg.concurrency_levels, "level_index": idx, "concurrency": c,
                                          "levels_done": len([lv for lv in levels])}}
            ctx.log("INFO", f"Sweep level {idx + 1}/{len(cfg.concurrency_levels)}: concurrency {c}, "
                            f"{lcfg.num_requests} requests x {lcfg.iterations} iterations")
            recs: list[Record] = []
            marks: dict[str, float] = {}
            per_level.append((c, recs, marks))
            try:
                pf = await run_preflight(lcfg, adapter)
                pf.registry_model = plan.model if plan else None
                before = read_gpus()
                await bench._body(lcfg, adapter, progress, recs, marks, ctx)
                after = read_gpus()
                res = bench._finalize(lcfg, pf, recs, marks, sampler, baseline, before, after)
                levels.append(self._level(c, lcfg, res, sampler, marks))
                failed_in_row = 0
            except (RuntimeError, PreflightError) as e:  # this level produced no valid measurement
                ctx.log("ERROR", f"Concurrency {c} failed: {e}")
                problem = await self._crash_after_failure(server, str(e)) if server else None
                levels.append({"concurrency": c, "status": "failed", "num_requests": lcfg.num_requests,
                               "error": f"server crashed: {problem}" if problem else str(e)})
                if problem:
                    state["stopped_early"] = f"server crashed during concurrency {c}"
                    ctx.log("ERROR", problem)
                    break
                failed_in_row += 1
                if failed_in_row >= STOP_AFTER_FAILED_LEVELS:
                    state["stopped_early"] = f"stopped after {failed_in_row} consecutive failed levels"
                    ctx.log("WARNING", state["stopped_early"])
                    break

    @staticmethod
    async def _crash_after_failure(server: "vm.ManagedVllmServer", error: str, settle_s: float = 6.0) -> str | None:
        """Did the server die? Connection-type failures can be seen a moment before Docker reports the
        container as exited, so allow a short settle window before concluding it is still healthy."""
        deadline = time.monotonic() + (settle_s if any(k in error.lower() for k in ("connection", "connect", "aborted after")) else 0)
        while True:
            problem = await asyncio.to_thread(server.check_alive)
            if problem or time.monotonic() >= deadline:
                return problem
            await asyncio.sleep(0.5)

    @staticmethod
    def _level(c: int, lcfg: InferenceBenchmarkConfig, res: dict[str, Any], sampler: GpuSampler, marks: dict[str, float]) -> dict[str, Any]:
        per_device = res["gpu"]["per_device"]  # keys are ints in memory (they become strings only after a JSON round trip)
        s, gpu0 = res["summary"], per_device.get(0) or per_device.get("0") or {}
        window = [x for x in sampler.samples if marks["measured_start"] <= x["t"] <= marks["measured_end"]]
        srv = [x["server_memory_mb"] for x in window if x.get("server_memory_mb") is not None]
        dists = {k: gate_dist(v) for k, v in s["latency"].items()}
        thr = dict(s["throughput"])
        thr["per_request_output_tok_s"] = gate_dist(thr["per_request_output_tok_s"])
        thr["per_request_decode_tok_s"] = gate_dist(thr["per_request_decode_tok_s"])
        return {
            "concurrency": c, "status": "ok", "num_requests": lcfg.num_requests, "warmup_requests": lcfg.warmup_requests,
            "iterations_run": len(res["iterations"]), "requests": s["requests"], "wall_s": s["wall_s"],
            "token_count_source": s["token_count_source"], "tokens": s["tokens"], "latency": dists, "throughput": thr,
            "iterations": res["iterations"], "stats": level_stats(res["iterations"]),
            "gpu": {**gpu0, "server_peak_memory_mb": max(srv) if srv else None,
                    "server_memory_attribution": "nvml per-process over the managed container's PIDs" if srv else None},
            "other_compute_processes": res["gpu"]["other_compute_processes"], "verification": res["workload"]["verification"],
            "max_observed_in_flight": res["workload"]["max_observed_in_flight"], "warnings": res["warnings"],
            "seed": lcfg.seed,
        }

    def _assemble(self, cfg, levels, stopped_early, plan, server, info, served, sampler) -> dict[str, Any]:
        ok = [lv for lv in levels if lv["status"] == "ok"]
        if not ok:
            raise ExpectedFailure("No concurrency level produced valid measurements: "
                                  + "; ".join(f"c={lv['concurrency']}: {lv['error']}" for lv in levels))
        others = {(p["pid"], p["name"], p["memory_mb"]) for lv in ok for p in lv["other_compute_processes"]}
        pre = plan.gpu.as_dict() if plan else None
        contended = bool(others) or bool(pre and pre["contended"])
        analysis = analyze(levels, contended)
        warnings = list(self._sample_warnings(cfg))
        for lv in levels:
            if lv["status"] == "failed":
                warnings.append(f"Concurrency {lv['concurrency']} failed: {lv['error']}")
        if stopped_early:
            warnings.append(stopped_early)
        if contended:
            warnings.append("GPU contention was detected; results are marked contended.")
        for w in {w for lv in ok for w in lv["warnings"] if "other compute process" not in w}:
            warnings.append(w)
        row = None
        if server and server.row_id:
            with session_scope() as s:
                r = s.get(ManagedServer, server.row_id)
                row = {"managed_server_id": r.id, "container_id": r.container_id, "container_name": r.container_name,
                       "image_ref": r.image_ref, "image_id": r.image_id, "image_digest": r.image_digest, "command": r.command,
                       "config": r.config, "host_port": r.host_port, "startup_s": r.startup_s}
        return {
            "kind": "concurrency_sweep", "schema_version": SCHEMA_VERSION, "mode": cfg.mode,
            "managed_server": row,
            "runtime": {"name": "vllm", "version": info.version, "kv_cache_tokens": info.kv_cache_tokens,
                        "server_config": info.extra.get("cache_config"), "served_model": next(
                            ({"id": m.id, "root": m.root, "max_model_len": m.max_model_len} for m in info.models if m.id == served), None),
                        "server_url": None if server else cfg.server_url},
            "model_registry": plan.model if plan else None,
            "workload": {"input_tokens": cfg.input_tokens, "output_tokens": cfg.output_tokens, "prompt_mode": cfg.prompt_mode,
                         "levels": cfg.concurrency_levels, "requests_per_level": cfg.requests_per_level,
                         "min_requests_per_slot": cfg.min_requests_per_slot, "warmup_requests": cfg.warmup_requests,
                         "iterations": cfg.iterations, "seed": cfg.seed,
                         "seed_scheme": "level seed = seed*1009 + level index; per-request seeds differ per phase/iteration/index",
                         "decoding": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k, "ignore_eos": True}},
            "gpu": {"preflight": pre, "contended": contended, "contention_override": bool(plan and cfg.runtime.allow_contention),
                    "other_compute_processes": [{"pid": p, "name": n, "memory_mb": m} for p, n, m in sorted(others)],
                    "scope": "peak_memory_mb is device-wide; server_peak_memory_mb is per-process (managed containers only)"},
            "levels": levels, "analysis": analysis, "stopped_early": stopped_early, "warnings": warnings,
            "definitions": {**DEFINITIONS, **SWEEP_DEFINITIONS},
        }

    @staticmethod
    def _write_artifacts(ctx, per_level, sampler, sweep_t0, complete, server) -> None:
        if any(recs for _, recs, _ in per_level):
            ctx.write_artifact("requests.json", "requests", {
                "complete": complete, "note": None if complete else "PARTIAL: run was cancelled, failed or interrupted",
                "levels": [{"concurrency": c, "requests": [r.to_dict() for r in recs]} for c, recs, _ in per_level if recs]})
        if sampler and sampler.samples:
            ctx.write_artifact("gpu_samples.json", "timeseries", {
                "origin": "seconds since sweep start (perf_counter)",
                "level_windows": [{"concurrency": c, "start_s": m["measured_start"] - sweep_t0,
                                   "end_s": m.get("measured_end", m["measured_start"]) - sweep_t0}
                                  for c, _, m in per_level if "measured_start" in m],
                "samples": [{**s, "t": round(s["t"] - sweep_t0, 3)} for s in sampler.samples]})
        if server and server.final_logs:
            ctx.write_artifact("server.log", "log", {"container": server.container_name,
                                                     "lines": server.final_logs.splitlines()[-3000:]})
