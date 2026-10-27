import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel

from locallab.experiments.base import Cancelled, Experiment, ExperimentContext, PreflightResult
from locallab.experiments.inference.config import PRESETS, InferenceBenchmarkConfig
from locallab.experiments.inference.metrics import DEFINITIONS, Record, summarize, verify_lengths
from locallab.experiments.inference.preflight import PreflightError, run_preflight
from locallab.experiments.inference.runner import Planned, Progress, run_phase
from locallab.experiments.inference.workload import request_seed, text_prompt, token_id_prompt
from locallab.runtimes.base import GenerationRequest
from locallab.runtimes.vllm_openai import VllmOpenAIAdapter
from locallab.services.gpu import GpuSampler, read_gpus

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def build_plans(cfg: InferenceBenchmarkConfig, phase: str, iteration: int, count: int) -> list[Planned]:
    plans = []
    for i in range(count):
        seed = request_seed(cfg.seed, phase, iteration, i)
        prompt = token_id_prompt(seed, cfg.input_tokens) if cfg.prompt_mode == "token_ids" else text_prompt(seed, cfg.input_tokens)
        plans.append(Planned(
            i, phase, iteration, seed, cfg.input_tokens if cfg.prompt_mode == "token_ids" else None, cfg.output_tokens,
            GenerationRequest(model=cfg.model, prompt=prompt, max_tokens=cfg.output_tokens, temperature=cfg.temperature,
                              top_p=cfg.top_p, top_k=cfg.top_k, seed=seed, timeout_s=cfg.request_timeout_s)))
    return plans


def _gpu_summary(samples: list[dict], t_start: float, t_end: float, baseline: list[dict]) -> dict[str, Any]:
    window = [s for s in samples if t_start <= s["t"] <= t_end]
    by_gpu: dict[int, Any] = {}
    for idx in sorted({s["index"] for s in samples}):
        rows = [s for s in window if s["index"] == idx]
        base = next((s["memory_used_mb"] for s in baseline if s["index"] == idx), None)
        mem = [r["memory_used_mb"] for r in rows]
        def mx(k): return max((r[k] for r in rows if r[k] is not None), default=None)
        by_gpu[idx] = {
            "baseline_memory_mb": base, "peak_memory_mb": max(mem) if mem else None,
            "peak_delta_mb": (max(mem) - base) if mem and base is not None else None,
            "peak_utilization_pct": mx("utilization_gpu_pct"), "peak_power_w": mx("power_w"),
            "peak_temperature_c": mx("temperature_c"), "samples": len(rows),
        }
    return by_gpu


class InferenceBenchmark(Experiment):
    type_name = "inference_benchmark"
    title = "Inference benchmark"
    description = ("Load-tests an OpenAI-compatible vLLM server with a fixed synthetic workload and records TTFT, "
                   "TPOT, latency, throughput and GPU memory per request.")
    config_model = InferenceBenchmarkConfig
    presets = PRESETS

    def preflight(self, config: BaseModel) -> PreflightResult:
        cfg = InferenceBenchmarkConfig.model_validate(config)
        try:
            pf = asyncio.run(run_preflight(cfg))
        except PreflightError as e:
            raise ValueError(str(e)) from e
        return PreflightResult(warnings=pf.warnings, info={"server": pf.info.__dict__ | {"models": [m.__dict__ for m in pf.info.models]},
                                                            "served_model": pf.served_model})

    def run(self, config: BaseModel, ctx: ExperimentContext) -> dict[str, Any]:
        cfg = InferenceBenchmarkConfig.model_validate(config)
        return asyncio.run(self._run(cfg, ctx))

    async def _run(self, cfg: InferenceBenchmarkConfig, ctx: ExperimentContext) -> dict[str, Any]:
        adapter = VllmOpenAIAdapter(cfg.server_url, max_connections=cfg.concurrency + 8)
        progress = Progress()
        records: list[Record] = []
        sampler = GpuSampler()
        marks: dict[str, float] = {}
        monitor_exc: list[BaseException] = []
        succeeded = False
        try:
            ctx.log("INFO", f"Preflight against {cfg.server_url} model={cfg.model}")
            pf = await run_preflight(cfg, adapter)
            for w in pf.warnings:
                ctx.log("WARNING", w)
            gpus_before = read_gpus()
            if not gpus_before.available:
                raise RuntimeError(f"GPU telemetry unavailable: {gpus_before.error}")
            sampler.start()
            baseline = [s for s in sampler.samples]
            body = asyncio.create_task(self._body(cfg, adapter, progress, records, marks, ctx))
            monitor = asyncio.create_task(self._monitor(ctx, progress, body, monitor_exc))
            try:
                await body
            except asyncio.CancelledError:
                if not monitor_exc:
                    raise
            finally:
                monitor.cancel()
            if monitor_exc:
                raise monitor_exc[0]  # Cancelled or WorkerShutdown, raised after data is preserved below
            gpus_after = read_gpus()
            result = self._finalize(cfg, pf, records, marks, sampler, baseline, gpus_before, gpus_after)
            succeeded = True
            return result
        finally:
            sampler.stop()
            await adapter.aclose()
            self._write_artifacts(ctx, cfg, records, sampler, marks, complete=succeeded)

    async def _monitor(self, ctx: ExperimentContext, progress: Progress, body: asyncio.Task, exc: list[BaseException]) -> None:
        while not body.done():
            try:
                if ctx.should_cancel():
                    exc.append(Cancelled())
                    body.cancel()
                    return
            except Exception as e:  # WorkerShutdown from the worker
                exc.append(e)
                body.cancel()
                return
            elapsed = time.perf_counter() - progress.phase_started if progress.phase_started else 0
            ctx.set_progress({
                "phase": progress.phase, "iteration": progress.iteration, "total": progress.total, "done": progress.done,
                "ok": progress.ok, "failed": progress.failed, "in_flight": progress.in_flight,
                "output_tokens": progress.output_tokens, "elapsed_s": round(elapsed, 1),
                "live_output_tok_s": round(progress.output_tokens / elapsed, 1) if elapsed > 1 else None,
                "last_error": progress.last_error,
            })
            await asyncio.sleep(0.5)

    async def _body(self, cfg, adapter, progress: Progress, records: list[Record], marks, ctx) -> None:
        t0 = marks["t0"] = time.perf_counter()
        eff = min(cfg.concurrency, cfg.num_requests)
        if cfg.warmup_requests:
            progress.start_phase("warmup", 0, cfg.warmup_requests, time.perf_counter())
            ctx.log("INFO", f"Warmup: {cfg.warmup_requests} requests (concurrency {min(eff, cfg.warmup_requests)})")
            warm: list[Record] = []
            await run_phase(adapter, build_plans(cfg, "warmup", 0, cfg.warmup_requests), min(eff, cfg.warmup_requests),
                            cfg.abort_after_failures, t0, progress, warm)
            records.extend(warm)
            if not any(r.outcome.ok for r in warm):
                raise RuntimeError(f"Warmup failed: all {len(warm)} requests failed. Last error: {progress.last_error}")
        for it in range(cfg.iterations):
            progress.start_phase("measured", it + 1, cfg.num_requests, time.perf_counter())
            marks.setdefault("measured_start", progress.phase_started)
            ctx.log("INFO", f"Measured iteration {it + 1}/{cfg.iterations}: {cfg.num_requests} requests, concurrency {eff}")
            await run_phase(adapter, build_plans(cfg, "measured", it, cfg.num_requests), eff,
                            cfg.abort_after_failures, t0, progress, records)
            marks["measured_end"] = time.perf_counter()
            marks["max_in_flight"] = max(marks.get("max_in_flight", 0), progress.max_in_flight)
            if progress.aborted:
                raise RuntimeError(f"Aborted after {cfg.abort_after_failures} consecutive failures. Last error: {progress.last_error}")

    def _finalize(self, cfg, pf, records, marks, sampler, baseline, gpus_before, gpus_after) -> dict[str, Any]:
        rows = [r.to_dict() for r in records if r.phase == "measured"]
        pooled = summarize(rows)
        if pooled["requests"]["succeeded"] == 0:
            first = next((r for r in rows if not r["ok"]), None)
            raise RuntimeError(
                "No measured request succeeded; there are no valid measurements to report. "
                f"Errors: {pooled['requests']['errors_by_type']}. First error: "
                f"{first['error_type']}: {first['error_message']}" if first else "No measured requests were recorded.")
        iterations = []
        for it in range(cfg.iterations):
            s = summarize([r for r in rows if r["iteration"] == it])
            iterations.append({"iteration": it + 1, "wall_s": s["wall_s"], "success_rate": s["requests"]["success_rate"],
                               "aggregate_output_tok_s": s["throughput"]["aggregate_output_tok_s"],
                               "total_tok_s": s["throughput"]["total_tok_s"],
                               "e2e_p50_ms": (s["latency"]["e2e_ms"] or {}).get("p50")})
        # pooled wall time = sum of per-iteration walls (iterations run back to back)
        wall = sum(i["wall_s"] for i in iterations)
        out_tok, in_tok = pooled["tokens"]["output_total"], pooled["tokens"]["input_total"]
        pooled["wall_s"] = wall
        pooled["throughput"].update(
            aggregate_output_tok_s=out_tok / wall if wall else None, total_tok_s=(in_tok + out_tok) / wall if wall else None,
            requests_per_s=pooled["requests"]["succeeded"] / wall if wall else None)
        warnings = list(pf.warnings)
        lengths = verify_lengths(rows)
        if cfg.prompt_mode == "token_ids" and lengths["input_mismatches"]:
            warnings.append(f"{lengths['input_mismatches']} requests had a server-reported prompt length different from the requested one.")
        if lengths["output_mismatches"]:
            warnings.append(f"{lengths['output_mismatches']} requests produced a different number of output tokens than requested.")
        if pooled["requests"]["failed"]:
            warnings.append(f"{pooled['requests']['failed']} of {pooled['requests']['attempted']} measured requests failed: {pooled['requests']['errors_by_type']}")
        if pooled["token_count_source"] == "estimated":
            warnings.append("Server did not return usage; output token counts are estimated from streamed chunks.")
        procs = {p.pid: p for g in (gpus_before.gpus + gpus_after.gpus) for p in g.processes if p.kind == "compute"}
        # The runtime's own processes are expected; anything else computing on the GPU may contend with it.
        others = sorted((p.pid, (p.name or "?")[:80], round(p.used_memory_mb or 0)) for p in procs.values()
                        if "vllm" not in (p.name or "").lower())
        gpu = _gpu_summary(sampler.samples, marks["measured_start"], marks["measured_end"], baseline)
        if others:
            warnings.append(f"{len(others)} other compute process(es) were using the GPU; peak memory is device-wide "
                            "and latency/throughput may be affected by contention.")
        if sampler.error:
            warnings.append(f"GPU sampling stopped early: {sampler.error}")
        return {
            "kind": "inference_benchmark", "schema_version": SCHEMA_VERSION,
            "runtime": {"name": "vllm", "version": pf.info.version, "server_url": cfg.server_url, "kv_cache_tokens": pf.info.kv_cache_tokens,
                        "server_config": pf.info.extra.get("cache_config"), "served_model": pf.served_model},
            "model_registry": pf.registry_model,
            "workload": {"profile": cfg.profile, "prompt_mode": cfg.prompt_mode, "input_tokens": cfg.input_tokens,
                         "output_tokens": cfg.output_tokens, "num_requests": cfg.num_requests, "concurrency": cfg.concurrency,
                         "effective_concurrency": min(cfg.concurrency, cfg.num_requests), "max_observed_in_flight": marks.get("max_in_flight"),
                         "warmup_requests": cfg.warmup_requests, "iterations": cfg.iterations, "seed": cfg.seed,
                         "decoding": {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k, "ignore_eos": True},
                         "verification": lengths},
            "summary": pooled, "iterations": iterations,
            "gpu": {"per_device": gpu, "other_compute_processes": [{"pid": p, "name": n, "memory_mb": m} for p, n, m in others], "scope": "device-wide NVML, measured phase only"},
            "warnings": warnings, "definitions": DEFINITIONS,
        }

    def _write_artifacts(self, ctx, cfg, records: list[Record], sampler: GpuSampler, marks, complete: bool) -> None:
        if not records:
            return
        t0 = marks.get("t0", 0.0)
        ctx.write_artifact("requests.json", "requests", {
            "complete": complete, "note": None if complete else "PARTIAL: run was cancelled, failed or interrupted",
            "requests": [r.to_dict() for r in records]})
        ctx.write_artifact("gpu_samples.json", "timeseries", {
            "origin": "seconds since run start (perf_counter)", "measured_start_s": marks.get("measured_start", t0) - t0,
            "measured_end_s": (marks["measured_end"] - t0) if "measured_end" in marks else None,
            "samples": [{**s, "t": round(s["t"] - t0, 3)} for s in sampler.samples]})
