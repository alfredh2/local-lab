"""Optimization matrix: compare model/runtime configurations on performance *and* task quality.

For each server configuration (model x context x weight quantization x KV dtype) the server is launched once;
synthetic performance cells (io pairs x concurrency levels, via the existing sweep code) and the quality evals
run against that same server, and the combined result is stored in one `config_results` row. Rows are written as
each configuration finishes (or fails), so nothing is lost to a later failure, cancellation or crash, and a new job
on the same experiment resumes with only the unfinished configurations.
"""
import asyncio
import logging
import sys
import time
import traceback
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select

from locallab.config import Settings, get_settings
from locallab.db import session_scope
from locallab.errors import ExpectedFailure, ServerCrashed, WorkerShutdown
from locallab.evals.registry import SUITES
from locallab.evals.runner import calibrate_chars_per_token, run_suite
from locallab.evals.sandbox import DockerSandbox
from locallab.experiments.base import Cancelled, Experiment, ExperimentContext, PreflightResult
from locallab.experiments.inference.experiment import InferenceBenchmark
from locallab.experiments.inference.matrix import MatrixConfig, ServerConfig, cell_plan, config_key, expand
from locallab.experiments.inference.runner import Progress
from locallab.experiments.inference.sweep import ConcurrencySweep, SweepConfig
from locallab.experiments.inference.sweep_stats import analyze as analyze_levels
from locallab.models import ConfigResult, Job, ManagedServer, ModelRecord, utcnow
from locallab.runtimes import vllm_managed as vm
from locallab.services.gpu import GpuSampler
from locallab.services.gpu_preflight import check_gpu

log = logging.getLogger(__name__)
SCHEMA_VERSION = 1


def _now() -> float:  # monkeypatched in tests to exercise the runtime budget deterministically
    return time.monotonic()


class MatrixPlan:
    def __init__(self, sc: ServerConfig, plan: vm.ManagedPlan, cells: dict[str, Any]) -> None:
        self.sc, self.plan, self.cells = sc, plan, cells

    @property
    def errors(self) -> list[str]:
        return self.plan.errors


def plan_all(cfg: MatrixConfig, settings: Settings, docker: Any, allow_run_oneoff: bool) -> list[MatrixPlan]:
    """Validate every configuration against the checkpoint, GPU and vLLM image without launching anything."""
    plans: list[MatrixPlan] = []
    for sc in expand(cfg):
        with session_scope() as s:
            rec = s.get(ModelRecord, sc.model_id)
            sc.model_name, sc.revision = (rec.name, rec.revision or "") if rec else (f"model {sc.model_id}", "")
        sc.key = config_key(sc.model_id, sc.revision, sc.max_model_len, sc.quantization, sc.kv_cache_dtype)
        plan = vm.build_plan(cfg.runtime_for(sc), sc.model_id, settings, docker, allow_run_oneoff=allow_run_oneoff)
        cells = cell_plan(cfg, sc)
        if cfg.perf_enabled and not cells["io_pairs"] and not cells["evals"]:
            plan.errors.append("nothing to measure at this context length: " + "; ".join(cells["notes"]))
        plans.append(MatrixPlan(sc, plan, cells))
    return plans


def _describe(c: dict[str, Any]) -> str:
    """One configuration's outcome for the job-level error (the error text usually starts with its own failure kind)."""
    err, kind = c["error"][:160], str(c["failure_kind"])
    return f"{c['label']}: {err if err.startswith(kind) else f'{kind}: {err}'}"


class OptimizationMatrix(Experiment):
    type_name = "optimization_matrix"
    title = "Optimization matrix"
    description = ("Compares model and runtime configurations (model, context length, quantization, KV-cache dtype) on synthetic "
                   "performance and on real task quality (structured JSON, Python coding, long-context retrieval). One managed "
                   "vLLM launch per configuration; results are saved per configuration.")
    config_model = MatrixConfig
    resumable = True

    # ------------------------------------------------------------------ validation
    def preflight(self, config: BaseModel) -> PreflightResult:
        cfg = MatrixConfig.model_validate(config)
        settings = get_settings()
        plans = plan_all(cfg, settings, vm.make_docker(settings), allow_run_oneoff=False)
        feasible = [p for p in plans if not p.errors]
        if not feasible:
            raise ValueError("No configuration in the matrix is feasible. " + " | ".join(f"{p.sc.label}: {'; '.join(p.errors)}" for p in plans))
        warnings: list[str] = []
        infeasible = [p for p in plans if p.errors]
        if infeasible:
            warnings.append(f"{len(infeasible)} of {len(plans)} configurations are infeasible and will be recorded as skipped.")
        for p in feasible:
            warnings += [f"{p.sc.label}: {w}" for w in p.plan.warnings + p.cells["notes"] if "supported-architecture list is not cached" not in w]
        if "retrieval_needle" in cfg.evals and cfg.eval_split == "dev":
            warnings.append("Evaluating on the dev split: use held-out cases when comparing configurations.")
        n_perf = sum(cfg.level_requests(c) * cfg.iterations + cfg.level_warmup(c) for c in cfg.concurrency_levels) * len(cfg.io_pairs) if cfg.perf_enabled else 0
        info = {"configs": [{"key": p.sc.key, "index": p.sc.index, "label": p.sc.label, "status": "infeasible" if p.errors else "feasible",
                             "errors": p.errors, "warnings": p.plan.warnings, "notes": p.cells["notes"], "estimates": p.plan.estimates,
                             "io_pairs": [i.model_dump() for i in p.cells["io_pairs"]], "evals": p.cells["evals"],
                             "retrieval_lengths": p.cells["retrieval_lengths"], **p.sc.loading_config()} for p in plans],
                "launches": len(feasible), "perf_requests_per_launch": n_perf, "gpu": feasible[0].plan.gpu.as_dict(), "split": cfg.eval_split}
        return PreflightResult(warnings=warnings, info=info)

    # ------------------------------------------------------------------ execution
    def run(self, config: BaseModel, ctx: ExperimentContext) -> dict[str, Any]:
        return asyncio.run(self._run(MatrixConfig.model_validate(config), ctx))

    async def _run(self, cfg: MatrixConfig, ctx: ExperimentContext) -> dict[str, Any]:
        settings = get_settings()
        docker = vm.owned_docker(settings)
        with session_scope() as s:
            experiment_id = s.get(Job, ctx.job_id).experiment_id  # type: ignore[union-attr]
        t_start = _now()
        ctx.set_progress({"phase": "planning"})
        plans = await asyncio.to_thread(plan_all, cfg, settings, docker.docker, True)
        row_ids = self._init_rows(experiment_id, ctx.job_id, plans)
        total = len(plans)
        stopped_reason: str | None = None
        user_cancelled = False
        try:
            for pos, mp in enumerate(plans):
                rid = row_ids[mp.sc.key]
                with session_scope() as s:
                    status = s.get(ConfigResult, rid).status  # type: ignore[union-attr]
                if status in ("completed", "skipped"):
                    ctx.log("INFO", f"[{pos + 1}/{total}] {mp.sc.label}: {status}, nothing to do")
                    continue
                if _now() - t_start >= cfg.max_runtime_s:
                    stopped_reason = "budget_exhausted"
                    self._mark_remaining(row_ids, plans[pos:], "skipped", "budget_exhausted",
                                         f"runtime budget of {cfg.max_runtime_s}s was exhausted before this configuration started")
                    ctx.log("WARNING", f"runtime budget exhausted; skipping {len(plans) - pos} remaining configuration(s)")
                    break
                if ctx.should_cancel():
                    user_cancelled = True
                    break
                await self._run_one(cfg, mp, rid, pos, total, ctx, settings, docker, experiment_id)
        except Cancelled:
            user_cancelled = True
            self._mark_remaining(row_ids, plans, "cancelled", "cancelled", "the job was cancelled", only_pending=True)
            raise
        if user_cancelled:
            self._mark_remaining(row_ids, plans, "cancelled", "cancelled", "the job was cancelled", only_pending=True)
            raise Cancelled()
        return self._summary(cfg, experiment_id, t_start, stopped_reason)

    def _init_rows(self, experiment_id: int, job_id: int, plans: list[MatrixPlan]) -> dict[str, int]:
        """Create (or on resume, reset) one row per configuration; completed rows are kept as they are."""
        ids: dict[str, int] = {}
        with session_scope() as s:
            existing = {r.key: r for r in s.scalars(select(ConfigResult).where(ConfigResult.experiment_id == experiment_id))}
            for mp in plans:
                r = existing.get(mp.sc.key)
                if r is None:
                    r = ConfigResult(experiment_id=experiment_id, key=mp.sc.key, label=mp.sc.label, config=mp.sc.loading_config(), attempts=0)
                    s.add(r)
                if r.status != "completed":
                    r.job_id, r.results, r.started_at, r.finished_at, r.managed_server_id = job_id, None, None, None, None
                    r.status, r.failure_kind, r.error = ("skipped", "infeasible", "; ".join(mp.errors)) if mp.errors else ("pending", None, None)
                s.flush()
                ids[mp.sc.key] = r.id
        return ids

    @staticmethod
    def _save(rid: int, **fields: Any) -> None:
        with session_scope() as s:
            row = s.get(ConfigResult, rid)
            for k, v in fields.items():
                setattr(row, k, v)

    def _mark_remaining(self, row_ids: dict[str, int], plans: list[MatrixPlan], status: str, kind: str, why: str, only_pending: bool = False) -> None:
        with session_scope() as s:
            for mp in plans:
                r = s.get(ConfigResult, row_ids[mp.sc.key])
                if r.status in ("pending",) or (not only_pending and r.status == "running"):
                    r.status, r.failure_kind, r.error, r.finished_at = status, kind, why, utcnow()

    async def _run_one(self, cfg: MatrixConfig, mp: MatrixPlan, rid: int, pos: int, total: int, ctx: ExperimentContext, settings: Settings,
                       docker: Any, experiment_id: int) -> None:
        sc = mp.sc
        ctx.log("INFO", f"[{pos + 1}/{total}] {sc.label}: starting")
        self._save(rid, status="running", started_at=utcnow(), job_id=ctx.job_id, error=None, failure_kind=None)
        with session_scope() as s:
            s.get(ConfigResult, rid).attempts += 1  # type: ignore[union-attr]
        server = vm.ManagedVllmServer(mp.plan, docker, settings, ctx.job_id, experiment_id, ctx.log)
        state: dict[str, Any] = {"performance": None, "quality": {"split": cfg.eval_split, "suites": {}}, "warnings": [], "cases": {},
                                 "perf_requests": [], "calibration": None}
        status, kind, error = "completed", None, None
        try:
            await asyncio.wait_for(self._session(cfg, mp, pos, total, ctx, settings, docker, experiment_id, server, state), timeout=cfg.config_timeout_s)
        except (Cancelled, WorkerShutdown) as e:
            status, kind, error = ("cancelled", "cancelled", "cancelled by user") if isinstance(e, Cancelled) else ("failed", "interrupted", "interrupted by worker shutdown")
            self._finalize_row(rid, cfg, mp, server, state, status, kind, error, ctx)
            raise
        except asyncio.TimeoutError:
            status, kind, error = "failed", "config_timeout", f"configuration exceeded config_timeout_s={cfg.config_timeout_s}"
        except vm.ManagedServerError as e:
            status, kind, error = ("cancelled", "cancelled", "cancelled during server startup") if e.kind == "cancelled" else ("failed", e.kind, str(e))
            if e.kind == "cancelled":
                self._finalize_row(rid, cfg, mp, server, state, status, kind, error, ctx)
                raise Cancelled() from e
        except ServerCrashed as e:
            status, kind, error = "failed", "server_crashed", str(e)
        except ExpectedFailure as e:
            status, kind, error = "failed", type(e).__name__, str(e)
        except Exception:  # unexpected: record and keep the rest of the matrix going
            status, kind, error = "failed", "unexpected_error", traceback.format_exc()[-1500:]
            log.exception("configuration %s failed unexpectedly", sc.label)
        self._finalize_row(rid, cfg, mp, server, state, status, kind, error, ctx)
        ctx.log("INFO" if status == "completed" else "ERROR", f"[{pos + 1}/{total}] {sc.label}: {status}" + (f" ({kind})" if kind else ""))

    def _finalize_row(self, rid: int, cfg: MatrixConfig, mp: MatrixPlan, server: vm.ManagedVllmServer, state: dict[str, Any], status: str,
                      kind: str | None, error: str | None, ctx: ExperimentContext) -> None:
        """Always runs: stop the server, write artifacts, persist this configuration's row."""
        server.shutdown("configuration finished" if status == "completed" else f"configuration {status}: {kind}")
        arts: dict[str, str] = {}
        key = mp.sc.key
        if state["cases"]:
            ctx.write_artifact(f"cfg_{key}_cases.json", "eval_cases", {"split": cfg.eval_split, "suites": state["cases"]})
            arts["cases"] = f"cfg_{key}_cases.json"
        if state["perf_requests"]:
            ctx.write_artifact(f"cfg_{key}_perf_requests.json", "requests", {"complete": status == "completed", "cells": state["perf_requests"]})
            arts["perf_requests"] = f"cfg_{key}_perf_requests.json"
        if server.final_logs:
            ctx.write_artifact(f"cfg_{key}_server.log", "log", {"container": server.container_name, "lines": server.final_logs.splitlines()[-2000:]})
            arts["server_log"] = f"cfg_{key}_server.log"
        results = self._results(cfg, mp, server, state, status != "completed", arts)
        with session_scope() as s:
            row = s.get(ConfigResult, rid)
            row.status, row.failure_kind, row.error, row.finished_at = status, kind, error, utcnow()
            row.results = results
            row.managed_server_id = server.row_id

    def _results(self, cfg: MatrixConfig, mp: MatrixPlan, server: vm.ManagedVllmServer, state: dict[str, Any], partial: bool, arts: dict[str, str]) -> dict[str, Any]:
        others: dict[tuple, None] = {}
        for cell in (state["performance"] or {}).get("cells", []):
            for lv in cell["levels"]:
                for p in lv.get("other_compute_processes", []) if lv["status"] == "ok" else []:
                    others[(p["pid"], p["name"], p["memory_mb"])] = None
        pre = mp.plan.gpu.as_dict()
        ms = None
        if server.row_id:
            with session_scope() as s:
                r = s.get(ManagedServer, server.row_id)
                ms = {"managed_server_id": r.id, "container_name": r.container_name, "image_ref": r.image_ref, "image_id": r.image_id,
                      "image_digest": r.image_digest, "command": r.command, "startup_s": r.startup_s}
        return {
            "schema_version": SCHEMA_VERSION, "partial": partial, "server_config": mp.sc.loading_config(),
            "runtime_config": mp.plan.config.model_dump(exclude={"allow_contention"}), "model_registry": mp.plan.model,
            "managed_server": ms, "startup_s": server.startup_s, "runtime": state.get("runtime"),
            "gpu": {"preflight": pre, "contended": bool(others) or pre["contended"], "contention_override": cfg.allow_contention,
                    "other_compute_processes": [{"pid": p, "name": n, "memory_mb": m} for p, n, m in others],
                    "scope": "peak_memory_mb is device-wide; server_peak_memory_mb is per-process (managed containers only)"},
            "performance": state["performance"], "quality": state["quality"] if state["quality"]["suites"] else None,
            "notes": mp.cells["notes"], "warnings": state["warnings"], "artifacts": arts, "calibration": state["calibration"],
        }

    async def _session(self, cfg: MatrixConfig, mp: MatrixPlan, pos: int, total: int, ctx: ExperimentContext, settings: Settings,
                       docker: Any, experiment_id: int, server: vm.ManagedVllmServer, state: dict[str, Any]) -> None:
        """Launch, measure, and always release the sampler/adapter (the server itself is stopped in _finalize_row)."""
        bench = InferenceBenchmark()
        sampler: GpuSampler | None = None
        adapter = None
        matrix_ctx = {"matrix": {"config_index": pos, "configs_total": total, "label": mp.sc.label}}
        try:
            # The GPU can change between configurations (another job starts); re-check right before every launch.
            pre = await asyncio.to_thread(check_gpu, cfg.gpu_index, cfg.gpu_memory_utilization, cfg.allow_contention)
            mp.plan.gpu = pre  # results record the state the launch actually saw
            if pre.errors:
                kind = "insufficient_free_memory" if any("cannot be overridden" in e for e in pre.errors) else "gpu_contended"
                raise vm.ManagedServerError(kind, "; ".join(pre.errors))
            for w in pre.warnings:
                ctx.log("WARNING", f"{mp.sc.label}: {w}")
            ctx.set_progress({"phase": "launching_server", **matrix_ctx})
            await asyncio.to_thread(server.launch)
            t_wait = time.monotonic()

            def tick() -> bool:
                ctx.set_progress({"phase": "waiting_for_server", "elapsed_s": round(time.monotonic() - t_wait, 1),
                                  "timeout_s": cfg.startup_timeout_s, **matrix_ctx})
                return ctx.should_cancel()

            await asyncio.to_thread(server.wait_ready, tick)
            adapter = server.adapter()
            info = await adapter.probe()
            served = mp.plan.served_name
            if served not in [m.id for m in info.models]:
                raise ExpectedFailure(f"Server does not serve {served!r}; it serves {[m.id for m in info.models]}")
            state["runtime"] = {"name": "vllm", "version": info.version, "kv_cache_tokens": info.kv_cache_tokens, "server_config": info.extra.get("cache_config"),
                                "served_model": next(({"id": m.id, "root": m.root, "max_model_len": m.max_model_len} for m in info.models if m.id == served), None)}
            sampler = GpuSampler(pid_provider=server.server_pids)
            sampler.start()
            baseline = list(sampler.samples)
            progress = Progress()
            progress.context = dict(matrix_ctx)
            exc: list[BaseException] = []
            body = asyncio.create_task(self._measure(cfg, mp, ctx, settings, docker, experiment_id, adapter, server, served, progress, state, sampler, baseline, bench))
            monitor = asyncio.create_task(bench._monitor(ctx, progress, body, exc, health=server.check_alive))
            try:
                await body
            except asyncio.CancelledError:
                if not exc:
                    raise
            finally:
                monitor.cancel()
            if exc:
                raise exc[0]
        finally:
            if sampler:
                sampler.stop()
            if adapter:
                await adapter.aclose()

    async def _measure(self, cfg, mp, ctx, settings, docker, experiment_id, adapter, server, served, progress, state, sampler, baseline, bench) -> None:
        sc, plan = mp.sc, mp.plan
        sweeper = ConcurrencySweep()
        url = server.base_url
        if cfg.perf_enabled:
            state["performance"] = {"workload_kind": "synthetic_random_token_ids (not evaluation prompts)", "cells": []}
            for io_i, pair in enumerate(mp.cells["io_pairs"]):
                sweep = SweepConfig(
                    mode="managed", registry_model_id=sc.model_id, runtime=plan.config, concurrency_levels=cfg.concurrency_levels,
                    input_tokens=pair.input_tokens, output_tokens=pair.output_tokens, requests_per_level=cfg.requests_per_level,
                    min_requests_per_slot=cfg.min_requests_per_slot, warmup_requests=cfg.warmup_requests, iterations=cfg.iterations,
                    temperature=cfg.temperature, seed=cfg.seed * 7919 + sc.index * 101 + io_i, request_timeout_s=cfg.request_timeout_s,
                    abort_after_failures=cfg.abort_after_failures)
                levels: list[dict[str, Any]] = []
                st: dict[str, Any] = {"stopped_early": None}
                per_level: list = []
                cell = {"io": pair.model_dump(), "levels": levels, "analysis": None, "stopped_early": None}
                state["performance"]["cells"].append(cell)  # partial levels survive a crash because `levels` is filled in place
                state["perf_requests"].append({"io": pair.model_dump(), "levels": [{"concurrency": c, "requests": [r.to_dict() for r in recs]} for c, recs, _ in per_level]})
                try:
                    await sweeper._levels(sweep, bench, ctx, adapter, url, served, plan, progress, per_level, sampler, baseline, server, levels, st)
                finally:
                    cell["stopped_early"] = st["stopped_early"]
                    cell["analysis"] = analyze_levels(levels, contended=plan.gpu.contended) if any(lv["status"] == "ok" for lv in levels) else None
                    state["perf_requests"][-1]["levels"] = [{"concurrency": c, "requests": [r.to_dict() for r in recs]} for c, recs, _ in per_level if recs]
                if st["stopped_early"] and st["stopped_early"].startswith("server crashed"):
                    raise ServerCrashed(f"Inference server crashed during performance cell {pair.input_tokens}/{pair.output_tokens}: {st['stopped_early']}")
                if not any(lv["status"] == "ok" for lv in levels):
                    state["warnings"].append(f"performance cell {pair.input_tokens}/{pair.output_tokens}: no level produced a valid measurement")
        sandbox = DockerSandbox(docker.docker, {**docker.labels_for(ctx.job_id, experiment_id), "locallab.role": "sandbox"}, settings.sandbox_image)
        for name in mp.cells["evals"]:
            problem = await asyncio.to_thread(server.check_alive)
            if problem:  # never score a dead server's connection errors as model failures
                raise ServerCrashed(f"Inference server crashed before evaluating {name}: {problem}")
            suite = SUITES[name]
            params: dict[str, Any] = {}
            if name == "retrieval_needle":
                if state["calibration"] is None:
                    cpt, measured = await calibrate_chars_per_token(adapter, served)
                    state["calibration"] = {"chars_per_token": cpt, "measured_with_server_tokenizer": measured}
                    if not measured:
                        state["warnings"].append("server tokenizer unavailable; retrieval prompt lengths use a 4 chars/token estimate (actual lengths are recorded)")
                params = {"lengths": mp.cells["retrieval_lengths"], "positions": cfg.retrieval_positions, "repeats": cfg.retrieval_repeats,
                          "chars_per_token": state["calibration"]["chars_per_token"]}
            cases = suite.cases(cfg.eval_split, **params)
            progress.start_phase(f"eval:{name}", 0, len(cases), time.perf_counter())
            ctx.log("INFO", f"{sc.label}: evaluating {name} ({len(cases)} {cfg.eval_split} cases)")

            def on_progress(done: int, tot: int, passed: int) -> None:
                progress.done, progress.ok, progress.failed = done, passed, done - passed

            result, recs = await run_suite(adapter, served, suite, cases, sandbox, concurrency=cfg.eval_concurrency, timeout_s=cfg.eval_timeout_s,
                                           enable_thinking=cfg.enable_thinking, split=cfg.eval_split, on_progress=on_progress)
            if name == "retrieval_needle":
                got = [r["prompt_tokens"] for r in recs if r.get("prompt_tokens")]
                if got:
                    err = sum(abs(r["prompt_tokens"] - r["tags"]["target_tokens"]) / r["tags"]["target_tokens"] for r in recs if r.get("prompt_tokens")) / len(got)
                    result["length_error"] = err
                    if err > 0.10:
                        state["warnings"].append(f"retrieval prompts differ from their target length by {err:.0%} on average; use the recorded prompt_tokens")
            if result["performance"]["requests"]["failed"]:
                state["warnings"].append(f"{name}: {result['performance']['requests']['failed']} of {result['n']} generations failed "
                                         f"({result['performance']['requests']['errors_by_type']}); they count as failures")
            state["quality"]["suites"][name] = result
            state["cases"][name] = recs

    # ------------------------------------------------------------------ summary
    def _summary(self, cfg: MatrixConfig, experiment_id: int, t_start: float, stopped_reason: str | None) -> dict[str, Any]:
        with session_scope() as s:
            rows = list(s.scalars(select(ConfigResult).where(ConfigResult.experiment_id == experiment_id).order_by(ConfigResult.id)))
            configs = [{"key": r.key, "label": r.label, "status": r.status, "failure_kind": r.failure_kind, "error": (r.error or "")[:300]} for r in rows]
        counts: dict[str, int] = {}
        for c in configs:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        if not counts.get("completed"):
            raise ExpectedFailure("No configuration completed. " + " | ".join(_describe(c) for c in configs))
        return {"kind": "optimization_matrix", "schema_version": SCHEMA_VERSION, "split": cfg.eval_split, "counts": counts, "configs": configs,
                "elapsed_s": round(_now() - t_start, 1), "stopped_reason": stopped_reason}
