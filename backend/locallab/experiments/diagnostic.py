"""Diagnostic experiment: samples real GPU/CPU/RAM telemetry for a fixed window.

Not a benchmark -- it loads no model and measures only host telemetry. It exists
to exercise the full experiment lifecycle with genuine measurements.
"""
import statistics
import time
from datetime import datetime, timezone
from typing import Any

import psutil
from pydantic import BaseModel, Field

from locallab.experiments.base import Experiment, ExperimentContext
from locallab.services.gpu import read_gpus


class DiagnosticConfig(BaseModel):
    duration_s: int = Field(10, ge=1, le=120, description="Sampling window in seconds")
    sample_interval_ms: int = Field(500, ge=250, le=10_000)
    include_cpu_mem: bool = True


def _summ(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


class DiagnosticExperiment(Experiment):
    type_name = "diagnostic"
    title = "System diagnostic"
    description = "Samples real GPU, CPU and RAM telemetry for a short window. Loads no model."
    config_model = DiagnosticConfig

    def run(self, config: BaseModel, ctx: ExperimentContext) -> dict[str, Any]:
        cfg = DiagnosticConfig.model_validate(config)
        interval = cfg.sample_interval_ms / 1000
        ctx.log("INFO", f"Sampling for {cfg.duration_s}s every {cfg.sample_interval_ms}ms")
        samples: list[dict[str, Any]] = []
        if cfg.include_cpu_mem:
            psutil.cpu_percent(None)  # prime; first call returns a meaningless 0.0
        start = time.monotonic()
        next_tick = start
        while True:
            ctx.check_cancel()
            snap = read_gpus()
            if not snap.available:
                raise RuntimeError(f"GPU telemetry unavailable: {snap.error}")
            s: dict[str, Any] = {
                "t": round(time.monotonic() - start, 3),
                "ts": datetime.now(timezone.utc).isoformat(),
                "gpus": [
                    {"index": g.index, "memory_used_mb": g.memory_used_mb, "utilization_gpu_pct": g.utilization_gpu_pct,
                     "temperature_c": g.temperature_c, "power_w": g.power_w, "clock_sm_mhz": g.clock_sm_mhz}
                    for g in snap.gpus
                ],
            }
            if cfg.include_cpu_mem:
                s["cpu_pct"] = psutil.cpu_percent(None)
                s["ram_used_gb"] = (psutil.virtual_memory().total - psutil.virtual_memory().available) / 1024**3
            samples.append(s)
            next_tick += interval
            if next_tick - start >= cfg.duration_s:
                break
            while (remaining := next_tick - time.monotonic()) > 0:
                ctx.check_cancel()
                time.sleep(min(remaining, 0.2))

        elapsed = time.monotonic() - start
        ctx.write_artifact("telemetry.json", "timeseries", {"interval_ms": cfg.sample_interval_ms, "samples": samples})
        gpus: dict[str, Any] = {}
        for idx in sorted({g["index"] for s in samples for g in s["gpus"]}):
            rows = [g for s in samples for g in s["gpus"] if g["index"] == idx]
            gpus[str(idx)] = {
                "memory_used_mb": _summ([r["memory_used_mb"] for r in rows]),
                "utilization_gpu_pct": _summ([r["utilization_gpu_pct"] for r in rows if r["utilization_gpu_pct"] is not None]),
                "temperature_c": _summ([r["temperature_c"] for r in rows if r["temperature_c"] is not None]),
                "power_w": _summ([r["power_w"] for r in rows if r["power_w"] is not None]),
            }
        results: dict[str, Any] = {
            "kind": "diagnostic",
            "requested_duration_s": cfg.duration_s,
            "actual_duration_s": round(elapsed, 3),
            "sample_count": len(samples),
            "gpus": gpus,
        }
        if cfg.include_cpu_mem:
            results["cpu_pct"] = _summ([s["cpu_pct"] for s in samples])
            results["ram_used_gb"] = _summ([s["ram_used_gb"] for s in samples])
        ctx.log("INFO", f"Collected {len(samples)} samples over {elapsed:.1f}s")
        return results
