"""Pre-launch GPU checks: free memory, competing compute processes, contention policy."""
import time
from dataclasses import asdict, dataclass, field

from locallab.services.gpu import compute_capability, read_gpus, sample_gpu_light

CONTENTION_UTIL_PCT = 25.0
SAFETY_MARGIN_MB = 256


@dataclass
class GpuPreflight:
    gpu_index: int
    name: str | None = None
    total_mb: float = 0.0
    free_mb: float = 0.0
    requested_mb: float = 0.0
    compute_capability: tuple[int, int] | None = None
    utilization_pct: float | None = None
    other_processes: list[dict] = field(default_factory=list)
    contended: bool = False
    override: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def check_gpu(gpu_index: int, requested_util: float, allow_contention: bool,
              exclude_pids: set[int] | None = None) -> GpuPreflight:
    pf = GpuPreflight(gpu_index=gpu_index, override=allow_contention)
    snap = read_gpus()
    if not snap.available:
        pf.errors.append(f"GPU telemetry unavailable: {snap.error}")
        return pf
    gpu = next((g for g in snap.gpus if g.index == gpu_index), None)
    if gpu is None:
        pf.errors.append(f"GPU {gpu_index} not found (have {[g.index for g in snap.gpus]})")
        return pf
    pf.name, pf.total_mb, pf.free_mb = gpu.name, gpu.memory_total_mb, gpu.memory_free_mb
    pf.compute_capability = compute_capability(gpu_index)
    pf.requested_mb = requested_util * gpu.memory_total_mb
    skip = exclude_pids or set()
    pf.other_processes = [{"pid": p.pid, "name": (p.name or "?")[:80], "memory_mb": round(p.used_memory_mb or 0)}
                          for p in gpu.processes if p.kind == "compute" and p.pid not in skip]
    utils = []
    for _ in range(3):  # NVML utilization is a short moving average; take a few readings
        s = next((x for x in sample_gpu_light() if x["index"] == gpu_index), None)
        if s and s["utilization_gpu_pct"] is not None:
            utils.append(s["utilization_gpu_pct"])
        time.sleep(0.15)
    pf.utilization_pct = sum(utils) / len(utils) if utils else None
    pf.contended = bool(pf.other_processes) or (pf.utilization_pct or 0) >= CONTENTION_UTIL_PCT

    if pf.requested_mb + SAFETY_MARGIN_MB > pf.free_mb:
        pf.errors.append(
            f"gpu_memory_utilization {requested_util:g} requests {pf.requested_mb:.0f} MB but only "
            f"{pf.free_mb:.0f} MB is free on GPU {gpu_index}; vLLM would refuse to start. "
            "Lower gpu_memory_utilization or wait for the GPU to free up. (This cannot be overridden.)")
    if pf.contended:
        who = ", ".join(f"{p['name'].split('/')[-1]} (pid {p['pid']}, {p['memory_mb']} MB)" for p in pf.other_processes) or "none visible"
        msg = (f"GPU {gpu_index} is in use by other work: processes [{who}], utilization "
               f"{pf.utilization_pct:.0f}%." if pf.utilization_pct is not None else f"GPU {gpu_index} has other compute processes [{who}].")
        if allow_contention:
            pf.warnings.append(msg + " Proceeding because allow_contention is set; results will be marked contended "
                                     "and latency/throughput are not representative of an idle GPU.")
        else:
            pf.errors.append(msg + " Refusing to benchmark on a contended GPU. Wait for it to be idle, or set "
                                   "allow_contention to run anyway (nothing will be stopped).")
    return pf
