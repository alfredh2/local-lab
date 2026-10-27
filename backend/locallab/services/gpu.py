"""NVML access. All reads are real; failures are reported, never faked."""
import logging
import threading
from collections import deque
from datetime import datetime, timezone

import pynvml

from locallab.schemas.system import GpuHistoryPoint, GpuInfo, GpuProcess, GpuSnapshot

log = logging.getLogger(__name__)
_MB = 1024 * 1024
_init_lock = threading.Lock()
_initialized = False


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ensure_init() -> None:
    global _initialized
    with _init_lock:
        if not _initialized:
            pynvml.nvmlInit()
            _initialized = True


def _opt(fn, *args) -> float | None:
    """NVML reports unsupported metrics via NVMLError; map to None, nothing else."""
    try:
        return float(fn(*args))
    except pynvml.NVMLError_NotSupported:
        return None


def _s(v: str | bytes) -> str:
    return v.decode() if isinstance(v, bytes) else v


def _processes(handle) -> list[GpuProcess]:
    procs: list[GpuProcess] = []
    for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        try:
            for p in fn(handle):
                try:
                    name = _s(pynvml.nvmlSystemGetProcessName(p.pid))
                except pynvml.NVMLError:
                    name = None
                mem = None if p.usedGpuMemory is None else p.usedGpuMemory / _MB
                procs.append(GpuProcess(pid=p.pid, name=name, used_memory_mb=mem))
        except pynvml.NVMLError_NotSupported:
            continue
    return procs


def read_gpus() -> GpuSnapshot:
    try:
        _ensure_init()
        driver = _s(pynvml.nvmlSystemGetDriverVersion())
        cuda_raw = pynvml.nvmlSystemGetCudaDriverVersion()
        cuda = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
        gpus: list[GpuInfo] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                ugpu, umem = float(util.gpu), float(util.memory)
            except pynvml.NVMLError_NotSupported:
                ugpu = umem = None
            gpus.append(
                GpuInfo(
                    index=i,
                    name=_s(pynvml.nvmlDeviceGetName(h)),
                    uuid=_s(pynvml.nvmlDeviceGetUUID(h)),
                    memory_total_mb=mem.total / _MB,
                    memory_used_mb=mem.used / _MB,
                    memory_free_mb=mem.free / _MB,
                    utilization_gpu_pct=ugpu,
                    utilization_mem_pct=umem,
                    temperature_c=_opt(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU),
                    power_w=(p / 1000 if (p := _opt(pynvml.nvmlDeviceGetPowerUsage, h)) is not None else None),
                    power_limit_w=(
                        p / 1000 if (p := _opt(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)) is not None else None
                    ),
                    clock_sm_mhz=_opt(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_SM),
                    clock_mem_mhz=_opt(pynvml.nvmlDeviceGetClockInfo, h, pynvml.NVML_CLOCK_MEM),
                    processes=_processes(h),
                )
            )
        return GpuSnapshot(
            available=True, timestamp=_now(), driver_version=driver, cuda_driver_version=cuda, gpus=gpus
        )
    except pynvml.NVMLError as e:
        log.warning("NVML unavailable: %s", e)
        return GpuSnapshot(available=False, error=str(e), timestamp=_now())


class GpuHistory:
    """In-memory ring buffer filled by a sampler thread in the API process."""

    def __init__(self, maxlen: int = 900) -> None:
        self._points: deque[GpuHistoryPoint] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def record(self, snap: GpuSnapshot) -> None:
        with self._lock:
            for g in snap.gpus:
                self._points.append(
                    GpuHistoryPoint(
                        timestamp=snap.timestamp,
                        index=g.index,
                        memory_used_mb=g.memory_used_mb,
                        utilization_gpu_pct=g.utilization_gpu_pct,
                        temperature_c=g.temperature_c,
                        power_w=g.power_w,
                    )
                )

    def points(self) -> list[GpuHistoryPoint]:
        with self._lock:
            return list(self._points)

    def start(self, interval_s: float = 2.0) -> None:
        def loop() -> None:
            while not self._stop.wait(interval_s):
                self.record(read_gpus())

        self._thread = threading.Thread(target=loop, name="gpu-history", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
