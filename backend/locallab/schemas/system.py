from datetime import datetime

from pydantic import BaseModel


class GpuProcess(BaseModel):
    pid: int
    name: str | None
    used_memory_mb: float | None
    kind: str = "compute"  # compute | graphics


class GpuInfo(BaseModel):
    index: int
    name: str
    uuid: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_gpu_pct: float | None
    utilization_mem_pct: float | None
    temperature_c: float | None
    power_w: float | None
    power_limit_w: float | None
    clock_sm_mhz: float | None
    clock_mem_mhz: float | None
    processes: list[GpuProcess]


class GpuSnapshot(BaseModel):
    available: bool
    error: str | None = None
    timestamp: datetime
    driver_version: str | None = None
    cuda_driver_version: str | None = None
    gpus: list[GpuInfo] = []


class GpuHistoryPoint(BaseModel):
    timestamp: datetime
    index: int
    memory_used_mb: float
    utilization_gpu_pct: float | None
    temperature_c: float | None
    power_w: float | None


class WorkerStatus(BaseModel):
    alive: bool
    worker_id: str | None
    last_heartbeat: datetime | None


class SystemStatus(BaseModel):
    hostname: str
    os: str
    kernel: str
    cpu_model: str | None
    cpu_threads: int
    memory_total_gb: float
    memory_used_gb: float
    disk_total_gb: float
    disk_free_gb: float
    python_version: str
    locallab_version: str
    packages: dict[str, str]
    gpu: GpuSnapshot
    worker: WorkerStatus
