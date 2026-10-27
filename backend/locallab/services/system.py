import importlib.metadata as md
import platform
import shutil
import socket
from datetime import timedelta

import psutil

from locallab import __version__
from locallab.config import Settings
from locallab.models import utcnow
from locallab.schemas.system import SystemStatus, WorkerStatus
from locallab.services.gpu import read_gpus
from locallab.services.worker_heartbeat import read_heartbeat

_GB = 1024**3
TRACKED_PACKAGES = ("fastapi", "sqlalchemy", "alembic", "pydantic", "nvidia-ml-py", "psutil", "vllm", "torch")


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return out


def cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def worker_status(settings: Settings) -> WorkerStatus:
    hb = read_heartbeat(settings)
    if hb is None:
        return WorkerStatus(alive=False, worker_id=None, last_heartbeat=None)
    alive = utcnow() - hb.timestamp < timedelta(seconds=settings.worker_stale_s)
    return WorkerStatus(alive=alive, worker_id=hb.worker_id, last_heartbeat=hb.timestamp)


def hardware_snapshot() -> dict:
    """Facts recorded with every experiment for reproducibility."""
    gpu = read_gpus()
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": cpu_model(),
        "cpu_threads": psutil.cpu_count(logical=True),
        "memory_total_gb": round(vm.total / _GB, 2),
        "python": platform.python_version(),
        "locallab_version": __version__,
        "packages": package_versions(),
        "nvidia_driver": gpu.driver_version,
        "cuda_driver_version": gpu.cuda_driver_version,
        "gpus": [
            {"index": g.index, "name": g.name, "uuid": g.uuid, "memory_total_mb": round(g.memory_total_mb)}
            for g in gpu.gpus
        ],
    }


def system_status(settings: Settings) -> SystemStatus:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(settings.data_dir if settings.data_dir.exists() else ".")
    return SystemStatus(
        hostname=socket.gethostname(),
        os=platform.platform(),
        kernel=platform.release(),
        cpu_model=cpu_model(),
        cpu_threads=psutil.cpu_count(logical=True) or 0,
        memory_total_gb=round(vm.total / _GB, 2),
        memory_used_gb=round((vm.total - vm.available) / _GB, 2),
        disk_total_gb=round(disk.total / _GB, 1),
        disk_free_gb=round(disk.free / _GB, 1),
        python_version=platform.python_version(),
        locallab_version=__version__,
        packages=package_versions(),
        gpu=read_gpus(),
        worker=worker_status(settings),
    )
