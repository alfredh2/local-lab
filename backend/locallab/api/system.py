from fastapi import APIRouter, Request

from locallab import __version__
from locallab.config import get_settings
from locallab.schemas.system import GpuHistoryPoint, GpuSnapshot, SystemStatus
from locallab.services import gpu as gpu_service
from locallab.services import system as system_service

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/system/status", response_model=SystemStatus)
def system_status() -> SystemStatus:
    return system_service.system_status(get_settings())


@router.get("/gpu", response_model=GpuSnapshot)
def gpu() -> GpuSnapshot:
    return gpu_service.read_gpus()


@router.get("/gpu/history", response_model=list[GpuHistoryPoint])
def gpu_history(request: Request) -> list[GpuHistoryPoint]:
    return request.app.state.gpu_history.points()
