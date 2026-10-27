from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class RuntimeConfigIn(BaseModel):
    name: str
    registry_model_id: int | None = None
    config: dict[str, Any]


class RuntimeConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    runtime: str
    registry_model_id: int | None
    config: dict[str, Any]
    created_at: datetime


class ManagedServerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int | None
    container_id: str | None
    container_name: str
    status: str
    image_ref: str
    image_id: str | None
    image_digest: str | None
    host_port: int | None
    config: dict[str, Any]
    command: list[str] | None
    model: dict[str, Any] | None
    launched_at: datetime
    ready_at: datetime | None
    startup_s: float | None
    stopped_at: datetime | None
    stop_reason: str | None
    exit_code: int | None
    oom_killed: bool | None
    log_tail: str | None = None
