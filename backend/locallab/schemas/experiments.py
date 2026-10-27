from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentCreate(BaseModel):
    type: str
    name: str | None = Field(None, max_length=255)
    config: dict[str, Any] = {}
    seed: int | None = None


class ExperimentTypeOut(BaseModel):
    type: str
    title: str
    description: str
    config_schema: dict[str, Any]
    presets: dict[str, dict[str, Any]] = {}


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    status: str
    cancel_requested: bool
    worker_id: str | None
    error: str | None
    seed: int | None
    results: dict[str, Any] | None
    progress: dict[str, Any] | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    type: str
    name: str
    config: dict[str, Any]
    config_hash: str
    created_at: datetime
    latest_job: JobOut | None = None


class ExperimentDetail(ExperimentOut):
    environment: dict[str, Any]
    jobs: list[JobOut]


class JobLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ts: datetime
    level: str
    message: str


class ArtifactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    name: str
    kind: str
    size_bytes: int
    sha256: str
    created_at: datetime


class ValidationReport(BaseModel):
    ok: bool
    errors: list[str]
    warnings: list[str]
    info: dict[str, Any] | None = None
