from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ConfigResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    job_id: int | None
    key: str
    label: str
    config: dict[str, Any]
    status: str
    failure_kind: str | None
    error: str | None
    attempts: int
    managed_server_id: int | None
    results: dict[str, Any] | None
    started_at: datetime | None
    finished_at: datetime | None
