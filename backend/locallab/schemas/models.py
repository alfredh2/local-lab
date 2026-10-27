from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: str
    ref: str
    revision: str | None
    format: str | None
    quantization: str | None
    size_bytes: int | None
    meta: dict[str, Any]
    available: bool
    discovered_at: datetime
    last_seen_at: datetime


class ManualModelIn(BaseModel):
    name: str
    path: str


class ScanResult(BaseModel):
    discovered: int
    added: int
    updated: int
    marked_unavailable: int
    warnings: list[str]
