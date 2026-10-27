"""Worker liveness via a small JSON file in the data dir (works while idle, no DB writes)."""
import json
import os
from dataclasses import dataclass
from datetime import datetime

from locallab.config import Settings
from locallab.models import utcnow


@dataclass
class Heartbeat:
    worker_id: str
    timestamp: datetime


def _path(settings: Settings):
    return settings.data_dir / "worker_heartbeat.json"


def write_heartbeat(settings: Settings, worker_id: str) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    tmp = _path(settings).with_suffix(".tmp")
    tmp.write_text(json.dumps({"worker_id": worker_id, "ts": utcnow().isoformat()}))
    os.replace(tmp, _path(settings))


def read_heartbeat(settings: Settings) -> Heartbeat | None:
    try:
        raw = json.loads(_path(settings).read_text())
        return Heartbeat(worker_id=raw["worker_id"], timestamp=datetime.fromisoformat(raw["ts"]))
    except FileNotFoundError:
        return None
