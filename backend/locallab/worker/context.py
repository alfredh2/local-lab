import hashlib
import json
import time
from pathlib import Path
from typing import Any

from locallab.config import Settings
from locallab.db import session_scope
from locallab.errors import WorkerShutdown  # noqa: F401  (re-exported for the runner)
from locallab.experiments.base import Cancelled
from locallab.models import Artifact, Job, utcnow
from locallab.services import jobs as job_service
from locallab.services.worker_heartbeat import write_heartbeat


class JobContext:
    HEARTBEAT_EVERY_S = 2.0

    def __init__(self, job_id: int, seed: int | None, settings: Settings, shutdown, worker_id: str | None = None) -> None:
        self.job_id = job_id
        self.seed = seed
        self._settings = settings
        self._shutdown = shutdown  # threading.Event
        self._worker_id = worker_id
        self._last_hb = 0.0
        self._last_progress = 0.0

    def log(self, level: str, message: str) -> None:
        with session_scope() as s:
            job_service.add_log(s, self.job_id, level, message)

    def should_cancel(self) -> bool:
        if self._shutdown.is_set():
            raise WorkerShutdown("worker shutting down")
        now = time.monotonic()
        if now - self._last_hb < self.HEARTBEAT_EVERY_S and self._last_hb:
            return False
        self._last_hb = now
        if self._worker_id:
            write_heartbeat(self._settings, self._worker_id)  # stay 'alive' while a long job runs
        with session_scope() as s:
            job = s.get(Job, self.job_id)
            assert job is not None
            job.heartbeat_at = utcnow()
            return job.cancel_requested

    def set_progress(self, progress: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last_progress < 0.9:
            return
        self._last_progress = now
        with session_scope() as s:
            job = s.get(Job, self.job_id)
            assert job is not None
            job.progress = progress

    def check_cancel(self) -> None:
        # Poll the DB flag on every check at the heartbeat cadence; cancellation latency <= ~2s.
        if self.should_cancel():
            raise Cancelled()

    def write_artifact(self, name: str, kind: str, data: Any) -> None:
        safe = Path(name).name  # no path components from experiment code
        directory = self._settings.artifacts_dir / str(self.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / safe
        raw = json.dumps(data).encode()
        path.write_bytes(raw)
        with session_scope() as s:
            s.add(Artifact(job_id=self.job_id, name=safe, kind=kind, path=str(path),
                           size_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
