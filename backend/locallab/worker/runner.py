import logging
import os
import signal
import socket
import threading
import time
import traceback

from locallab.config import Settings, get_settings
from locallab.db import init_db, session_scope
from locallab.errors import ExpectedFailure
from locallab.experiments.base import Cancelled
from locallab.experiments.registry import get_experiment
from locallab.logging import configure_logging
from locallab.models import Job, JobStatus
from locallab.runtimes.docker import DockerError
from locallab.runtimes.vllm_managed import owned_docker, reconcile_orphans
from locallab.services import jobs as job_service
from locallab.services.worker_heartbeat import write_heartbeat
from locallab.worker.context import JobContext, WorkerShutdown

log = logging.getLogger("locallab.worker")


class Worker:
    def __init__(self, settings: Settings, worker_id: str | None = None) -> None:
        self.settings = settings
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self.shutdown = threading.Event()
        self._last_reconcile = 0.0

    def run_forever(self) -> None:
        log.info("worker %s started", self.worker_id)
        while not self.shutdown.is_set():
            self.tick()
            self.shutdown.wait(self.settings.worker_poll_s)
        log.info("worker %s stopped", self.worker_id)

    def tick(self) -> bool:
        """One poll iteration. Returns True if a job was processed."""
        write_heartbeat(self.settings, self.worker_id)
        with session_scope() as s:
            for jid in job_service.recover_stale(s, self.settings.worker_stale_s):
                log.warning("recovered stale job %s -> failed", jid)
        self._reconcile()
        with session_scope() as s:
            job = job_service.claim_next(s, self.worker_id)
            job_id = job.id if job else None
        if job_id is None:
            return False
        self.execute(job_id)
        return True

    def _reconcile(self) -> None:
        """Remove LocalLab-owned containers left behind by finished/dead jobs (runs after stale-job recovery)."""
        if time.monotonic() - self._last_reconcile < self.settings.reconcile_interval_s:
            return
        self._last_reconcile = time.monotonic()
        try:
            for name in reconcile_orphans(owned_docker(self.settings), self.settings):
                log.warning("removed orphaned managed container %s", name)
        except DockerError as e:
            log.warning("container reconciliation skipped: %s", e)

    def execute(self, job_id: int) -> None:
        ctx: JobContext | None = None
        try:
            with session_scope() as s:
                job = s.get(Job, job_id)
                assert job is not None
                exp = job.experiment
                exp_type, config_raw, seed = get_experiment(exp.type), exp.config, job.seed
                if job.cancel_requested:
                    job_service.transition(s, job, JobStatus.cancelled)
                    return
                config = exp_type.config_model.model_validate(config_raw)
                job_service.transition(s, job, JobStatus.running)
            ctx = JobContext(job_id, seed, self.settings, self.shutdown, self.worker_id)
            results = exp_type.run(config, ctx)
        except Cancelled:
            self._finish(job_id, JobStatus.cancelled)
        except WorkerShutdown:
            self._finish(job_id, JobStatus.failed, error="Interrupted by worker shutdown; results incomplete")
        except ExpectedFailure as e:
            log.error("job %s failed: %s", job_id, e)
            self._finish(job_id, JobStatus.failed, error=str(e))
        except Exception:
            tb = traceback.format_exc()
            log.exception("job %s failed", job_id)
            self._finish(job_id, JobStatus.failed, error=tb)
        else:
            self._finish(job_id, JobStatus.completed, results=results)

    def _finish(self, job_id: int, status: JobStatus, *, error: str | None = None,
                results: dict | None = None) -> None:
        with session_scope() as s:
            job = s.get(Job, job_id)
            assert job is not None
            if JobStatus(job.status).terminal:
                log.warning("job %s already %s; not overwriting with %s", job_id, job.status, status.value)
                return
            if job.status == JobStatus.preparing.value and status is JobStatus.completed:
                raise job_service.InvalidTransition("completed without running")
            if error:
                job_service.add_log(s, job_id, "ERROR", error.strip().splitlines()[-1])
            job_service.transition(s, job, status, error=error, results=results)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    init_db(settings)
    worker = Worker(settings)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.shutdown.set())
    worker.run_forever()
