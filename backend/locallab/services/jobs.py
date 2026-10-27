"""Job state machine and queue operations. Shared by API and worker."""
import logging
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from locallab.models import Job, JobLog, JobStatus, utcnow

log = logging.getLogger(__name__)
ACTIVE = (JobStatus.preparing.value, JobStatus.running.value)

ALLOWED: dict[str, set[str]] = {
    JobStatus.queued.value: {JobStatus.preparing.value, JobStatus.cancelled.value},
    JobStatus.preparing.value: {JobStatus.running.value, JobStatus.failed.value, JobStatus.cancelled.value},
    JobStatus.running.value: {JobStatus.completed.value, JobStatus.failed.value, JobStatus.cancelled.value},
}


class InvalidTransition(Exception):
    pass


class JobConflict(Exception):
    """Operation not valid for the job's current state."""


def add_log(session: Session, job_id: int, level: str, message: str) -> None:
    session.add(JobLog(job_id=job_id, level=level.upper(), message=message))
    log.log(getattr(logging, level.upper(), logging.INFO), "job %s: %s", job_id, message)


def transition(session: Session, job: Job, new: JobStatus, *, error: str | None = None,
               results: dict | None = None) -> None:
    if new.value not in ALLOWED.get(job.status, set()):
        raise InvalidTransition(f"job {job.id}: {job.status} -> {new.value}")
    now = utcnow()
    job.status = new.value
    job.heartbeat_at = now
    if new is JobStatus.running:
        job.started_at = now
    if new.terminal:
        job.finished_at = now
        job.error = error
        if new is JobStatus.completed:
            job.results = results
    add_log(session, job.id, "ERROR" if new is JobStatus.failed else "INFO", f"status -> {new.value}")


def claim_next(session: Session, worker_id: str) -> Job | None:
    """Atomically move the oldest queued job to `preparing`. Safe with multiple workers."""
    now = utcnow()
    candidate = (
        select(Job.id).where(Job.status == JobStatus.queued.value, Job.cancel_requested.is_(False))
        .order_by(Job.id).limit(1).scalar_subquery()
    )
    res = session.execute(
        update(Job).where(Job.id == candidate, Job.status == JobStatus.queued.value)
        .values(status=JobStatus.preparing.value, worker_id=worker_id, heartbeat_at=now)
        .returning(Job.id)
    ).first()
    if res is None:
        return None
    session.flush()
    job = session.get(Job, res[0], populate_existing=True)
    assert job is not None
    add_log(session, job.id, "INFO", f"claimed by {worker_id}; status -> preparing")
    return job


def request_cancel(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise LookupError(f"job {job_id} not found")
    if JobStatus(job.status).terminal:
        raise JobConflict(f"job already {job.status}")
    if job.status == JobStatus.queued.value:
        now = utcnow()
        done = session.execute(
            update(Job).where(Job.id == job_id, Job.status == JobStatus.queued.value)
            .values(status=JobStatus.cancelled.value, cancel_requested=True, finished_at=now)
        ).rowcount
        if done:
            session.refresh(job)
            add_log(session, job_id, "INFO", "cancelled while queued")
            return job
        session.refresh(job)  # a worker claimed it between our read and update
    job.cancel_requested = True
    add_log(session, job_id, "INFO", "cancellation requested")
    return job


def recover_stale(session: Session, stale_after_s: float) -> list[int]:
    """Fail active jobs whose worker stopped heartbeating. Never requeue or complete them."""
    cutoff = utcnow() - timedelta(seconds=stale_after_s)
    stale = session.scalars(
        select(Job).where(Job.status.in_(ACTIVE), (Job.heartbeat_at.is_(None)) | (Job.heartbeat_at < cutoff))
    ).all()
    for job in stale:
        job.error = (f"Worker {job.worker_id} stopped heartbeating while job was {job.status}; "
                     "run was interrupted and results are incomplete")
        job.status = JobStatus.failed.value
        job.finished_at = utcnow()
        add_log(session, job.id, "ERROR", "marked failed: worker died or restarted mid-run")
    return [j.id for j in stale]
