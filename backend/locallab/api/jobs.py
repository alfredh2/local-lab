from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from locallab.config import get_settings
from locallab.db import get_session
from locallab.models import Artifact, Job, JobLog
from locallab.schemas.experiments import ArtifactOut, JobLogOut, JobOut
from locallab.services import jobs as job_service

router = APIRouter(tags=["jobs"])


def _job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: int, session: Session = Depends(get_session)):
    return _job(session, job_id)


@router.get("/jobs/{job_id}/logs", response_model=list[JobLogOut])
def get_logs(job_id: int, after_id: int = 0, session: Session = Depends(get_session)):
    _job(session, job_id)
    return session.scalars(select(JobLog).where(JobLog.job_id == job_id, JobLog.id > after_id).order_by(JobLog.id)).all()


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel(job_id: int, session: Session = Depends(get_session)):
    try:
        return job_service.request_cancel(session, job_id)
    except LookupError as e:
        raise HTTPException(404, str(e)) from e
    except job_service.JobConflict as e:
        raise HTTPException(409, str(e)) from e


@router.get("/jobs/{job_id}/artifacts", response_model=list[ArtifactOut])
def list_artifacts(job_id: int, session: Session = Depends(get_session)):
    _job(session, job_id)
    return session.scalars(select(Artifact).where(Artifact.job_id == job_id).order_by(Artifact.id)).all()


@router.get("/artifacts/{artifact_id}/content")
def artifact_content(artifact_id: int, session: Session = Depends(get_session)):
    art = session.get(Artifact, artifact_id)
    if art is None:
        raise HTTPException(404, "Artifact not found")
    path = Path(art.path).resolve()
    if not path.is_relative_to(get_settings().artifacts_dir.resolve()) or not path.is_file():
        raise HTTPException(404, "Artifact file missing")
    return FileResponse(path, media_type="application/json", filename=art.name)
