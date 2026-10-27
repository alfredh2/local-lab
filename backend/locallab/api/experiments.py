from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from locallab.db import get_session
from locallab.experiments.registry import list_experiments
from locallab.models import Experiment, Job, JobStatus
from locallab.schemas.experiments import ExperimentCreate, ExperimentDetail, ExperimentOut, ExperimentTypeOut, ValidationReport
from locallab.services import experiments as experiment_service
from locallab.services import jobs as job_service

router = APIRouter(tags=["experiments"])


def _out(exp: Experiment) -> ExperimentOut:
    out = ExperimentOut.model_validate(exp)
    out.latest_job = exp.jobs[-1] if exp.jobs else None  # type: ignore[assignment]
    return out


@router.get("/experiment-types", response_model=list[ExperimentTypeOut])
def experiment_types():
    return [
        ExperimentTypeOut(type=e.type_name, title=e.title, description=e.description,
                          config_schema=e.config_model.model_json_schema(), presets=e.presets)
        for e in list_experiments()
    ]


@router.post("/experiments/validate", response_model=ValidationReport)
def validate(body: ExperimentCreate):
    return experiment_service.validate_experiment(body)


@router.post("/experiments", response_model=ExperimentDetail, status_code=201)
def create_experiment(body: ExperimentCreate, session: Session = Depends(get_session)):
    try:
        exp = experiment_service.create_experiment(session, body)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return _detail(exp)


def _detail(exp: Experiment) -> ExperimentDetail:
    d = ExperimentDetail.model_validate(exp)
    d.latest_job = d.jobs[-1] if d.jobs else None
    return d


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments_(limit: int = 100, type: str | None = None, session: Session = Depends(get_session)):
    q = select(Experiment).order_by(Experiment.id.desc()).limit(min(limit, 500))
    if type:
        q = q.where(Experiment.type == type)
    rows = session.scalars(q).all()
    return [_out(e) for e in rows]


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: int, session: Session = Depends(get_session)):
    exp = session.get(Experiment, experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    return _detail(exp)


@router.post("/experiments/{experiment_id}/resume", response_model=ExperimentDetail, status_code=201)
def resume_experiment(experiment_id: int, session: Session = Depends(get_session)):
    """Queue a new job on the same experiment; it re-runs only what did not complete (resumable types only)."""
    exp = session.get(Experiment, experiment_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    from locallab.experiments.registry import get_experiment as _get_type

    if not _get_type(exp.type).resumable:
        raise HTTPException(409, f"experiment type {exp.type!r} cannot be resumed")
    last = exp.jobs[-1]
    if not JobStatus(last.status).terminal:
        raise HTTPException(409, f"job {last.id} is still {last.status}")
    job = Job(seed=last.seed)
    exp.jobs.append(job)
    session.flush()
    job_service.add_log(session, job.id, "INFO", f"queued (resume of job {last.id})")
    return _detail(exp)
