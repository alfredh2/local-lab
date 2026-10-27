import hashlib
import json
import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from locallab.experiments.registry import get_experiment
from locallab.models import Experiment, Job
from locallab.schemas.experiments import ExperimentCreate
from locallab.services import jobs as job_service
from locallab.services.system import hardware_snapshot

REPO_ROOT = Path(__file__).resolve().parents[3]


def git_commit() -> str | None:
    """Fixed, allowlisted command; no user input reaches the subprocess."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def config_hash(type_name: str, config: dict, seed: int | None) -> str:
    blob = json.dumps({"type": type_name, "config": config, "seed": seed}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def create_experiment(session: Session, body: ExperimentCreate) -> Experiment:
    """Validate config against the experiment type, snapshot the environment, enqueue a job."""
    exp_type = get_experiment(body.type)  # ValueError for unknown types
    cfg = exp_type.config_model.model_validate(body.config)  # ValidationError -> 422 at API
    normalized = cfg.model_dump()
    exp = Experiment(
        type=body.type,
        name=body.name or exp_type.title,
        config=normalized,
        config_hash=config_hash(body.type, normalized, body.seed),
        environment={**hardware_snapshot(), "git_commit": git_commit()},
    )
    exp.jobs.append(Job(seed=body.seed))
    session.add(exp)
    session.flush()
    if not body.name:
        exp.name = f"{exp_type.title} #{exp.id}"
    job_service.add_log(session, exp.jobs[0].id, "INFO", "queued")
    return exp
