from sqlalchemy import inspect

import locallab.db as db
from locallab.models import Experiment, Job, JobLog, JobStatus


def test_migration_creates_all_tables(settings):
    names = set(inspect(db.get_engine()).get_table_names())
    assert {"models", "experiments", "jobs", "job_logs", "artifacts"} <= names


def test_job_defaults_and_relations(session):
    exp = Experiment(type="diagnostic", name="t", config={"a": 1}, config_hash="x")
    exp.jobs.append(Job())
    session.add(exp)
    session.flush()
    job = exp.jobs[0]
    session.add(JobLog(job_id=job.id, level="INFO", message="hi"))
    session.flush()
    assert job.status == JobStatus.queued.value
    assert job.cancel_requested is False
    assert [l.message for l in job.logs] == ["hi"]


def test_terminal_states():
    assert JobStatus.completed.terminal and JobStatus.failed.terminal and JobStatus.cancelled.terminal
    assert not JobStatus.running.terminal and not JobStatus.queued.terminal
