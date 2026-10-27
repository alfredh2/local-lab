import threading
import time
from datetime import timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

import locallab.db as db
from locallab.experiments import registry as exp_registry
from locallab.experiments.base import Experiment
from locallab.main import create_app
from locallab.models import Job, utcnow
from locallab.services.gpu import read_gpus
from locallab.worker.runner import Worker


class NoConfig(BaseModel):
    pass


class Failing(Experiment):
    type_name, title, description, config_model = "failing", "Failing", "always raises", NoConfig

    def run(self, config, ctx) -> dict[str, Any]:
        raise RuntimeError("boom")


class Slow(Experiment):
    type_name, title, description, config_model = "slow", "Slow", "loops until cancelled", NoConfig

    def run(self, config, ctx) -> dict[str, Any]:
        ctx.log("INFO", "started")
        for _ in range(200):
            ctx.check_cancel()
            time.sleep(0.05)
        return {"done": True}


@pytest.fixture(autouse=True)
def fake_types(monkeypatch):
    monkeypatch.setitem(exp_registry._TYPES, "failing", Failing())
    monkeypatch.setitem(exp_registry._TYPES, "slow", Slow())


@pytest.fixture
def client(settings):
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def worker(settings):
    return Worker(settings, worker_id="test-worker")


def submit(client, type_="failing", config=None, **kw):
    r = client.post("/api/experiments", json={"type": type_, "config": config or {}, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def job_of(client, exp):
    return client.get(f"/api/jobs/{exp['jobs'][0]['id']}").json()


def test_diagnostic_end_to_end_real_measurements(client, worker):
    if not read_gpus().available:
        pytest.skip("no NVML")
    exp = submit(client, "diagnostic", {"duration_s": 1, "sample_interval_ms": 250})
    assert exp["jobs"][0]["status"] == "queued" and exp["environment"]["gpus"]
    assert worker.tick() is True
    job = job_of(client, exp)
    assert job["status"] == "completed" and job["error"] is None
    res = job["results"]
    assert res["kind"] == "diagnostic" and res["sample_count"] >= 3
    mem = res["gpus"]["0"]["memory_used_mb"]
    assert 0 < mem["min"] <= mem["mean"] <= mem["max"] <= exp["environment"]["gpus"][0]["memory_total_mb"]
    arts = client.get(f"/api/jobs/{job['id']}/artifacts").json()
    assert arts[0]["name"] == "telemetry.json"
    series = client.get(f"/api/artifacts/{arts[0]['id']}/content").json()
    assert len(series["samples"]) == res["sample_count"]
    logs = [l["message"] for l in client.get(f"/api/jobs/{job['id']}/logs").json()]
    assert "status -> running" in logs and "status -> completed" in logs


def test_failure_is_recorded_not_completed(client, worker):
    exp = submit(client, "failing")
    worker.tick()
    job = job_of(client, exp)
    assert job["status"] == "failed" and "boom" in job["error"] and job["results"] is None
    assert job["finished_at"] is not None


def test_invalid_config_and_type_rejected(client):
    assert client.post("/api/experiments", json={"type": "diagnostic", "config": {"duration_s": 0}}).status_code == 422
    assert client.post("/api/experiments", json={"type": "diagnostic", "config": {"sample_interval_ms": 10}}).status_code == 422
    assert client.post("/api/experiments", json={"type": "rm -rf /"}).status_code == 422
    assert client.get("/api/experiments").json() == []  # nothing persisted for rejected requests


def test_cancel_queued_job(client, worker):
    exp = submit(client, "slow")
    jid = exp["jobs"][0]["id"]
    assert client.post(f"/api/jobs/{jid}/cancel").json()["status"] == "cancelled"
    assert worker.tick() is False  # never claimed
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 409
    assert client.post("/api/jobs/9999/cancel").status_code == 404


def test_cancel_running_job(client, worker):
    exp = submit(client, "slow")
    jid = exp["jobs"][0]["id"]
    t = threading.Thread(target=worker.tick)
    t.start()
    for _ in range(100):
        if client.get(f"/api/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.05)
    else:
        pytest.fail("job never reached running")
    assert client.post(f"/api/jobs/{jid}/cancel").json()["cancel_requested"] is True
    t.join(timeout=10)
    assert not t.is_alive()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "cancelled" and job["results"] is None


def test_worker_shutdown_marks_failed(client, worker):
    exp = submit(client, "slow")
    t = threading.Thread(target=worker.tick)
    t.start()
    time.sleep(0.5)
    worker.shutdown.set()
    t.join(timeout=10)
    job = job_of(client, exp)
    assert job["status"] == "failed" and "shutdown" in job["error"]


def test_stale_running_job_recovered_as_failed(client, worker, settings):
    exp = submit(client, "slow")
    jid = exp["jobs"][0]["id"]
    with db.session_scope() as s:
        job = s.get(Job, jid)
        job.status, job.worker_id = "running", "dead-worker"
        job.heartbeat_at = utcnow() - timedelta(seconds=settings.worker_stale_s + 5)
    worker.tick()
    job = client.get(f"/api/jobs/{jid}").json()
    assert job["status"] == "failed" and "dead-worker" in job["error"]


def test_fresh_active_job_not_recovered(client, worker):
    exp = submit(client, "slow")
    jid = exp["jobs"][0]["id"]
    with db.session_scope() as s:
        job = s.get(Job, jid)
        job.status, job.worker_id, job.heartbeat_at = "running", "other", utcnow()
    worker.tick()
    assert client.get(f"/api/jobs/{jid}").json()["status"] == "running"


def test_claim_is_exclusive_and_fifo(client, settings):
    from locallab.services.jobs import claim_next

    submit(client, "failing")
    submit(client, "failing")
    with db.session_scope() as s:
        a = claim_next(s, "w1")
        b = claim_next(s, "w2")
        c = claim_next(s, "w3")
    assert (a.id, b.id, c) == (1, 2, None)


def test_history_survives_api_restart(settings):
    with TestClient(create_app()) as c1:
        exp = submit(c1, "failing", name="persist me")
    db.init_db(settings)  # fresh engine == new process
    with TestClient(create_app()) as c2:
        got = c2.get(f"/api/experiments/{exp['id']}").json()
        assert got["name"] == "persist me" and got["jobs"][0]["status"] == "queued"
        assert [e["id"] for e in c2.get("/api/experiments").json()] == [exp["id"]]
