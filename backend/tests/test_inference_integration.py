import threading
import time

import pytest
from fastapi.testclient import TestClient

from locallab.main import create_app
from locallab.services.gpu import read_gpus
from locallab.worker.runner import Worker
from tests.mock_vllm import MockServer

pytestmark = pytest.mark.skipif(not read_gpus().available, reason="benchmark samples real NVML")


@pytest.fixture
def client(settings):
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def worker(settings):
    return Worker(settings, worker_id="bench-worker")


def cfg(srv, **kw):
    base = {"server_url": srv.url, "model": "mock-model", "input_tokens": 16, "output_tokens": 8,
            "num_requests": 6, "concurrency": 2, "warmup_requests": 2}
    return {"type": "inference_benchmark", "config": {**base, **kw}}


def submit(client, body):
    r = client.post("/api/experiments", json=body)
    assert r.status_code == 201, r.text
    return r.json()["jobs"][0]["id"]


def requests_artifact(client, job_id):
    arts = client.get(f"/api/jobs/{job_id}/artifacts").json()
    art = next(a for a in arts if a["name"] == "requests.json")
    return client.get(f"/api/artifacts/{art['id']}/content").json()


def test_full_run_records_real_measurements(client, worker):
    with MockServer(ttft_s=0.05, token_s=0.01) as srv:
        jid = submit(client, cfg(srv, num_requests=8, concurrency=4, iterations=2))
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed", job["error"]
        r = job["results"]
        s = r["summary"]
        assert s["requests"]["attempted"] == 16 and s["requests"]["success_rate"] == 1.0
        assert s["token_count_source"] == "server" and s["tokens"]["output_total"] == 16 * 8
        assert len(r["iterations"]) == 2
        # ttft ~ 50ms + overhead; decode ~10ms/token; allow generous scheduler slack
        assert 40 < s["latency"]["ttft_ms"]["p50"] < 400
        assert 5 < s["latency"]["tpot_ms"]["p50"] < 100
        assert s["throughput"]["aggregate_output_tok_s"] > 0
        assert r["runtime"]["version"] == "0.0.mock" and r["runtime"]["kv_cache_tokens"] == 16000
        assert r["workload"]["verification"]["input_length_exact"] is True
        assert r["gpu"]["per_device"]["0"]["peak_memory_mb"] > 0
        art = requests_artifact(client, jid)
        assert art["complete"] is True
        measured = [x for x in art["requests"] if x["phase"] == "measured"]
        warm = [x for x in art["requests"] if x["phase"] == "warmup"]
        assert len(measured) == 16 and len(warm) == 2
        assert {x["seed"] for x in measured}.isdisjoint({x["seed"] for x in warm})
        assert srv.state.requests == 18
        # forced-length decoding parameters were actually sent
        assert srv.state.bodies[0]["ignore_eos"] is True and srv.state.bodies[0]["min_tokens"] == 8
        assert len(srv.state.bodies[0]["prompt"]) == 16


def test_concurrency_is_bounded_and_reached(client, worker):
    with MockServer(token_s=0.01) as srv:
        jid = submit(client, cfg(srv, num_requests=12, concurrency=3, warmup_requests=0))
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed"
        assert srv.state.max_in_flight == 3  # never more, and the limit was actually reached
        assert job["results"]["workload"]["max_observed_in_flight"] == 3


def test_partial_failures_are_counted_not_hidden(client, worker):
    with MockServer(mode="fail_every_other") as srv:
        jid = submit(client, cfg(srv, num_requests=10, concurrency=1, warmup_requests=0, abort_after_failures=50))
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "completed"
        req = job["results"]["summary"]["requests"]
        assert req["failed"] == 5 and req["succeeded"] == 5 and req["success_rate"] == 0.5
        assert req["errors_by_type"] == {"server_error": 5}
        assert any("failed" in w for w in job["results"]["warnings"])


def test_oom_is_classified(client, worker):
    with MockServer(fail_first_n=0) as srv:
        jid = submit(client, cfg(srv, num_requests=4, warmup_requests=0, concurrency=1))
        srv.state.mode = "oom"
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed" and job["results"] is None and "oom" in job["error"]
        art = requests_artifact(client, jid)
        assert art["requests"][0]["error_type"] == "oom" and art["complete"] is False


def test_all_requests_failing_fails_job_and_aborts_early(client, worker):
    with MockServer() as srv:
        jid = submit(client, cfg(srv, num_requests=200, concurrency=1, warmup_requests=0, abort_after_failures=3))
        srv.state.mode = "http500"
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed" and "3 consecutive failures" in job["error"]
        assert srv.state.requests == 3  # stopped hammering the server


def test_warmup_failure_fails_job(client, worker):
    with MockServer() as srv:
        jid = submit(client, cfg(srv, warmup_requests=2, concurrency=1, abort_after_failures=5))
        srv.state.mode = "http500"
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed" and "Warmup failed" in job["error"]


def test_timeout_is_reported(client, worker):
    with MockServer() as srv:
        jid = submit(client, cfg(srv, num_requests=2, warmup_requests=0, concurrency=1, request_timeout_s=5, abort_after_failures=1))
        srv.state.mode = "hang"
        t0 = time.time()
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed" and "timeout" in job["error"] and time.time() - t0 < 15


def test_connection_drop_mid_stream(client, worker):
    with MockServer() as srv:
        jid = submit(client, cfg(srv, num_requests=2, warmup_requests=0, concurrency=1, abort_after_failures=1))
        srv.state.mode = "drop"
        worker.tick()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed"
        art = requests_artifact(client, jid)
        assert art["requests"][0]["error_type"] in ("connection_lost", "incomplete_stream")


def test_missing_usage_marks_tokens_estimated(client, worker):
    with MockServer(usage=False) as srv:
        jid = submit(client, cfg(srv, warmup_requests=0))
        worker.tick()
        r = client.get(f"/api/jobs/{jid}").json()["results"]
        assert r["summary"]["token_count_source"] == "estimated"
        assert any("estimated" in w for w in r["warnings"])


def test_prompt_length_mismatch_is_flagged(client, worker):
    with MockServer(prompt_token_offset=2) as srv:
        jid = submit(client, cfg(srv, warmup_requests=0))
        worker.tick()
        r = client.get(f"/api/jobs/{jid}").json()["results"]
        assert r["workload"]["verification"]["input_length_exact"] is False
        assert any("prompt length different" in w for w in r["warnings"])


def test_cancel_mid_run_stops_server_work_and_keeps_partial_data(client, worker):
    with MockServer(ttft_s=0.05, token_s=0.05) as srv:
        jid = submit(client, cfg(srv, num_requests=200, concurrency=4, output_tokens=40, warmup_requests=0))
        t = threading.Thread(target=worker.tick)
        t.start()
        for _ in range(100):
            p = client.get(f"/api/jobs/{jid}").json().get("progress")
            if p and p["done"] >= 3:
                break
            time.sleep(0.1)
        else:
            pytest.fail("no progress observed")
        assert p["phase"] == "measured" and p["total"] == 200
        client.post(f"/api/jobs/{jid}/cancel")
        t.join(timeout=15)
        assert not t.is_alive()
        job = client.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "cancelled" and job["results"] is None
        time.sleep(0.5)
        assert srv.state.requests < 200 and srv.state.in_flight == 0  # in-flight streams were aborted
        assert srv.state.disconnects >= 1
        art = requests_artifact(client, jid)
        assert art["complete"] is False and "PARTIAL" in art["note"] and len(art["requests"]) >= 3


def test_validation_rejects_infeasible_and_unreachable(client):
    with MockServer(max_model_len=1000) as srv:
        body = cfg(srv, input_tokens=900, output_tokens=200)
        r = client.post("/api/experiments/validate", json=body).json()
        assert r["ok"] is False and "max_model_len" in r["errors"][0]
        assert client.post("/api/experiments", json=body).status_code == 422
        assert client.get("/api/experiments").json() == []
        wrong = client.post("/api/experiments/validate", json=cfg(srv, model="other")).json()
        assert wrong["ok"] is False and "mock-model" in wrong["errors"][0]
    down = client.post("/api/experiments/validate", json={"type": "inference_benchmark", "config": {
        "model": "m", "server_url": "http://127.0.0.1:1"}}).json()
    assert down["ok"] is False and "Cannot reach" in down["errors"][0]
    blocked = client.post("/api/experiments/validate", json={"type": "inference_benchmark", "config": {
        "model": "m", "server_url": "http://10.0.0.5:8000"}}).json()
    assert blocked["ok"] is False and "ALLOWED_RUNTIME_HOSTS" in blocked["errors"][0]


def test_validation_warns_on_kv_overcommit(client):
    with MockServer(num_gpu_blocks=10) as srv:  # 160 token cache
        r = client.post("/api/experiments/validate", json=cfg(srv, concurrency=4, input_tokens=64, output_tokens=64)).json()
        assert r["ok"] is True and any("KV cache" in w for w in r["warnings"])


def test_probe_endpoint_and_allowlist(client):
    with MockServer() as srv:
        body = client.get("/api/runtimes/vllm/probe", params={"url": srv.url}).json()
        assert body["models"][0]["id"] == "mock-model" and body["version"] == "0.0.mock"
    assert client.get("/api/runtimes/vllm/probe", params={"url": "http://192.168.1.9:8000"}).status_code == 422
    assert client.get("/api/runtimes/vllm/probe", params={"url": "http://127.0.0.1:1"}).status_code == 502


def test_experiment_types_expose_presets(client):
    t = {x["type"]: x for x in client.get("/api/experiment-types").json()}["inference_benchmark"]
    assert set(t["presets"]) == {"short_chat", "long_prefill", "long_decode", "concurrent"}
    assert "model" in t["config_schema"]["required"]
