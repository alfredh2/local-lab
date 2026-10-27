import json
import math
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import locallab.db as db
from locallab.experiments.inference import sweep_stats as st
from locallab.experiments.inference.sweep import SweepConfig
from locallab.main import create_app
from locallab.models import Job, ManagedServer, ModelRecord, utcnow
from locallab.runtimes import vllm_managed as vm
from locallab.runtimes.docker import OwnedDocker
from locallab.services.gpu import read_gpus
from locallab.services.gpu_preflight import GpuPreflight
from locallab.services.registry import sync_registry
from locallab.worker.runner import Worker
from tests.test_registry import make_repo

CFG = {"architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
       "hidden_size": 64, "max_position_embeddings": 32768}
needs_gpu = pytest.mark.skipif(not read_gpus().available, reason="sweep samples real NVML")


# ---------------- config & statistics (pure) ----------------
def test_config_validation(settings):
    c = SweepConfig(registry_model_id=1, concurrency_levels=[8, 1, 4, 4])
    assert c.concurrency_levels == [1, 4, 8]
    with pytest.raises(ValueError, match="registry_model_id is required"):
        SweepConfig()
    with pytest.raises(ValueError, match="served model name"):
        SweepConfig(mode="external")
    with pytest.raises(ValueError, match="1-12 values"):
        SweepConfig(registry_model_id=1, concurrency_levels=[0])
    with pytest.raises(ValueError, match="total requests"):
        SweepConfig(registry_model_id=1, concurrency_levels=[512], requests_per_level=5000, iterations=20)
    with pytest.raises(ValueError, match="ALLOWED_RUNTIME_HOSTS"):
        SweepConfig(mode="external", model="m", server_url="http://10.1.1.1:8000")


def test_level_sizing_and_seeds(settings):
    c = SweepConfig(registry_model_id=1, concurrency_levels=[1, 16], requests_per_level=32, min_requests_per_slot=4, warmup_requests=4)
    assert c.level_requests(1) == 32 and c.level_requests(16) == 64  # 16 slots x 4 each
    assert c.level_warmup(1) == 4 and c.level_warmup(16) == 16
    seeds = {c.level_config(i, lv, "http://127.0.0.1:1", "m").seed for i, lv in enumerate([1, 2, 4, 8, 16])}
    assert len(seeds) == 5  # levels never replay each other's prompts


def test_mean_ci_matches_hand_calculation():
    r = st.mean_ci([10.0, 12.0, 14.0])
    assert r["mean"] == 12 and r["sd"] == pytest.approx(2.0)
    assert r["ci95_half"] == pytest.approx(4.303 * 2 / math.sqrt(3))
    assert r["cv"] == pytest.approx(2 / 12)
    two = st.mean_ci([1.0, 3.0])
    assert two["sd"] is not None and two["ci95_half"] is None  # <3 iterations: no CI claimed
    assert st.mean_ci([5.0])["sd"] is None and st.mean_ci([])["n"] == 0


def test_percentiles_gated_by_sample_size():
    full = {"n": 0, "mean": 1, "min": 1, "p50": 1, "p95": 9, "p99": 10, "max": 11}
    assert st.gate_dist({**full, "n": 10})["p95"] is None
    d = st.gate_dist({**full, "n": 20})
    assert d["p95"] == 9 and d["p99"] is None and d["reported"] == {"p95": True, "p99": False}
    assert st.gate_dist({**full, "n": 100})["p99"] == 10 and st.gate_dist(None) is None


def lvl(c, mean, half=None, n_req=64, cv=None):
    iters = [mean] * 3
    stats = {"aggregate_output_tok_s": {"n": 3, "mean": mean, "sd": 1.0, "ci95_half": half, "ci95_rel": None, "cv": cv or 0.01}}
    return {"concurrency": c, "status": "ok", "stats": stats, "latency": {"e2e_ms": {"n": n_req}}, "iterations": iters}


def test_analysis_finds_saturation_and_efficiency():
    levels = [lvl(1, 100, 2), lvl(2, 190, 2), lvl(4, 340, 3), lvl(8, 360, 3), lvl(16, 365, 3)]
    a = st.analyze(levels, contended=False)
    assert a["saturation_after_concurrency"] == 4 and a["peak_throughput_concurrency"] == 16
    assert levels[2]["scaling_efficiency"] == pytest.approx(340 / 400)
    assert a["steps"][2]["gain_pct"] == pytest.approx(5.88, abs=0.01) and a["reliable"] is True
    assert any("stopped scaling" in n for n in a["notes"])


def test_analysis_flags_noise_and_contention():
    levels = [lvl(1, 100, 30, cv=0.25), lvl(2, 110, 30, cv=0.25)]
    a = st.analyze(levels, contended=True)
    assert a["saturation_after_concurrency"] == 1  # +10% is within the noise of the two CIs
    assert a["reliable"] is False
    assert any("noisy" in n for n in a["notes"]) and any("contended" in n for n in a["notes"])
    assert st.analyze([lvl(1, 100, 2)], False)["saturation_after_concurrency"] is None


def test_few_samples_are_called_out():
    levels = [lvl(1, 100, 2, n_req=10)]
    st.analyze(levels, False)
    assert any("p95/p99 not reported" in i for i in levels[0]["quality_issues"])


# ---------------- integration: managed lifecycle through the real worker ----------------
@pytest.fixture
def gpu(monkeypatch):
    state = GpuPreflight(gpu_index=0, name="Fake 5090", total_mb=32768, free_mb=20000, compute_capability=(12, 0))

    def fake(index, util, allow, exclude_pids=None):
        errors, warnings = list(state.errors), list(state.warnings)
        if state.contended:
            msg = "GPU 0 is in use by other work"
            (warnings if allow else errors).append(msg + (" (override)" if allow else ". Refusing to benchmark on a contended GPU."))
        return GpuPreflight(**{**state.__dict__, "requested_mb": util * state.total_mb, "override": allow,
                               "errors": errors, "warnings": warnings})

    monkeypatch.setattr(vm, "check_gpu", fake)
    return state


@pytest.fixture
def gpu_procs(monkeypatch):
    """Controls which compute processes the benchmark 'sees' on the GPU (the real GPU may be busy)."""
    import locallab.experiments.inference.experiment as exp_mod
    import locallab.experiments.inference.sweep as sweep_mod
    from locallab.schemas.system import GpuProcess

    procs: list[GpuProcess] = []
    real = read_gpus

    def fake_read():
        snap = real()
        for g in snap.gpus:
            g.processes = list(procs)
        return snap

    monkeypatch.setattr(exp_mod, "read_gpus", fake_read)
    monkeypatch.setattr(sweep_mod, "read_gpus", fake_read)
    return procs


@pytest.fixture
def env(settings, session, fake_docker, gpu, gpu_procs):
    make_repo(Path(settings.hf_cache_dir), "Qwen/Mini-Instruct", CFG)
    sync_registry(session, settings)
    session.commit()
    mid = session.scalars(select(ModelRecord).where(ModelRecord.name == "Qwen/Mini-Instruct")).one().id
    (settings.data_dir / f"vllm_archs_{fake_docker.image.id.split(':')[-1][:16]}.json").write_text(json.dumps(fake_docker.arch_list))
    with TestClient(create_app()) as client:
        yield client, fake_docker, mid, Worker(settings, worker_id="sweep-worker")


def sweep_body(mid, **kw):
    cfg = {"mode": "managed", "registry_model_id": mid, "concurrency_levels": [1, 2, 4], "input_tokens": 16, "output_tokens": 8,
           "requests_per_level": 8, "min_requests_per_slot": 2, "warmup_requests": 2, "iterations": 3,
           "runtime": {"max_model_len": 2048, "gpu_memory_utilization": 0.3, "startup_timeout_s": 30}}
    runtime = {**cfg["runtime"], **kw.pop("runtime", {})}
    return {"type": "concurrency_sweep", "config": {**cfg, **kw, "runtime": runtime}}


def submit(client, body):
    r = client.post("/api/experiments", json=body)
    assert r.status_code == 201, r.text
    return r.json()["jobs"][0]["id"]


def job(client, jid):
    return client.get(f"/api/jobs/{jid}").json()


def artifact(client, jid, name):
    arts = client.get(f"/api/jobs/{jid}/artifacts").json()
    a = next((x for x in arts if x["name"] == name), None)
    return None if a is None else client.get(f"/api/artifacts/{a['id']}/content").json()


def mrow(jid) -> ManagedServer:
    with db.session_scope() as s:
        return s.scalars(select(ManagedServer).where(ManagedServer.job_id == jid)).one()


def wait_for(pred, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    pytest.fail("condition not reached")


@needs_gpu
def test_full_managed_sweep_lifecycle(env):
    client, fd, mid, worker = env
    jid = submit(client, sweep_body(mid))
    fd.server_kw.update(ttft_s=0.02, token_s=0.01)
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "completed", j["error"]
    r = j["results"]
    assert r["kind"] == "concurrency_sweep" and [lv["concurrency"] for lv in r["levels"]] == [1, 2, 4]
    assert all(lv["status"] == "ok" and lv["iterations_run"] == 3 for lv in r["levels"])
    tput = [lv["stats"]["aggregate_output_tok_s"]["mean"] for lv in r["levels"]]
    assert tput[1] > tput[0] * 1.4 and tput[2] > tput[1] * 1.4  # mock server scales with concurrency
    l4 = r["levels"][2]
    assert l4["num_requests"] == 8 and l4["max_observed_in_flight"] == 4
    assert l4["stats"]["aggregate_output_tok_s"]["ci95_half"] is not None
    assert l4["latency"]["e2e_ms"]["n"] == 24 and l4["latency"]["e2e_ms"]["p95"] is not None and l4["latency"]["e2e_ms"]["p99"] is None
    assert r["levels"][0]["latency"]["e2e_ms"]["p95"] is not None  # 8 req x 3 iterations = 24 >= 20
    assert r["analysis"]["scaling_baseline_concurrency"] == 1
    # metadata: ownership, image, revision, command, timing
    ms = r["managed_server"]
    assert ms["image_digest"].startswith("vllm/vllm-openai@sha256:") and ms["command"][:3] == ["Qwen/Mini-Instruct", "--revision", "abc123"]
    assert r["model_registry"]["revision"] == "abc123" and r["runtime"]["version"] == "0.0.mock"
    assert r["gpu"]["contended"] is False and r["gpu"]["contention_override"] is False
    # cleanup
    assert fd.containers == {} and mrow(jid).status == "stopped" and mrow(jid).startup_s is not None
    # artifacts: per-level requests, gpu samples, server log
    req = artifact(client, jid, "requests.json")
    assert req["complete"] is True and [lv["concurrency"] for lv in req["levels"]] == [1, 2, 4]
    assert len(req["levels"][2]["requests"]) == 4 + 24  # warmup max(2, concurrency) + 8 x 3
    assert artifact(client, jid, "gpu_samples.json")["level_windows"][0]["concurrency"] == 1
    assert "Application startup complete" in "\n".join(artifact(client, jid, "server.log")["lines"])


@needs_gpu
def test_all_prompts_unique_across_levels(env):
    client, fd, mid, worker = env
    bodies = []
    import tests.mock_vllm as mv

    orig = mv.MockState.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.bodies = bodies  # share one list across all server instances

    mv.MockState.__init__ = init
    try:
        jid = submit(client, sweep_body(mid, concurrency_levels=[1, 2], iterations=2, requests_per_level=4))
        worker.tick()
    finally:
        mv.MockState.__init__ = orig
    assert job(client, jid)["status"] == "completed"
    prompts = [tuple(b["prompt"]) for b in bodies]
    assert len(prompts) >= 20 and len(prompts) == len(set(prompts))


def test_contention_blocks_unless_overridden_and_is_recorded(env, gpu):
    client, fd, mid, worker = env
    gpu.contended = True
    v = client.post("/api/experiments/validate", json=sweep_body(mid)).json()
    assert v["ok"] is False and "Refusing to benchmark on a contended GPU" in v["errors"][0]
    assert client.post("/api/experiments", json=sweep_body(mid)).status_code == 422
    assert client.get("/api/experiments").json() == [] and fd.calls == []  # nothing queued, nothing launched

    if not read_gpus().available:
        pytest.skip("no NVML")
    body = sweep_body(mid, concurrency_levels=[1, 2], runtime={"allow_contention": True})
    v = client.post("/api/experiments/validate", json=body).json()
    assert v["ok"] is True and any("override" in w for w in v["warnings"])
    jid = submit(client, body)
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "completed", j["error"]
    g = j["results"]["gpu"]
    assert g["contended"] is True and g["contention_override"] is True
    assert any("contended" in n for n in j["results"]["analysis"]["notes"]) and j["results"]["analysis"]["reliable"] is False


def test_preflight_failure_starts_nothing(env, gpu):
    client, fd, mid, worker = env
    gpu.errors = ["gpu_memory_utilization 0.3 requests 9830 MB but only 100 MB is free. (This cannot be overridden.)"]
    r = client.post("/api/experiments", json=sweep_body(mid, runtime={"allow_contention": True}))
    assert r.status_code == 422 and "cannot be overridden" in r.text
    assert fd.calls == []


@needs_gpu
def test_cancel_while_queued_never_launches(env):
    client, fd, mid, worker = env
    jid = submit(client, sweep_body(mid))
    assert client.post(f"/api/jobs/{jid}/cancel").json()["status"] == "cancelled"
    assert worker.tick() is False and fd.calls == []


@needs_gpu
def test_cancel_during_server_startup(env):
    client, fd, mid, worker = env
    fd.start_mode = "never_ready"
    jid = submit(client, sweep_body(mid))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: (job(client, jid).get("progress") or {}).get("phase") == "waiting_for_server")
    client.post(f"/api/jobs/{jid}/cancel")
    t.join(timeout=15)
    assert not t.is_alive()
    j = job(client, jid)
    assert j["status"] == "cancelled" and j["results"] is None
    assert fd.containers == {} and mrow(jid).status == "stopped"


@needs_gpu
def test_cancel_during_benchmark_removes_container_and_keeps_partial_data(env):
    client, fd, mid, worker = env
    fd.server_kw.update(ttft_s=0.05, token_s=0.05)
    jid = submit(client, sweep_body(mid, concurrency_levels=[2], output_tokens=30, requests_per_level=40, iterations=5))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: (job(client, jid).get("progress") or {}).get("done", 0) >= 3 and job(client, jid)["progress"]["phase"] == "measured")
    p = job(client, jid)["progress"]
    assert p["sweep"]["concurrency"] == 2 and p["sweep"]["levels"] == [2]
    client.post(f"/api/jobs/{jid}/cancel")
    t.join(timeout=20)
    assert not t.is_alive()
    assert job(client, jid)["status"] == "cancelled"
    assert fd.containers == {} and mrow(jid).status == "stopped"
    art = artifact(client, jid, "requests.json")
    assert art["complete"] is False and "PARTIAL" in art["note"] and art["levels"][0]["requests"]


@needs_gpu
def test_worker_shutdown_cleans_up_container(env):
    client, fd, mid, worker = env
    fd.server_kw.update(ttft_s=0.05, token_s=0.05)
    jid = submit(client, sweep_body(mid, concurrency_levels=[2], output_tokens=30, requests_per_level=40, iterations=5))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: (job(client, jid).get("progress") or {}).get("phase") == "measured")
    worker.shutdown.set()
    t.join(timeout=20)
    j = job(client, jid)
    assert j["status"] == "failed" and "shutdown" in j["error"] and fd.containers == {}


@needs_gpu
def test_server_crash_at_later_level_keeps_finished_levels(env):
    client, fd, mid, worker = env
    fd.server_kw.update(ttft_s=0.05, token_s=0.03)
    jid = submit(client, sweep_body(mid, concurrency_levels=[1, 2, 4], output_tokens=40, requests_per_level=4, iterations=1,
                                    min_requests_per_slot=1, warmup_requests=1))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: ((job(client, jid).get("progress") or {}).get("sweep") or {}).get("concurrency") == 2)
    cid = next(iter(fd.containers))
    fd.crash(cid, oom=True, logs="torch.OutOfMemoryError: CUDA out of memory")
    t.join(timeout=30)
    j = job(client, jid)
    assert j["status"] == "completed", j["error"]
    r = j["results"]
    assert [(lv["concurrency"], lv["status"]) for lv in r["levels"]] == [(1, "ok"), (2, "failed")]
    assert "server crashed" in r["levels"][1]["error"] and r["stopped_early"].startswith("server crashed")
    assert any("Concurrency 2 failed" in w for w in r["warnings"])
    assert fd.containers == {} and mrow(jid).status == "stopped"
    assert mrow(jid).oom_killed is True and "out of memory" in mrow(jid).log_tail


@needs_gpu
def test_server_crash_before_any_level_completes_fails_job(env):
    client, fd, mid, worker = env
    fd.server_kw.update(ttft_s=0.05, token_s=0.05)
    jid = submit(client, sweep_body(mid, concurrency_levels=[1], output_tokens=30, requests_per_level=20, iterations=1))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: (job(client, jid).get("progress") or {}).get("phase") in ("warmup", "measured"))
    fd.crash(next(iter(fd.containers)), oom=False, logs="RuntimeError: CUDA error: illegal memory access")
    t.join(timeout=30)
    j = job(client, jid)
    assert j["status"] == "failed" and "crashed" in j["error"].lower() and j["results"] is None
    assert fd.containers == {}


@needs_gpu
def test_docker_failure_fails_job_cleanly(env):
    client, fd, mid, worker = env
    jid = submit(client, sweep_body(mid))
    fd.fail_run = "docker: Error response from daemon: could not select device driver"
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "failed" and "docker_failed" in j["error"] and "Traceback" not in j["error"]
    assert mrow(jid).status == "failed" and fd.containers == {}


@needs_gpu
def test_startup_oom_is_reported_with_launch_log(env):
    client, fd, mid, worker = env
    fd.start_mode, fd.exit_logs = "exit_at_start", "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"
    jid = submit(client, sweep_body(mid))
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "failed" and "oom" in j["error"]
    assert "CUDA out of memory" in mrow(jid).log_tail and fd.containers == {}
    assert "CUDA out of memory" in "\n".join(artifact(client, jid, "server.log")["lines"])


@needs_gpu
def test_startup_timeout_is_bounded(env):
    client, fd, mid, worker = env
    fd.start_mode = "never_ready"
    jid = submit(client, sweep_body(mid, runtime={"startup_timeout_s": 5}))
    t0 = time.time()
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "failed" and "startup_timeout" in j["error"] and time.time() - t0 < 20
    assert fd.containers == {}


@needs_gpu
def test_worker_crash_is_reconciled_on_restart(env, settings):
    """Simulate a worker that died mid-job: container left running, job heartbeat stale."""
    client, fd, mid, worker = env
    jid = submit(client, sweep_body(mid))
    with db.session_scope() as s:
        j = s.get(Job, jid)
        j.status, j.worker_id, j.heartbeat_at = "running", "dead-worker", utcnow()
        eid = j.experiment_id
    plan = vm.build_plan(vm.ManagedVllmConfig(startup_timeout_s=30), mid, settings, fd, allow_run_oneoff=False)
    srv = vm.ManagedVllmServer(plan, vm.owned_docker(settings), settings, jid, eid)
    srv.launch()
    foreign = fd.add_foreign("open-webui")
    worker.tick()  # fresh heartbeat: a live worker may own it, so it must stay
    assert srv.container_id in fd.containers
    with db.session_scope() as s:
        s.get(Job, jid).heartbeat_at = utcnow() - timedelta(seconds=settings.worker_stale_s + 5)
    worker._last_reconcile = 0.0
    worker.tick()
    assert job(client, jid)["status"] == "failed"
    assert srv.container_id not in fd.containers and foreign.id in fd.containers
    assert mrow(jid).status == "orphan_removed"


@needs_gpu
def test_external_mode_sweep_works_without_docker(env):
    client, fd, mid, worker = env
    from tests.mock_vllm import MockServer
    with MockServer(ttft_s=0.01, token_s=0.005) as srv:
        body = {"type": "concurrency_sweep", "config": {"mode": "external", "server_url": srv.url, "model": "mock-model",
                "concurrency_levels": [1, 2], "input_tokens": 8, "output_tokens": 4, "requests_per_level": 6,
                "min_requests_per_slot": 1, "warmup_requests": 1, "iterations": 3}}
        v = client.post("/api/experiments/validate", json=body).json()
        assert v["ok"] and any("prefix cache" in w for w in v["warnings"])
        jid = submit(client, body)
        worker.tick()
    j = job(client, jid)
    assert j["status"] == "completed", j["error"]
    assert j["results"]["managed_server"] is None and j["results"]["mode"] == "external"
    assert fd.calls == []  # Docker untouched


def test_types_endpoint_lists_sweep_and_existing_types(env):
    client, *_ = env
    types = {t["type"] for t in client.get("/api/experiment-types").json()}
    assert {"diagnostic", "inference_benchmark", "concurrency_sweep"} <= types


@needs_gpu
def test_observed_foreign_gpu_process_marks_results_contended(env, gpu_procs):
    from locallab.schemas.system import GpuProcess

    client, fd, mid, worker = env
    gpu_procs.append(GpuProcess(pid=4321, name="/venvs/other/bin/python", used_memory_mb=17000, kind="compute"))
    gpu_procs.append(GpuProcess(pid=1, name="Xorg", used_memory_mb=500, kind="graphics"))  # desktop: not contention
    jid = submit(client, sweep_body(mid, concurrency_levels=[1], runtime={"allow_contention": True}))
    worker.tick()
    r = job(client, jid)["results"]
    assert r["gpu"]["contended"] is True
    assert [p["pid"] for p in r["gpu"]["other_compute_processes"]] == [4321]
    assert r["levels"][0]["other_compute_processes"][0]["memory_mb"] == 17000
