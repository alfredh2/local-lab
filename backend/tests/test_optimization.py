import json
import re
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import locallab.db as db
from locallab.evals.datasets.coding import TASKS, CodingSuite
from locallab.evals.datasets.structured import StructuredSuite
from locallab.experiments.inference import matrix as MX
from locallab.experiments.inference import optimization as opt
from locallab.main import create_app
from locallab.models import ConfigResult, Experiment, Job, ManagedServer, ModelRecord, utcnow
from locallab.runtimes import vllm_managed as vm
from locallab.runtimes.docker import LABEL_INSTANCE, LABEL_MANAGED, OwnedDocker
from locallab.services.gpu import read_gpus
from locallab.services.registry import sync_registry
from locallab.worker.runner import Worker
from tests.test_registry import make_repo

CFG = {"architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
       "hidden_size": 64, "max_position_embeddings": 32768}
needs_gpu = pytest.mark.skipif(not read_gpus().available, reason="matrix runs sample real NVML")
REF = {t.key: t.reference for t in TASKS}


def oracle(degrade_coding=False, break_json=False):
    """A mock 'model' that answers every eval case correctly, optionally degraded to model a worse configuration."""
    struct = {c.messages[-1]["content"]: json.dumps(c.meta["reference"]) for sp in ("dev", "heldout") for c in StructuredSuite().cases(sp)}
    code = {c.messages[-1]["content"]: c.meta["key"] for sp in ("dev", "heldout") for c in CodingSuite().cases(sp)}

    def respond(messages, body):
        text = messages[-1]["content"]
        if text in struct:
            return "Here: " + struct[text] if break_json else struct[text]
        if text in code:
            return "```python\ndef nope():\n    pass\n```" if degrade_coding else f"```python\n{REF[code[text]]}\n```"
        if "access code for vault" in text:
            key = re.search(r"vault (V\d+)\?", text).group(1)
            return re.search(rf"access code for vault {key} is ([a-z]+-[a-z]+-\d+)\.", text).group(1)
        return "unknown"
    return respond


@pytest.fixture
def mx(settings, session, fake_docker, gpu, gpu_procs):
    ids = {}
    for key, name in [("a", "Qwen/Mini-Instruct"), ("b", "Qwen/Mini-B")]:
        make_repo(Path(settings.hf_cache_dir), name, CFG)
    sync_registry(session, settings)
    session.commit()
    for key, name in [("a", "Qwen/Mini-Instruct"), ("b", "Qwen/Mini-B")]:
        ids[key] = session.scalars(select(ModelRecord).where(ModelRecord.name == name)).one().id
    (settings.data_dir / f"vllm_archs_{fake_docker.image.id.split(':')[-1][:16]}.json").write_text(json.dumps(fake_docker.arch_list))
    fake_docker.server_kw.update(max_model_len=8192, responder=oracle())
    with TestClient(create_app()) as client:
        yield client, fake_docker, ids, Worker(settings, worker_id="mx-worker")


def body(ids, **kw):
    cfg = {"registry_model_ids": [ids["a"]], "max_model_lens": [8192], "quantizations": ["none"], "kv_cache_dtypes": ["auto"],
           "gpu_memory_utilization": 0.3, "startup_timeout_s": 30, "concurrency_levels": [1, 2], "io_pairs": [{"input_tokens": 16, "output_tokens": 8}],
           "requests_per_level": 4, "min_requests_per_slot": 1, "warmup_requests": 1, "iterations": 2, "evals": ["structured_json"]}
    cfg.update(kw)
    return {"type": "optimization_matrix", "config": cfg}


def submit(client, b):
    r = client.post("/api/experiments", json=b)
    assert r.status_code == 201, r.text
    return r.json()["id"], r.json()["jobs"][0]["id"]


def job(client, jid):
    return client.get(f"/api/jobs/{jid}").json()


def rows(client, eid):
    return {r["label"]: r for r in client.get(f"/api/optimization/{eid}").json()}


def runs(fd):
    return [n for c, n in fd.calls if c == "run"]


def wait_for(pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.05)
    pytest.fail("condition not reached")


# ---------------- configuration expansion (pure) ----------------
def test_expansion_groups_by_loading_config_and_dedupes(settings):
    c = MX.MatrixConfig(registry_model_ids=[1, 2, 1], max_model_lens=[4096, 8192, 4096], kv_cache_dtypes=["auto", "fp8", "auto"],
                        concurrency_levels=[16, 1, 4, 4], io_pairs=[{"input_tokens": 8, "output_tokens": 8}, {"input_tokens": 8, "output_tokens": 8}])
    ex = MX.expand(c)
    assert len(ex) == 2 * 2 * 1 * 2 == 8  # duplicates collapsed; concurrency and io are NOT multiplied into launches
    assert [e.model_id for e in ex] == [1, 1, 1, 1, 2, 2, 2, 2]  # model-major: one model's launches are contiguous
    assert c.concurrency_levels == [1, 4, 16] and len(c.io_pairs) == 1
    assert len({(e.model_id, e.max_model_len, e.quantization, e.kv_cache_dtype) for e in ex}) == 8


def test_config_keys_depend_on_revision_and_loading_axes():
    k = MX.config_key(1, "rev1", 8192, "none", "auto")
    assert k == MX.config_key(1, "rev1", 8192, "none", "auto")
    assert len({k, MX.config_key(1, "rev2", 8192, "none", "auto"), MX.config_key(1, "rev1", 4096, "none", "auto"),
                MX.config_key(1, "rev1", 8192, "fp8", "auto"), MX.config_key(1, "rev1", 8192, "none", "fp8"), MX.config_key(2, "rev1", 8192, "none", "auto")}) == 6


def test_matrix_limits_and_validation(settings):
    with pytest.raises(ValueError, match="expands to 12 configurations, above max_configurations=8"):
        MX.MatrixConfig(registry_model_ids=[1, 2, 3], kv_cache_dtypes=["auto", "fp8"], max_model_lens=[2048, 4096], max_configurations=8)
    with pytest.raises(ValueError, match="at least one eval"):
        MX.MatrixConfig(registry_model_ids=[1], perf_enabled=False, evals=[])
    with pytest.raises(ValueError):
        MX.MatrixConfig(registry_model_ids=[])
    with pytest.raises(ValueError, match="between 0 and 1"):
        MX.MatrixConfig(registry_model_ids=[1], retrieval_positions=[1.5])
    assert MX.MatrixConfig(registry_model_ids=[1]).eval_split == "heldout"  # held-out by default


def test_cell_plan_skips_what_does_not_fit_the_context(settings):
    c = MX.MatrixConfig(registry_model_ids=[1], max_model_lens=[2048], retrieval_lengths=[1024, 4096],
                        io_pairs=[{"input_tokens": 128, "output_tokens": 128}, {"input_tokens": 4096, "output_tokens": 64}])
    plan = MX.cell_plan(c, MX.expand(c)[0])
    assert [p.input_tokens for p in plan["io_pairs"]] == [128] and plan["retrieval_lengths"] == [1024]
    assert any("4096" in n and "exceeds max_model_len" in n for n in plan["notes"]) and any("[4096] skipped" in n for n in plan["notes"])


# ---------------- validation through the API ----------------
def test_validate_reports_each_configuration_and_blocks_all_infeasible(mx):
    client, fd, ids, _ = mx
    b = body(ids, registry_model_ids=[ids["a"], ids["b"]], quantizations=["none", "awq"])
    v = client.post("/api/experiments/validate", json=b).json()
    assert v["ok"] is True
    cfgs = v["info"]["configs"]
    assert len(cfgs) == 4 and sum(c["status"] == "feasible" for c in cfgs) == 2 and v["info"]["launches"] == 2
    bad = next(c for c in cfgs if c["quantization"] == "awq")
    assert bad["status"] == "infeasible" and "requires a awq checkpoint" in bad["errors"][0]
    assert any("2 of 4 configurations are infeasible" in w for w in v["warnings"])
    only_bad = client.post("/api/experiments/validate", json=body(ids, quantizations=["awq", "gptq"])).json()
    assert only_bad["ok"] is False and "No configuration in the matrix is feasible" in only_bad["errors"][0]
    assert client.post("/api/experiments", json=body(ids, quantizations=["awq"])).status_code == 422 and fd.calls == []


def test_validate_contention_blocks_unless_overridden(mx, gpu):
    client, fd, ids, _ = mx
    gpu.contended = True
    v = client.post("/api/experiments/validate", json=body(ids)).json()
    assert v["ok"] is False and "Refusing to benchmark on a contended GPU" in v["errors"][0]
    assert client.post("/api/experiments/validate", json=body(ids, allow_contention=True)).json()["ok"] is True


def test_validate_warns_about_dev_split_and_cell_skips(mx):
    client, fd, ids, _ = mx
    v = client.post("/api/experiments/validate", json=body(ids, max_model_lens=[2048], evals=["retrieval_needle"], eval_split="dev")).json()
    assert v["ok"] and any("dev split" in w for w in v["warnings"]) and any("retrieval suite skipped" in w for w in v["warnings"])


# ---------------- full runs ----------------
@needs_gpu
def test_one_launch_per_loading_config_with_quality_and_performance_linked(mx):
    client, fd, ids, worker = mx
    b = body(ids, registry_model_ids=[ids["a"], ids["b"]], kv_cache_dtypes=["auto", "fp8"], concurrency_levels=[1, 2, 4],
             io_pairs=[{"input_tokens": 16, "output_tokens": 8}, {"input_tokens": 32, "output_tokens": 8}], evals=["structured_json", "coding_python", "retrieval_needle"],
             retrieval_lengths=[2048], retrieval_repeats=1)
    eid, jid = submit(client, b)
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "completed", j["error"]
    assert j["results"]["counts"] == {"completed": 4} and j["results"]["split"] == "heldout"
    # 4 loading configs -> exactly 4 launches, although each measures 3 concurrency levels x 2 io pairs x 3 suites
    assert len(runs(fd)) == 4 and fd.containers == {}
    order = [(c, n) for c, n in fd.calls if c in ("run", "rm")]
    assert [c for c, _ in order] == ["run", "rm"] * 4  # strictly sequential: a server is removed before the next starts
    rs = rows(client, eid)
    assert len(rs) == 4
    for label, r in rs.items():
        assert r["status"] == "completed" and r["attempts"] == 1
        res = r["results"]
        # the row's config, the server command and the measured results all describe the same configuration
        assert res["server_config"] == r["config"]
        cmd = res["managed_server"]["command"]
        assert cmd[0] == "Qwen/" + ("Mini-Instruct" if "Mini-Instruct" in label else "Mini-B") and ("--kv-cache-dtype" in cmd) == ("fp8" in label)
        assert res["runtime_config"]["kv_cache_dtype"] == r["config"]["kv_cache_dtype"]
        cells = res["performance"]["cells"]
        assert [c["io"]["input_tokens"] for c in cells] == [16, 32] and [lv["concurrency"] for lv in cells[0]["levels"]] == [1, 2, 4]
        assert set(res["quality"]["suites"]) == {"structured_json", "coding_python", "retrieval_needle"} and res["quality"]["split"] == "heldout"
        for name, s in res["quality"]["suites"].items():
            assert s["interval"]["rate"] == 1.0 and s["split"] == "heldout" and s["fingerprint"] and s["version"] == "1", name
        # synthetic benchmark prompts and real evaluation prompts are labelled differently and measured separately
        assert "synthetic" in res["performance"]["workload_kind"]
        assert "evaluation_prompts" in res["quality"]["suites"]["coding_python"]["performance"]["workload_kind"]
        assert res["quality"]["suites"]["coding_python"]["n"] == 13 and res["calibration"]["measured_with_server_tokenizer"] is True
        assert set(res["artifacts"]) >= {"cases", "perf_requests", "server_log"}
    assert sum(1 for c, _ in fd.calls if c == "sandbox") == 4 * 13  # every coding case ran in a sandbox container


@needs_gpu
def test_quality_differences_between_configs_are_recorded_per_config(mx):
    client, fd, ids, worker = mx
    fd.kw_fn = lambda spec: {"responder": oracle(degrade_coding=True)} if "--kv-cache-dtype" in spec.command else {}
    eid, jid = submit(client, body(ids, kv_cache_dtypes=["auto", "fp8"], evals=["coding_python", "structured_json"]))
    worker.tick()
    rs = rows(client, eid)
    auto = next(r for k, r in rs.items() if "kv:" not in k)["results"]["quality"]["suites"]
    fp8 = next(r for k, r in rs.items() if "kv:fp8" in k)["results"]["quality"]["suites"]
    assert auto["coding_python"]["interval"]["rate"] == 1.0 and fp8["coding_python"]["interval"]["rate"] == 0.0
    assert fp8["coding_python"]["aggregate"]["failures_by_type"] == {"missing_function": 13}
    assert auto["structured_json"]["interval"]["rate"] == fp8["structured_json"]["interval"]["rate"] == 1.0


@needs_gpu
def test_infeasible_configs_are_skipped_and_never_launched(mx):
    client, fd, ids, worker = mx
    eid, jid = submit(client, body(ids, quantizations=["none", "awq"]))
    worker.tick()
    rs = rows(client, eid)
    sk = next(r for k, r in rs.items() if "w:awq" in k)
    assert sk["status"] == "skipped" and sk["failure_kind"] == "infeasible" and "awq" in sk["error"] and sk["results"] is None
    assert len(runs(fd)) == 1 and job(client, jid)["results"]["counts"] == {"completed": 1, "skipped": 1}


# ---------------- failure isolation ----------------
@needs_gpu
def test_launch_failure_of_one_config_does_not_affect_others(mx):
    client, fd, ids, worker = mx
    fd.fail_fn = lambda spec: "could not select device driver" if spec.command[0] == "Qwen/Mini-B" else None
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]], kv_cache_dtypes=["auto", "fp8"]))
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "completed" and j["results"]["counts"] == {"completed": 2, "failed": 2}
    for label, r in rows(client, eid).items():
        if "Mini-B" in label:
            assert r["status"] == "failed" and r["failure_kind"] == "docker_failed" and "could not select device driver" in r["error"]
        else:
            assert r["status"] == "completed" and r["results"]["quality"]["suites"]["structured_json"]["n"] == 18
    assert fd.containers == {}


@needs_gpu
def test_server_crash_midway_keeps_partial_data_and_other_configs(mx):
    client, fd, ids, worker = mx
    fd.server_kw.update(ttft_s=0.05, token_s=0.05)
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]], concurrency_levels=[1, 2], iterations=1, requests_per_level=12,
                                   io_pairs=[{"input_tokens": 16, "output_tokens": 16}]))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: ((job(client, jid).get("progress") or {}).get("matrix") or {}).get("config_index") == 1 and
             ((job(client, jid)["progress"].get("sweep") or {}).get("concurrency") == 2))
    fd.crash(next(iter(fd.containers)), oom=True, logs="torch.OutOfMemoryError: CUDA out of memory")
    t.join(timeout=40)
    j = job(client, jid)
    assert j["status"] == "completed" and j["results"]["counts"] == {"completed": 1, "failed": 1}
    rs = rows(client, eid)
    bad = next(r for r in rs.values() if r["status"] == "failed")
    assert bad["failure_kind"] in ("server_crashed", "ServerCrashed") and "crashed" in bad["error"].lower()
    assert bad["results"]["partial"] is True and bad["results"]["performance"]["cells"][0]["levels"][0]["status"] == "ok"  # level 1 survived
    good = next(r for r in rs.values() if r["status"] == "completed")
    assert good["results"]["partial"] is False and fd.containers == {}


@needs_gpu
def test_every_config_failing_fails_the_job_with_a_summary(mx):
    client, fd, ids, worker = mx
    fd.fail_run = "docker daemon unavailable"
    eid, jid = submit(client, body(ids, kv_cache_dtypes=["auto", "fp8"]))
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "failed" and "No configuration completed" in j["error"] and "docker_failed" in j["error"] and j["results"] is None
    assert {r["status"] for r in rows(client, eid).values()} == {"failed"}  # per-config records still persist


@needs_gpu
def test_startup_oom_is_classified_per_config(mx):
    client, fd, ids, worker = mx
    fd.mode_fn = lambda spec: "exit_at_start" if "--kv-cache-dtype" in spec.command else None
    fd.exit_logs = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"
    eid, jid = submit(client, body(ids, kv_cache_dtypes=["auto", "fp8"]))
    worker.tick()
    rs = rows(client, eid)
    assert next(r for k, r in rs.items() if "kv:fp8" in k)["failure_kind"] == "oom" and next(r for k, r in rs.items() if "kv:" not in k)["status"] == "completed"
    srv = client.get(f"/api/managed-servers/{next(r for k, r in rs.items() if 'kv:fp8' in k)['managed_server_id']}").json()
    assert "CUDA out of memory" in srv["log_tail"]


@needs_gpu
def test_sandbox_infrastructure_failure_is_not_blamed_on_the_model(mx):
    client, fd, ids, worker = mx
    fd.sandbox_error = "docker: cannot connect to the daemon"
    eid, jid = submit(client, body(ids, evals=["coding_python"]))
    worker.tick()
    r = next(iter(rows(client, eid).values()))
    s = r["results"]["quality"]["suites"]["coding_python"]
    assert s["aggregate"]["failures_by_type"] == {"sandbox_error": 13} and r["status"] == "completed"


# ---------------- budget, cancellation, shutdown ----------------
@needs_gpu
def test_runtime_budget_stops_new_configs_but_keeps_finished_ones(mx, monkeypatch):
    client, fd, ids, worker = mx
    t = {"v": 0.0}

    def clock():
        t["v"] += 6.0  # every look at the clock costs 6 'seconds'
        return t["v"]

    monkeypatch.setattr(opt, "_now", clock)
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]], kv_cache_dtypes=["auto", "fp8"], max_runtime_s=10))
    worker.tick()
    j = job(client, jid)
    assert j["status"] == "completed" and j["results"]["stopped_reason"] == "budget_exhausted"
    assert j["results"]["counts"] == {"completed": 1, "skipped": 3} and len(runs(fd)) == 1
    sk = [r for r in rows(client, eid).values() if r["status"] == "skipped"]
    assert {r["failure_kind"] for r in sk} == {"budget_exhausted"} and "runtime budget" in sk[0]["error"]


@needs_gpu
def test_cancel_during_startup_of_second_config(mx):
    client, fd, ids, worker = mx
    n = {"i": 0}

    def mode(spec):
        n["i"] += 1
        return "never_ready" if n["i"] == 2 else None

    fd.mode_fn = mode
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]], kv_cache_dtypes=["auto", "fp8"]))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: ((job(client, jid).get("progress") or {}).get("matrix") or {}).get("config_index") == 1 and job(client, jid)["progress"]["phase"] == "waiting_for_server")
    client.post(f"/api/jobs/{jid}/cancel")
    t.join(timeout=30)
    assert not t.is_alive() and job(client, jid)["status"] == "cancelled"
    statuses = [r["status"] for r in sorted(rows(client, eid).values(), key=lambda r: r["id"])]
    assert statuses == ["completed", "cancelled", "cancelled", "cancelled"]  # finished work is kept; nothing else was started
    assert len(runs(fd)) == 2 and fd.containers == {}


@needs_gpu
def test_cancel_during_evaluation_cleans_up(mx):
    client, fd, ids, worker = mx
    fd.server_kw.update(chat_token_s=0.05)
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]], perf_enabled=False, evals=["structured_json"]))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: (job(client, jid).get("progress") or {}).get("phase") == "eval:structured_json")
    client.post(f"/api/jobs/{jid}/cancel")
    t.join(timeout=40)
    assert not t.is_alive() and job(client, jid)["status"] == "cancelled" and fd.containers == {}
    assert {r["status"] for r in rows(client, eid).values()} == {"cancelled"} and len(runs(fd)) == 1


@needs_gpu
def test_worker_shutdown_leaves_resumable_state(mx):
    client, fd, ids, worker = mx
    fd.server_kw.update(chat_token_s=0.05)
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]], perf_enabled=False, evals=["structured_json"]))
    t = threading.Thread(target=worker.tick)
    t.start()
    wait_for(lambda: (job(client, jid).get("progress") or {}).get("phase") == "eval:structured_json")
    worker.shutdown.set()
    t.join(timeout=40)
    j = job(client, jid)
    assert j["status"] == "failed" and "shutdown" in j["error"] and fd.containers == {}
    st = {r["label"]: r for r in rows(client, eid).values()}
    assert sorted(r["status"] for r in st.values()) == ["failed", "pending"]
    assert next(r for r in st.values() if r["status"] == "failed")["failure_kind"] == "interrupted"


# ---------------- resume and persistence ----------------
@needs_gpu
def test_resume_reruns_only_unfinished_configs(mx):
    client, fd, ids, worker = mx
    fd.fail_fn = lambda spec: "could not select device driver" if spec.command[0] == "Qwen/Mini-B" else None
    eid, jid = submit(client, body(ids, registry_model_ids=[ids["a"], ids["b"]]))
    worker.tick()
    first = rows(client, eid)
    a_before = next(r for k, r in first.items() if "Mini-Instruct" in k)
    assert a_before["status"] == "completed" and next(r for k, r in first.items() if "Mini-B" in k)["status"] == "failed"
    launches_before = len(runs(fd))
    fd.fail_fn = None
    r = client.post(f"/api/experiments/{eid}/resume")
    assert r.status_code == 201 and len(r.json()["jobs"]) == 2
    worker.tick()
    j2 = job(client, r.json()["jobs"][1]["id"])
    assert j2["status"] == "completed" and j2["results"]["counts"] == {"completed": 2}
    assert len(runs(fd)) == launches_before + 1  # only the failed configuration was launched again
    second = rows(client, eid)
    a_after = next(r for k, r in second.items() if "Mini-Instruct" in k)
    assert a_after["finished_at"] == a_before["finished_at"] and a_after["attempts"] == 1 and a_after["results"] == a_before["results"]  # untouched
    b_after = next(r for k, r in second.items() if "Mini-B" in k)
    assert b_after["status"] == "completed" and b_after["attempts"] == 2 and b_after["failure_kind"] is None and b_after["error"] is None


@needs_gpu
def test_resume_guards(mx):
    client, fd, ids, worker = mx
    eid, jid = submit(client, body(ids))
    assert client.post(f"/api/experiments/{eid}/resume").status_code == 409  # latest job is still queued
    worker.tick()
    d = client.post("/api/experiments", json={"type": "diagnostic", "config": {"duration_s": 1}}).json()
    assert client.post(f"/api/experiments/{d['id']}/resume").status_code == 409  # not a resumable type
    assert client.post("/api/experiments/9999/resume").status_code == 404


@needs_gpu
def test_results_survive_api_restart(mx, settings):
    client, fd, ids, worker = mx
    eid, jid = submit(client, body(ids))
    worker.tick()
    before = rows(client, eid)
    db.init_db(settings)  # a fresh engine behaves like a restarted process
    with TestClient(create_app()) as c2:
        after = {r["label"]: r for r in c2.get(f"/api/optimization/{eid}").json()}
        assert after == before and c2.get(f"/api/optimization/{eid}/pareto").status_code == 200
    assert client.get("/api/optimization/9999").status_code == 404


# ---------------- reconciliation of sandbox containers ----------------
def test_reconcile_treats_sandbox_containers_via_their_job_label(mx, settings):
    client, fd, ids, worker = mx
    eid, jid = submit(client, body(ids))
    iid = vm.owned_docker(settings).instance_id
    labels = {LABEL_MANAGED: "true", LABEL_INSTANCE: iid, "locallab.job_id": str(jid), "locallab.role": "sandbox"}
    live = fd.add_foreign("locallab-managed-sbx-live", labels)
    with db.session_scope() as s:
        j = s.get(Job, jid)
        j.status, j.heartbeat_at = "running", utcnow()
    assert vm.reconcile_orphans(vm.owned_docker(settings), settings) == [] and live.id in fd.containers  # active job: keep
    with db.session_scope() as s:
        s.get(Job, jid).status = "failed"
    assert vm.reconcile_orphans(vm.owned_docker(settings), settings) == ["locallab-managed-sbx-live"] and live.id not in fd.containers


# ---------------- Pareto / export API over stored results ----------------
def synthetic_row(eid, key, label, cfg, coding, thr, p95, vram, split="heldout", contended=False, n=13):
    from locallab.evals.base import wilson

    def lvl(c, t):
        st = lambda m, h=3.0: {"n": 3, "mean": m, "ci95_half": h}  # noqa: E731
        return {"concurrency": c, "status": "ok", "stats": {"aggregate_output_tok_s": st(t), "ttft_p50_ms": st(10, 1), "tpot_p50_ms": st(3, 0.1), "e2e_p95_ms": st(p95, 5)},
                "latency": {"e2e_ms": {"n": 48, "p95": p95}}, "gpu": {"peak_memory_mb": vram, "server_peak_memory_mb": vram - 2000}, "requests": {"attempted": 48, "failed": 0}}
    res = {"partial": False, "server_config": cfg, "startup_s": 90.0, "runtime": {"kv_cache_tokens": 100000},
           "gpu": {"contended": contended}, "performance": {"cells": [{"io": {"input_tokens": 128, "output_tokens": 128}, "levels": [lvl(1, thr / 4), lvl(8, thr)]}]},
           "quality": {"split": split, "suites": {"coding_python": {"n": n, "interval": wilson(round(coding * n), n), "aggregate": {},
                       "performance": {"requests": {"attempted": n, "failed": 0}, "latency": {"e2e_ms": {"p50": 900.0}}}}}}}
    return ConfigResult(experiment_id=eid, key=key, label=label, config=cfg, status="completed", attempts=1, results=res)


def seed_matrix(settings):
    with db.session_scope() as s:
        e = Experiment(type="optimization_matrix", name="synthetic", config={}, config_hash="x")
        e.jobs.append(Job())
        s.add(e)
        s.flush()
        cfg = lambda m: {"model_name": m, "revision": "r", "max_model_len": 8192, "quantization": "none", "kv_cache_dtype": "auto"}  # noqa: E731
        s.add_all([synthetic_row(e.id, "k1", "big", cfg("big"), 0.85, 900, 400, 24000),
                   synthetic_row(e.id, "k2", "small", cfg("small"), 0.55, 2400, 150, 9000),
                   synthetic_row(e.id, "k3", "bad", cfg("bad"), 0.50, 800, 500, 26000, contended=True),
                   ConfigResult(experiment_id=e.id, key="k4", label="crashed", config=cfg("crashed"), status="failed", failure_kind="oom", error="boom", attempts=1)])
        return e.id


def test_pareto_endpoint_front_constraints_and_notes(mx, settings):
    client, *_ = mx
    eid = seed_matrix(settings)
    r = client.get(f"/api/optimization/{eid}/pareto", params={"objectives": "coding_pass_rate:max,throughput_tok_s:max", "concurrency": 8}).json()
    a = r["analysis"]
    assert set(a["front"]) == {"k1", "k2"} and [d["id"] for d in a["dominated"]] == ["k3"]
    assert {x["id"] for x in r["not_analysed"]} == {"k4"} and r["not_analysed"][0]["reason"] == "oom"  # failed configs are listed, not hidden
    assert r["cell"]["common_levels"] == [1, 8] and r["cell"]["io_pairs"] == [{"input_tokens": 128, "output_tokens": 128}]
    pt = next(p for p in r["points"] if p["id"] == "k1")["values"]["coding_pass_rate"]
    assert pt["display"] == pytest.approx(84.6, abs=0.1) and pt["lo"] < pt["display"] < pt["hi"] and pt["n"] == 13  # shown as %, with its interval
    assert any("contended" in n for n in r["notes"]) and any("only 13 samples" in w for w in a["warnings"])
    c = client.get(f"/api/optimization/{eid}/pareto", params={"objectives": "coding_pass_rate:max,throughput_tok_s:max", "constraints": "peak_vram_mb:<=:24000", "concurrency": 8}).json()
    assert set(c["analysis"]["feasible"]) == {"k1", "k2"} and {e["id"] for e in c["analysis"]["excluded"]} == {"k3"}
    pct = client.get(f"/api/optimization/{eid}/pareto", params={"objectives": "throughput_tok_s:max", "constraints": "coding_pass_rate:>=:80", "concurrency": 8}).json()
    assert pct["analysis"]["feasible"] == ["k1"]  # '80' means 80%
    w = client.get(f"/api/optimization/{eid}/pareto", params={"objectives": "coding_pass_rate:max,throughput_tok_s:max", "weights": "coding_pass_rate:3,throughput_tok_s:1", "concurrency": 8}).json()
    assert w["analysis"]["weighted"]["weights"] == {"coding_pass_rate": 0.75, "throughput_tok_s": 0.25} and "formula" in w["analysis"]["weighted"]
    assert client.get(f"/api/optimization/{eid}/pareto", params={"concurrency": 8}).json()["analysis"] is None  # nothing is ranked unless asked


def test_pareto_endpoint_validation_and_defaults(mx, settings):
    client, *_ = mx
    eid = seed_matrix(settings)
    g = lambda **p: client.get(f"/api/optimization/{eid}/pareto", params=p)  # noqa: E731
    assert g(objectives="nonsense:max").status_code == 422
    assert g(objectives="coding_pass_rate:max", constraints="startup_s:<=").status_code == 422
    assert g(objectives="coding_pass_rate:max", constraints="startup_s:~:5").status_code == 422
    assert g(objectives="coding_pass_rate:max", weights="coding_pass_rate:abc").status_code == 422
    assert g(objectives="coding_pass_rate:sideways").status_code == 422
    r = g(objectives="coding_pass_rate").json()  # direction defaults from the metric catalog
    assert r["analysis"]["objectives"][0]["direction"] == "max" and r["cell"]["default_concurrency"] in (1, 8)
    missing = g(objectives="coding_pass_rate:max,throughput_tok_s:max", concurrency=4).json()
    assert any("not measured for every configuration" in n for n in missing["notes"]) and missing["analysis"]["excluded"]


def test_dev_split_results_are_flagged(mx, settings):
    client, *_ = mx
    eid = seed_matrix(settings)
    with db.session_scope() as s:
        r = s.scalars(select(ConfigResult).where(ConfigResult.key == "k1")).one()
        res = json.loads(json.dumps(r.results))
        res["quality"]["split"] = "dev"
        r.results = res
    assert any("dev split" in n for n in client.get(f"/api/optimization/{eid}/pareto").json()["notes"])


def test_export_csv_and_json(mx, settings):
    client, *_ = mx
    eid = seed_matrix(settings)
    csv_text = client.get(f"/api/optimization/{eid}/export", params={"format": "csv", "concurrency": 8}).text.splitlines()
    head = csv_text[0].split(",")
    assert head[:4] == ["key", "label", "status", "failure_kind"] and "coding_pass_rate" in head and "throughput_tok_s_lo" in head
    rows_ = {l.split(",")[0]: l.split(",") for l in csv_text[1:]}
    assert float(rows_["k1"][head.index("throughput_tok_s")]) == 900.0 and rows_["k1"][head.index("status")] == "completed"
    assert rows_["k4"][head.index("failure_kind")] == "oom" and rows_["k4"][head.index("throughput_tok_s")] == ""  # failed: empty, never zero
    js = client.get(f"/api/optimization/{eid}/export").json()
    assert [r["key"] for r in js] == ["k1", "k2", "k3", "k4"] and js[3]["error"] == "boom" and js[0]["results"]["quality"]["split"] == "heldout"
    assert client.get(f"/api/optimization/{eid}/export", params={"format": "xml"}).status_code == 422


def test_metric_catalog_endpoint(mx):
    client, *_ = mx
    ids = {m["id"] for m in client.get("/api/optimization/metrics").json()}
    assert {"coding_pass_rate", "structured_schema_rate", "retrieval_accuracy", "throughput_tok_s", "peak_vram_mb", "eval_error_rate"} <= ids
