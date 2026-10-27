import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import select

import locallab.db as db
from locallab.models import Experiment, Job, ManagedServer, ModelRecord, utcnow
from locallab.runtimes import vllm_managed as vm
from locallab.runtimes.docker import LABEL_INSTANCE, LABEL_MANAGED, OwnedDocker
from locallab.runtimes.vllm_compat import ManagedVllmConfig, classify_launch_failure, validate_runtime
from locallab.services import gpu_preflight as gp
from locallab.services.gpu_preflight import GpuPreflight
from locallab.services.registry import sync_registry
from tests.fake_docker import FakeDocker
from tests.test_registry import make_repo

CFG = {"architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2,
       "hidden_size": 64, "max_position_embeddings": 32768}
IID = "test-instance"


@pytest.fixture
def gpu(monkeypatch):
    state = GpuPreflight(gpu_index=0, name="Fake 5090", total_mb=32768, free_mb=20000, compute_capability=(12, 0))

    def fake(index, util, allow, exclude_pids=None):
        s = GpuPreflight(**{**state.__dict__, "requested_mb": util * state.total_mb, "override": allow,
                            "errors": list(state.errors), "warnings": list(state.warnings)})
        return s

    monkeypatch.setattr(vm, "check_gpu", fake)
    return state


@pytest.fixture
def fd(settings, fake_docker):
    d = fake_docker
    (settings.data_dir / f"vllm_archs_{d.image.id.split(':')[-1][:16]}.json").write_text(json.dumps(d.arch_list))
    return d


@pytest.fixture
def model_id(settings, session):
    make_repo(Path(settings.hf_cache_dir), "Qwen/Mini-Instruct", CFG)
    sync_registry(session, settings)
    session.commit()
    return session.scalars(select(ModelRecord).where(ModelRecord.name == "Qwen/Mini-Instruct")).one().id


def make_job() -> tuple[int, int]:
    with db.session_scope() as s:
        e = Experiment(type="concurrency_sweep", name="t", config={}, config_hash="h")
        e.jobs.append(Job())
        s.add(e)
        s.flush()
        return e.id, e.jobs[0].id


def plan_for(settings, fd, model_id, **cfg):
    return vm.build_plan(ManagedVllmConfig(**cfg), model_id, settings, fd, allow_run_oneoff=True)


# ---------------- planning / validation ----------------
def test_plan_ok_builds_exact_command(settings, fd, model_id, gpu):
    p = plan_for(settings, fd, model_id, max_model_len=2048, gpu_memory_utilization=0.3)
    assert p.errors == []
    cmd = p.command()
    assert cmd[:2] == ["Qwen/Mini-Instruct", "--revision"] and cmd[2] == "abc123"
    assert "--served-model-name" in cmd and cmd[cmd.index("--served-model-name") + 1] == "mini-instruct"
    assert "--quantization" not in cmd and "--kv-cache-dtype" not in cmd
    assert p.estimates["kv_tokens"] > 2048 and p.estimates["kv_bytes_per_token"] == 256  # 2*4 layers*2 heads*8*2B


def test_plan_rejects_non_hf_and_missing_pieces(settings, fd, gpu, session):
    r = vm.build_plan(ManagedVllmConfig(), None, settings, fd, allow_run_oneoff=False)
    assert any("registry_model_id is required" in e for e in r.errors)
    rec = ModelRecord(name="llama3:latest", source="ollama", ref="llama3:latest", revision=None)
    session.add(rec)
    session.commit()
    r = vm.build_plan(ManagedVllmConfig(), rec.id, settings, fd, allow_run_oneoff=False)
    assert any("Only Hugging Face cache models" in e for e in r.errors)
    fd.image_present = False
    r = vm.build_plan(ManagedVllmConfig(), rec.id, settings, fd, allow_run_oneoff=False)
    assert any("not available locally" in e for e in r.errors)


def test_incomplete_download_rejected(settings, fd, gpu, session):
    make_repo(Path(settings.hf_cache_dir), "org/noweights", CFG, weights=False)
    sync_registry(session, settings)
    session.commit()
    mid = session.scalars(select(ModelRecord).where(ModelRecord.name == "org/noweights")).one().id
    r = vm.build_plan(ManagedVllmConfig(), mid, settings, fd, allow_run_oneoff=False)
    assert any("no safetensors weights" in e for e in r.errors)


@pytest.mark.parametrize("kw,needle", [
    ({"quantization": "awq"}, "requires a awq checkpoint"),
    ({"quantization": "gptq"}, "requires a gptq checkpoint"),
    ({"max_model_len": 65536}, "exceeds the model's max_position_embeddings"),
])
def test_compat_errors(settings, fd, model_id, gpu, kw, needle):
    p = plan_for(settings, fd, model_id, **kw)
    assert any(needle in e for e in p.errors), p.errors


def test_fp8_needs_capability_and_quant_mismatch():
    ck = {"architectures": ["A"], "quant_method": None, "max_position_embeddings": 32768, "kv_layers": 4, "kv_heads": 2,
          "head_dim": 8, "weights_mb": 100, "dtype": "bfloat16"}
    errs, warns, _ = validate_runtime(ManagedVllmConfig(quantization="fp8", kv_cache_dtype="fp8"), ck, (7, 5), 32768, ["A"])
    assert any("compute capability >= 8.9" in e for e in errs)
    errs, warns, _ = validate_runtime(ManagedVllmConfig(quantization="fp8", kv_cache_dtype="fp8"), ck, (12, 0), 32768, ["A"])
    assert errs == [] and any("Online fp8" in w for w in warns) and any("fp8 KV cache" in w for w in warns)
    errs, _, _ = validate_runtime(ManagedVllmConfig(quantization="fp8"), {**ck, "quant_method": "awq"}, (12, 0), 32768, ["A"])
    assert any("cannot be applied to a awq-quantized" in e for e in errs)


def test_memory_arithmetic_errors():
    ck = {"architectures": ["A"], "quant_method": None, "max_position_embeddings": 100000, "kv_layers": 32, "kv_heads": 8,
          "head_dim": 128, "weights_mb": 8000, "dtype": "bfloat16"}
    errs, _, est = validate_runtime(ManagedVllmConfig(gpu_memory_utilization=0.2, max_model_len=4096), ck, (12, 0), 32768, ["A"])
    assert any("do not fit" in e for e in errs)  # 6553 MB budget < 8000 weights
    errs, _, est = validate_runtime(ManagedVllmConfig(gpu_memory_utilization=0.3, max_model_len=65536), ck, (12, 0), 32768, ["A"])
    assert est["kv_bytes_per_token"] == 131072 and any("below max_model_len" in e for e in errs)


def test_unsupported_architecture_uses_cached_list(settings, fd, model_id, gpu):
    fd.arch_list = ["LlamaForCausalLM"]
    for f in settings.data_dir.glob("vllm_archs_*.json"):
        f.write_text(json.dumps(fd.arch_list))
    assert any("not supported by this vLLM image" in e for e in plan_for(settings, fd, model_id).errors)


def test_arch_list_fetched_once_and_cached(settings, session, model_id, gpu):
    d = FakeDocker()
    d.arch_list = ["Qwen2ForCausalLM"]
    for _ in range(2):
        assert plan_for(settings, d, model_id).errors == []
    assert d.oneoff_calls == 1
    for f in settings.data_dir.glob("vllm_archs_*.json"):
        f.unlink()
    p = vm.build_plan(ManagedVllmConfig(), model_id, settings, FakeDocker(), allow_run_oneoff=False)
    assert any("not cached yet" in w for w in p.warnings)  # API-side validation never runs containers


def test_failure_classification():
    assert classify_launch_failure("RuntimeError: Error 803: system has unsupported display driver")[0] == "cuda_driver_mismatch"
    assert classify_launch_failure("ValueError: Free memory on device (5/32 GiB) on startup is less than desired GPU memory utilization")[0] == "insufficient_free_memory"
    assert classify_launch_failure("ValueError: No available memory for the cache blocks")[0] == "insufficient_kv_cache"
    assert classify_launch_failure("torch.OutOfMemoryError: CUDA out of memory")[0] == "oom"
    assert classify_launch_failure("Model architectures ['X'] are not supported")[0] == "unsupported_model"
    assert classify_launch_failure("AssertionError: Error in memory profiling. Initial free memory 11.1 GiB, current free memory 28.07 GiB")[0] == "gpu_memory_changed_during_startup"
    assert classify_launch_failure("boom", oom_killed=True)[0] == "host_oom"
    assert classify_launch_failure("???")[0] == "launch_failed"


# ---------------- GPU preflight ----------------
def Proc(pid, name, mb, kind="compute"):
    from locallab.schemas.system import GpuProcess
    return GpuProcess(pid=pid, name=name, used_memory_mb=mb, kind=kind)


def fake_gpu_env(monkeypatch, procs, util, free=20000.0):
    from locallab.schemas.system import GpuInfo, GpuSnapshot

    g = GpuInfo(index=0, name="5090", uuid="u", memory_total_mb=32768, memory_used_mb=32768 - free, memory_free_mb=free,
                utilization_gpu_pct=util, utilization_mem_pct=0, temperature_c=40, power_w=100, power_limit_w=600,
                clock_sm_mhz=1, clock_mem_mhz=1, processes=procs)
    monkeypatch.setattr(gp, "read_gpus", lambda: GpuSnapshot(available=True, timestamp=utcnow(), gpus=[g]))
    monkeypatch.setattr(gp, "sample_gpu_light", lambda: [{"index": 0, "utilization_gpu_pct": util}])
    monkeypatch.setattr(gp, "compute_capability", lambda i: (12, 0))
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)


def test_preflight_idle_gpu_ok(monkeypatch):
    fake_gpu_env(monkeypatch, [Proc(1, "Xorg", 500, "graphics")], util=3)
    r = gp.check_gpu(0, 0.3, False)
    assert r.errors == [] and r.contended is False and r.other_processes == []


def test_preflight_contention_blocks_unless_overridden(monkeypatch):
    fake_gpu_env(monkeypatch, [Proc(99, "/venv/bin/python", 17000)], util=99)
    r = gp.check_gpu(0, 0.3, False)
    assert r.contended and any("Refusing to benchmark" in e for e in r.errors)
    r = gp.check_gpu(0, 0.3, True)
    assert r.contended and r.errors == [] and any("results will be marked contended" in w for w in r.warnings)
    assert r.other_processes[0]["pid"] == 99


def test_preflight_high_util_without_visible_process_is_contention(monkeypatch):
    fake_gpu_env(monkeypatch, [], util=80)
    assert gp.check_gpu(0, 0.3, False).contended is True


def test_preflight_insufficient_memory_cannot_be_overridden(monkeypatch):
    fake_gpu_env(monkeypatch, [], util=1, free=5000)
    r = gp.check_gpu(0, 0.3, True)
    assert any("cannot be overridden" in e for e in r.errors)


def test_preflight_excludes_own_pids(monkeypatch):
    fake_gpu_env(monkeypatch, [Proc(7, "VLLM::EngineCore", 9000)], util=1)
    assert gp.check_gpu(0, 0.3, False, exclude_pids={7}).contended is False


# ---------------- lifecycle ----------------
def make_server(settings, fd, model_id, **cfg):
    plan = plan_for(settings, fd, model_id, **cfg)
    assert plan.errors == [], plan.errors
    eid, jid = make_job()
    return vm.ManagedVllmServer(plan, OwnedDocker(fd, IID), settings, jid, eid), jid


def row(server) -> ManagedServer:
    with db.session_scope() as s:
        return s.get(ManagedServer, server.row_id)


def test_launch_ready_shutdown_records_everything(settings, fd, model_id, gpu):
    fd.ready_delay_s = 0.3
    srv, jid = make_server(settings, fd, model_id)
    srv.launch()
    info = srv.wait_ready(lambda: False, poll_s=0.1)
    assert info.models[0].id == "mini-instruct" or info.models[0].id == "mock-model"
    r = row(srv)
    assert r.status == "ready" and r.startup_s >= 0.3 and r.ready_at is not None
    assert r.container_id == srv.container_id and r.image_digest.startswith("vllm/vllm-openai@sha256:")
    assert r.command[:3] == ["Qwen/Mini-Instruct", "--revision", "abc123"] and r.model["revision"] == "abc123"
    labels = fd.containers[srv.container_id].labels
    assert labels[LABEL_MANAGED] == "true" and labels[LABEL_INSTANCE] == IID and labels["locallab.job_id"] == str(jid)
    spec = fd.containers[srv.container_id].spec
    assert spec.mounts[0][2] is True and spec.env["HF_HUB_OFFLINE"] == "1" and "LD_LIBRARY_PATH" in spec.env
    srv.shutdown("done")
    srv.shutdown("again")  # idempotent
    assert fd.containers == {} and row(srv).status == "stopped" and "Application startup complete" in row(srv).log_tail
    assert [c for c in fd.calls if c[0] == "rm"] == [("rm", srv.container_id)]


def test_docker_run_failure_marks_row_failed(settings, fd, model_id, gpu):
    fd.fail_run = "docker: Error response from daemon: could not select device driver"
    srv, _ = make_server(settings, fd, model_id)
    with pytest.raises(vm.ManagedServerError) as e:
        srv.launch()
    assert e.value.kind == "docker_failed"
    srv.shutdown("cleanup")
    assert row(srv).status == "failed" and "could not select device driver" in row(srv).stop_reason


def test_startup_timeout_is_bounded_and_cleaned_up(settings, fd, model_id, gpu):
    fd.start_mode = "never_ready"
    srv, _ = make_server(settings, fd, model_id, startup_timeout_s=30)
    srv.plan.config.startup_timeout_s = 0.6  # bypass the 30 s schema floor for the test
    srv.launch()
    t0 = time.time()
    with pytest.raises(vm.ManagedServerError) as e:
        srv.wait_ready(lambda: False, poll_s=0.1)
    assert e.value.kind == "startup_timeout" and time.time() - t0 < 5
    srv.shutdown("timeout")
    assert fd.containers == {}


def test_cancel_during_startup(settings, fd, model_id, gpu):
    fd.start_mode = "never_ready"
    srv, _ = make_server(settings, fd, model_id)
    srv.launch()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(vm.ManagedServerError) as e:
        srv.wait_ready(cancel, poll_s=0.05)
    assert e.value.kind == "cancelled"
    srv.shutdown("cancelled")
    assert fd.containers == {}


def test_container_exits_during_startup_with_oom(settings, fd, model_id, gpu):
    fd.start_mode = "exit_at_start"
    fd.exit_logs = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"
    srv, _ = make_server(settings, fd, model_id)
    srv.launch()
    with pytest.raises(vm.ManagedServerError) as e:
        srv.wait_ready(lambda: False, poll_s=0.05)
    assert e.value.kind == "oom"
    srv.shutdown("failed")
    r = row(srv)
    assert r.exit_code == 1 and "CUDA out of memory" in r.log_tail and fd.containers == {}


def test_watchdog_detects_crash_after_ready(settings, fd, model_id, gpu):
    srv, _ = make_server(settings, fd, model_id)
    srv.launch()
    srv.wait_ready(lambda: False, poll_s=0.05)
    assert srv.check_alive() is None
    fd.crash(srv.container_id, oom=False, logs="RuntimeError: CUDA error: an illegal memory access\ntorch.OutOfMemoryError")
    msg = srv.check_alive()
    assert "exited" in msg and "oom" in msg
    srv.shutdown("crashed")
    assert srv.check_alive() is None and fd.containers == {}


# ---------------- reconciliation ----------------
def test_reconcile_removes_only_orphaned_owned_containers(settings, fd, model_id, gpu):
    od = OwnedDocker(fd, IID)
    # 1) orphan: owned container whose job already failed
    srv, jid = make_server(settings, fd, model_id)
    srv.launch()
    with db.session_scope() as s:
        s.get(Job, jid).status = "failed"
    # 2) active job with a fresh heartbeat: must be kept
    srv2, jid2 = make_server(settings, fd, model_id)
    srv2.launch()
    with db.session_scope() as s:
        j = s.get(Job, jid2)
        j.status, j.heartbeat_at = "running", utcnow()
    # 3) unrelated containers incl. one that looks similar
    foreign = fd.add_foreign("open-webui")
    lookalike = fd.add_foreign("locallab-managed-9-zzz", {LABEL_MANAGED: "true", LABEL_INSTANCE: "other-install"})
    removed = vm.reconcile_orphans(od, settings)
    assert removed == [srv.container_name]
    assert srv.container_id not in fd.containers and srv2.container_id in fd.containers
    assert foreign.id in fd.containers and lookalike.id in fd.containers
    assert row(srv).status == "orphan_removed" and "orphaned" in row(srv).stop_reason
    assert vm.reconcile_orphans(od, settings) == []  # idempotent
    # worker died mid-job: heartbeat goes stale -> removed on the next pass
    with db.session_scope() as s:
        s.get(Job, jid2).heartbeat_at = utcnow() - timedelta(seconds=settings.worker_stale_s + 5)
    assert vm.reconcile_orphans(od, settings) == [srv2.container_name]


def test_reconcile_marks_rows_whose_container_vanished(settings, fd, model_id, gpu):
    srv, jid = make_server(settings, fd, model_id)
    srv.launch()
    fd.containers.pop(srv.container_id)
    with db.session_scope() as s:
        s.get(Job, jid).status = "failed"
    vm.reconcile_orphans(OwnedDocker(fd, IID), settings)
    assert row(srv).status == "failed" and "not found" in row(srv).stop_reason
