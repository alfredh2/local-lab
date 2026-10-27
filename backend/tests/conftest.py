from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

import locallab.db as db
from locallab.config import Settings, get_settings
from locallab.models import utcnow  # noqa: F401
from locallab.runtimes import vllm_managed as vm
from locallab.services.gpu import read_gpus
from locallab.services.gpu_preflight import GpuPreflight

BACKEND = Path(__file__).resolve().parent.parent


def migrate(url: str) -> None:
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("LOCALLAB_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LOCALLAB_DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setenv("LOCALLAB_HF_CACHE_DIR", str(tmp_path / "hf"))
    monkeypatch.setenv("LOCALLAB_OLLAMA_URL", "http://127.0.0.1:1")  # unreachable on purpose
    get_settings.cache_clear()
    s = get_settings()
    migrate(s.database_url)
    db.init_db(s)
    yield s
    get_settings.cache_clear()


@pytest.fixture
def session(settings: Settings) -> Iterator[Session]:
    with db.session_scope() as s:
        yield s


@pytest.fixture(autouse=True)
def fake_docker(monkeypatch: pytest.MonkeyPatch):
    """No test may touch the real Docker daemon: every Docker call goes to this in-memory fake."""
    import locallab.runtimes.vllm_managed as vm
    from tests.fake_docker import FakeDocker

    fd = FakeDocker(model="mini-instruct")
    monkeypatch.setattr(vm, "make_docker", lambda s: fd)
    return fd


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


