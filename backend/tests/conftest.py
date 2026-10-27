from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import Session

import locallab.db as db
from locallab.config import Settings, get_settings

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
