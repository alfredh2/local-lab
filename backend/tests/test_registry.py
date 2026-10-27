import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from locallab.main import create_app
from locallab.services import registry


def make_repo(root: Path, name: str, cfg: dict, weights: bool = True) -> Path:
    repo = root / f"models--{name.replace('/', '--')}"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123")
    (snap / "config.json").write_text(json.dumps(cfg))
    if weights:
        blob = repo / "blobs"
        blob.mkdir()
        (blob / "deadbeef").write_bytes(b"x" * 1000)
        (snap / "model.safetensors").symlink_to(blob / "deadbeef")
    return snap


def test_scan_hf_cache(tmp_path):
    make_repo(tmp_path, "Qwen/Q-AWQ", {"architectures": ["QwenForCausalLM"], "quantization_config": {"quant_method": "awq", "bits": 4}})
    make_repo(tmp_path, "org/config-only", {"model_type": "x"}, weights=False)
    found = {d.name: d for d in registry.scan_hf_cache(tmp_path)}
    q = found["Qwen/Q-AWQ"]
    assert q.available and q.size_bytes == 1000 and q.quantization == "awq-int4" and q.revision == "abc123"
    assert q.format == "safetensors"
    assert not found["org/config-only"].available  # incomplete download is not "installed"


def test_scan_missing_dir(tmp_path):
    assert registry.scan_hf_cache(tmp_path / "nope") == []


def test_ollama_unreachable_reports_warning():
    d, warn = registry.scan_ollama("http://127.0.0.1:1", timeout=0.5)
    assert d == [] and "not reachable" in warn


@pytest.fixture
def client(settings):
    with TestClient(create_app()) as c:
        yield c


def test_api_scan_and_lifecycle(client, settings):
    make_repo(Path(settings.hf_cache_dir), "a/b", {})
    r = client.post("/api/models/scan").json()
    assert r["added"] == 1 and r["warnings"]  # ollama unreachable in tests
    models = client.get("/api/models").json()
    assert [m["name"] for m in models] == ["a/b"] and models[0]["available"]
    # rescanning is idempotent
    assert client.post("/api/models/scan").json()["added"] == 0
    # removing from disk marks unavailable but keeps history
    import shutil
    shutil.rmtree(Path(settings.hf_cache_dir) / "models--a--b")
    assert client.post("/api/models/scan").json()["marked_unavailable"] == 1
    assert client.get("/api/models").json()[0]["available"] is False


def test_manual_add_validation(client, tmp_path):
    assert client.post("/api/models", json={"name": "x", "path": str(tmp_path / "missing")}).status_code == 422
    (tmp_path / "m").mkdir()
    r = client.post("/api/models", json={"name": "x", "path": str(tmp_path / "m")})
    assert r.status_code == 201 and r.json()["available"] is False
    assert client.delete(f"/api/models/{r.json()['id']}").status_code == 204
    assert client.delete("/api/models/999").status_code == 404
