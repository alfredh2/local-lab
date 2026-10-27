import pytest
from fastapi.testclient import TestClient

from locallab.main import create_app
from locallab.services import gpu as gpu_service


@pytest.fixture
def client(settings):
    with TestClient(create_app()) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_system_status_shape(client):
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    assert body["cpu_threads"] > 0 and body["worker"]["alive"] is False
    assert "fastapi" in body["packages"]


def test_gpu_unavailable_is_reported_not_faked(client, monkeypatch):
    import pynvml

    def boom():
        raise pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED)

    monkeypatch.setattr(gpu_service, "_ensure_init", boom)
    body = client.get("/api/gpu").json()
    assert body["available"] is False and body["gpus"] == [] and body["error"]


def test_gpu_real_reading_if_present(client):
    body = client.get("/api/gpu").json()
    if not body["available"]:
        pytest.skip("no NVML on this machine")
    g = body["gpus"][0]
    assert g["memory_total_mb"] > 0 and 0 <= g["memory_used_mb"] <= g["memory_total_mb"]


def test_gpu_history_ring_buffer():
    h = gpu_service.GpuHistory(maxlen=2)
    snap = gpu_service.read_gpus()
    for _ in range(3):
        h.record(snap)
    assert len(h.points()) <= 2
