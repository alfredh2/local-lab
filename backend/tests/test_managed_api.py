import pytest
from fastapi.testclient import TestClient

import locallab.db as db
from locallab.main import create_app
from locallab.models import ManagedServer


@pytest.fixture
def client(settings):
    with TestClient(create_app()) as c:
        yield c


def test_runtime_config_crud_and_validation(client):
    body = {"name": "qwen 3b fast", "config": {"max_model_len": 4096, "gpu_memory_utilization": 0.4, "allow_contention": True}}
    r = client.post("/api/runtime-configs", json=body)
    assert r.status_code == 201
    cfg = r.json()["config"]
    assert cfg["max_model_len"] == 4096 and cfg["gpu_memory_utilization"] == 0.4
    assert "allow_contention" not in cfg  # an override is a per-run decision, never saved
    assert client.post("/api/runtime-configs", json=body).status_code == 409
    assert client.post("/api/runtime-configs", json={"name": "bad", "config": {"quantization": "magic"}}).status_code == 422
    assert client.post("/api/runtime-configs", json={"name": "x", "config": {}, "registry_model_id": 999}).status_code == 422
    assert [c["name"] for c in client.get("/api/runtime-configs").json()] == ["qwen 3b fast"]
    assert client.delete(f"/api/runtime-configs/{r.json()['id']}").status_code == 204
    assert client.delete("/api/runtime-configs/999").status_code == 404


def test_image_must_be_a_vllm_image(client):
    r = client.post("/api/runtime-configs", json={"name": "evil", "config": {"image": "alpine:latest"}})
    assert r.status_code == 422


def test_managed_server_listing_hides_logs_in_list_only(client):
    with db.session_scope() as s:
        s.add(ManagedServer(container_name="locallab-managed-1-aaa", instance_id="i", image_ref="vllm/vllm-openai:latest",
                            status="stopped", config={}, log_tail="line1\nline2"))
    lst = client.get("/api/managed-servers").json()
    assert lst[0]["log_tail"] is None and lst[0]["container_name"] == "locallab-managed-1-aaa"
    assert client.get(f"/api/managed-servers/{lst[0]['id']}").json()["log_tail"] == "line1\nline2"
    assert client.get("/api/managed-servers/999").status_code == 404
    assert client.get("/api/managed-servers", params={"job_id": 12345}).json() == []


def test_gpu_preflight_endpoint_shape(client):
    r = client.get("/api/gpu/preflight", params={"utilization": 0.2}).json()
    assert {"free_mb", "total_mb", "other_processes", "contended", "errors", "warnings"} <= set(r)
