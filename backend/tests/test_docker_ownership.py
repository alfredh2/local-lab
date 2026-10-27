from pathlib import Path

import pytest

from locallab.runtimes.docker import (
    CUDA_COMPAT_FIX_LD_LIBRARY_PATH, LABEL_INSTANCE, LABEL_MANAGED, ContainerSpec, DockerCLI, DockerError, NotOwnedError,
    OwnedDocker, new_container_name,
)
from tests.fake_docker import FakeDocker

IID = "inst-1"


def owned_labels(iid=IID):
    return {LABEL_MANAGED: "true", LABEL_INSTANCE: iid}


def test_foreign_containers_are_never_stopped_or_removed():
    fd = FakeDocker()
    od = OwnedDocker(fd, IID)
    cases = {
        "no labels": fd.add_foreign("open-webui"),
        "other instance": fd.add_foreign("locallab-managed-1-abc", owned_labels("someone-else")),
        "label but wrong name": fd.add_foreign("my-own-vllm", owned_labels()),
        "managed=false": fd.add_foreign("locallab-managed-2-abc", {LABEL_MANAGED: "false", LABEL_INSTANCE: IID}),
    }
    for label, c in cases.items():
        with pytest.raises(NotOwnedError):
            od.stop_and_remove(c.id)
        assert c.id in fd.containers and c.status == "running", label
    assert [call for call in fd.calls if call[0] in ("stop", "rm")] == []
    assert od.list_owned() == []


def test_owned_container_removal_is_idempotent():
    fd = FakeDocker()
    od = OwnedDocker(fd, IID)
    c = fd.add_foreign("locallab-managed-7-abc", owned_labels())
    assert [x.id for x in od.list_owned()] == [c.id]
    assert od.stop_and_remove(c.id) == "removed"
    assert od.stop_and_remove(c.id) == "already_gone"
    assert ("stop", c.id) in fd.calls and ("rm", c.id) in fd.calls


def test_cli_builds_fixed_argv_without_shell(monkeypatch):
    seen = {}

    def fake_run(self, argv, timeout=60, check=True):
        seen["argv"] = argv

        class P:
            stdout, stderr, returncode = "cid123\n", "", 0
        return P()

    monkeypatch.setattr(DockerCLI, "_run", fake_run)
    spec = ContainerSpec(name="locallab-managed-3-aa", image="vllm/vllm-openai:latest",
                         labels={LABEL_MANAGED: "true"}, env={"HF_HUB_OFFLINE": "1"}, mounts=[("/h", "/c", True)],
                         host_port=5555, container_port=8000, command=["org/m", "--revision", "r"], gpu_index=1)
    assert DockerCLI().create_and_start(spec) == "cid123"
    a = seen["argv"]
    assert a[:3] == ["run", "-d", "--name"] and "127.0.0.1:5555:8000" in a and "device=1" in a
    assert "/h:/c:ro" in a and "no-new-privileges" in a and a[-3:] == ["org/m", "--revision", "r"]
    assert "--privileged" not in a and not any(x.startswith("--network=host") for x in a)


def test_cli_rejects_unsafe_tokens():
    bad = ContainerSpec(name="x; rm -rf /", image="vllm/vllm-openai", labels={}, env={}, mounts=[], host_port=1,
                        container_port=2, command=[])
    with pytest.raises(DockerError, match="unsafe"):
        DockerCLI().create_and_start(bad)


def test_container_names_unique_and_prefixed():
    names = {new_container_name(5) for _ in range(50)}
    assert len(names) == 50 and all(n.startswith("locallab-managed-5-") for n in names)


def test_cuda_workaround_stays_in_sync_with_launch_script():
    script = (Path(__file__).resolve().parents[2] / "scripts" / "run_vllm_server.sh").read_text()
    assert CUDA_COMPAT_FIX_LD_LIBRARY_PATH in script
