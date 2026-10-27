"""Docker access for LocalLab-managed containers.

Only the `docker` CLI is used, always with a fixed argv list (never a shell). Mutating calls on an
existing container go through OwnedDocker, which re-inspects the container and refuses unless it
carries this install's ownership labels, so unrelated containers can never be stopped or removed.
"""
import json
import logging
import re
import secrets
import subprocess
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)

# Same workaround as scripts/run_vllm_server.sh: put the host driver's libcuda ahead of the image's
# bundled CUDA 12.9 compat library, otherwise torch fails with "Error 803" on driver 595 / RTX 5090.
CUDA_COMPAT_FIX_LD_LIBRARY_PATH = "/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"

LABEL_MANAGED = "locallab.managed"
LABEL_INSTANCE = "locallab.instance"
NAME_PREFIX = "locallab-managed-"
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@=+-]*$")


class DockerError(Exception):
    """A docker command failed (message includes stderr)."""


class NotOwnedError(DockerError):
    """Refused to act on a container that LocalLab does not own."""


@dataclass
class ContainerInfo:
    id: str
    name: str
    status: str  # created|running|exited|dead|...
    exit_code: int | None = None
    oom_killed: bool = False
    labels: dict[str, str] = field(default_factory=dict)
    image_id: str | None = None
    error: str = ""

    @property
    def running(self) -> bool:
        return self.status == "running"


@dataclass
class ContainerSpec:
    name: str
    image: str
    labels: dict[str, str]
    env: dict[str, str]
    mounts: list[tuple[str, str, bool]]  # (host, container, read_only)
    host_port: int
    container_port: int
    command: list[str]
    gpu_index: int = 0
    shm_size: str = "8g"


@dataclass
class ImageInfo:
    id: str
    repo_digests: list[str]


def new_container_name(job_id: int) -> str:
    return f"{NAME_PREFIX}{job_id}-{secrets.token_hex(3)}"


def _check_token(value: str, what: str) -> str:
    if not _SAFE.match(value):
        raise DockerError(f"unsafe {what}: {value!r}")
    return value


class DockerAPI(Protocol):
    def create_and_start(self, spec: ContainerSpec) -> str: ...
    def inspect(self, container: str) -> ContainerInfo | None: ...
    def logs(self, container: str, tail: int | None = None) -> str: ...
    def top_pids(self, container: str) -> set[int]: ...
    def stop(self, container: str, timeout_s: int = 30) -> None: ...
    def remove(self, container: str, force: bool = True) -> None: ...
    def list_by_label(self, label: str) -> list[ContainerInfo]: ...
    def image_inspect(self, ref: str) -> ImageInfo | None: ...
    def run_oneoff(self, image: str, entrypoint: str, args: list[str], env: dict[str, str], timeout_s: int) -> str: ...


class DockerCLI:
    def __init__(self, binary: str = "docker") -> None:
        self.binary = binary

    def _run(self, argv: list[str], timeout: float = 60, check: bool = True) -> subprocess.CompletedProcess:
        try:
            p = subprocess.run([self.binary, *argv], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError as e:
            raise DockerError(f"docker binary not found: {self.binary}") from e
        except subprocess.TimeoutExpired as e:
            raise DockerError(f"docker {argv[0]} timed out after {timeout}s") from e
        if check and p.returncode != 0:
            raise DockerError(f"docker {argv[0]} failed ({p.returncode}): {(p.stderr or p.stdout).strip()[:1000]}")
        return p

    def create_and_start(self, spec: ContainerSpec) -> str:
        _check_token(spec.name, "container name")
        _check_token(spec.image, "image")
        argv = ["run", "-d", "--name", spec.name, "--runtime", "nvidia", "--gpus", f"device={int(spec.gpu_index)}",
                "--shm-size", spec.shm_size, "--security-opt", "no-new-privileges", "--pids-limit", "4096",
                "-p", f"127.0.0.1:{int(spec.host_port)}:{int(spec.container_port)}"]
        for k, v in spec.labels.items():
            argv += ["--label", f"{_check_token(k, 'label key')}={v}"]
        for k, v in spec.env.items():
            argv += ["-e", f"{_check_token(k, 'env key')}={v}"]
        for host, ctr, ro in spec.mounts:
            argv += ["-v", f"{host}:{ctr}{':ro' if ro else ''}"]
        argv += [spec.image, *spec.command]
        return self._run(argv, timeout=120).stdout.strip()

    def inspect(self, container: str) -> ContainerInfo | None:
        p = self._run(["inspect", container], check=False)
        if p.returncode != 0:
            if "No such" in p.stderr or "no such" in p.stderr:
                return None
            raise DockerError(f"docker inspect failed: {p.stderr.strip()[:500]}")
        d = json.loads(p.stdout)[0]
        st = d.get("State", {})
        return ContainerInfo(id=d["Id"], name=d["Name"].lstrip("/"), status=st.get("Status", "unknown"),
                             exit_code=st.get("ExitCode") if st.get("Status") != "running" else None,
                             oom_killed=bool(st.get("OOMKilled")), labels=d.get("Config", {}).get("Labels") or {},
                             image_id=d.get("Image"), error=st.get("Error", ""))

    def logs(self, container: str, tail: int | None = None) -> str:
        argv = ["logs"] + (["--tail", str(tail)] if tail else []) + [container]
        p = self._run(argv, timeout=60, check=False)
        return (p.stdout or "") + (p.stderr or "")  # docker logs writes the container's stderr to stderr

    def top_pids(self, container: str) -> set[int]:
        p = self._run(["top", container, "-eo", "pid"], check=False)
        return {int(x) for x in p.stdout.split()[1:] if x.isdigit()} if p.returncode == 0 else set()

    def stop(self, container: str, timeout_s: int = 30) -> None:
        self._run(["stop", "-t", str(int(timeout_s)), container], timeout=timeout_s + 30)

    def remove(self, container: str, force: bool = True) -> None:
        self._run(["rm"] + (["-f"] if force else []) + [container], timeout=60)

    def list_by_label(self, label: str) -> list[ContainerInfo]:
        ids = self._run(["ps", "-a", "--filter", f"label={label}", "--format", "{{.ID}}"]).stdout.split()
        infos = [self.inspect(i) for i in ids]
        return [i for i in infos if i is not None]

    def image_inspect(self, ref: str) -> ImageInfo | None:
        p = self._run(["image", "inspect", ref], check=False)
        if p.returncode != 0:
            return None
        d = json.loads(p.stdout)[0]
        return ImageInfo(id=d["Id"], repo_digests=d.get("RepoDigests") or [])

    def run_oneoff(self, image: str, entrypoint: str, args: list[str], env: dict[str, str], timeout_s: int) -> str:
        argv = ["run", "--rm", "--entrypoint", entrypoint]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        return self._run(argv + [image, *args], timeout=timeout_s).stdout


class OwnedDocker:
    """Ownership-checked operations. `instance_id` is unique per LocalLab install (data dir)."""

    def __init__(self, docker: DockerAPI, instance_id: str) -> None:
        self.docker = docker
        self.instance_id = instance_id

    def is_owned(self, info: ContainerInfo) -> bool:
        return (info.labels.get(LABEL_MANAGED) == "true" and info.labels.get(LABEL_INSTANCE) == self.instance_id
                and info.name.startswith(NAME_PREFIX))

    def labels_for(self, job_id: int, experiment_id: int) -> dict[str, str]:
        return {LABEL_MANAGED: "true", LABEL_INSTANCE: self.instance_id, "locallab.job_id": str(job_id),
                "locallab.experiment_id": str(experiment_id)}

    def list_owned(self) -> list[ContainerInfo]:
        return [c for c in self.docker.list_by_label(f"{LABEL_MANAGED}=true") if self.is_owned(c)]

    def stop_and_remove(self, container: str, stop_timeout_s: int = 30) -> str:
        """Idempotent. Returns 'removed' | 'already_gone'. Raises NotOwnedError for foreign containers."""
        info = self.docker.inspect(container)
        if info is None:
            return "already_gone"
        if not self.is_owned(info):
            raise NotOwnedError(f"refusing to touch container {info.name!r}: not owned by this LocalLab instance")
        if info.running:
            try:
                self.docker.stop(container, stop_timeout_s)
            except DockerError:
                log.warning("graceful stop of %s failed; forcing removal", info.name)
        try:
            self.docker.remove(container, force=True)
        except DockerError:
            if self.docker.inspect(container) is not None:  # gone concurrently is fine
                raise
        return "removed"
