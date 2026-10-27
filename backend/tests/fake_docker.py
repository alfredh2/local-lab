"""In-memory Docker stand-in. Containers 'run' a MockServer on the requested host port."""
import itertools
import threading
import time
from dataclasses import dataclass, field

from locallab.evals.sandbox import LocalSubprocessSandbox, SandboxResult
from locallab.runtimes.docker import ContainerInfo, ContainerSpec, DockerError, ImageInfo, SandboxSpec
from tests.mock_vllm import MockServer

_ids = itertools.count(1)


@dataclass
class FakeContainer:
    id: str
    name: str
    labels: dict[str, str]
    status: str = "running"
    exit_code: int | None = None
    oom: bool = False
    logs: str = ""
    spec: ContainerSpec | None = None
    server: MockServer | None = None


class FakeDocker:
    def __init__(self, **server_kw) -> None:
        self.containers: dict[str, FakeContainer] = {}
        self.server_kw = server_kw
        self.ready_delay_s = 0.0  # server answers only after this long
        self.fail_run: str | None = None  # raise DockerError on create
        self.start_mode = "ok"  # ok | never_ready | exit_at_start
        self.exit_logs = "RuntimeError: CUDA out of memory"
        self.calls: list[tuple[str, str]] = []
        self.arch_list = ["Qwen2ForCausalLM"]
        self.image = ImageInfo(id="sha256:" + "ab" * 32, repo_digests=["vllm/vllm-openai@sha256:" + "cd" * 32])
        self.image_present = True
        self.oneoff_calls = 0
        self.sandbox_error: str | None = None
        self._lock = threading.Lock()

    # --- helpers for tests ---
    def add_foreign(self, name: str, labels: dict[str, str] | None = None, status: str = "running") -> FakeContainer:
        c = FakeContainer(id=f"foreign{next(_ids)}", name=name, labels=labels or {}, status=status)
        self.containers[c.id] = c
        return c

    def crash(self, cid: str, oom: bool = False, logs: str = "") -> None:
        c = self.containers[cid]
        if c.server:
            c.server.__exit__()
            c.server = None
        c.status, c.exit_code, c.oom, c.logs = "exited", 137 if oom else 1, oom, logs or c.logs

    # --- DockerAPI ---
    def create_and_start(self, spec: ContainerSpec) -> str:
        self.calls.append(("run", spec.name))
        if self.fail_run:
            raise DockerError(self.fail_run)
        cid = f"fake{next(_ids):04d}"
        c = FakeContainer(id=cid, name=spec.name, labels=dict(spec.labels), spec=spec, logs="INFO vLLM starting\n")
        if self.start_mode == "exit_at_start":
            c.status, c.exit_code, c.logs = "exited", 1, self.exit_logs
        elif self.start_mode == "ok":
            def boot() -> None:
                time.sleep(self.ready_delay_s)
                if c.status == "running":
                    srv = MockServer(port=spec.host_port, **self.server_kw)
                    srv.__enter__()
                    c.server = srv
                    c.logs += "INFO Application startup complete.\n"
            threading.Thread(target=boot, daemon=True).start()
        self.containers[cid] = c
        return cid

    def inspect(self, container: str) -> ContainerInfo | None:
        c = self.containers.get(container) or next((x for x in self.containers.values() if x.name == container), None)
        if c is None:
            return None
        return ContainerInfo(id=c.id, name=c.name, status=c.status, exit_code=c.exit_code if c.status != "running" else None,
                             oom_killed=c.oom, labels=c.labels, image_id=self.image.id)

    def logs(self, container: str, tail: int | None = None) -> str:
        c = self.containers.get(container)
        return c.logs if c else ""

    def top_pids(self, container: str) -> set[int]:
        return {424242} if container in self.containers and self.containers[container].status == "running" else set()

    def stop(self, container: str, timeout_s: int = 30) -> None:
        self.calls.append(("stop", container))
        c = self.containers[container]
        if c.server:
            c.server.__exit__()
            c.server = None
        c.status, c.exit_code = "exited", c.exit_code if c.exit_code is not None else 0

    def remove(self, container: str, force: bool = True) -> None:
        self.calls.append(("rm", container))
        c = self.containers.pop(container, None)
        if c and c.server:
            c.server.__exit__()

    def list_by_label(self, label: str) -> list[ContainerInfo]:
        k, _, v = label.partition("=")
        return [self.inspect(c.id) for c in list(self.containers.values()) if c.labels.get(k) == v]  # type: ignore[misc]

    def image_inspect(self, ref: str) -> ImageInfo | None:
        return self.image if self.image_present else None

    def run_oneoff(self, image, entrypoint, args, env, timeout_s) -> str:
        import json
        self.oneoff_calls += 1
        return "noise\n" + json.dumps(self.arch_list) + "\n"

    def run_sandbox(self, spec: SandboxSpec) -> SandboxResult:
        """Executes the harness on the host (tests only; trusted reference/mock code, never real model output)."""
        from pathlib import Path

        self.calls.append(("sandbox", spec.name))
        if self.sandbox_error:
            return SandboxResult(None, "", "", error=self.sandbox_error)
        files = {p.name: p.read_text() for p in Path(spec.host_dir).iterdir()}
        return LocalSubprocessSandbox().run_python(files, spec.entry, spec.timeout_s)

    def alive(self) -> list[FakeContainer]:
        return [c for c in self.containers.values() if c.status == "running"]
