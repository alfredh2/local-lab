"""Sandboxed execution of model-written code.

Production uses DockerSandbox (no network, read-only root, memory/CPU/PID limits, all capabilities
dropped, non-root, hard timeout). LocalSubprocessSandbox exists only to validate trusted reference
solutions in tests; never run model output with it.
"""
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SENTINEL = "@@LOCALLAB_RESULT@@"
MAX_OUTPUT = 64_000


@dataclass
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    oom_killed: bool = False
    error: str | None = None  # sandbox infrastructure failure (not a model failure)


class Sandbox(Protocol):
    def run_python(self, files: dict[str, str], entry: str, timeout_s: float) -> SandboxResult: ...


class LocalSubprocessSandbox:
    """TRUSTED CODE ONLY (reference solutions in tests)."""

    def run_python(self, files: dict[str, str], entry: str, timeout_s: float) -> SandboxResult:
        with tempfile.TemporaryDirectory() as d:
            for name, text in files.items():
                (Path(d) / name).write_text(text)
            try:
                p = subprocess.run([sys.executable, "-I", "-B", str(Path(d) / entry)], capture_output=True, text=True,
                                   timeout=timeout_s, cwd=d, env={"PATH": os.environ.get("PATH", "")})
            except subprocess.TimeoutExpired as e:
                return SandboxResult(None, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "", timed_out=True)
            return SandboxResult(p.returncode, p.stdout[:MAX_OUTPUT], p.stderr[:MAX_OUTPUT])


class DockerSandbox:
    def __init__(self, docker, instance_labels: dict[str, str], image: str) -> None:
        self.docker, self.labels, self.image = docker, instance_labels, image

    def run_python(self, files: dict[str, str], entry: str, timeout_s: float) -> SandboxResult:
        from locallab.runtimes.docker import SandboxSpec, new_sandbox_name

        with tempfile.TemporaryDirectory(prefix="locallab-sbx-") as d:
            os.chmod(d, 0o755)
            for name, text in files.items():
                path = Path(d) / name
                path.write_text(text)
                os.chmod(path, 0o644)
            return self.docker.run_sandbox(SandboxSpec(name=new_sandbox_name(), image=self.image, host_dir=d, entry=entry,
                                                       labels=self.labels, timeout_s=timeout_s))


def parse_harness_output(stdout: str) -> list[dict] | None:
    """The harness prints SENTINEL + JSON on its last line; anything else in stdout is the model's own prints."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(SENTINEL):
            try:
                return json.loads(line[len(SENTINEL):])["results"]
            except (ValueError, KeyError):
                return None
    return None
