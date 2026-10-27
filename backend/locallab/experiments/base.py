"""Experiment interface shared by all experiment types.

Kept deliberately small: a type declares a validated config model and a `run`
that returns a results dict. The worker owns lifecycle/status; `run` never
sets a terminal status itself.
"""
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel


class Cancelled(Exception):
    """Raised by the context when the job should stop (user cancel)."""


class ExperimentContext(Protocol):
    job_id: int
    seed: int | None

    def log(self, level: str, message: str) -> None: ...

    def should_cancel(self) -> bool:
        """True if cancellation was requested. Also refreshes the job heartbeat."""

    def check_cancel(self) -> None:
        """Raise Cancelled if cancellation was requested."""

    def write_artifact(self, name: str, kind: str, data: Any) -> None:
        """Persist JSON-serialisable `data` as a job artifact."""


class Experiment(ABC):
    type_name: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    config_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def run(self, config: BaseModel, ctx: ExperimentContext) -> dict[str, Any]:
        """Execute and return the results dict. Raise on failure; never fake success."""
