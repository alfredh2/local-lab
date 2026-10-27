"""Evaluation framework: stable cases, deterministic scoring, no LLM judge.

A suite produces `EvalCase`s for a split ("dev" for iterating on prompts, "heldout" for comparing
configurations) and scores a model response with plain code (JSON schema validation, hidden unit
tests run in a sandbox, exact-match retrieval). `meta` carries scorer inputs and is never shown to
the model.
"""
import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

SPLITS = ("dev", "heldout")


@dataclass
class EvalCase:
    id: str  # stable, e.g. "json-v1-heldout-007"
    suite: str
    version: str
    split: str
    messages: list[dict[str, str]]
    max_tokens: int
    meta: dict[str, Any] = field(default_factory=dict)  # scorer inputs; never sent to the model
    tags: dict[str, Any] = field(default_factory=dict)  # breakdown keys (length, position, template, ...)


@dataclass
class CaseScore:
    passed: bool
    components: dict[str, float | bool] = field(default_factory=dict)
    failure_type: str | None = None
    detail: str | None = None


class EvalSuite(ABC):
    name: str
    version: str
    title: str
    primary_metric: str  # key in aggregate() holding the headline rate

    @abstractmethod
    def cases(self, split: str, **params: Any) -> list[EvalCase]: ...

    @abstractmethod
    def score(self, case: EvalCase, response: str, sandbox: Any = None) -> CaseScore: ...

    def aggregate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Suite-specific breakdowns over scored records (each has case tags, `passed`, `components`)."""
        return {}


def fingerprint(cases: list[EvalCase]) -> str:
    """Hash of everything that defines the task set, so results name the exact dataset they used."""
    blob = json.dumps([asdict(c) for c in cases], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def wilson(passed: int, n: int, z: float = 1.96) -> dict[str, Any]:
    """Wilson score interval for a proportion; honest about small n (wide intervals)."""
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "lo": None, "hi": None}
    p = passed / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"n": n, "k": passed, "rate": p, "lo": max(0.0, centre - half), "hi": min(1.0, centre + half)}


def split_for(index: int, heldout_every: int = 3) -> str:
    """Fixed, documented assignment: every `heldout_every`-th case is dev, the rest are held out."""
    return "dev" if index % heldout_every == 0 else "heldout"
