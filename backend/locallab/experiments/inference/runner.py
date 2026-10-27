"""Closed-loop request scheduler: `concurrency` workers pull requests until the plan is exhausted."""
import asyncio
from dataclasses import dataclass

from locallab.experiments.inference.metrics import Record
from locallab.runtimes.base import GenerationRequest, RuntimeAdapter


@dataclass
class Planned:
    index: int
    phase: str
    iteration: int
    seed: int
    input_expected: int | None
    output_expected: int
    request: GenerationRequest


@dataclass
class Progress:
    """Live counters, read by the monitor coroutine for job progress."""

    phase: str = "starting"
    iteration: int = 0
    total: int = 0
    done: int = 0
    ok: int = 0
    failed: int = 0
    in_flight: int = 0
    max_in_flight: int = 0
    output_tokens: int = 0
    phase_started: float = 0.0
    consecutive_failures: int = 0
    last_error: str | None = None
    aborted: bool = False

    def start_phase(self, phase: str, iteration: int, total: int, now: float) -> None:
        self.phase, self.iteration, self.total, self.phase_started = phase, iteration, total, now
        self.done = self.ok = self.failed = self.in_flight = self.output_tokens = self.consecutive_failures = 0


async def run_phase(adapter: RuntimeAdapter, plans: list[Planned], concurrency: int, abort_after: int,
                    t0: float, progress: Progress, records: list[Record]) -> None:
    """Append a Record per finished request to `records` (so partial data survives cancellation)."""
    cursor = 0

    async def worker() -> None:
        nonlocal cursor
        while not progress.aborted and cursor < len(plans):
            plan = plans[cursor]
            cursor += 1
            progress.in_flight += 1
            progress.max_in_flight = max(progress.max_in_flight, progress.in_flight)
            try:
                outcome = await adapter.generate(plan.request)
            finally:
                progress.in_flight -= 1
            records.append(Record(plan.index, plan.phase, plan.iteration, plan.seed, plan.input_expected,
                                  plan.output_expected, outcome, t0))
            progress.done += 1
            if outcome.ok:
                progress.ok += 1
                progress.consecutive_failures = 0
                progress.output_tokens += outcome.completion_tokens or 0
            else:
                progress.failed += 1
                progress.consecutive_failures += 1
                progress.last_error = f"{outcome.error_type}: {outcome.error_message}"
                if progress.consecutive_failures >= abort_after:
                    progress.aborted = True

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(plans)))))
