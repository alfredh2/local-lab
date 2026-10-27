"""Run an eval suite against a server: generate every case, score it deterministically, and aggregate.

Quality and the evaluation workload's own performance are reported side by side but kept apart from the
synthetic benchmark numbers (these are real prompts with real output lengths).
"""
import asyncio
import hashlib
import time
from typing import Any, Callable

from locallab.evals.base import CaseScore, EvalCase, EvalSuite, fingerprint, wilson
from locallab.evals.datasets.retrieval import FILLER_SAMPLE_CHARS, filler_text
from locallab.experiments.inference.metrics import dist
from locallab.experiments.inference.sweep_stats import gate_dist
from locallab.runtimes.base import GenerationRequest, RuntimeAdapter

RESPONSE_KEEP = 2000


async def calibrate_chars_per_token(adapter: Any, model: str) -> tuple[float, bool]:
    """Characters per token of the retrieval filler text under the server's tokenizer. (ratio, measured?)"""
    sample = filler_text("calibration", FILLER_SAMPLE_CHARS)
    n = await adapter.count_tokens(model, sample) if hasattr(adapter, "count_tokens") else None
    return (len(sample) / n, True) if n else (4.0, False)


def _seed(case_id: str) -> int:
    return int(hashlib.sha256(case_id.encode()).hexdigest()[:8], 16)


async def run_suite(adapter: RuntimeAdapter, model: str, suite: EvalSuite, cases: list[EvalCase], sandbox: Any, *,
                    concurrency: int = 8, timeout_s: float = 180.0, enable_thinking: bool = False, split: str = "heldout",
                    on_progress: Callable[[int, int, int], None] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Returns (suite result for storage, per-case records for the artifact)."""
    records: list[dict[str, Any] | None] = [None] * len(cases)
    cursor = 0
    done = passed = 0
    t_start = time.perf_counter()

    async def worker() -> None:
        nonlocal cursor, done, passed
        while cursor < len(cases):
            i = cursor
            cursor += 1
            case = cases[i]
            out = await adapter.generate(GenerationRequest(
                model=model, prompt="", messages=case.messages, max_tokens=case.max_tokens, temperature=0.0, seed=_seed(case.id),
                force_length=False, timeout_s=timeout_s, chat_template_kwargs={"enable_thinking": enable_thinking}))
            if out.ok:
                try:
                    score = await asyncio.to_thread(suite.score, case, out.text, sandbox)
                except Exception as e:  # a scorer/sandbox fault must not look like a model failure
                    score = CaseScore(False, {}, "scoring_error", f"{type(e).__name__}: {e}"[:300])
            else:
                score = CaseScore(False, {}, f"generation_{out.error_type}", out.error_message)
            e2e = (out.t_end - out.t_start) * 1000
            ttft = (out.t_first_token - out.t_start) * 1000 if out.t_first_token else None
            ntok = out.completion_tokens
            records[i] = {
                "case_id": case.id, "split": case.split, "tags": case.tags, "passed": score.passed, "components": score.components,
                "failure_type": score.failure_type, "detail": score.detail, "generation_ok": out.ok, "error_type": out.error_type,
                "prompt_tokens": out.prompt_tokens, "completion_tokens": ntok, "finish_reason": out.finish_reason,
                "truncated": out.finish_reason == "length", "e2e_ms": e2e, "ttft_ms": ttft,
                "tpot_ms": (e2e - ttft) / (ntok - 1) if out.ok and ttft is not None and ntok and ntok > 1 else None,
                "response": out.text[:RESPONSE_KEEP]}
            done += 1
            passed += 1 if score.passed else 0
            if on_progress:
                on_progress(done, len(cases), passed)

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(cases)))))
    wall = time.perf_counter() - t_start
    recs = [r for r in records if r is not None]
    ok = [r for r in recs if r["generation_ok"]]
    errors: dict[str, int] = {}
    for r in recs:
        if not r["generation_ok"]:
            errors[r["error_type"] or "unknown"] = errors.get(r["error_type"] or "unknown", 0) + 1
    k = sum(1 for r in recs if r["passed"])
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    result = {
        "suite": suite.name, "version": suite.version, "split": split, "fingerprint": fingerprint(cases),
        "primary_metric": suite.primary_metric, "n": len(recs), "passed": k, "interval": wilson(k, len(recs)),
        "aggregate": suite.aggregate(recs), "truncated": sum(1 for r in recs if r["truncated"]),
        "performance": {
            "workload_kind": "evaluation_prompts (real prompts and output lengths; not comparable to synthetic cells)",
            "requests": {"attempted": len(recs), "succeeded": len(ok), "failed": len(recs) - len(ok), "errors_by_type": errors},
            "latency": {"e2e_ms": gate_dist(dist([r["e2e_ms"] for r in ok])), "ttft_ms": gate_dist(dist([r["ttft_ms"] for r in ok])),
                        "tpot_ms": gate_dist(dist([r["tpot_ms"] for r in ok if r["tpot_ms"] is not None]))},
            "tokens": {"prompt_total": sum(r["prompt_tokens"] or 0 for r in ok), "completion_total": out_tok},
            "output_tok_s": out_tok / wall if wall > 0 else None, "concurrency": min(concurrency, len(cases)), "wall_s": wall},
    }
    return result, recs
