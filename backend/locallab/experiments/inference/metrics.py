"""Metric definitions and statistics. All times are seconds on time.perf_counter()."""
import math
from dataclasses import dataclass
from typing import Any

from locallab.runtimes.base import RequestOutcome

DEFINITIONS = {
    "ttft_ms": "Time to first token: request sent -> first streamed chunk that carries a generated token. Includes queueing and prefill.",
    "tpot_ms": "Time per output token: (end-to-end latency - TTFT) / (output_tokens - 1). Undefined for 1-token outputs.",
    "itl_ms": "Inter-chunk latency: gap between consecutive streamed chunks (one token each in vLLM unless the server batches).",
    "e2e_ms": "End-to-end latency: request sent -> final stream chunk received.",
    "output_tok_s_per_request": "output_tokens / end-to-end latency for one request (includes TTFT; not pure decode speed).",
    "decode_tok_s_per_request": "1000 / TPOT: steady-state decode speed seen by one request.",
    "aggregate_output_tok_s": "Sum of output tokens of successful requests / wall time of the measured phase (first send -> last completion).",
    "total_tok_s": "(input + output tokens of successful requests) / wall time. Input tokens count prefill work.",
    "requests_per_s": "Successful requests / wall time.",
    "success_rate": "Successful requests / attempted measured requests. Failed requests are excluded from latency/token statistics.",
    "gpu_peak_memory_mb": "Peak device-wide GPU memory in use (NVML) during the measured phase; includes other processes and vLLM's preallocated KV cache.",
    "token_counts": "Taken from the server's usage object ('server'). If absent, output tokens are the number of streamed chunks ('estimated').",
    "percentiles": "Linear interpolation between closest ranks over successful requests.",
}


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    if not s:
        raise ValueError("empty")
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(values: list[float]) -> dict[str, float | int] | None:
    v = [x for x in values if x is not None and math.isfinite(x)]
    if not v:
        return None
    return {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "p50": percentile(v, 50),
            "p95": percentile(v, 95), "p99": percentile(v, 99), "max": max(v)}


@dataclass
class Record:
    """One request's measurements (what goes into requests.json)."""

    index: int
    phase: str
    iteration: int
    seed: int
    input_tokens_expected: int | None  # exact only in token_ids mode
    output_tokens_expected: int
    outcome: RequestOutcome
    t0: float  # perf_counter origin of the run

    def to_dict(self) -> dict[str, Any]:
        o = self.outcome
        e2e = (o.t_end - o.t_start) * 1000
        ttft = (o.t_first_token - o.t_start) * 1000 if o.t_first_token else None
        out_tok = o.completion_tokens
        tpot = (e2e - ttft) / (out_tok - 1) if o.ok and ttft is not None and out_tok and out_tok > 1 else None
        itl = [(b - a) * 1000 for a, b in zip(o.chunk_times, o.chunk_times[1:])]
        return {
            "index": self.index, "phase": self.phase, "iteration": self.iteration, "seed": self.seed,
            "ok": o.ok, "error_type": o.error_type, "error_message": o.error_message, "http_status": o.http_status,
            "start_s": round(o.t_start - self.t0, 4), "end_s": round(o.t_end - self.t0, 4),
            "input_tokens_expected": self.input_tokens_expected, "input_tokens": o.prompt_tokens,
            "output_tokens_expected": self.output_tokens_expected, "output_tokens": out_tok,
            "token_source": o.token_source, "finish_reason": o.finish_reason,
            "e2e_ms": e2e, "ttft_ms": ttft, "tpot_ms": tpot,
            "output_tok_s": out_tok / (e2e / 1000) if o.ok and out_tok and e2e > 0 else None,
            "decode_tok_s": 1000 / tpot if tpot else None,
            "itl_mean_ms": sum(itl) / len(itl) if itl else None,
            "itl_p95_ms": percentile(itl, 95) if itl else None,
            "itl_max_ms": max(itl) if itl else None,
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one set of measured request rows (a single iteration, or pooled iterations)."""
    ok = [r for r in rows if r["ok"]]
    errors: dict[str, int] = {}
    for r in rows:
        if not r["ok"]:
            errors[r["error_type"] or "unknown"] = errors.get(r["error_type"] or "unknown", 0) + 1
    wall = (max(r["end_s"] for r in rows) - min(r["start_s"] for r in rows)) if rows else 0.0
    out_tok = sum(r["output_tokens"] or 0 for r in ok)
    in_tok = sum(r["input_tokens"] or r["input_tokens_expected"] or 0 for r in ok)
    sources = {r["token_source"] for r in ok}
    return {
        "requests": {"attempted": len(rows), "succeeded": len(ok), "failed": len(rows) - len(ok),
                     "success_rate": len(ok) / len(rows) if rows else 0.0, "errors_by_type": errors},
        "wall_s": wall,
        "token_count_source": "estimated" if "estimated" in sources else "server",
        "tokens": {"input_total": in_tok, "output_total": out_tok},
        "latency": {
            "e2e_ms": dist([r["e2e_ms"] for r in ok]), "ttft_ms": dist([r["ttft_ms"] for r in ok]),
            "tpot_ms": dist([r["tpot_ms"] for r in ok]), "itl_mean_ms": dist([r["itl_mean_ms"] for r in ok]),
        },
        "throughput": {
            "aggregate_output_tok_s": out_tok / wall if wall > 0 else None,
            "total_tok_s": (in_tok + out_tok) / wall if wall > 0 else None,
            "requests_per_s": len(ok) / wall if wall > 0 else None,
            "per_request_output_tok_s": dist([r["output_tok_s"] for r in ok]),
            "per_request_decode_tok_s": dist([r["decode_tok_s"] for r in ok]),
        },
    }


def verify_lengths(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare server-reported token counts with what was requested."""
    ok = [r for r in rows if r["ok"]]
    in_bad = [r for r in ok if r["input_tokens_expected"] is not None and r["input_tokens"] is not None
              and r["input_tokens"] != r["input_tokens_expected"]]
    out_bad = [r for r in ok if r["output_tokens"] is not None and r["output_tokens"] != r["output_tokens_expected"]]
    in_known = [r["input_tokens"] for r in ok if r["input_tokens"] is not None]
    return {
        "input_tokens_mean": sum(in_known) / len(in_known) if in_known else None,
        "input_length_exact": bool(ok) and not in_bad and any(r["input_tokens_expected"] is not None for r in ok),
        "input_mismatches": len(in_bad),
        "output_length_exact": bool(ok) and not out_bad,
        "output_mismatches": len(out_bad),
    }
