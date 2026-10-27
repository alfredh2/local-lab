"""Uncertainty and scaling statistics for concurrency sweeps.

Everything here is computed from recorded measurements; thresholds are explicit constants so the
UI can state exactly why a number was (not) reported.
"""
import math
from typing import Any

MIN_N_P95 = 20  # successful requests needed before p95 is reported
MIN_N_P99 = 100  # ... before p99 is reported
MIN_ITERATIONS_FOR_CI = 3
NOISY_CV = 0.10  # coefficient of variation of throughput across iterations above which a level is flagged
SATURATION_GAIN = 0.10  # a step adding less than 10% throughput is treated as "stopped scaling"

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

THRESHOLD_NOTE = (f"p95 is reported with >= {MIN_N_P95} successful requests and p99 with >= {MIN_N_P99}; confidence "
                  f"intervals need >= {MIN_ITERATIONS_FOR_CI} iterations; a level is 'noisy' if throughput varies by more "
                  f"than {NOISY_CV:.0%} (CV) across iterations.")


def t_crit95(df: int) -> float:
    return _T95.get(df, 1.96)


def mean_ci(values: list[float | None]) -> dict[str, Any]:
    v = [x for x in values if x is not None and math.isfinite(x)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95_half": None, "ci95_rel": None, "cv": None}
    mean = sum(v) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "ci95_half": None, "ci95_rel": None, "cv": None}
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))
    half = t_crit95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "ci95_half": half if n >= MIN_ITERATIONS_FOR_CI else None,
            "ci95_rel": half / mean if mean and n >= MIN_ITERATIONS_FOR_CI else None, "cv": sd / mean if mean else None}


def gate_dist(d: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop percentiles the sample size cannot support (they become None, with `reported` flags)."""
    if d is None:
        return None
    n = d["n"]
    out = {k: d[k] for k in ("n", "mean", "min", "p50", "max")}
    out["p95"] = d["p95"] if n >= MIN_N_P95 else None
    out["p99"] = d["p99"] if n >= MIN_N_P99 else None
    out["reported"] = {"p95": n >= MIN_N_P95, "p99": n >= MIN_N_P99}
    return out


def level_stats(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "aggregate_output_tok_s": mean_ci([i["aggregate_output_tok_s"] for i in iterations]),
        "ttft_p50_ms": mean_ci([i.get("ttft_p50_ms") for i in iterations]),
        "tpot_p50_ms": mean_ci([i.get("tpot_p50_ms") for i in iterations]),
        "e2e_p95_ms": mean_ci([i.get("e2e_p95_ms") for i in iterations]),
    }


def level_quality(level: dict[str, Any]) -> list[str]:
    issues = []
    st = level["stats"]["aggregate_output_tok_s"]
    if st["n"] < MIN_ITERATIONS_FOR_CI:
        issues.append(f"only {st['n']} iteration(s): no confidence interval")
    elif st["cv"] is not None and st["cv"] > NOISY_CV:
        issues.append(f"noisy: throughput CV {st['cv']:.0%} across iterations")
    n = level["latency"]["e2e_ms"]["n"] if level["latency"]["e2e_ms"] else 0
    if n < MIN_N_P95:
        issues.append(f"only {n} successful requests: p95/p99 not reported")
    return issues


def _interval(level: dict[str, Any]) -> tuple[float, float] | None:
    st = level["stats"]["aggregate_output_tok_s"]
    return None if st["ci95_half"] is None else (st["mean"] - st["ci95_half"], st["mean"] + st["ci95_half"])


def analyze(levels: list[dict[str, Any]], contended: bool) -> dict[str, Any]:
    """Scaling analysis over successful levels. Statements are observations about these runs only."""
    ok = [lv for lv in levels if lv["status"] == "ok"]
    thr = lambda lv: lv["stats"]["aggregate_output_tok_s"]["mean"]  # noqa: E731
    base = next((lv for lv in ok if lv["concurrency"] == 1), None)
    for lv in ok:
        lv["scaling_efficiency"] = thr(lv) / (lv["concurrency"] * thr(base)) if base and thr(base) else None
        lv["quality_issues"] = level_quality(lv)
    steps: list[dict[str, Any]] = []
    saturation = None
    for a, b in zip(ok, ok[1:]):
        gain = (thr(b) - thr(a)) / thr(a) if thr(a) else None
        ia, ib = _interval(a), _interval(b)
        significant = None if ia is None or ib is None else ib[0] > ia[1]  # CIs separated and b higher
        steps.append({"from": a["concurrency"], "to": b["concurrency"], "gain_pct": None if gain is None else gain * 100,
                      "significant": significant})
        if saturation is None and gain is not None and (gain < SATURATION_GAIN or significant is False):
            saturation = a["concurrency"]
    peak = max(ok, key=thr) if ok else None
    notes: list[str] = []
    if saturation is not None:
        step = next(s for s in steps if s["from"] == saturation)
        why = "gain under 10%" if step["gain_pct"] is not None and step["gain_pct"] < SATURATION_GAIN * 100 else "gain not distinguishable from noise"
        notes.append(f"Throughput stopped scaling meaningfully after concurrency {saturation} ({why}: "
                     f"{step['gain_pct']:+.1f}% going to {step['to']}).")
    elif len(ok) > 1:
        notes.append(f"No saturation observed up to concurrency {ok[-1]['concurrency']}; higher levels may still scale.")
    if peak is not None and len(ok) > 1:
        notes.append(f"Highest measured throughput: {thr(peak):.0f} tok/s at concurrency {peak['concurrency']}.")
    flagged = {lv["concurrency"]: lv["quality_issues"] for lv in ok if lv["quality_issues"]}
    if flagged:
        notes.append("Treat conclusions with caution: " + "; ".join(f"c={c}: {', '.join(i)}" for c, i in flagged.items()) + ".")
    if contended:
        notes.append("The GPU was contended during this sweep; relative scaling may be meaningful but absolute numbers are not "
                     "representative of an idle GPU.")
    return {"scaling_baseline_concurrency": base["concurrency"] if base else None, "steps": steps,
            "saturation_after_concurrency": saturation, "peak_throughput_concurrency": peak["concurrency"] if peak else None,
            "reliable": not flagged and not contended, "notes": notes, "thresholds": THRESHOLD_NOTE}
