"""Comparable metrics extracted from stored configuration results.

Quality metrics are proportions with Wilson intervals; performance metrics come from the *synthetic*
benchmark cell selected by (io index, concurrency) with CIs across iterations; `eval_*` metrics are
measured on the real evaluation prompts and are kept separate on purpose.
"""
from typing import Any

from locallab.analysis.pareto import MAX, MIN, Metric, Point
from locallab.evals.base import wilson

METRICS: dict[str, Metric] = {m.id: m for m in [
    Metric("structured_schema_rate", "Structured output: schema compliance", "%", MAX, "quality"),
    Metric("structured_extraction_acc", "Structured output: field accuracy", "%", MAX, "quality"),
    Metric("coding_pass_rate", "Coding: pass rate (hidden tests)", "%", MAX, "quality"),
    Metric("retrieval_accuracy", "Long-context retrieval accuracy", "%", MAX, "quality"),
    Metric("throughput_tok_s", "Aggregate output throughput (synthetic)", "tok/s", MAX, "performance"),
    Metric("ttft_p50_ms", "TTFT p50 (synthetic)", "ms", MIN, "performance"),
    Metric("tpot_p50_ms", "TPOT p50 (synthetic)", "ms", MIN, "performance"),
    Metric("e2e_p95_ms", "End-to-end latency p95 (synthetic)", "ms", MIN, "performance"),
    Metric("eval_e2e_p50_ms", "End-to-end latency p50 on eval prompts", "ms", MIN, "performance"),
    Metric("peak_vram_mb", "Peak GPU memory, device-wide", "MB", MIN, "resource"),
    Metric("server_vram_mb", "Peak GPU memory, server process (attributed)", "MB", MIN, "resource"),
    Metric("kv_cache_tokens", "KV cache capacity", "tokens", MAX, "resource"),
    Metric("synthetic_error_rate", "Failed synthetic requests", "%", MIN, "reliability"),
    Metric("eval_error_rate", "Failed eval generations (timeouts/errors)", "%", MIN, "reliability"),
    Metric("startup_s", "Server startup time", "s", MIN, "reliability"),
]}
PERCENT = {"structured_schema_rate", "structured_extraction_acc", "coding_pass_rate", "retrieval_accuracy", "synthetic_error_rate", "eval_error_rate"}


def _suite(res: dict[str, Any], name: str) -> dict[str, Any] | None:
    return (res.get("quality") or {}).get("suites", {}).get(name)


def _rate_point(s: dict[str, Any] | None, key: str = "pass") -> Point | None:
    if not s or not s.get("n"):
        return None
    w = s["interval"]
    return Point(w["rate"], w["lo"], w["hi"], w["n"])


def _level(res: dict[str, Any], io_index: int, concurrency: int) -> dict[str, Any] | None:
    cells = (res.get("performance") or {}).get("cells") or []
    if io_index >= len(cells):
        return None
    return next((lv for lv in cells[io_index]["levels"] if lv["concurrency"] == concurrency and lv["status"] == "ok"), None)


def available_levels(res: dict[str, Any], io_index: int = 0) -> list[int]:
    cells = (res.get("performance") or {}).get("cells") or []
    if io_index >= len(cells):
        return []
    return sorted(lv["concurrency"] for lv in cells[io_index]["levels"] if lv["status"] == "ok")


def _ci(st: dict[str, Any]) -> Point | None:
    if st.get("mean") is None:
        return None
    h = st.get("ci95_half")
    return Point(st["mean"], None if h is None else st["mean"] - h, None if h is None else st["mean"] + h, st.get("n"))


def extract_points(res: dict[str, Any], io_index: int = 0, concurrency: int | None = None) -> dict[str, Point]:
    """Metric id -> Point for one configuration. Metrics that were not recorded are simply absent."""
    pts: dict[str, Point] = {}
    for mid, suite_name, field in [("structured_schema_rate", "structured_json", None), ("coding_pass_rate", "coding_python", None),
                                   ("retrieval_accuracy", "retrieval_needle", None)]:
        p = _rate_point(_suite(res, suite_name))
        if p:
            pts[mid] = p
    st = _suite(res, "structured_json")
    if st and st.get("aggregate", {}).get("extraction_accuracy") is not None:
        pts["structured_extraction_acc"] = Point(st["aggregate"]["extraction_accuracy"], n=st["n"])
    suites = (res.get("quality") or {}).get("suites", {}).values()
    attempted = sum(s["performance"]["requests"]["attempted"] for s in suites if s.get("performance"))
    failed = sum(s["performance"]["requests"]["failed"] for s in suites if s.get("performance"))
    if attempted:
        w = wilson(failed, attempted)
        pts["eval_error_rate"] = Point(w["rate"], w["lo"], w["hi"], attempted)
    lat = [s["performance"]["latency"]["e2e_ms"]["p50"] for s in suites if s.get("performance") and s["performance"]["latency"].get("e2e_ms")]
    if lat:
        pts["eval_e2e_p50_ms"] = Point(sum(lat) / len(lat))
    if res.get("startup_s") is not None:
        pts["startup_s"] = Point(res["startup_s"])
    if res.get("runtime", {}).get("kv_cache_tokens"):
        pts["kv_cache_tokens"] = Point(float(res["runtime"]["kv_cache_tokens"]))
    if concurrency is not None:
        lv = _level(res, io_index, concurrency)
        if lv:
            s = lv["stats"]
            for mid, key in [("throughput_tok_s", "aggregate_output_tok_s"), ("ttft_p50_ms", "ttft_p50_ms"), ("tpot_p50_ms", "tpot_p50_ms"), ("e2e_p95_ms", "e2e_p95_ms")]:
                p = _ci(s[key])
                if p and (mid != "e2e_p95_ms" or lv["latency"]["e2e_ms"].get("p95") is not None):  # p95 only if the sample size supports it
                    pts[mid] = p
            g = lv["gpu"]
            if g.get("peak_memory_mb") is not None:
                pts["peak_vram_mb"] = Point(g["peak_memory_mb"])
            if g.get("server_peak_memory_mb") is not None:
                pts["server_vram_mb"] = Point(g["server_peak_memory_mb"])
            r = lv["requests"]
            w = wilson(r["failed"], r["attempted"])
            pts["synthetic_error_rate"] = Point(w["rate"], w["lo"], w["hi"], r["attempted"])
    return pts


def to_display(metric_id: str, value: float) -> float:
    """Percent-type metrics are stored as 0-1 fractions and shown as percentages."""
    return value * 100 if metric_id in PERCENT else value
