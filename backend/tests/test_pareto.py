import random

import pytest

from locallab.analysis import metrics as M
from locallab.analysis import pareto as P
from locallab.analysis.pareto import Point
from locallab.evals.base import wilson


def cfg(i, **m):
    return {"id": i, "label": i, "metrics": {k: (v if isinstance(v, Point) else Point(v)) for k, v in m.items()}}


OBJ = [{"metric": "acc", "direction": "max"}, {"metric": "lat", "direction": "min"}]


def test_dominance_basics_and_ties():
    d = lambda a, b: P.dominates(a, b, OBJ)  # noqa: E731
    assert d({"acc": 0.9, "lat": 100}, {"acc": 0.8, "lat": 120})
    assert d({"acc": 0.9, "lat": 100}, {"acc": 0.9, "lat": 120})  # equal on one, strictly better on the other
    assert not d({"acc": 0.9, "lat": 100}, {"acc": 0.9, "lat": 100})  # identical: neither dominates
    assert not d({"acc": 0.9, "lat": 200}, {"acc": 0.8, "lat": 100})  # a real trade-off
    assert not d({"acc": 0.8, "lat": 100}, {"acc": 0.9, "lat": 200})


def test_front_and_dominated_sets():
    cs = [cfg("a", acc=0.9, lat=300), cfg("b", acc=0.8, lat=100), cfg("c", acc=0.7, lat=400), cfg("d", acc=0.9, lat=300), cfg("e", acc=0.85, lat=150)]
    r = P.analyze(cs, OBJ)
    assert set(r["front"]) == {"a", "b", "d", "e"}  # c is dominated by a, b, d, e; a and d tie so both stay
    assert [x["id"] for x in r["dominated"]] == ["c"] and {x["by"] for x in r["dominated"][0]["dominated_by"]} == {"a", "b", "d", "e"}


def test_directions_flip_the_front():
    cs = [cfg("fast", acc=0.5, lat=50), cfg("good", acc=0.9, lat=500)]
    assert set(P.analyze(cs, OBJ)["front"]) == {"fast", "good"}
    both_max = [{"metric": "acc", "direction": "max"}, {"metric": "lat", "direction": "max"}]
    assert P.analyze(cs, both_max)["front"] == ["good"]


def test_single_objective_front_is_the_best_value():
    cs = [cfg("a", acc=0.5), cfg("b", acc=0.9), cfg("c", acc=0.9)]
    r = P.analyze(cs, [{"metric": "acc", "direction": "max"}])
    assert set(r["front"]) == {"b", "c"}


def test_constraints_filter_before_dominance_and_are_explained():
    cs = [cfg("a", acc=0.95, lat=900, vram=30000), cfg("b", acc=0.8, lat=100, vram=9000), cfg("c", acc=0.7, lat=120, vram=8000)]
    r = P.analyze(cs, OBJ, [{"metric": "vram", "op": "<=", "value": 24000}])
    assert r["feasible"] == ["b", "c"] and r["front"] == ["b"]
    ex = r["excluded"][0]
    assert ex["id"] == "a" and ex["reason"] == "constraint" and "vram" in ex["detail"] and "24000" in ex["detail"]


def test_missing_metrics_excluded_not_treated_as_zero():
    cs = [cfg("a", acc=0.9, lat=100), cfg("b", acc=0.99)]  # b never measured latency
    r = P.analyze(cs, OBJ)
    assert r["front"] == ["a"] and r["excluded"] == [{"id": "b", "reason": "missing_metrics", "detail": "no recorded value for lat"}]


def test_validation_errors():
    with pytest.raises(ValueError, match="at least one"):
        P.analyze([], [])
    with pytest.raises(ValueError, match="once"):
        P.analyze([], [OBJ[0], OBJ[0]])
    with pytest.raises(ValueError, match="direction"):
        P.analyze([], [{"metric": "acc", "direction": "up"}])
    with pytest.raises(ValueError, match="operator"):
        P.analyze([], OBJ, [{"metric": "acc", "op": "~", "value": 1}])


def test_dominance_support_uses_intervals():
    clear = [cfg("a", acc=Point(0.9, 0.85, 0.95, 100), lat=100), cfg("b", acc=Point(0.5, 0.4, 0.6, 100), lat=200)]
    # latency has no interval -> that objective cannot be verified
    r = P.analyze(clear, OBJ)["dominated"][0]["dominated_by"][0]
    assert r["support"] == "unverified"
    noisy = [cfg("a", acc=Point(0.9, 0.6, 1.0, 10), lat=Point(100, 90, 110)), cfg("b", acc=Point(0.8, 0.5, 0.95, 10), lat=Point(200, 190, 210))]
    assert P.analyze(noisy, OBJ)["dominated"][0]["dominated_by"][0]["support"] == "within_noise"  # accuracy intervals overlap
    firm = [cfg("a", acc=Point(0.9, 0.85, 0.95, 100), lat=Point(100, 90, 110)), cfg("b", acc=Point(0.6, 0.5, 0.7, 100), lat=Point(200, 190, 210))]
    assert P.analyze(firm, OBJ)["dominated"][0]["dominated_by"][0]["support"] == "clear"


def test_small_sample_quality_warning():
    info = {"acc": P.Metric("acc", "Acc", "%", "max", "quality")}
    cs = [cfg("a", acc=Point(0.9, 0.6, 0.98, 10), lat=100), cfg("b", acc=Point(0.9, 0.85, 0.95, 200), lat=100)]
    w = P.analyze(cs, OBJ, metric_info=info)["warnings"]
    assert len(w) == 1 and "a: acc rests on only 10 samples" in w[0]


def test_weighted_ranking_is_explicit_and_explained():
    cs = [cfg("a", acc=0.9, lat=300), cfg("b", acc=0.8, lat=100), cfg("c", acc=0.7, lat=200)]
    r = P.analyze(cs, OBJ, weights={"acc": 3, "lat": 1})["weighted"]
    assert r["weights"] == {"acc": 0.75, "lat": 0.25} and "formula" in r and "caveat" in r
    assert r["ranking"][0]["id"] == "a"  # accuracy-heavy weighting prefers the most accurate
    r2 = P.analyze(cs, OBJ, weights={"acc": 1, "lat": 3})["weighted"]
    assert r2["ranking"][0]["id"] == "b"  # a different weighting changes the winner: the score is not objective
    assert "weighted" not in P.analyze(cs, OBJ)  # never produced unless the user asks
    with pytest.raises(ValueError):
        P.analyze(cs, OBJ, weights={"acc": 0, "lat": 0})


def test_property_front_is_correct_on_random_data():
    rnd = random.Random(7)
    for _ in range(200):
        n = rnd.randint(1, 12)
        cs = [cfg(f"c{i}", acc=round(rnd.random(), 1), lat=rnd.randint(1, 5) * 100, vram=rnd.randint(1, 4)) for i in range(n)]
        objs = [{"metric": "acc", "direction": "max"}, {"metric": "lat", "direction": "min"}, {"metric": "vram", "direction": rnd.choice(["min", "max"])}]
        r = P.analyze(cs, objs)
        vals = {c["id"]: {o["metric"]: c["metrics"][o["metric"]].value for o in objs} for c in cs}
        front = set(r["front"])
        for a in front:  # nothing dominates a front member
            assert not any(P.dominates(vals[b], vals[a], objs) for b in vals if b != a)
        for d in r["dominated"]:  # every dominated config really is dominated by a front member (transitivity)
            assert any(P.dominates(vals[f], vals[d["id"]], objs) for f in front)
        assert front | {d["id"] for d in r["dominated"]} == set(vals) and not front & {d["id"] for d in r["dominated"]}


# ---------------- metric extraction ----------------
def level(c, thr, ttft=10.0, tpot=3.0, e2e95=400.0, n=24, failed=0, half=5.0, mem=11000.0, srv=9000.0):
    st = lambda m, h=half: {"n": 3, "mean": m, "ci95_half": h}  # noqa: E731
    return {"concurrency": c, "status": "ok", "stats": {"aggregate_output_tok_s": st(thr), "ttft_p50_ms": st(ttft, 1), "tpot_p50_ms": st(tpot, 0.1), "e2e_p95_ms": st(e2e95, 5)},
            "latency": {"e2e_ms": {"n": n, "p95": e2e95 if n >= 20 else None}}, "gpu": {"peak_memory_mb": mem, "server_peak_memory_mb": srv},
            "requests": {"attempted": 32, "failed": failed}}


def result(rate=(8, 10), coding=(6, 13), thr=1000.0, n=24):
    def suite(k, nn, perf_fail=0):
        return {"n": nn, "interval": wilson(k, nn), "aggregate": {"extraction_accuracy": 0.9},
                "performance": {"requests": {"attempted": nn, "failed": perf_fail}, "latency": {"e2e_ms": {"p50": 800.0}}}}
    return {"startup_s": 95.0, "runtime": {"kv_cache_tokens": 150000}, "quality": {"suites": {"structured_json": suite(*rate), "coding_python": suite(*coding, 1)}},
            "performance": {"cells": [{"levels": [level(1, thr / 10, n=n), level(8, thr, n=n)]}]}}


def test_extract_quality_and_performance_points():
    pts = M.extract_points(result(), 0, 8)
    assert pts["structured_schema_rate"].value == 0.8 and pts["structured_schema_rate"].n == 10 and pts["structured_schema_rate"].lo == pytest.approx(0.49, abs=0.01)
    assert pts["coding_pass_rate"].n == 13 and pts["throughput_tok_s"].value == 1000 and pts["throughput_tok_s"].lo == 995 and pts["throughput_tok_s"].hi == 1005
    assert pts["peak_vram_mb"].value == 11000 and pts["server_vram_mb"].value == 9000 and pts["startup_s"].value == 95
    assert pts["eval_error_rate"].n == 23 and pts["eval_error_rate"].value == pytest.approx(1 / 23)
    assert "retrieval_accuracy" not in pts  # never measured: absent, not zero


def test_cell_selection_and_missing_levels():
    r = result()
    assert M.extract_points(r, 0, 1)["throughput_tok_s"].value == 100
    assert "throughput_tok_s" not in M.extract_points(r, 0, 4)  # level 4 was not run
    assert "throughput_tok_s" not in M.extract_points(r, 0, None) and "coding_pass_rate" in M.extract_points(r, 0, None)
    assert M.available_levels(r) == [1, 8] and M.available_levels(r, io_index=3) == []


def test_p95_dropped_when_sample_size_cannot_support_it():
    assert "e2e_p95_ms" in M.extract_points(result(n=24), 0, 8)
    assert "e2e_p95_ms" not in M.extract_points(result(n=10), 0, 8)


def test_display_scaling():
    assert M.to_display("coding_pass_rate", 0.5) == 50 and M.to_display("startup_s", 95) == 95


def test_end_to_end_analysis_over_extracted_points():
    a, b = result(rate=(9, 10), thr=1500.0), result(rate=(5, 10), thr=900.0)
    configs = [{"id": i, "label": i, "metrics": M.extract_points(r, 0, 8)} for i, r in [("fast_good", a), ("slow_bad", b)]]
    out = P.analyze(configs, [{"metric": "structured_schema_rate", "direction": "max"}, {"metric": "throughput_tok_s", "direction": "max"}], metric_info=M.METRICS)
    assert out["front"] == ["fast_good"] and out["dominated"][0]["id"] == "slow_bad"
    assert any("only 10 samples" in w for w in out["warnings"])
