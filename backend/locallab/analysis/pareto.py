"""Pareto-frontier analysis over recorded configuration results.

The user chooses the objectives and constraints; nothing is collapsed into a single score unless the
user supplies explicit weights, and then the exact formula is returned with the result.
Intervals (Wilson for rates, CI across iterations for performance) are used only to qualify dominance,
never to change which points are listed as non-dominated.
"""
from dataclasses import dataclass
from typing import Any, Callable

MAX, MIN = "max", "min"


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    unit: str
    direction: str  # default direction: max | min
    kind: str  # quality | performance | resource | reliability


@dataclass
class Point:
    """A configuration's value for one metric, with optional uncertainty."""

    value: float
    lo: float | None = None
    hi: float | None = None
    n: int | None = None  # sample size behind the value, when known


def _better(a: float, b: float, direction: str) -> bool:
    return a > b if direction == MAX else a < b


def _geq(a: float, b: float, direction: str) -> bool:
    return a >= b if direction == MAX else a <= b


def dominates(a: dict[str, float], b: dict[str, float], objectives: list[dict[str, str]]) -> bool:
    """a dominates b: at least as good on every objective and strictly better on one."""
    strictly = False
    for o in objectives:
        m, d = o["metric"], o["direction"]
        if not _geq(a[m], b[m], d):
            return False
        if _better(a[m], b[m], d):
            strictly = True
    return strictly


def _separated(a: Point, b: Point, direction: str) -> bool | None:
    """Is a's advantage over b outside both intervals? None if either interval is unknown."""
    if a.lo is None or a.hi is None or b.lo is None or b.hi is None:
        return None
    return a.lo > b.hi if direction == MAX else a.hi < b.lo


OPS: dict[str, Callable[[float, float], bool]] = {">=": lambda x, v: x >= v, "<=": lambda x, v: x <= v, ">": lambda x, v: x > v,
                                                    "<": lambda x, v: x < v, "==": lambda x, v: x == v}


def analyze(configs: list[dict[str, Any]], objectives: list[dict[str, str]], constraints: list[dict[str, Any]] | None = None,
            weights: dict[str, float] | None = None, metric_info: dict[str, Metric] | None = None) -> dict[str, Any]:
    """configs: [{"id": ..., "label": ..., "metrics": {metric_id: Point}}].

    Returns feasible/excluded sets, the non-dominated front, for every dominated config who dominates it
    and how well supported that is, and optionally an explicit weighted ranking.
    """
    constraints = constraints or []
    if len(objectives) < 1:
        raise ValueError("at least one objective is required")
    if len({o["metric"] for o in objectives}) != len(objectives):
        raise ValueError("each metric may appear once in objectives")
    for o in objectives:
        if o["direction"] not in (MAX, MIN):
            raise ValueError(f"direction must be 'max' or 'min', got {o['direction']!r}")
    for c in constraints:
        if c["op"] not in OPS:
            raise ValueError(f"unknown constraint operator {c['op']!r}")

    needed = {o["metric"] for o in objectives} | {c["metric"] for c in constraints}
    excluded: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for cfg in configs:
        ms: dict[str, Point] = cfg["metrics"]
        missing = sorted(m for m in needed if m not in ms or ms[m] is None)
        if missing:
            excluded.append({"id": cfg["id"], "reason": "missing_metrics", "detail": f"no recorded value for {', '.join(missing)}"})
            continue
        failed = [c for c in constraints if not OPS[c["op"]](ms[c["metric"]].value, c["value"])]
        if failed:
            c = failed[0]
            excluded.append({"id": cfg["id"], "reason": "constraint",
                             "detail": f"{c['metric']} = {ms[c['metric']].value:.4g} violates {c['op']} {c['value']}"})
            continue
        feasible.append(cfg)

    vals = {c["id"]: {o["metric"]: c["metrics"][o["metric"]].value for o in objectives} for c in feasible}
    dominators: dict[Any, list[Any]] = {c["id"]: [] for c in feasible}
    for a in feasible:
        for b in feasible:
            if a["id"] != b["id"] and dominates(vals[a["id"]], vals[b["id"]], objectives):
                dominators[b["id"]].append(a["id"])
    front = [c["id"] for c in feasible if not dominators[c["id"]]]

    by_id = {c["id"]: c for c in feasible}
    dominated = []
    for cid, doms in dominators.items():
        if not doms:
            continue
        best = []
        for did in doms:
            sup = []
            for o in objectives:
                a, b = by_id[did]["metrics"][o["metric"]], by_id[cid]["metrics"][o["metric"]]
                if _better(a.value, b.value, o["direction"]):
                    sup.append(_separated(a, b, o["direction"]))
            # clear only if every objective on which the dominator is ahead is separated beyond uncertainty
            support = "clear" if sup and all(s is True for s in sup) else "unverified" if any(s is None for s in sup) else "within_noise"
            best.append({"by": did, "support": support})
        dominated.append({"id": cid, "dominated_by": best})

    warnings: list[str] = []
    for cfg in feasible:
        for o in objectives:
            p = cfg["metrics"][o["metric"]]
            if p.n is not None and p.n < 20 and o["metric"] in (metric_info or {}) and (metric_info or {})[o["metric"]].kind == "quality":
                warnings.append(f"{cfg['id']}: {o['metric']} rests on only {p.n} samples (interval {p.lo:.2f}–{p.hi:.2f})"
                                if p.lo is not None and p.hi is not None else f"{cfg['id']}: {o['metric']} rests on only {p.n} samples")
    out: dict[str, Any] = {"objectives": objectives, "constraints": constraints, "feasible": [c["id"] for c in feasible],
                           "excluded": excluded, "front": front, "dominated": dominated, "warnings": warnings}
    if weights:
        out["weighted"] = weighted_ranking(feasible, objectives, weights)
    return out


def weighted_ranking(feasible: list[dict[str, Any]], objectives: list[dict[str, str]], weights: dict[str, float]) -> dict[str, Any]:
    """Explicit user-weighted ranking. Each metric is min-max normalised over the feasible set (best = 1)."""
    ws = {o["metric"]: float(weights.get(o["metric"], 0)) for o in objectives}
    if any(w < 0 for w in ws.values()) or sum(ws.values()) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    total = sum(ws.values())
    ws = {k: v / total for k, v in ws.items()}
    rng: dict[str, tuple[float, float]] = {}
    for o in objectives:
        xs = [c["metrics"][o["metric"]].value for c in feasible]
        rng[o["metric"]] = (min(xs), max(xs)) if xs else (0.0, 0.0)
    rows = []
    for c in feasible:
        parts = {}
        for o in objectives:
            lo, hi = rng[o["metric"]]
            v = c["metrics"][o["metric"]].value
            norm = 1.0 if hi == lo else ((v - lo) / (hi - lo) if o["direction"] == MAX else (hi - v) / (hi - lo))
            parts[o["metric"]] = norm
        rows.append({"id": c["id"], "score": sum(ws[m] * parts[m] for m in ws), "normalized": parts})
    rows.sort(key=lambda r: -r["score"])
    return {"weights": ws, "ranking": rows, "formula": "score = sum_m w_m * norm_m, norm_m = (v - min)/(max - min) for 'max' metrics, "
            "(max - v)/(max - min) for 'min' metrics, min/max taken over the feasible configurations (1.0 when all are equal)",
            "caveat": "A weighted score depends on the chosen weights and on which configurations were tested; it is not a measured quantity."}
