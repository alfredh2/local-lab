import csv
import io as _io
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from locallab.analysis import metrics as M
from locallab.analysis import pareto as P
from locallab.db import get_session
from locallab.models import ConfigResult, Experiment
from locallab.schemas.optimization import ConfigResultOut

router = APIRouter(prefix="/optimization", tags=["optimization"])


def _experiment(session: Session, eid: int) -> Experiment:
    exp = session.get(Experiment, eid)
    if exp is None or exp.type != "optimization_matrix":
        raise HTTPException(404, "optimization experiment not found")
    return exp


def _rows(session: Session, eid: int) -> list[ConfigResult]:
    return list(session.scalars(select(ConfigResult).where(ConfigResult.experiment_id == eid).order_by(ConfigResult.id)))


def _catalog() -> list[dict[str, Any]]:
    return [{"id": m.id, "label": m.label, "unit": m.unit, "direction": m.direction, "kind": m.kind, "percent": m.id in M.PERCENT} for m in M.METRICS.values()]


@router.get("/metrics")
def metric_catalog():
    return _catalog()


@router.get("/{eid}", response_model=list[ConfigResultOut])
def config_results(eid: int, session: Session = Depends(get_session)):
    _experiment(session, eid)
    return _rows(session, eid)


def _parse_objectives(s: str | None) -> list[dict[str, str]]:
    out = []
    for part in filter(None, (s or "").split(",")):
        m, _, d = part.partition(":")
        if m not in M.METRICS:
            raise HTTPException(422, f"unknown metric {m!r}")
        out.append({"metric": m, "direction": d or M.METRICS[m].direction})
    return out


def _parse_constraints(s: str | None) -> list[dict[str, Any]]:
    out = []
    for part in filter(None, (s or "").split(",")):
        bits = part.split(":")
        if len(bits) != 3 or bits[0] not in M.METRICS:
            raise HTTPException(422, f"constraint must look like metric:op:value with a known metric, got {part!r}")
        try:
            val = float(bits[2])
        except ValueError:
            raise HTTPException(422, f"constraint value must be a number, got {bits[2]!r}") from None
        if bits[0] in M.PERCENT:
            val /= 100.0  # the API takes percentages for rate metrics; values are stored as fractions
        out.append({"metric": bits[0], "op": bits[1], "value": val})
    return out


def _parse_weights(s: str | None) -> dict[str, float] | None:
    if not s:
        return None
    w = {}
    for part in s.split(","):
        m, _, v = part.partition(":")
        if m not in M.METRICS:
            raise HTTPException(422, f"unknown metric {m!r}")
        try:
            w[m] = float(v)
        except ValueError:
            raise HTTPException(422, f"weight must be a number, got {v!r}") from None
    return w


def _point_dict(mid: str, p: P.Point) -> dict[str, Any]:
    f = lambda v: None if v is None else M.to_display(mid, v)  # noqa: E731
    return {"value": p.value, "display": f(p.value), "lo": f(p.lo), "hi": f(p.hi), "n": p.n}


@router.get("/{eid}/pareto")
def pareto(eid: int, objectives: str | None = None, constraints: str | None = None, weights: str | None = None,
           io: int = 0, concurrency: int | None = None, session: Session = Depends(get_session)):
    _experiment(session, eid)
    rows = _rows(session, eid)
    done = [r for r in rows if r.status == "completed" and r.results]
    level_sets = [set(M.available_levels(r.results, io)) for r in done]
    common = sorted(set.intersection(*level_sets)) if level_sets else []
    default_c = common[len(common) // 2] if common else None
    c = concurrency if concurrency is not None else default_c
    notes: list[str] = []
    if concurrency is not None and common and concurrency not in common:
        notes.append(f"concurrency {concurrency} was not measured for every configuration; those lacking it are excluded from performance objectives")
    points = []
    for r in done:
        pts = M.extract_points(r.results, io, c)
        points.append({"id": r.key, "label": r.label, "status": r.status, "contended": bool(r.results["gpu"]["contended"]),
                       "config": r.config, "values": {mid: _point_dict(mid, p) for mid, p in pts.items()}, "_pts": pts})
    skipped = [{"id": r.key, "label": r.label, "status": r.status, "reason": r.failure_kind or r.status} for r in rows if r.status != "completed"]
    if any(p["contended"] for p in points):
        notes.append("Some configurations ran on a contended GPU; performance differences between them may reflect other GPU work.")
    split = {r.results.get("quality", {}) and r.results["quality"].get("split") for r in done if r.results.get("quality")}
    if "dev" in split:
        notes.append("Some quality results are from the dev split; use held-out results to compare configurations.")
    objs, cons, wts = _parse_objectives(objectives), _parse_constraints(constraints), _parse_weights(weights)
    analysis = None
    if objs:
        try:
            analysis = P.analyze([{"id": p["id"], "label": p["label"], "metrics": p["_pts"]} for p in points], objs, cons, wts, M.METRICS)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
    present = sorted({mid for p in points for mid in p["values"]}, key=list(M.METRICS).index)
    for p in points:
        p.pop("_pts")
    ios = (done[0].results.get("performance") or {}).get("cells", []) if done else []
    return {"cell": {"io_index": io, "concurrency": c, "common_levels": common, "default_concurrency": default_c,
                     "io_pairs": [x["io"] for x in ios]},
            "metrics": [m for m in _catalog() if m["id"] in present], "points": points, "not_analysed": skipped, "analysis": analysis, "notes": notes}


@router.get("/{eid}/export")
def export(eid: int, format: str = Query("json", pattern="^(json|csv)$"), io: int = 0, concurrency: int | None = None,
           session: Session = Depends(get_session)):
    _experiment(session, eid)
    rows = _rows(session, eid)
    if format == "json":
        data = [{"key": r.key, "label": r.label, "status": r.status, "failure_kind": r.failure_kind, "error": r.error, "attempts": r.attempts,
                 "config": r.config, "results": r.results} for r in rows]
        return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="optimization-{eid}.json"'})
    buf = _io.StringIO()
    cols = ["key", "label", "status", "failure_kind", "model", "revision", "max_model_len", "quantization", "kv_cache_dtype", "contended", "split"]
    mids = list(M.METRICS)
    w = csv.writer(buf)
    w.writerow(cols + mids + [f"{m}_lo" for m in mids] + [f"{m}_hi" for m in mids])
    for r in rows:
        res = r.results or {}
        pts = M.extract_points(res, io, concurrency) if res else {}
        cfg = r.config
        base = [r.key, r.label, r.status, r.failure_kind or "", cfg.get("model_name"), cfg.get("revision"), cfg.get("max_model_len"),
                cfg.get("quantization"), cfg.get("kv_cache_dtype"), (res.get("gpu") or {}).get("contended", ""), ((res.get("quality") or {}).get("split") or "")]
        g = lambda m, k: "" if m not in pts or getattr(pts[m], k) is None else M.to_display(m, getattr(pts[m], k))  # noqa: E731
        w.writerow(base + [g(m, "value") for m in mids] + [g(m, "lo") for m in mids] + [g(m, "hi") for m in mids])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="optimization-{eid}.csv"'})
