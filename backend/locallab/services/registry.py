"""Model registry: read-only discovery of local models, persisted to the DB."""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from locallab.config import Settings
from locallab.models import ModelRecord, utcnow
from locallab.schemas.models import ScanResult

log = logging.getLogger(__name__)
WEIGHT_SUFFIXES = {".safetensors": "safetensors", ".gguf": "gguf", ".bin": "pytorch"}


@dataclass
class Discovered:
    name: str
    source: str
    ref: str
    revision: str | None = None
    format: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    available: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


def _quantization_from_config(cfg: dict[str, Any]) -> str | None:
    q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    if not q:
        return None
    method = q.get("quant_method")
    bits = q.get("bits")
    return f"{method}-int{bits}" if method and bits else method


def inspect_model_dir(path: Path, name: str, source: str, revision: str | None = None) -> Discovered:
    """Describe a directory holding HF-format weights. Missing weights => unavailable."""
    cfg: dict[str, Any] = {}
    try:
        cfg = json.loads((path / "config.json").read_text())
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        log.warning("Bad config.json in %s: %s", path, e)
    weights = [p for p in path.iterdir() if p.suffix in WEIGHT_SUFFIXES]
    size = sum(p.stat().st_size for p in weights)  # stat follows HF's blob symlinks
    fmt = WEIGHT_SUFFIXES[weights[0].suffix] if weights else None
    quant = _quantization_from_config(cfg)
    if quant is None and "fp8" in name.lower():
        quant = "fp8 (from name)"
    archs = cfg.get("architectures") or []
    return Discovered(
        name=name, source=source, ref=str(path), revision=revision, format=fmt, quantization=quant,
        size_bytes=size or None, available=bool(weights),
        meta={"architecture": archs[0] if archs else None, "model_type": cfg.get("model_type"),
              "weight_files": len(weights), **({} if weights else {"note": "no weight files in snapshot"})},
    )


def scan_hf_cache(cache_dir: Path) -> list[Discovered]:
    root = cache_dir.expanduser()
    found: list[Discovered] = []
    if not root.is_dir():
        return found
    for repo in sorted(root.glob("models--*")):
        name = repo.name.removeprefix("models--").replace("--", "/")
        snaps = repo / "snapshots"
        if not snaps.is_dir():
            continue
        main_ref = repo / "refs" / "main"
        rev = main_ref.read_text().strip() if main_ref.is_file() else None
        snap = snaps / rev if rev and (snaps / rev).is_dir() else None
        if snap is None:
            candidates = sorted((p for p in snaps.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
            if not candidates:
                continue
            snap = candidates[-1]
            rev = snap.name
        found.append(inspect_model_dir(snap, name, "hf_cache", rev))
    return found


def scan_ollama(url: str, timeout: float = 3.0) -> tuple[list[Discovered], str | None]:
    try:
        r = httpx.get(f"{url}/api/tags", timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as e:
        return [], f"Ollama not reachable at {url}: {e.__class__.__name__}"
    out = []
    for m in r.json().get("models", []):
        d = m.get("details", {})
        out.append(Discovered(
            name=m["name"], source="ollama", ref=m["name"], revision=m.get("digest", "")[:12] or None,
            format=d.get("format"), quantization=d.get("quantization_level"), size_bytes=m.get("size"),
            meta={"family": d.get("family"), "parameter_size": d.get("parameter_size"),
                  "context_length": d.get("context_length")},
        ))
    return out, None


def sync_registry(session: Session, settings: Settings) -> ScanResult:
    warnings: list[str] = []
    discovered = scan_hf_cache(settings.hf_cache_dir)
    ollama, warn = scan_ollama(settings.ollama_url)
    if warn:
        warnings.append(warn)
    discovered += ollama
    scanned_sources = {"hf_cache"} | ({"ollama"} if not warn else set())

    existing = {(m.source, m.ref): m for m in session.scalars(select(ModelRecord))}
    seen: set[tuple[str, str]] = set()
    added = updated = 0
    now = utcnow()
    for d in discovered:
        key = (d.source, d.ref)
        seen.add(key)
        rec = existing.get(key)
        if rec is None:
            session.add(ModelRecord(**{k: getattr(d, k) for k in
                ("name", "source", "ref", "revision", "format", "quantization", "size_bytes", "available", "meta")},
                last_seen_at=now))
            added += 1
        else:
            for k in ("name", "revision", "format", "quantization", "size_bytes", "available", "meta"):
                setattr(rec, k, getattr(d, k))
            rec.last_seen_at = now
            updated += 1
    gone = 0
    for key, rec in existing.items():
        if key not in seen and rec.source in scanned_sources and rec.available:
            rec.available = False
            gone += 1
    return ScanResult(discovered=len(discovered), added=added, updated=updated,
                      marked_unavailable=gone, warnings=warnings)


def add_manual(session: Session, name: str, path: str) -> ModelRecord:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")
    d = inspect_model_dir(p, name, "manual")
    if session.scalars(select(ModelRecord).where(ModelRecord.source == "manual", ModelRecord.ref == d.ref)).first():
        raise ValueError(f"Already registered: {d.ref}")
    rec = ModelRecord(name=d.name, source=d.source, ref=d.ref, format=d.format, quantization=d.quantization,
                      size_bytes=d.size_bytes, available=d.available, meta=d.meta)
    session.add(rec)
    session.flush()
    return rec
