# Architecture

```
 Browser ──HTTP──▶ Next.js (3210) ──fetch──▶ FastAPI (8321) ──▶ SQLite ◀── Worker process
                                              │  services layer             │ runs experiments
                                              └──▶ NVML / HF cache / Ollama ◀┘
```

## Components
- **API** (`locallab/api`): thin FastAPI routers; all logic lives in `locallab/services` so a future MCP server can reuse it unchanged.
- **Worker** (`python -m locallab.worker`): separate process polling the `jobs` table. It is the only component that executes experiments.
- **Database**: SQLite in WAL mode (API + worker share the file safely) via SQLAlchemy 2; Alembic migrations; switching to PostgreSQL is a `LOCALLAB_DATABASE_URL` change plus review of `claim_next` (it uses `UPDATE … RETURNING`).
- **Artifacts**: JSON files under `data/artifacts/<job_id>/`, indexed in the `artifacts` table with size and sha256.

## Data model
`models` (registry) · `experiments` (type, validated config, config hash, environment snapshot) · `jobs` (status, heartbeat, error, results, seed) · `job_logs` · `artifacts`. An experiment has one or more jobs; Phase 1 creates one job per experiment.

## Job lifecycle
`queued → preparing → running → completed | failed | cancelled`, enforced by `services/jobs.py` (`ALLOWED`).
- **Claim**: a single atomic `UPDATE … WHERE id = (oldest queued)`; safe with several workers.
- **Success**: only recorded when `run()` returns normally. Exceptions become `failed` with the traceback.
- **Cancellation**: queued jobs cancel immediately; running jobs get `cancel_requested`, which experiments observe via `ctx.check_cancel()` (≈2 s latency, also the heartbeat cadence).
- **Crash recovery**: active jobs with a stale heartbeat (`LOCALLAB_WORKER_STALE_S`) are marked `failed`, not requeued, because partial runs would corrupt measurements. A SIGTERM'd worker fails its current job as interrupted.

## Experiments
`experiments/base.py` defines a minimal interface: a pydantic `config_model` and `run(config, ctx) -> results`. `experiments/registry.py` holds the types (only `diagnostic` today). No generic framework beyond this until a second type needs it.

## Reproducibility
At creation each experiment stores: normalized config, config hash (type+config+seed), seed, hardware/OS/driver/CUDA versions, tracked package versions, Python version and the repo git commit. Later phases add model revision, runtime version and dataset version; reproducibility is not claimed for results lacking these.

## Safety
No endpoint accepts commands. The only subprocess is a fixed `git rev-parse HEAD`. Model scanning is read-only. The API binds to localhost by default and has no auth yet.

## Key decisions
- Polling DB queue instead of Redis/Celery: zero extra services, durable, adequate for a single workstation.
- Frontend polls (2 s) instead of WebSockets: simpler and sufficient.
- Timestamps are stored as naive UTC; the frontend normalizes them.
- GPU history for charts is an in-memory ring buffer in the API process (resets on restart); durable measurements live in experiment artifacts.
