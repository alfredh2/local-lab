# LocalLab

A local AI experimentation platform: discover local models, run reproducible experiments on your own GPU, and keep a durable record of every run. Phase 1 (this release) is the foundation: GPU monitoring, a model registry, a persistent job queue with a separate worker, and a real diagnostic experiment that exercises the whole lifecycle. **No inference benchmarking exists yet** (see [roadmap](docs/roadmap.md)).

## What works today
- Live GPU telemetry via NVML (memory, utilization, temperature, power, clocks, processes). If NVML is unavailable the UI says so; nothing is mocked.
- Model registry that scans the Hugging Face cache and Ollama (read-only). Incomplete downloads (config but no weights) are flagged unavailable.
- Experiments with a persistent SQLite-backed job queue: `queued → preparing → running → completed | failed | cancelled`, cancellation, and recovery of jobs orphaned by a dead worker (marked **failed**, never completed).
- A `diagnostic` experiment that samples real GPU/CPU/RAM telemetry for a chosen window and stores the time series as an artifact.
- Next.js dashboard: Overview, GPU, Models, Experiments, Experiment details.

## Requirements
Python ≥ 3.11, Node ≥ 20, an NVIDIA GPU + driver (NVML) for GPU features. Nothing here touches drivers, CUDA, or other services.

## Setup
```bash
cp .env.example .env        # optional; defaults work
make setup                  # venv + pip + npm install
```

## Run (three processes)
```bash
make api      # applies migrations, serves http://127.0.0.1:8321  (docs at /docs)
make worker   # executes queued jobs
make web      # dashboard at http://localhost:3210
```
Ports 8321 and 3210 were chosen to avoid common services (Open WebUI on 3000, Ollama on 11434); change them in `.env`. For remote access (e.g. Tailscale) set `LOCALLAB_API_HOST`, add the dashboard origin to `LOCALLAB_CORS_ORIGINS`, and point `NEXT_PUBLIC_API_URL` at the API; there is no authentication yet, so keep it on a trusted network.

## Try it
1. Open the dashboard → **Models** → *Scan for models*.
2. **Experiments** → *System diagnostic* → Launch. Watch it go queued → running → completed.
3. Stop and restart the API (and/or worker); experiment history remains in `data/locallab.db`.

## Tests
```bash
make test    # backend: unit + API/worker integration tests
make lint    # frontend lint + typecheck
make build   # frontend production build
```

## Layout
`backend/locallab/` — `api/` routers, `services/` shared service layer, `experiments/` experiment types, `worker/` job runner, `models/` ORM. `frontend/` — Next.js app. See [docs/architecture.md](docs/architecture.md).
