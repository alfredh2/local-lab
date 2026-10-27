# LocalLab

A local AI experimentation platform: discover local models, run reproducible experiments on your own GPU, and keep a durable record of every run. Phase 1 is the foundation (GPU monitoring, model registry, persistent job queue + worker). Phase 2 adds real **vLLM inference benchmarking** against an externally managed OpenAI-compatible server, with run comparison. See the [roadmap](docs/roadmap.md).

## What works today
- Live GPU telemetry via NVML (memory, utilization, temperature, power, clocks, processes). If NVML is unavailable the UI says so; nothing is mocked.
- Model registry that scans the Hugging Face cache and Ollama (read-only). Incomplete downloads (config but no weights) are flagged unavailable.
- Experiments with a persistent SQLite-backed job queue: `queued → preparing → running → completed | failed | cancelled`, cancellation, and recovery of jobs orphaned by a dead worker (marked **failed**, never completed).
- A `diagnostic` experiment that samples real GPU/CPU/RAM telemetry for a chosen window and stores the time series as an artifact.
- **Inference benchmark** (`inference_benchmark`): closed-loop load test of a vLLM server measuring TTFT, TPOT, end-to-end latency (p50/p95/p99), per-request and aggregate output tok/s, total tok/s, success rate and device-wide peak GPU memory. Profiles: short chat, long prefill, long decode, concurrent. Per-request data is stored as an artifact; partial data is kept when a run fails or is cancelled.
- Side-by-side **comparison** of completed benchmarks (`/compare`).
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

## Benchmarking vLLM
LocalLab benchmarks a server you run; it does not launch one itself yet. If you have the `vllm/vllm-openai` Docker image:
```bash
scripts/run_vllm_server.sh Qwen/Qwen2.5-1.5B-Instruct qwen2.5-1.5b-instruct 0.25   # http://127.0.0.1:8100
```
Then **Experiments → Inference benchmark → Connect**, pick the served model and a profile, and Launch. The form checks feasibility first (reachable server, served model, `max_model_len`, KV-cache capacity). Only hosts in `LOCALLAB_ALLOWED_RUNTIME_HOSTS` (localhost by default) may be targeted.

Read results carefully: **peak GPU memory is device-wide** (vLLM preallocates its KV cache), and if other processes compute on the GPU, latency and throughput are contended; the run lists them as a warning. Metric definitions are stored with every result and shown in the UI.

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
