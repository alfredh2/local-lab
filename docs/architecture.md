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

## Phase 2: inference benchmarking
- **Runtime adapters** (`runtimes/`): `RuntimeAdapter` (`probe`, `generate`, `aclose`). `VllmOpenAIAdapter` streams `/v1/completions`; failures are returned as classified outcomes (`timeout`, `connection`, `connection_lost`, `oom`, `bad_request`, `server_error`, `incomplete_stream`), cancellation propagates. A managed adapter can later add launch/shutdown without touching experiments.
- **Experiment** (`experiments/inference/`): `config` (validated, host-allowlisted) -> `preflight` (reachability, model, `max_model_len`, KV capacity) -> `runner` (closed-loop scheduler) -> `metrics` (definitions, percentiles) -> `experiment` (warmup, iterations, GPU sampler, artifacts). Preflight runs at API creation and again at job start.
- **Workload**: token-ID prompts with per-request seeds give an exact input length (verified against server `usage`) and defeat prefix caching; output length is forced with `ignore_eos`/`min_tokens`. Text mode is approximate and labelled so.
- **Measurement**: TTFT = send -> first streamed chunk carrying a token; TPOT = (E2E - TTFT)/(n-1); aggregate throughput = successful output tokens / measured wall time. Token counts come from the server's usage object and are flagged `estimated` otherwise. GPU memory is a 250 ms device-wide NVML sampler over the measured phase.
- **Honest failure**: all-failed, warmup-failed and consecutive-failure runs end `failed` (no results row); the `requests.json` artifact is still written and marked `complete: false`. Partial failures complete with `success_rate < 1` and a warning.
- **Progress**: `jobs.progress` (JSON) updated about once a second; the worker also refreshes its heartbeat during long jobs.

## Phase 3: managed serving and sweeps
```
 job (concurrency_sweep, managed)
   preflight -> launch container -> wait ready -> [level 1..N: InferenceBenchmark._body/_finalize] -> assemble -> shutdown
        |              |                |                    |                                          ^ always (finally)
   GPU + compat    DockerCLI        watchdog (2s)      shared GPU sampler (device-wide + per-PID)
```
- **Docker layer** (`runtimes/docker.py`): `docker` CLI with fixed argv, no shell, validated tokens. `OwnedDocker` re-inspects before every stop/remove and refuses unless labels `locallab.managed=true` + `locallab.instance=<this install>` and the `locallab-managed-` name prefix match. The container runs with `no-new-privileges`, a PID limit, `--shm-size 8g` (no `--ipc=host`), localhost-only port, read-only HF cache and `HF_HUB_OFFLINE=1`. The CUDA 803 workaround (`LD_LIBRARY_PATH` host libcuda first) is a shared constant, and a test keeps `scripts/run_vllm_server.sh` in sync.
- **Planning** (`runtimes/vllm_managed.py`, `vllm_compat.py`, `services/gpu_preflight.py`): registry model -> checkpoint facts -> `ManagedPlan` with errors/warnings/estimates. Hard errors (cannot be overridden): insufficient free memory, quantization mismatch, `max_model_len` > model limit, weights not fitting, unsupported architecture, missing image. Contention is an error unless `allow_contention` is set, in which case it is a recorded warning. vLLM's supported-architecture list is read from the image once and cached per image id.
- **Lifecycle**: `ManagedVllmServer.launch/wait_ready/check_alive/shutdown`. Startup is bounded (`startup_timeout_s`) and cancellable; `shutdown` is idempotent, captures logs, and keeps the first terminal reason. Failures are classified from logs (`oom`, `insufficient_kv_cache`, `gpu_memory_changed_during_startup`, `cuda_driver_mismatch`, ...). `managed_servers` stores container id, image id/digest, exact command, model revision, timings, exit code and log tail.
- **Reconciliation**: each worker tick (every `LOCALLAB_RECONCILE_INTERVAL_S`) runs after stale-job recovery and removes owned containers whose job is terminal, missing or stale; containers of jobs with a fresh heartbeat are left alone.
- **Sweep** (`experiments/inference/sweep.py`): one `InferenceBenchmark` measurement per level, so per-level numbers come from the same code as a single benchmark. Level seeds differ so levels never replay prompts (prefix-cache safe). A server crash keeps finished levels and records the crash; if no level finished the job fails. `sweep_stats.py` holds the thresholds: p95 needs >= 20 and p99 >= 100 successful requests, CIs need >= 3 iterations, CV > 10% flags a level as noisy, a step under +10% (or within CI noise) marks saturation.
- **Memory honesty**: `peak_memory_mb` is device-wide NVML. For managed servers the sampler also sums NVML per-process memory over the container's PIDs (`docker top`) as `server_peak_memory_mb`, labelled as attributed.

## Phase 4: optimization matrix and quality evaluation
```
 matrix config -> expand (model x ctx x weight-quant x kv-dtype) -> plan/validate each -> for each feasible config:
     launch once -> [synthetic perf cells: io pair x concurrency (existing sweep code)] -> [eval suites on real prompts] -> persist row -> stop server
```
- **Grouping**: only axes that change what the server loads (model, context, weight quantization, KV dtype) multiply launches; concurrency, I/O lengths and eval suites are measured inside one launch. Expansion is model-major and deduplicated; configuration keys hash (model id, revision, context, quantization, KV dtype).
- **Persistence and isolation**: `config_results` has one row per (experiment, key) with status `pending|running|completed|failed|skipped|cancelled`, `failure_kind`, attempts and the full per-configuration results. Rows are written when a configuration ends (success, failure, cancellation or timeout), including partial data. A launch failure, crash, timeout or unexpected error is recorded on that row and the matrix continues; the job fails only if no configuration completed. `POST /experiments/{id}/resume` queues a new job that skips `completed` rows and re-runs the rest. User cancel marks unstarted configurations `cancelled`; a worker shutdown leaves them `pending` so resume picks them up. The runtime budget stops starting new configurations (remaining rows become `skipped: budget_exhausted`).
- **Evals** (`locallab/evals`): a suite yields `EvalCase`s (stable ids, version, split, messages, hidden `meta`) and scores responses with plain code. Structured JSON: strict parse then Draft 2020-12 validation, plus field accuracy against a reference answer. Coding: model code and a trusted harness with hidden asserts run in a sandbox container (`DockerSandbox` -> `DockerCLI.run_sandbox`: no network, read-only root, tmpfs `/tmp`, 512 MB, 1 CPU, 64 PIDs, `--cap-drop ALL`, non-root, hard timeout then `docker rm -f`); every task ships a reference solution that the tests run against its own hidden asserts. Retrieval: a deterministic document with a target fact and distractors at controlled positions, length calibrated with the server's `/tokenize`, scored by exact match; the server-reported prompt length is stored per case. Dev/held-out splits are fixed by construction and the matrix defaults to held-out. Generation failures (timeouts, errors, truncated output) count against accuracy; a scorer/sandbox fault is reported as `scoring_error`/`sandbox_error`, not as a model failure.
- **Synthetic vs real workloads**: the performance block is labelled `synthetic_random_token_ids`; each suite carries its own latency/throughput measured on the evaluation prompts, labelled `evaluation_prompts`. The two are never mixed in a metric.
- **Pareto** (`analysis/pareto.py`, `analysis/metrics.py`): metrics come from stored rows (quality: Wilson intervals; performance: CIs across iterations at a chosen I/O cell and concurrency; memory: device-wide and attributed per-process). The user picks objectives (direction per metric) and constraints; the result returns the feasible set, exclusions with reasons (constraint or missing metric, never treated as zero), the non-dominated front, and for each dominated configuration its dominators with `clear | within_noise | unverified` support. Weighted ranking exists only when weights are supplied and returns its formula and a caveat.
