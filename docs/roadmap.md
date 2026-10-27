# Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | API, NVML monitoring, model registry, job queue + worker, diagnostic experiment, dashboard | **Done** |
| 2 Inference benchmarking | vLLM adapter (external server); TTFT/TPOT/throughput/p50/p95; peak GPU memory; OOM/timeouts; run comparison | **Done** |
| 3 Managed serving & sweeps | Managed vLLM containers, GPU preflight, concurrency sweeps with uncertainty, orphan reconciliation | **Done** |
| 4 Optimization & quality evaluation | Configuration matrices, structured/coding/retrieval evals, Pareto analysis, Optimization Lab | **Done** |
| 5 Coding-agent arena | Sandboxed repo tasks, shared scaffold, test-based scoring, traces | Planned |
| 6 MCP & remote compute | MCP server over the same service layer, auth | Planned |
| 7 Experiment planner | Deterministic search baselines first, then LLM planner using stored results only | Planned |
| 8 Advanced optimization | Quantization/KV-cache/speculative decoding studies, routing, LoRA, Pareto analysis | Planned |

Phase 1 follow-ups: authentication for non-local access, persisted GPU history, PostgreSQL validation, experiment re-run and compare UI.

Phase 2 follow-ups: managed vLLM launching (adapter interface is ready for it), open-loop request-rate mode, per-process GPU memory attribution, a concurrency sweep experiment, SGLang/llama.cpp adapters, tokenizer-based chat-template workloads, quality (accuracy) checks alongside speed.

Phase 3 follow-ups: sweeps over other axes (context length, quantization, KV dtype) with automatic server restarts, multi-GPU placement, open-loop request-rate mode, a quality check alongside speed, image pulling/version pinning policy, authentication.

Phase 4 follow-ups: more eval tasks and suites (multi-file coding, tool use), repeated eval runs with sampling to estimate run-to-run variance, held-out dataset rotation, additional runtimes (SGLang/llama.cpp), speculative decoding as a matrix axis, multi-GPU placement, authentication.
