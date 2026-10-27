# Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | API, NVML monitoring, model registry, job queue + worker, diagnostic experiment, dashboard | **Done** |
| 2 Inference benchmarking | vLLM adapter (external server); TTFT/TPOT/throughput/p50/p95; peak GPU memory; OOM/timeouts; run comparison | **Done** |
| 3 Coding-agent arena | Sandboxed repo tasks, shared scaffold, test-based scoring, traces | Planned |
| 4 MCP & remote compute | MCP server over the same service layer, auth | Planned |
| 5 Experiment planner | Deterministic search baselines first, then LLM planner using stored results only | Planned |
| 6 Advanced optimization | Quantization/KV-cache/speculative decoding studies, routing, LoRA, Pareto analysis | Planned |

Phase 1 follow-ups: authentication for non-local access, persisted GPU history, PostgreSQL validation, experiment re-run and compare UI.

Phase 2 follow-ups: managed vLLM launching (adapter interface is ready for it), open-loop request-rate mode, per-process GPU memory attribution, a concurrency sweep experiment, SGLang/llama.cpp adapters, tokenizer-based chat-template workloads, quality (accuracy) checks alongside speed.
