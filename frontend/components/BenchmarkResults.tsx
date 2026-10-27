"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { API_URL, type Artifact, type Job } from "@/lib/api";
import { fmtInt, fmtMb, fmtMs, fmtNum } from "@/lib/format";
import LineChartCard from "@/components/GpuChart";
import { Card, Empty, Pill, Stat } from "@/components/ui";

type D = { n: number; mean: number; min: number; p50: number; p95: number; p99: number; max: number } | null;

export function useJson<T = any>(art: Artifact | undefined) {
  return useQuery({
    queryKey: ["artifact", art?.id], enabled: !!art, staleTime: Infinity,
    queryFn: async () => (await fetch(`${API_URL}/api/artifacts/${art!.id}/content`)).json() as Promise<T>,
  });
}

export function BenchmarkProgress({ job }: { job: Job }) {
  const p = job.progress;
  if (!p) return <Empty title={job.status === "queued" ? "Waiting for the worker" : "Starting…"}>Progress appears once requests begin.</Empty>;
  const pct = p.total ? (p.done / p.total) * 100 : 0;
  return (
    <Card title={`Running — ${p.phase}${p.phase === "measured" ? ` (iteration ${p.iteration})` : ""}`}>
      <div className="h-2 overflow-hidden rounded-full bg-zinc-800"><div className="h-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} /></div>
      <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-5">
        <Stat label="Completed" value={`${p.done} / ${p.total}`} />
        <Stat label="In flight" value={p.in_flight} />
        <Stat label="Failed" value={<span className={p.failed ? "text-red-300" : ""}>{p.failed}</span>} />
        <Stat label="Live output tok/s" value={fmtNum(p.live_output_tok_s, 0)} sub="running average" />
        <Stat label="Elapsed" value={`${p.elapsed_s}s`} />
      </div>
      {p.last_error && <div className="mt-3 text-xs text-red-300">Last error: {p.last_error}</div>}
    </Card>
  );
}

const ROWS: [string, string, (s: any) => D][] = [
  ["TTFT", "ttft_ms", (s) => s.latency.ttft_ms], ["TPOT", "tpot_ms", (s) => s.latency.tpot_ms],
  ["End-to-end", "e2e_ms", (s) => s.latency.e2e_ms], ["Inter-chunk (per-request mean)", "itl_mean_ms", (s) => s.latency.itl_mean_ms],
];

export function DistTable({ s }: { s: any }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[480px] text-sm">
        <thead className="text-left text-xs text-zinc-500"><tr>
          {["Metric", "mean", "p50", "p95", "p99", "max"].map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-zinc-800 tabular-nums">
          {ROWS.map(([label, , get]) => { const d = get(s); return (
            <tr key={label}><td className="py-1.5 pr-3 text-zinc-300">{label}</td>
              {(["mean", "p50", "p95", "p99", "max"] as const).map((k) => <td key={k} className="pr-3">{fmtMs(d?.[k])}</td>)}</tr>); })}
          {(["per_request_output_tok_s", "per_request_decode_tok_s"] as const).map((k) => { const d: D = s.throughput[k]; return (
            <tr key={k}><td className="py-1.5 pr-3 text-zinc-300">{k === "per_request_output_tok_s" ? "Output tok/s per request" : "Decode tok/s per request"}</td>
              {(["mean", "p50", "p95", "p99", "max"] as const).map((x) => <td key={x} className="pr-3">{d ? d[x].toFixed(1) : "—"}</td>)}</tr>); })}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchmarkResults({ job, artifacts }: { job: Job; artifacts: Artifact[] }) {
  const r = job.results!;
  const s = r.summary;
  const reqArt = useJson(artifacts.find((a) => a.name === "requests.json"));
  const gpuArt = useJson(artifacts.find((a) => a.name === "gpu_samples.json"));
  const [showAll, setShowAll] = useState(false);
  const measured: any[] = (reqArt.data?.requests ?? []).filter((x: any) => x.phase === "measured");
  const perReq = measured.map((x, i) => ({ n: i + 1, e2e: x.e2e_ms, ttft: x.ttft_ms, tpot: x.tpot_ms, toks: x.output_tok_s }));
  const g0 = r.gpu.per_device["0"];
  const gs = (gpuArt.data?.samples ?? []).filter((x: any) => x.index === 0 && x.t >= gpuArt.data.measured_start_s - 1 &&
    (gpuArt.data.measured_end_s == null || x.t <= gpuArt.data.measured_end_s + 1))
    .map((x: any) => ({ t: x.t, mem: x.memory_used_mb / 1024, util: x.utilization_gpu_pct, power: x.power_w }));
  const v = r.workload.verification;
  const w = r.workload;
  return (
    <div className="space-y-6">
      {r.warnings.length > 0 && (
        <Card title="Warnings"><ul className="space-y-1 text-sm text-amber-300">{r.warnings.map((x: string) => <li key={x}>⚠ {x}</li>)}</ul></Card>
      )}
      <Card title="Throughput & reliability">
        <div className="grid grid-cols-2 gap-5 md:grid-cols-4">
          <Stat label="Aggregate output tok/s" value={fmtNum(s.throughput.aggregate_output_tok_s, 1)} sub="all requests / wall time" />
          <Stat label="Total tok/s" value={fmtNum(s.throughput.total_tok_s, 1)} sub="input + output" />
          <Stat label="Requests/s" value={fmtNum(s.throughput.requests_per_s, 2)} />
          <Stat label="Success rate" value={<span className={s.requests.success_rate < 1 ? "text-amber-300" : ""}>{(s.requests.success_rate * 100).toFixed(1)}%</span>}
            sub={`${s.requests.succeeded}/${s.requests.attempted}${Object.keys(s.requests.errors_by_type).length ? " · " + JSON.stringify(s.requests.errors_by_type) : ""}`} />
          <Stat label="TTFT p50 / p95" value={<span className="text-xl">{fmtMs(s.latency.ttft_ms?.p50)} / {fmtMs(s.latency.ttft_ms?.p95)}</span>} />
          <Stat label="TPOT p50 / p95" value={<span className="text-xl">{fmtMs(s.latency.tpot_ms?.p50)} / {fmtMs(s.latency.tpot_ms?.p95)}</span>} />
          <Stat label="E2E p50 / p95" value={<span className="text-xl">{fmtMs(s.latency.e2e_ms?.p50)} / {fmtMs(s.latency.e2e_ms?.p95)}</span>} />
          <Stat label="Peak GPU memory" value={g0 ? fmtMb(g0.peak_memory_mb) : "—"} sub="device-wide, measured phase" />
        </div>
      </Card>

      <Card title="Latency & per-request speed distribution (successful requests)"><DistTable s={s} /></Card>

      {measured.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="Latency per request (ms), in issue order">
            <LineChartCard data={perReq} xKey="n" xLabel={(x) => `req ${x}`}
              series={[{ key: "e2e", label: "End-to-end (left)", color: "#60a5fa" }, { key: "ttft", label: "TTFT (right)", color: "#fbbf24", axis: "right" }]} />
          </Card>
          <Card title="Per-request speed">
            <LineChartCard data={perReq} xKey="n" xLabel={(x) => `req ${x}`}
              series={[{ key: "toks", label: "Output tok/s (left)", color: "#34d399" }, { key: "tpot", label: "TPOT ms (right)", color: "#f87171", axis: "right" }]} />
          </Card>
        </div>
      )}
      {gs.length > 1 && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title="GPU 0 memory (GB) & utilization (%) — measured phase, device-wide">
            <LineChartCard data={gs} xKey="t" xLabel={(x) => `${Number(x).toFixed(0)}s`}
              series={[{ key: "mem", label: "Memory GB (left)", color: "#60a5fa" }, { key: "util", label: "Util % (right)", color: "#34d399", axis: "right" }]} />
            <p className="mt-2 text-xs text-zinc-500">vLLM preallocates its KV cache, so memory is usually flat during a run. Baseline {fmtMb(g0.baseline_memory_mb)}, peak {fmtMb(g0.peak_memory_mb)}.</p>
          </Card>
          <Card title="GPU 0 power (W)">
            <LineChartCard data={gs} xKey="t" xLabel={(x) => `${Number(x).toFixed(0)}s`} series={[{ key: "power", label: "Power W", color: "#fbbf24" }]} />
          </Card>
        </div>
      )}
      {r.iterations.length > 1 && (
        <Card title="Iterations">
          <table className="w-full text-sm tabular-nums"><thead className="text-left text-xs text-zinc-500"><tr>
            {["#", "Wall (s)", "Output tok/s", "Total tok/s", "E2E p50", "Success"].map((h) => <th key={h} className="py-1 font-medium">{h}</th>)}</tr></thead>
            <tbody className="divide-y divide-zinc-800">{r.iterations.map((i: any) => (
              <tr key={i.iteration}><td className="py-1.5">{i.iteration}</td><td>{fmtNum(i.wall_s, 2)}</td><td>{fmtNum(i.aggregate_output_tok_s, 1)}</td>
                <td>{fmtNum(i.total_tok_s, 1)}</td><td>{fmtMs(i.e2e_p50_ms)}</td><td>{(i.success_rate * 100).toFixed(0)}%</td></tr>))}</tbody></table>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Workload & verification">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-zinc-500">Profile</dt><dd>{w.profile}</dd>
            <dt className="text-zinc-500">Input / output tokens</dt><dd>{w.input_tokens} / {w.output_tokens}</dd>
            <dt className="text-zinc-500">Requests × iterations</dt><dd>{w.num_requests} × {w.iterations} (+{w.warmup_requests} warmup)</dd>
            <dt className="text-zinc-500">Concurrency</dt><dd>{w.concurrency} (max observed in flight {w.max_observed_in_flight})</dd>
            <dt className="text-zinc-500">Decoding</dt><dd>T={w.decoding.temperature}, top_p={w.decoding.top_p}, ignore_eos, seed {w.seed}</dd>
            <dt className="text-zinc-500">Prompt mode</dt><dd>{w.prompt_mode}</dd>
            <dt className="text-zinc-500">Input length</dt><dd>{v.input_length_exact ? <Pill tone="green">exact, server-verified</Pill> : w.prompt_mode === "text" ? <Pill tone="amber">approximate (mean {fmtNum(v.input_tokens_mean, 0)})</Pill> : <Pill tone="red">{v.input_mismatches} mismatches</Pill>}</dd>
            <dt className="text-zinc-500">Output length</dt><dd>{v.output_length_exact ? <Pill tone="green">exact</Pill> : <Pill tone="red">{v.output_mismatches} mismatches</Pill>}</dd>
            <dt className="text-zinc-500">Token counts</dt><dd>{s.token_count_source === "server" ? <Pill tone="green">server-reported</Pill> : <Pill tone="amber">estimated</Pill>}</dd>
          </dl>
        </Card>
        <Card title="Runtime & model">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-zinc-500">Runtime</dt><dd>vLLM {r.runtime.version ?? "(unknown)"}</dd>
            <dt className="text-zinc-500">Served model</dt><dd>{r.runtime.served_model.id}</dd>
            <dt className="text-zinc-500">Model root</dt><dd className="break-all">{r.runtime.served_model.root ?? "—"}</dd>
            <dt className="text-zinc-500">Max model length</dt><dd>{fmtInt(r.runtime.served_model.max_model_len)}</dd>
            <dt className="text-zinc-500">KV cache</dt><dd>{r.runtime.kv_cache_tokens ? `${fmtInt(r.runtime.kv_cache_tokens)} tokens` : "unknown"}</dd>
            {r.model_registry && (<><dt className="text-zinc-500">Registry</dt><dd>{r.model_registry.name} · {r.model_registry.quantization ?? "no quant info"}</dd></>)}
            <dt className="text-zinc-500">Other GPU compute procs</dt>
            <dd>{r.gpu.other_compute_processes.length ? r.gpu.other_compute_processes.map((p: any) => `${p.name.split("/").pop()} (${fmtMb(p.memory_mb)})`).join(", ") : "none"}</dd>
          </dl>
          {r.runtime.server_config && (
            <details className="mt-2 text-xs text-zinc-500"><summary className="cursor-pointer">Server-reported cache config</summary>
              <pre className="mt-1 overflow-auto">{JSON.stringify(r.runtime.server_config, null, 1)}</pre></details>
          )}
        </Card>
      </div>

      <Card title="Metric definitions">
        <details><summary className="cursor-pointer text-sm text-zinc-400">How each number is measured</summary>
          <dl className="mt-3 space-y-2 text-sm">{Object.entries<string>(r.definitions).map(([k, d]) => (
            <div key={k}><dt className="font-mono text-xs text-zinc-300">{k}</dt><dd className="text-zinc-500">{d}</dd></div>))}</dl>
        </details>
      </Card>

      {measured.length > 0 && (
        <Card title={`Per-request measurements (${measured.length})`} action={measured.length > 25 &&
          <button className="text-xs text-blue-400 hover:underline" onClick={() => setShowAll(!showAll)}>{showAll ? "Show fewer" : "Show all"}</button>}>
          <div className="max-h-96 overflow-auto">
            <table className="w-full min-w-[640px] text-xs tabular-nums">
              <thead className="sticky top-0 bg-zinc-900 text-left text-zinc-500"><tr>
                {["#", "iter", "status", "in tok", "out tok", "TTFT ms", "TPOT ms", "E2E ms", "tok/s"].map((h) => <th key={h} className="py-1 pr-2 font-medium">{h}</th>)}</tr></thead>
              <tbody className="divide-y divide-zinc-800">
                {(showAll ? measured : measured.slice(0, 25)).map((x, i) => (
                  <tr key={i} className={x.ok ? "" : "text-red-300"}>
                    <td className="py-1 pr-2">{x.index + 1}</td><td>{x.iteration + 1}</td>
                    <td title={x.error_message ?? ""}>{x.ok ? "ok" : x.error_type}</td>
                    <td>{x.input_tokens ?? "—"}</td><td>{x.output_tokens ?? "—"}</td>
                    <td>{fmtNum(x.ttft_ms, 1)}</td><td>{fmtNum(x.tpot_ms, 2)}</td><td>{fmtNum(x.e2e_ms, 1)}</td><td>{fmtNum(x.output_tok_s, 1)}</td>
                  </tr>))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
