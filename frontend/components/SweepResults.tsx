"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type Artifact, type Job, type ManagedServerRec } from "@/lib/api";
import { fmtInt, fmtMb, fmtMs, fmtNum } from "@/lib/format";
import { useJson } from "@/components/BenchmarkResults";
import { Card, Empty, Pill, Stat } from "@/components/ui";

const TIP = { background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 };
export const C = { blue: "#60a5fa", amber: "#fbbf24", green: "#34d399", red: "#f87171", violet: "#c084fc", gray: "#71717a" };

type Series = { key: string; label: string; color: string; axis?: "left" | "right"; dashed?: boolean; err?: string };
function ConcChart({ data, series, unit, height = 230 }: { data: Record<string, any>[]; series: Series[]; unit?: string; height?: number }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 16, left: 0 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
          <XAxis dataKey="c" stroke="#71717a" fontSize={11} label={{ value: "concurrency", position: "insideBottom", offset: -8, fill: "#71717a", fontSize: 11 }} />
          <YAxis yAxisId="left" stroke="#71717a" fontSize={11} width={48} unit={unit} />
          {series.some((s) => s.axis === "right") && <YAxis yAxisId="right" orientation="right" stroke="#71717a" fontSize={11} width={44} />}
          <Tooltip contentStyle={TIP} formatter={(v) => (typeof v === "number" ? v.toFixed(1) : String(v))} labelFormatter={(c) => `concurrency ${c}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
          {series.map((s) => (
            <Line key={s.key} yAxisId={s.axis ?? "left"} dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2}
              strokeDasharray={s.dashed ? "5 4" : undefined} dot={{ r: 3 }} isAnimationActive={false}>
              {s.err && <ErrorBar dataKey={s.err} width={5} stroke={s.color} strokeWidth={1.5} />}
            </Line>
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function SweepProgress({ job }: { job: Job }) {
  const p = job.progress;
  if (!p) return <Empty title={job.status === "queued" ? "Waiting for the worker" : "Starting…"}>Progress appears once the job begins.</Empty>;
  const sw = p.sweep;
  const startup = ["preflight", "launching_server", "waiting_for_server"].includes(p.phase);
  const label: Record<string, string> = {
    preflight: "Checking GPU and configuration", launching_server: "Launching vLLM container",
    waiting_for_server: "Waiting for the model to load",
  };
  const pct = startup ? 0 : p.total ? (p.done / p.total) * 100 : 0;
  return (
    <Card title={startup ? label[p.phase] : `Running — concurrency ${sw?.concurrency} (level ${(sw?.level_index ?? 0) + 1} of ${sw?.levels.length}) · ${p.phase}${p.phase === "measured" ? ` iteration ${p.iteration}` : ""}`}>
      {startup ? (
        <div className="text-sm text-zinc-400">
          {p.phase === "waiting_for_server" ? `${p.elapsed_s}s elapsed of a ${p.timeout_s}s limit. First start compiles kernels and can take a minute or more.` : "This takes a few seconds."}
        </div>
      ) : (
        <>
          <div className="h-2 overflow-hidden rounded-full bg-zinc-800"><div className="h-full bg-blue-500 transition-all" style={{ width: `${pct}%` }} /></div>
          <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-5">
            <Stat label="Completed" value={`${p.done} / ${p.total}`} />
            <Stat label="In flight" value={p.in_flight} />
            <Stat label="Failed" value={<span className={p.failed ? "text-red-300" : ""}>{p.failed}</span>} />
            <Stat label="Live output tok/s" value={fmtNum(p.live_output_tok_s, 0)} />
            <Stat label="Levels" value={`${sw?.levels_done ?? 0} / ${sw?.levels.length}`} sub={sw?.levels.join(", ")} />
          </div>
          {p.last_error && <div className="mt-3 text-xs text-red-300">Last error: {p.last_error}</div>}
        </>
      )}
    </Card>
  );
}

export function useManagedServer(jobId: number | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ["managed-server", jobId], enabled: enabled && !!jobId,
    queryFn: async () => {
      const list = await api<ManagedServerRec[]>(`/managed-servers?job_id=${jobId}`);
      return list[0] ? api<ManagedServerRec>(`/managed-servers/${list[0].id}`) : null;
    },
  });
}

export function ServerCard({ ms, defaultOpen = false }: { ms: ManagedServerRec; defaultOpen?: boolean }) {
  return (
    <Card title="Managed server container">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-zinc-500">Status</dt><dd><Pill tone={ms.status === "stopped" ? "green" : ms.status === "failed" ? "red" : "amber"}>{ms.status}</Pill> {ms.stop_reason && <span className="text-xs text-zinc-500">{ms.stop_reason}</span>}</dd>
        <dt className="text-zinc-500">Container</dt><dd className="break-all font-mono text-xs">{ms.container_name} · {ms.container_id?.slice(0, 12)}</dd>
        <dt className="text-zinc-500">Image</dt><dd className="break-all font-mono text-xs">{ms.image_digest ?? ms.image_ref}</dd>
        <dt className="text-zinc-500">Startup</dt><dd>{ms.startup_s != null ? `${ms.startup_s.toFixed(1)}s to ready` : "never became ready"}{ms.exit_code != null && ` · exit code ${ms.exit_code}`}{ms.oom_killed ? " · host OOM-killed" : ""}</dd>
        <dt className="text-zinc-500">Command</dt><dd className="break-all font-mono text-xs">vllm serve {ms.command?.join(" ")}</dd>
      </dl>
      {ms.log_tail && (
        <details className="mt-3" open={defaultOpen}><summary className="cursor-pointer text-sm text-zinc-400">Launch / server log (last {ms.log_tail.split("\n").length} lines)</summary>
          <pre className="mt-2 max-h-80 overflow-auto rounded-lg bg-zinc-950 p-3 text-[11px] leading-snug text-zinc-400">{ms.log_tail}</pre></details>
      )}
    </Card>
  );
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function SweepResults({ job, artifacts }: { job: Job; artifacts: Artifact[] }) {
  const r = job.results!;
  const levels: any[] = r.levels;
  const ok = levels.filter((l) => l.status === "ok");
  const a = r.analysis;
  const base = ok.find((l) => l.concurrency === 1);
  const thr1 = base?.stats.aggregate_output_tok_s.mean;
  const ms = useManagedServer(job.id, r.mode === "managed");
  const reqArt = useJson(artifacts.find((x) => x.name === "requests.json"));
  const [lv, setLv] = useState<number | null>(null);
  const selLevel = lv ?? ok[0]?.concurrency;

  const data = levels.map((l) => {
    const good = l.status === "ok";
    const st = good ? l.stats : null;
    const total = good ? l.requests.attempted : null;
    return {
      c: l.concurrency,
      thr: st?.aggregate_output_tok_s.mean ?? null, thr_err: st?.aggregate_output_tok_s.ci95_half ?? null,
      ideal: thr1 ? thr1 * l.concurrency : null,
      e2e_p50: good ? l.latency.e2e_ms?.p50 : null, e2e_p95: good ? l.latency.e2e_ms?.p95 : null, e2e_p99: good ? l.latency.e2e_ms?.p99 : null,
      ttft: st?.ttft_p50_ms.mean ?? null, tpot: st?.tpot_p50_ms.mean ?? null,
      mem: good ? (l.gpu.peak_memory_mb ?? 0) / 1024 : null, srv: good && l.gpu.server_peak_memory_mb != null ? l.gpu.server_peak_memory_mb / 1024 : null,
      util: good ? l.gpu.peak_utilization_pct : null,
      fail: good ? (total ? (l.requests.failed / total) * 100 : 0) : 100,
    };
  });
  const hasSrvMem = data.some((d) => d.srv != null);
  const gp = r.gpu;
  const rows = (reqArt.data?.levels ?? []).find((x: any) => x.concurrency === selLevel)?.requests?.filter((x: any) => x.phase === "measured") ?? [];

  return (
    <div className="space-y-6">
      {gp.contended && (
        <div className="rounded-xl border border-amber-800 bg-amber-950/30 p-4 text-sm text-amber-200">
          <div className="font-medium">Contended GPU{gp.contention_override ? " (override was set)" : ""}</div>
          Other compute processes were using the GPU: {gp.other_compute_processes.length ? gp.other_compute_processes.map((p: any) => `${p.name.split("/").pop()} (${fmtMb(p.memory_mb)})`).join(", ") : "activity detected before launch"}.
          Absolute latency and throughput are not representative of an idle GPU.
        </div>
      )}
      <Card title="What this sweep shows">
        <ul className="space-y-1 text-sm text-zinc-300">{a.notes.map((n: string) => <li key={n}>• {n}</li>)}</ul>
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          <Pill tone={a.reliable ? "green" : "amber"}>{a.reliable ? "no quality flags" : "has quality flags"}</Pill>
          {r.stopped_early && <Pill tone="red">{r.stopped_early}</Pill>}
        </div>
        <p className="mt-2 text-xs text-zinc-500">{a.thresholds}</p>
      </Card>
      {r.warnings.length > 0 && <Card title="Warnings"><ul className="space-y-1 text-sm text-amber-300">{r.warnings.map((x: string) => <li key={x}>⚠ {x}</li>)}</ul></Card>}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Throughput vs concurrency (aggregate output tok/s)">
          <ConcChart data={data} series={[{ key: "thr", label: "Measured (mean ± 95% CI)", color: C.blue, err: "thr_err" }, { key: "ideal", label: "Linear scaling from c=1", color: C.gray, dashed: true }]} />
        </Card>
        <Card title="End-to-end latency vs concurrency (ms)">
          <ConcChart data={data} series={[{ key: "e2e_p50", label: "p50", color: C.green }, { key: "e2e_p95", label: "p95", color: C.amber }, { key: "e2e_p99", label: "p99 (≥100 samples)", color: C.red }]} />
        </Card>
        <Card title="TTFT and TPOT, p50 (ms)">
          <ConcChart data={data} series={[{ key: "ttft", label: "TTFT (left)", color: C.violet }, { key: "tpot", label: "TPOT (right)", color: C.green, axis: "right" }]} />
        </Card>
        <Card title="GPU memory (GB) and utilization">
          <ConcChart data={data} series={[
            { key: "mem", label: "Device-wide peak GB", color: C.blue }, ...(hasSrvMem ? [{ key: "srv", label: "Server process GB (attributed)", color: C.green }] : []),
            { key: "util", label: "Peak util % (right)", color: C.amber, axis: "right" as const, dashed: true }]} />
          <p className="mt-1 text-xs text-zinc-500">Device-wide memory includes every process on the GPU and vLLM&apos;s preallocated KV cache{hasSrvMem ? "; the server line is per-process via NVML for the managed container" : ""}.</p>
        </Card>
        <Card title="Failed requests (%)" className="lg:col-span-2">
          <div style={{ height: 160 }}><ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 6, right: 12, bottom: 16, left: 0 }}>
              <CartesianGrid stroke="#27272a" strokeDasharray="3 3" /><XAxis dataKey="c" stroke="#71717a" fontSize={11} label={{ value: "concurrency", position: "insideBottom", offset: -8, fill: "#71717a", fontSize: 11 }} />
              <YAxis stroke="#71717a" fontSize={11} width={40} domain={[0, 100]} unit="%" /><Tooltip contentStyle={TIP} formatter={(v) => `${Number(v).toFixed(1)}%`} labelFormatter={(c) => `concurrency ${c}`} />
              <Bar dataKey="fail" name="failed" fill={C.red} isAnimationActive={false} />
            </BarChart></ResponsiveContainer></div>
        </Card>
      </div>

      <Card title="Per-level results" className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-sm tabular-nums">
          <thead className="text-left text-xs text-zinc-500"><tr>
            {["c", "requests × iters", "success", "output tok/s (±95% CI)", "scaling eff.", "TTFT p50", "TPOT p50", "E2E p50", "E2E p95", "E2E p99", "peak mem"].map((h) => <th key={h} className="py-1 pr-3 font-medium">{h}</th>)}</tr></thead>
          <tbody className="divide-y divide-zinc-800">
            {levels.map((l) => l.status !== "ok" ? (
              <tr key={l.concurrency} className="text-red-300"><td className="py-1.5 pr-3">{l.concurrency}</td><td colSpan={10}>failed: {l.error}</td></tr>
            ) : (
              <tr key={l.concurrency}>
                <td className="py-1.5 pr-3 font-medium">{l.concurrency}</td>
                <td className="pr-3">{l.num_requests} × {l.iterations_run}</td>
                <td className={`pr-3 ${l.requests.success_rate < 1 ? "text-amber-300" : ""}`}>{pct(l.requests.success_rate)}</td>
                <td className="pr-3">{fmtNum(l.stats.aggregate_output_tok_s.mean, 0)}{l.stats.aggregate_output_tok_s.ci95_half != null ? <span className="text-xs text-zinc-500"> ±{fmtNum(l.stats.aggregate_output_tok_s.ci95_half, 0)}</span> : <span className="text-xs text-zinc-600"> no CI</span>}
                  {(l.stats.aggregate_output_tok_s.cv ?? 0) > 0.1 && <span title="throughput varies >10% across iterations"> <Pill tone="amber">noisy</Pill></span>}</td>
                <td className="pr-3">{l.scaling_efficiency != null ? l.scaling_efficiency.toFixed(2) : "—"}</td>
                <td className="pr-3">{fmtMs(l.latency.ttft_ms?.p50)}</td><td className="pr-3">{fmtMs(l.latency.tpot_ms?.p50)}</td>
                <td className="pr-3">{fmtMs(l.latency.e2e_ms?.p50)}</td>
                <td className="pr-3">{l.latency.e2e_ms?.p95 != null ? fmtMs(l.latency.e2e_ms.p95) : <span title={`needs ≥20 successful requests (have ${l.latency.e2e_ms?.n})`} className="text-zinc-600">n too small</span>}</td>
                <td className="pr-3">{l.latency.e2e_ms?.p99 != null ? fmtMs(l.latency.e2e_ms.p99) : <span title={`needs ≥100 successful requests (have ${l.latency.e2e_ms?.n})`} className="text-zinc-600">n too small</span>}</td>
                <td className="pr-3">{fmtMb(l.gpu.peak_memory_mb)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {a.steps.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-zinc-400">
            {a.steps.map((s: any) => (
              <span key={s.to} className="rounded-md bg-zinc-800 px-2 py-1">{s.from}→{s.to}: {s.gain_pct == null ? "—" : `${s.gain_pct >= 0 ? "+" : ""}${s.gain_pct.toFixed(1)}%`}
                {s.significant === false && " (within noise)"}{s.significant === true && " (significant)"}</span>))}
          </div>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Configuration">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-zinc-500">Mode</dt><dd>{r.mode}</dd>
            <dt className="text-zinc-500">Model</dt><dd className="break-all">{r.model_registry?.name ?? r.runtime.served_model.id}{r.model_registry?.quantization ? ` · ${r.model_registry.quantization}` : ""}</dd>
            <dt className="text-zinc-500">Revision</dt><dd className="break-all font-mono text-xs">{r.model_registry?.revision ?? "unknown (external server)"}</dd>
            <dt className="text-zinc-500">Input / output</dt><dd>{r.workload.input_tokens} / {r.workload.output_tokens} tokens ({r.workload.prompt_mode})</dd>
            <dt className="text-zinc-500">Per level</dt><dd>≥{r.workload.requests_per_level} requests (≥{r.workload.min_requests_per_slot} per slot) × {r.workload.iterations} iterations, {r.workload.warmup_requests}+ warmup</dd>
            <dt className="text-zinc-500">Decoding</dt><dd>T={r.workload.decoding.temperature}, top_p={r.workload.decoding.top_p}, ignore_eos, seed {r.workload.seed}</dd>
            <dt className="text-zinc-500">Seeds</dt><dd className="text-xs text-zinc-400">{r.workload.seed_scheme}</dd>
            {r.managed_server && (<><dt className="text-zinc-500">vLLM config</dt><dd className="text-xs">ctx {r.managed_server.config.max_model_len}, mem {r.managed_server.config.gpu_memory_utilization}, dtype {r.managed_server.config.dtype}, quant {r.managed_server.config.quantization}, kv {r.managed_server.config.kv_cache_dtype}, seqs {r.managed_server.config.max_num_seqs}</dd></>)}
          </dl>
        </Card>
        <Card title="Environment & contention">
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-zinc-500">vLLM</dt><dd>{r.runtime.version ?? "unknown"}</dd>
            <dt className="text-zinc-500">KV cache</dt><dd>{r.runtime.kv_cache_tokens ? `${fmtInt(r.runtime.kv_cache_tokens)} tokens (server-reported)` : "unknown"}</dd>
            <dt className="text-zinc-500">Max model length</dt><dd>{fmtInt(r.runtime.served_model?.max_model_len)}</dd>
            {gp.preflight && (<><dt className="text-zinc-500">GPU at launch</dt><dd>{gp.preflight.name} · {fmtMb(gp.preflight.free_mb)} free of {fmtMb(gp.preflight.total_mb)} · util {gp.preflight.utilization_pct?.toFixed(0)}%</dd></>)}
            <dt className="text-zinc-500">Contention</dt><dd>{gp.contended ? <Pill tone="amber">contended{gp.contention_override ? " · override" : ""}</Pill> : <Pill tone="green">none detected</Pill>}</dd>
            <dt className="text-zinc-500">Memory scope</dt><dd className="text-xs text-zinc-400">{gp.scope}</dd>
          </dl>
          <p className="mt-2 text-xs text-zinc-500">Environment snapshot (OS, driver, package versions, git commit) is in the experiment record below.</p>
        </Card>
      </div>

      {ms.data && <ServerCard ms={ms.data} />}

      {ok.length > 0 && rows.length > 0 && (
        <Card title="Per-request measurements" action={
          <select className="rounded border border-zinc-700 bg-zinc-900 px-2 py-1 text-xs" value={selLevel} onChange={(e) => setLv(Number(e.target.value))}>
            {ok.map((l) => <option key={l.concurrency} value={l.concurrency}>concurrency {l.concurrency}</option>)}</select>}>
          <div className="max-h-72 overflow-auto"><table className="w-full min-w-[560px] text-xs tabular-nums">
            <thead className="sticky top-0 bg-zinc-900 text-left text-zinc-500"><tr>{["#", "iter", "status", "TTFT ms", "TPOT ms", "E2E ms", "tok/s"].map((h) => <th key={h} className="py-1 pr-2 font-medium">{h}</th>)}</tr></thead>
            <tbody className="divide-y divide-zinc-800">{rows.slice(0, 200).map((x: any, i: number) => (
              <tr key={i} className={x.ok ? "" : "text-red-300"}><td className="py-1 pr-2">{x.index + 1}</td><td>{x.iteration + 1}</td><td title={x.error_message ?? ""}>{x.ok ? "ok" : x.error_type}</td>
                <td>{fmtNum(x.ttft_ms, 1)}</td><td>{fmtNum(x.tpot_ms, 2)}</td><td>{fmtNum(x.e2e_ms, 1)}</td><td>{fmtNum(x.output_tok_s, 1)}</td></tr>))}</tbody></table></div>
        </Card>
      )}
      <Card title="Metric definitions"><details><summary className="cursor-pointer text-sm text-zinc-400">How each number is measured</summary>
        <dl className="mt-3 space-y-2 text-sm">{Object.entries<string>(r.definitions).map(([k, d]) => <div key={k}><dt className="font-mono text-xs text-zinc-300">{k}</dt><dd className="text-zinc-500">{d}</dd></div>)}</dl></details></Card>
    </div>
  );
}
