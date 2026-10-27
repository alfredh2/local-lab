"use client";
import Link from "next/link";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { useQueries } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type ExperimentDetail } from "@/lib/api";
import { Card, Empty, ErrorBox } from "@/components/ui";

const COLORS = ["#60a5fa", "#fbbf24", "#34d399", "#f87171", "#c084fc", "#22d3ee"];
type Metric = { label: string; unit: string; higherIsBetter: boolean; get: (r: any) => number | null | undefined; digits: number };
const METRICS: Metric[] = [
  { label: "Aggregate output tok/s", unit: "", higherIsBetter: true, digits: 1, get: (r) => r.summary.throughput.aggregate_output_tok_s },
  { label: "Total tok/s (in+out)", unit: "", higherIsBetter: true, digits: 1, get: (r) => r.summary.throughput.total_tok_s },
  { label: "Requests/s", unit: "", higherIsBetter: true, digits: 2, get: (r) => r.summary.throughput.requests_per_s },
  { label: "Success rate", unit: "%", higherIsBetter: true, digits: 1, get: (r) => r.summary.requests.success_rate * 100 },
  { label: "TTFT p50", unit: " ms", higherIsBetter: false, digits: 1, get: (r) => r.summary.latency.ttft_ms?.p50 },
  { label: "TTFT p95", unit: " ms", higherIsBetter: false, digits: 1, get: (r) => r.summary.latency.ttft_ms?.p95 },
  { label: "TPOT p50", unit: " ms", higherIsBetter: false, digits: 2, get: (r) => r.summary.latency.tpot_ms?.p50 },
  { label: "TPOT p95", unit: " ms", higherIsBetter: false, digits: 2, get: (r) => r.summary.latency.tpot_ms?.p95 },
  { label: "E2E latency p50", unit: " ms", higherIsBetter: false, digits: 0, get: (r) => r.summary.latency.e2e_ms?.p50 },
  { label: "E2E latency p95", unit: " ms", higherIsBetter: false, digits: 0, get: (r) => r.summary.latency.e2e_ms?.p95 },
  { label: "Decode tok/s per request (p50)", unit: "", higherIsBetter: true, digits: 1, get: (r) => r.summary.throughput.per_request_decode_tok_s?.p50 },
  { label: "Peak GPU memory", unit: " MB", higherIsBetter: false, digits: 0, get: (r) => r.gpu.per_device["0"]?.peak_memory_mb },
];

function CompareInner() {
  const ids = (useSearchParams().get("ids") ?? "").split(",").map(Number).filter(Boolean).slice(0, 6);
  const results = useQueries({ queries: ids.map((id) => ({ queryKey: ["experiment", String(id)], queryFn: () => api<ExperimentDetail>(`/experiments/${id}`) })) });
  const err = results.find((r) => r.isError);
  if (ids.length < 2) return <Empty title="Pick at least two runs">Select completed inference benchmarks on the <Link href="/experiments" className="text-blue-400 hover:underline">Experiments</Link> page.</Empty>;
  if (err) return <ErrorBox error={err.error} />;
  if (results.some((r) => !r.data)) return <div className="h-32 animate-pulse rounded-xl bg-zinc-900" />;
  const runs = results.map((r) => r.data!).map((e) => ({ e, r: e.jobs.at(-1)?.results, ok: e.type === "inference_benchmark" && e.jobs.at(-1)?.status === "completed" }));
  const bad = runs.filter((x) => !x.ok);
  if (bad.length) return <ErrorBox error={`Only completed inference benchmarks can be compared. Not comparable: ${bad.map((x) => "#" + x.e.id).join(", ")}`} />;
  const label = (x: (typeof runs)[number]) => `#${x.e.id} ${x.r!.runtime.served_model.id}`;

  // Configuration fields that differ between runs; the comparison is only meaningful relative to these.
  const facts: [string, (r: any) => string][] = [
    ["Model", (r) => r.runtime.served_model.id], ["Model root", (r) => r.runtime.served_model.root ?? "—"],
    ["Registry quantization", (r) => r.model_registry?.quantization ?? "—"], ["vLLM version", (r) => r.runtime.version ?? "—"],
    ["Profile", (r) => r.workload.profile], ["Input tokens", (r) => String(r.workload.input_tokens)], ["Output tokens", (r) => String(r.workload.output_tokens)],
    ["Requests × iterations", (r) => `${r.workload.num_requests} × ${r.workload.iterations}`], ["Concurrency", (r) => String(r.workload.concurrency)],
    ["Temperature", (r) => String(r.workload.decoding.temperature)], ["Seed", (r) => String(r.workload.seed)], ["Prompt mode", (r) => r.workload.prompt_mode],
    ["KV cache tokens", (r) => String(r.runtime.kv_cache_tokens ?? "unknown")],
  ];
  const differing = facts.filter(([, f]) => new Set(runs.map((x) => f(x.r))).size > 1);
  const workloadDiffers = differing.some(([n]) => ["Profile", "Input tokens", "Output tokens", "Requests × iterations", "Concurrency", "Prompt mode"].includes(n));
  const barCard = (title: string, ms: Metric[]) => (
    <Card title={title}>
      <div style={{ height: 240 }}><ResponsiveContainer width="100%" height="100%">
        <BarChart data={ms.map((m) => ({ name: m.label, ...Object.fromEntries(runs.map((x) => [label(x), m.get(x.r) ?? null])) }))} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" /><XAxis dataKey="name" stroke="#71717a" fontSize={11} /><YAxis stroke="#71717a" fontSize={11} width={52} />
          <Tooltip contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }} /><Legend wrapperStyle={{ fontSize: 12 }} />
          {runs.map((x, i) => <Bar key={x.e.id} dataKey={label(x)} fill={COLORS[i]} isAnimationActive={false} />)}
        </BarChart></ResponsiveContainer></div>
    </Card>
  );
  const best = (m: Metric) => { const vals = runs.map((x) => m.get(x.r)).map((v) => (v == null ? NaN : v)); const f = m.higherIsBetter ? Math.max : Math.min; const b = f(...vals.filter((v) => !isNaN(v))); return vals.map((v) => v === b && runs.length > 1 && new Set(vals).size > 1); };
  return (
    <div className="space-y-6">
      <div><Link href="/experiments" className="text-xs text-zinc-500 hover:underline">← Experiments</Link><h1 className="mt-1 text-xl font-semibold">Compare benchmarks</h1></div>
      {workloadDiffers && <div className="rounded-xl border border-amber-900 bg-amber-950/30 p-3 text-sm text-amber-300">
        These runs used different workloads ({differing.map(([n]) => n).join(", ")}). Differences below mix model/runtime effects with workload effects.</div>}
      <Card title="Metrics" className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm tabular-nums">
          <thead className="text-left text-xs text-zinc-500"><tr><th className="py-1 font-medium">Metric</th>
            {runs.map((x, i) => <th key={x.e.id} className="py-1 pr-3 font-medium"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COLORS[i] }} />
              <Link href={`/experiments/${x.e.id}`} className="hover:underline">{label(x)}</Link></th>)}</tr></thead>
          <tbody className="divide-y divide-zinc-800">
            {METRICS.map((m) => { const b = best(m); const base = m.get(runs[0].r); return (
              <tr key={m.label}><td className="py-1.5 pr-3 text-zinc-300">{m.label}</td>
                {runs.map((x, i) => { const v = m.get(x.r); const d = i > 0 && v != null && base ? ((v - base) / base) * 100 : null; return (
                  <td key={x.e.id} className={`pr-3 ${b[i] ? "font-semibold text-emerald-300" : ""}`}>
                    {v == null ? "—" : v.toFixed(m.digits) + m.unit}
                    {d != null && <span className="ml-1.5 text-xs text-zinc-500">{d >= 0 ? "+" : ""}{d.toFixed(1)}%</span>}</td>); })}</tr>); })}
          </tbody>
        </table>
        <p className="mt-2 text-xs text-zinc-500">Percentages are relative to the first run. Green marks the best value per row. Latencies are over successful requests only.</p>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        {barCard("Throughput (tok/s)", [METRICS[0], METRICS[1]])}
        {barCard("Latency (ms)", [METRICS[4], METRICS[5], METRICS[8], METRICS[9]])}
      </div>
      <Card title={differing.length ? "What differs between runs" : "Runs share the same configuration"}>
        <table className="w-full text-sm"><thead className="text-left text-xs text-zinc-500"><tr><th className="py-1 font-medium">Setting</th>{runs.map((x) => <th key={x.e.id} className="py-1 font-medium">{label(x)}</th>)}</tr></thead>
          <tbody className="divide-y divide-zinc-800">{(differing.length ? differing : facts).map(([n, f]) => (
            <tr key={n}><td className="py-1.5 pr-3 text-zinc-300">{n}</td>{runs.map((x) => <td key={x.e.id} className="pr-3 break-all">{f(x.r)}</td>)}</tr>))}</tbody></table>
      </Card>
    </div>
  );
}

export default function ComparePage() { return <Suspense><CompareInner /></Suspense>; }
