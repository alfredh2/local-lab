"use client";
import Link from "next/link";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ExperimentDetail } from "@/lib/api";
import { fmtMb, fmtMs } from "@/lib/format";
import { Card, Pill } from "@/components/ui";

const COLORS = ["#60a5fa", "#fbbf24", "#34d399", "#f87171", "#c084fc", "#22d3ee"];
const TIP = { background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 };

export default function SweepCompare({ runs }: { runs: ExperimentDetail[] }) {
  const rs = runs.map((e) => ({ e, r: e.jobs.at(-1)!.results! }));
  const label = (x: (typeof rs)[number]) => `#${x.e.id} ${x.r.model_registry?.name?.split("/").pop() ?? x.r.runtime.served_model.id}`;
  const conc = [...new Set(rs.flatMap((x) => x.r.levels.map((l: any) => l.concurrency as number)))].sort((a, b) => a - b);
  const pick = (x: (typeof rs)[number], c: number) => x.r.levels.find((l: any) => l.concurrency === c && l.status === "ok");
  const series = (f: (l: any) => number | null | undefined) =>
    conc.map((c) => ({ c, ...Object.fromEntries(rs.map((x) => { const l = pick(x, c); return [label(x), l ? f(l) ?? null : null]; })) }));
  const chart = (title: string, data: any[], unit = "") => (
    <Card title={title}>
      <div style={{ height: 240 }}><ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 16, left: 0 }}>
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
          <XAxis dataKey="c" stroke="#71717a" fontSize={11} label={{ value: "concurrency", position: "insideBottom", offset: -8, fill: "#71717a", fontSize: 11 }} />
          <YAxis stroke="#71717a" fontSize={11} width={52} unit={unit} />
          <Tooltip contentStyle={TIP} formatter={(v) => (typeof v === "number" ? v.toFixed(1) : String(v))} labelFormatter={(c) => `concurrency ${c}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
          {rs.map((x, i) => <Line key={x.e.id} dataKey={label(x)} stroke={COLORS[i]} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />)}
        </LineChart></ResponsiveContainer></div>
    </Card>
  );
  const facts: [string, (r: any) => string][] = [
    ["Model", (r) => r.model_registry?.name ?? r.runtime.served_model.id], ["Revision", (r) => r.model_registry?.revision?.slice(0, 10) ?? "—"],
    ["Quantization", (r) => r.managed_server?.config.quantization ?? r.model_registry?.quantization ?? "—"], ["dtype", (r) => r.managed_server?.config.dtype ?? "—"],
    ["KV dtype", (r) => r.managed_server?.config.kv_cache_dtype ?? "—"], ["Context length", (r) => String(r.managed_server?.config.max_model_len ?? r.runtime.served_model?.max_model_len ?? "—")],
    ["GPU memory share", (r) => String(r.managed_server?.config.gpu_memory_utilization ?? "—")], ["vLLM", (r) => r.runtime.version ?? "—"],
    ["Input / output", (r) => `${r.workload.input_tokens} / ${r.workload.output_tokens}`], ["Per level", (r) => `${r.workload.requests_per_level} req × ${r.workload.iterations} it`],
    ["Seed", (r) => String(r.workload.seed)], ["KV cache tokens", (r) => String(r.runtime.kv_cache_tokens ?? "?")],
  ];
  const differing = facts.filter(([, f]) => new Set(rs.map((x) => f(x.r))).size > 1);
  const workloadDiff = differing.some(([n]) => ["Input / output", "Per level", "Seed"].includes(n));
  const anyContended = rs.filter((x) => x.r.gpu.contended);

  return (
    <div className="space-y-6">
      {anyContended.length > 0 && <div className="rounded-xl border border-amber-800 bg-amber-950/30 p-3 text-sm text-amber-200">
        Contended GPU in {anyContended.map((x) => `#${x.e.id}`).join(", ")}: differences between runs may reflect other GPU work rather than the configuration.</div>}
      {workloadDiff && <div className="rounded-xl border border-amber-900 bg-amber-950/30 p-3 text-sm text-amber-300">These sweeps used different workloads ({differing.map(([n]) => n).join(", ")}).</div>}
      <div className="grid gap-4 lg:grid-cols-2">
        {chart("Throughput vs concurrency (aggregate output tok/s)", series((l) => l.stats.aggregate_output_tok_s.mean))}
        {chart("End-to-end p95 latency vs concurrency (ms)", series((l) => l.latency.e2e_ms?.p95 ?? l.latency.e2e_ms?.p50))}
        {chart("TTFT p50 vs concurrency (ms)", series((l) => l.stats.ttft_p50_ms.mean))}
        {chart("TPOT p50 vs concurrency (ms)", series((l) => l.stats.tpot_p50_ms.mean))}
      </div>
      <Card title="Throughput by level (tok/s, mean ± 95% CI) and change vs first run" className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm tabular-nums">
          <thead className="text-left text-xs text-zinc-500"><tr><th className="py-1 font-medium">c</th>
            {rs.map((x, i) => <th key={x.e.id} className="py-1 pr-3 font-medium"><span className="mr-1.5 inline-block h-2 w-2 rounded-full" style={{ background: COLORS[i] }} />
              <Link href={`/experiments/${x.e.id}`} className="hover:underline">{label(x)}</Link>{x.r.gpu.contended && <> <Pill tone="amber">contended</Pill></>}</th>)}</tr></thead>
          <tbody className="divide-y divide-zinc-800">
            {conc.map((c) => { const base = pick(rs[0], c)?.stats.aggregate_output_tok_s.mean; return (
              <tr key={c}><td className="py-1.5 pr-3 font-medium">{c}</td>
                {rs.map((x, i) => { const l = pick(x, c); const s = l?.stats.aggregate_output_tok_s; const d = i > 0 && base && s ? ((s.mean - base) / base) * 100 : null; return (
                  <td key={x.e.id} className="pr-3">{s ? <>{s.mean.toFixed(0)}{s.ci95_half != null && <span className="text-xs text-zinc-500"> ±{s.ci95_half.toFixed(0)}</span>}
                    {d != null && <span className={`ml-1.5 text-xs ${d >= 0 ? "text-emerald-400" : "text-zinc-500"}`}>{d >= 0 ? "+" : ""}{d.toFixed(0)}%</span>}
                    <div className="text-xs text-zinc-500">E2E p50 {fmtMs(l.latency.e2e_ms?.p50)} · peak {fmtMb(l.gpu.peak_memory_mb)}</div></> : <span className="text-zinc-600">—</span>}</td>); })}</tr>); })}
          </tbody>
        </table>
      </Card>
      <Card title="Scaling summary"><table className="w-full text-sm"><thead className="text-left text-xs text-zinc-500"><tr><th className="py-1 font-medium">Run</th><th className="font-medium">Peak throughput</th><th className="font-medium">Saturation</th><th className="font-medium">Quality flags</th></tr></thead>
        <tbody className="divide-y divide-zinc-800">{rs.map((x) => <tr key={x.e.id}><td className="py-1.5 pr-3">{label(x)}</td>
          <td className="pr-3">c={x.r.analysis.peak_throughput_concurrency}</td>
          <td className="pr-3">{x.r.analysis.saturation_after_concurrency ? `after c=${x.r.analysis.saturation_after_concurrency}` : "not observed"}</td>
          <td>{x.r.analysis.reliable ? <Pill tone="green">none</Pill> : <Pill tone="amber">see run</Pill>}</td></tr>)}</tbody></table></Card>
      <Card title={differing.length ? "What differs between runs" : "Runs share the same configuration"}>
        <table className="w-full text-sm"><thead className="text-left text-xs text-zinc-500"><tr><th className="py-1 font-medium">Setting</th>{rs.map((x) => <th key={x.e.id} className="py-1 font-medium">{label(x)}</th>)}</tr></thead>
          <tbody className="divide-y divide-zinc-800">{(differing.length ? differing : facts).map(([n, f]) => <tr key={n}><td className="py-1.5 pr-3 text-zinc-300">{n}</td>{rs.map((x) => <td key={x.e.id} className="pr-3 break-all">{f(x.r)}</td>)}</tr>)}</tbody></table>
      </Card>
    </div>
  );
}
