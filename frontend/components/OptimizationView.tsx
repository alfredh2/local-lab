"use client";
import Link from "next/link";
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CartesianGrid, LabelList, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { api, API_URL, type ConfigResultRec, type ExperimentDetail, type ManagedServerRec, type MetricInfo, type ParetoResponse } from "@/lib/api";
import { fmtDuration } from "@/lib/format";
import { btn, btnGhost, Card, Empty, ErrorBox, Pill, StatusBadge } from "@/components/ui";

const TIP = { background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 };
const STATUS_TONE: Record<string, "zinc" | "green" | "red" | "amber"> = { completed: "green", failed: "red", skipped: "zinc", cancelled: "amber", pending: "zinc", running: "amber" };
const QUALITY = ["structured_schema_rate", "coding_pass_rate", "retrieval_accuracy"];
const PERF = ["throughput_tok_s", "ttft_p50_ms", "tpot_p50_ms", "e2e_p95_ms", "peak_vram_mb", "server_vram_mb"];

function fmtMetric(m: MetricInfo | undefined, v: number | null | undefined) {
  if (v == null || !m) return "—";
  if (m.percent) return `${v.toFixed(v < 10 ? 1 : 0)}%`;
  if (m.unit === "tok/s" || m.unit === "MB" || m.unit === "tokens") return Math.round(v).toLocaleString();
  return v.toFixed(v < 10 ? 2 : 1);
}
const why = (r: ConfigResultRec) => (r.error && r.failure_kind && r.error.startsWith(r.failure_kind) ? r.error : `${r.failure_kind ?? ""}${r.error ? ": " + r.error : ""}`);
const hasData = (r: ConfigResultRec) => !!(r.results?.performance || r.results?.quality);
const short = (l: string) => l.replace("Qwen/", "").replace(" · ctx ", " ctx ").replace("Instruct", "I");

export default function OptimizationView({ id }: { id: number }) {
  const qc = useQueryClient();
  const exp = useQuery({ queryKey: ["experiment", String(id)], queryFn: () => api<ExperimentDetail>(`/experiments/${id}`),
    refetchInterval: (q) => { const s = q.state.data?.jobs.at(-1)?.status; return s && ["completed", "failed", "cancelled"].includes(s) ? false : 2000; } });
  const job = exp.data?.jobs.at(-1);
  const active = !!job && !["completed", "failed", "cancelled"].includes(job.status);
  const rows = useQuery({ queryKey: ["opt-rows", id], queryFn: () => api<ConfigResultRec[]>(`/optimization/${id}`), refetchInterval: active ? 3000 : false });
  const cancel = useMutation({ mutationFn: () => api(`/jobs/${job!.id}/cancel`, { method: "POST" }), onSuccess: () => qc.invalidateQueries({ queryKey: ["experiment", String(id)] }) });
  const resume = useMutation({ mutationFn: () => api(`/experiments/${id}/resume`, { method: "POST" }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["experiment", String(id)] }); qc.invalidateQueries({ queryKey: ["opt-rows", id] }); } });

  const [io, setIo] = useState(0);
  const [conc, setConc] = useState<number | null>(null);
  if (exp.isError) return <ErrorBox error={exp.error} />;
  if (!exp.data || !job) return <div className="h-24 animate-pulse rounded-xl bg-zinc-900" />;
  const list = rows.data ?? [];
  const done = list.filter((r) => r.status === "completed");
  const unfinished = list.some((r) => !["completed", "skipped"].includes(r.status));
  const q = `io=${io}${conc != null ? `&concurrency=${conc}` : ""}`;

  return (
    <div className="space-y-6">
      <div>
        <Link href="/optimization" className="text-xs text-zinc-500 hover:underline">← Optimization Lab</Link>
        <div className="mt-1 flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-semibold">{exp.data.name}</h1><StatusBadge status={job.status} />
          {active && <button className={btnGhost} disabled={cancel.isPending || job.cancel_requested} onClick={() => cancel.mutate()}>{job.cancel_requested ? "Cancelling…" : "Cancel"}</button>}
          {!active && unfinished && <button className={btn} disabled={resume.isPending} onClick={() => resume.mutate()}>Resume unfinished</button>}
          {done.length > 0 && <>
            <a className={btnGhost} href={`${API_URL}/api/optimization/${id}/export?format=csv&${q}`} download>Export CSV</a>
            <a className={btnGhost} href={`${API_URL}/api/optimization/${id}/export?format=json`} download>Export JSON</a></>}
        </div>
        <p className="mt-1 text-xs text-zinc-500">Experiment #{id} · job #{job.id}{exp.data.jobs.length > 1 ? ` (attempt ${exp.data.jobs.length})` : ""} · {fmtDuration(job.started_at, job.finished_at)}</p>
      </div>
      {(cancel.isError || resume.isError) && <ErrorBox error={cancel.error ?? resume.error} />}
      {job.status === "failed" && job.error && <Card title="Job error"><pre className="max-h-48 overflow-auto whitespace-pre-wrap text-xs text-red-300">{job.error}</pre></Card>}
      {active && <ProgressCard job={job} rows={list} />}
      {list.length > 0 && <StatusTable rows={list} />}
      {done.length === 0 ? (
        !active && <Empty title="No configuration completed">There are no measurements to compare. See the failure analysis below for why.</Empty>
      ) : (
        <>
          <Lab id={id} io={io} setIo={setIo} conc={conc} setConc={setConc} />
          <QualityDetails rows={done} />
        </>
      )}
      <Failures rows={list} />
    </div>
  );
}

function ProgressCard({ job, rows }: { job: any; rows: ConfigResultRec[] }) {
  const p = job.progress ?? {};
  const m = p.matrix;
  let stage = "Planning and validating configurations…";
  if (p.phase === "launching_server") stage = "Launching the vLLM container";
  else if (p.phase === "waiting_for_server") stage = `Waiting for the model to load (${p.elapsed_s}s of ${p.timeout_s}s)`;
  else if (String(p.phase).startsWith("eval:")) stage = `Evaluating ${String(p.phase).slice(5)}: ${p.done}/${p.total} cases scored, ${p.ok} passed`;
  else if (p.phase === "warmup" || p.phase === "measured") stage = `Performance at concurrency ${p.sweep?.concurrency}: ${p.phase} ${p.done}/${p.total}${p.live_output_tok_s ? ` · ${p.live_output_tok_s} tok/s` : ""}`;
  return (
    <Card title={m ? `Configuration ${m.config_index + 1} of ${m.configs_total}: ${m.label}` : "Starting"}>
      <div className="text-sm text-zinc-300">{stage}</div>
      <div className="mt-3 flex flex-wrap gap-1.5">{rows.map((r) => <span key={r.key} title={r.label}><Pill tone={STATUS_TONE[r.status]}>{r.status === "running" ? "● " : ""}{short(r.label)}</Pill></span>)}</div>
    </Card>
  );
}

function StatusTable({ rows }: { rows: ConfigResultRec[] }) {
  return (
    <Card title="Configurations" className="overflow-x-auto p-0">
      <table className="w-full min-w-[640px] text-sm"><thead className="text-left text-xs text-zinc-500"><tr className="border-b border-zinc-800">
        {["Configuration", "Status", "Attempts", "Startup", "Note"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-zinc-800">{rows.map((r) => (
          <tr key={r.key}><td className="px-4 py-2">{r.label}</td><td className="px-4 py-2"><Pill tone={STATUS_TONE[r.status]}>{r.status}</Pill></td>
            <td className="px-4 py-2 tabular-nums">{r.attempts}</td><td className="px-4 py-2 tabular-nums">{r.results?.startup_s ? `${r.results.startup_s.toFixed(0)}s` : "—"}</td>
            <td className="max-w-md truncate px-4 py-2 text-xs text-zinc-500" title={r.error ?? ""}>{r.status === "completed" ? (r.results?.gpu?.contended ? "contended GPU" : "") : why(r)}</td></tr>))}</tbody></table>
    </Card>
  );
}

function Lab({ id, io, setIo, conc, setConc }: { id: number; io: number; setIo: (n: number) => void; conc: number | null; setConc: (n: number | null) => void }) {
  const [objs, setObjs] = useState<Record<string, "max" | "min">>({ coding_pass_rate: "max", throughput_tok_s: "max" });
  const [cons, setCons] = useState<{ metric: string; op: string; value: string }[]>([]);
  const [useW, setUseW] = useState(false);
  const [w, setW] = useState<Record<string, number>>({});
  const [x, setX] = useState("throughput_tok_s");
  const [y, setY] = useState("coding_pass_rate");
  const params = new URLSearchParams({ io: String(io) });
  if (conc != null) params.set("concurrency", String(conc));
  const oStr = Object.entries(objs).map(([m, d]) => `${m}:${d}`).join(",");
  if (oStr) params.set("objectives", oStr);
  const cStr = cons.filter((c) => c.value !== "" && !isNaN(Number(c.value))).map((c) => `${c.metric}:${c.op}:${c.value}`).join(",");
  if (cStr) params.set("constraints", cStr);
  if (useW && oStr) params.set("weights", Object.keys(objs).map((m) => `${m}:${w[m] ?? 1}`).join(","));
  const res = useQuery({ queryKey: ["pareto", id, params.toString()], queryFn: () => api<ParetoResponse>(`/optimization/${id}/pareto?${params}`), placeholderData: (p) => p });
  if (res.isError) return <ErrorBox error={res.error} />;
  const d = res.data;
  if (!d) return <div className="h-40 animate-pulse rounded-xl bg-zinc-900" />;
  const mi = Object.fromEntries(d.metrics.map((m) => [m.id, m]));
  const a = d.analysis;
  const label = (k: string) => d.points.find((p) => p.id === k)?.label ?? k;

  return (
    <>
      <Card title="Comparison at one workload" action={
        <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400">
          {d.cell.io_pairs.length > 1 && <label>I/O <select className="rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5" value={io} onChange={(e) => setIo(Number(e.target.value))}>
            {d.cell.io_pairs.map((p, i) => <option key={i} value={i}>{p.input_tokens}/{p.output_tokens}</option>)}</select></label>}
          {d.cell.common_levels.length > 0 && <label>concurrency <select className="rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5" value={d.cell.concurrency ?? ""} onChange={(e) => setConc(Number(e.target.value))}>
            {d.cell.common_levels.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>}
        </div>}>
        {d.notes.map((n) => <div key={n} className="mb-1 text-xs text-amber-300">⚠ {n}</div>)}
        <ComparisonTable d={d} mi={mi} front={a?.front} />
        <p className="mt-2 text-xs text-zinc-500">Quality: rate with 95% Wilson interval and number of cases. Performance: synthetic prompts at the selected cell, mean ± 95% CI across iterations. VRAM is device-wide unless marked server.</p>
      </Card>

      <Card title="Pareto analysis">
        <div className="grid gap-5 lg:grid-cols-[300px_1fr]">
          <div className="space-y-4 text-sm">
            <div>
              <div className="mb-1 font-medium text-zinc-300">Objectives <span className="text-xs font-normal text-zinc-500">pick what matters</span></div>
              {d.metrics.map((m) => (
                <div key={m.id} className="flex items-center justify-between gap-2 py-0.5">
                  <label className="flex items-center gap-2"><input type="checkbox" checked={m.id in objs} onChange={(e) => { const n = { ...objs }; if (e.target.checked) n[m.id] = m.direction; else delete n[m.id]; setObjs(n); }} />
                    <span className="text-xs">{m.label}</span></label>
                  {m.id in objs && <select className="rounded border border-zinc-700 bg-zinc-900 px-1 text-xs" value={objs[m.id]} onChange={(e) => setObjs({ ...objs, [m.id]: e.target.value as "max" | "min" })}><option value="max">maximize</option><option value="min">minimize</option></select>}
                </div>))}
            </div>
            <div>
              <div className="mb-1 font-medium text-zinc-300">Constraints <span className="text-xs font-normal text-zinc-500">hard limits</span></div>
              {cons.map((c, i) => (
                <div key={i} className="mb-1 flex gap-1">
                  <select className="w-28 rounded border border-zinc-700 bg-zinc-900 px-1 text-xs" value={c.metric} onChange={(e) => setCons(cons.map((x2, j) => (j === i ? { ...x2, metric: e.target.value } : x2)))}>{d.metrics.map((m) => <option key={m.id} value={m.id}>{m.label.slice(0, 28)}</option>)}</select>
                  <select className="rounded border border-zinc-700 bg-zinc-900 px-1 text-xs" value={c.op} onChange={(e) => setCons(cons.map((x2, j) => (j === i ? { ...x2, op: e.target.value } : x2)))}>{[">=", "<=", ">", "<"].map((o) => <option key={o}>{o}</option>)}</select>
                  <input className="w-20 rounded border border-zinc-700 bg-zinc-900 px-1 text-xs" value={c.value} placeholder={mi[c.metric]?.percent ? "% e.g. 70" : "value"} onChange={(e) => setCons(cons.map((x2, j) => (j === i ? { ...x2, value: e.target.value } : x2)))} />
                  <button type="button" className="text-xs text-zinc-500 hover:text-red-300" onClick={() => setCons(cons.filter((_, j) => j !== i))}>✕</button>
                </div>))}
              <button type="button" className="text-xs text-blue-400 hover:underline" onClick={() => setCons([...cons, { metric: d.metrics.find((m) => m.kind === "resource")?.id ?? d.metrics[0].id, op: "<=", value: "" }])}>+ add constraint</button>
            </div>
            <details>
              <summary className="cursor-pointer text-zinc-400">Weighted ranking (optional)</summary>
              <p className="mt-1 text-xs text-zinc-500">Off by default. A weighted score is a preference, not a measurement: it depends entirely on the weights you choose.</p>
              <label className="mt-1 flex items-center gap-2 text-xs"><input type="checkbox" checked={useW} onChange={(e) => setUseW(e.target.checked)} />Rank with my weights</label>
              {useW && Object.keys(objs).map((m) => <label key={m} className="mt-1 flex items-center justify-between gap-2 text-xs"><span>{mi[m]?.label}</span><input type="number" min={0} step={0.5} className="w-16 rounded border border-zinc-700 bg-zinc-900 px-1" value={w[m] ?? 1} onChange={(e) => setW({ ...w, [m]: Number(e.target.value) })} /></label>)}
            </details>
          </div>
          <div>
            <ParetoChart d={d} x={x} y={y} setX={setX} setY={setY} mi={mi} />
            {a && <Dominance a={a} label={label} />}
            {!a && <div className="mt-3 text-sm text-zinc-500">Choose at least one objective to see which configurations are non-dominated.</div>}
          </div>
        </div>
        {a?.weighted && (
          <div className="mt-4 rounded-lg border border-zinc-800 p-3 text-sm">
            <div className="font-medium text-zinc-300">Weighted ranking</div>
            <ol className="mt-1 list-decimal pl-5">{a.weighted.ranking.map((r) => <li key={r.id}>{label(r.id)} <span className="text-xs text-zinc-500">score {r.score.toFixed(3)}</span></li>)}</ol>
            <p className="mt-2 text-xs text-zinc-500">Weights (normalised): {Object.entries(a.weighted.weights).map(([k, v]) => `${mi[k]?.label} ${v.toFixed(2)}`).join("; ")}. {a.weighted.formula}. {a.weighted.caveat}</p>
          </div>)}
      </Card>
    </>
  );
}

function ComparisonTable({ d, mi, front }: { d: ParetoResponse; mi: Record<string, MetricInfo>; front?: string[] }) {
  const cols = [...QUALITY, ...PERF].filter((c) => c in mi);
  const [sort, setSort] = useState<string | null>(null);
  const pts = useMemo(() => {
    const p = [...d.points];
    if (sort) p.sort((a, b) => { const va = a.values[sort]?.value, vb = b.values[sort]?.value; if (va == null) return 1; if (vb == null) return -1; return (mi[sort].direction === "max" ? vb - va : va - vb); });
    return p;
  }, [d.points, sort, mi]);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[820px] text-sm tabular-nums">
        <thead className="text-left text-xs text-zinc-500"><tr><th className="py-1 pr-3 font-medium">Configuration</th>
          {cols.map((c) => <th key={c} className="cursor-pointer py-1 pr-3 font-medium hover:text-zinc-300" title={`${mi[c].label} (click to sort, best first)`} onClick={() => setSort(c)}>{mi[c].label.replace("Structured output: ", "JSON ").replace("Coding: ", "Code ").replace("Long-context retrieval accuracy", "Retrieval").replace(" (synthetic)", "").replace("Aggregate output throughput", "Throughput")}{sort === c && " ↓"}<div className="font-normal text-zinc-600">{mi[c].unit}</div></th>)}</tr></thead>
        <tbody className="divide-y divide-zinc-800">{pts.map((p) => (
          <tr key={p.id} className={front?.includes(p.id) ? "bg-blue-500/5" : ""}>
            <td className="py-1.5 pr-3">{short(p.label)} {front?.includes(p.id) && <Pill tone="green">non-dominated</Pill>} {p.contended && <Pill tone="amber">contended</Pill>}</td>
            {cols.map((c) => { const v = p.values[c]; return (
              <td key={c} className="py-1.5 pr-3">{v ? <>{fmtMetric(mi[c], v.display)}
                {v.lo != null && v.hi != null && mi[c].kind === "quality" && <span className="text-xs text-zinc-500"> [{v.lo.toFixed(0)}–{v.hi.toFixed(0)}] n={v.n}</span>}
                {v.lo != null && v.hi != null && mi[c].kind === "performance" && <span className="text-xs text-zinc-500"> ±{((v.hi - v.lo) / 2).toFixed(v.value < 100 ? 1 : 0)}</span>}</> : <span className="text-zinc-600">—</span>}</td>); })}</tr>))}</tbody>
      </table>
    </div>
  );
}

function ParetoChart({ d, x, y, setX, setY, mi }: { d: ParetoResponse; x: string; y: string; setX: (s: string) => void; setY: (s: string) => void; mi: Record<string, MetricInfo> }) {
  const a = d.analysis;
  const status = (id: string) => (!a ? "plain" : a.front.includes(id) ? "front" : a.excluded.some((e) => e.id === id) ? "excluded" : "dominated");
  const multiCtx = new Set(d.points.map((p) => p.config.max_model_len)).size > 1;
  const tiny = (l: string) => l.replace(/^[^/]+\//, "").replace(/-Instruct/, "").replace(multiCtx ? /$^/ : / · ctx \d+/, "");
  const mk = (p: any) => { const vx = p.values[x], vy = p.values[y]; if (!vx || !vy) return null;
    return { id: p.id, name: tiny(p.label), x: vx.display, y: vy.display, s: status(p.id),
      xi: vx.lo != null && vx.hi != null ? [{ x: vx.lo, y: vy.display }, { x: vx.hi, y: vy.display }] : null,
      yi: vy.lo != null && vy.hi != null ? [{ x: vx.display, y: vy.lo }, { x: vx.display, y: vy.hi }] : null }; };
  const data = d.points.map(mk).filter(Boolean) as any[];
  const by = (s: string) => data.filter((p) => p.s === s);
  const opts = d.metrics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>);
  const axisProps = (m: MetricInfo | undefined) => ({ reversed: m?.direction === "min", domain: m?.percent ? [0, 100] : ["auto", "auto"] as any });
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-3 text-xs text-zinc-400">
        <label>X <select className="ml-1 rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5" value={x} onChange={(e) => setX(e.target.value)}>{opts}</select></label>
        <label>Y <select className="ml-1 rounded border border-zinc-700 bg-zinc-900 px-1 py-0.5" value={y} onChange={(e) => setY(e.target.value)}>{opts}</select></label>
        <span className="text-zinc-500">axes are oriented so up/right is better; bars are 95% intervals (Wilson for rates, CI across iterations for performance)</span>
      </div>
      <div style={{ height: 340 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 28, right: 70, bottom: 28, left: 8 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis type="number" dataKey="x" name={mi[x]?.label} stroke="#71717a" fontSize={11} {...axisProps(mi[x])} label={{ value: `${mi[x]?.label ?? ""} (${mi[x]?.percent ? "%" : mi[x]?.unit})${mi[x]?.direction === "min" ? " — lower is better →" : ""}`, position: "insideBottom", offset: -16, fill: "#a1a1aa", fontSize: 11 }} />
            <YAxis type="number" dataKey="y" name={mi[y]?.label} stroke="#71717a" fontSize={11} width={52} {...(mi[y]?.direction === "min" ? { reversed: true } : {})} domain={mi[y]?.percent ? [0, 100] : ["auto", "auto"]} />
            <ZAxis range={[90, 90]} />
            <Tooltip contentStyle={TIP} cursor={{ strokeDasharray: "3 3" }} content={({ payload }) => { const p: any = payload?.[0]?.payload; return p ? (
              <div style={TIP} className="p-2"><div className="font-medium">{p.name}</div><div>{mi[x]?.label}: {fmtMetric(mi[x], p.x)}</div><div>{mi[y]?.label}: {fmtMetric(mi[y], p.y)}</div><div className="text-zinc-400">{p.s === "front" ? "non-dominated" : p.s === "dominated" ? "dominated" : p.s === "excluded" ? "excluded by constraint / missing metric" : ""}</div></div>) : null; }} />
            <Legend wrapperStyle={{ fontSize: 12 }} verticalAlign="top" height={24} />
            {data.filter((p) => p.xi).map((p) => <Scatter key={`xi${p.id}`} data={p.xi} line={{ stroke: p.s === "front" || !a ? "#60a5fa" : "#71717a", strokeWidth: 1.5 }} lineType="joint" shape={() => <g />} legendType="none" isAnimationActive={false} />)}
            {data.filter((p) => p.yi).map((p) => <Scatter key={`yi${p.id}`} data={p.yi} line={{ stroke: p.s === "front" || !a ? "#60a5fa" : "#71717a", strokeWidth: 1.5 }} lineType="joint" shape={() => <g />} legendType="none" isAnimationActive={false} />)}
            {a && <Scatter name="dominated" data={by("dominated")} fill="transparent" stroke="#a1a1aa" strokeWidth={2} isAnimationActive={false}><LabelList dataKey="name" position="top" fill="#a1a1aa" fontSize={10} /></Scatter>}
            {a && <Scatter name="excluded" data={by("excluded")} fill="#3f3f46" shape="cross" isAnimationActive={false}><LabelList dataKey="name" position="top" fill="#71717a" fontSize={10} /></Scatter>}
            <Scatter name={a ? "non-dominated" : "configurations"} data={by(a ? "front" : "plain")} fill="#60a5fa" isAnimationActive={false}><LabelList dataKey="name" position="top" offset={8} fill="#e4e4e7" fontSize={11} /></Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
            {data.length === 0 && <div className="mt-2 text-sm text-amber-300">No configuration has both selected metrics recorded.</div>}
    </div>
  );
}

function Dominance({ a, label }: { a: NonNullable<ParetoResponse["analysis"]>; label: (k: string) => string }) {
  const supportTip: Record<string, string> = { clear: "better on every objective by more than the uncertainty", within_noise: "advantage is within the uncertainty on at least one objective", unverified: "uncertainty is unknown for at least one objective" };
  return (
    <div className="mt-4 space-y-2 text-sm">
      <div><span className="font-medium text-zinc-300">Non-dominated ({a.front.length}):</span> {a.front.map(label).map(short).join(", ") || "none"}</div>
      {a.dominated.length > 0 && <div><span className="font-medium text-zinc-300">Dominated:</span>
        <ul className="mt-1 space-y-0.5 text-xs text-zinc-400">{a.dominated.map((x) => <li key={x.id}>{short(label(x.id))} is dominated by {x.dominated_by.map((b) => <span key={b.by} title={supportTip[b.support]}>{short(label(b.by))} <Pill tone={b.support === "clear" ? "green" : "amber"}>{b.support.replace("_", " ")}</Pill> </span>)}</li>)}</ul></div>}
      {a.excluded.length > 0 && <div><span className="font-medium text-zinc-300">Excluded:</span>
        <ul className="mt-1 text-xs text-zinc-400">{a.excluded.map((x) => <li key={x.id}>{short(label(x.id))}: {x.detail}</li>)}</ul></div>}
      {(a.warnings.length > 0) && <ul className="space-y-0.5 text-xs text-amber-300">{a.warnings.map((w) => <li key={w}>⚠ {w}</li>)}</ul>}
      <p className="text-xs text-zinc-500">A configuration is dominated when another is at least as good on every chosen objective and strictly better on one. &quot;Within noise&quot; means the measured advantage is smaller than the recorded uncertainty, so treat the ordering as tentative.</p>
    </div>
  );
}

function heat(rate: number) { const g = Math.round(40 + rate * 150); return `rgb(${Math.round(150 - rate * 110)}, ${g}, 70)`; }

function QualityDetails({ rows }: { rows: ConfigResultRec[] }) {
  return (
    <Card title="Quality details per configuration">
      <div className="space-y-2">{rows.map((r, idx) => { const suites = r.results?.quality?.suites ?? {}; return (
        <details key={r.key} open={idx === 0} className="rounded-lg border border-zinc-800 p-3"><summary className="cursor-pointer text-sm">{r.label}<span className="ml-2 text-xs text-zinc-500">{Object.entries<any>(suites).map(([k, s]) => `${k.split("_")[0]} ${(s.interval.rate * 100).toFixed(0)}% (${s.passed}/${s.n})`).join(" · ")}</span></summary>
          <div className="mt-3 space-y-4">{Object.entries<any>(suites).map(([name, s]) => (
            <div key={name}>
              <div className="text-sm font-medium text-zinc-300">{name} <span className="text-xs font-normal text-zinc-500">v{s.version} · {s.split} split · dataset {s.fingerprint}</span></div>
              <div className="mt-1 grid gap-x-6 text-xs text-zinc-400 md:grid-cols-2">
                <div>Pass rate {(s.interval.rate * 100).toFixed(1)}% (95% CI {(s.interval.lo * 100).toFixed(0)}–{(s.interval.hi * 100).toFixed(0)}%, n={s.n}){s.truncated ? ` · ${s.truncated} truncated outputs` : ""}</div>
                <div>Eval-prompt latency p50 {s.performance.latency.e2e_ms ? `${s.performance.latency.e2e_ms.p50.toFixed(0)} ms` : "—"} · {s.performance.output_tok_s ? `${s.performance.output_tok_s.toFixed(0)} tok/s` : ""} · {s.performance.requests.failed} failed generations <Pill>evaluation prompts</Pill></div>
              </div>
              {Object.keys(s.aggregate.failures_by_type ?? {}).length > 0 && <div className="mt-1 text-xs text-red-300">Failures: {Object.entries<number>(s.aggregate.failures_by_type).map(([k, v]) => `${k} ×${v}`).join(", ")}</div>}
              {name === "structured_json" && <div className="mt-1 text-xs text-zinc-400">parse {(s.aggregate.parse_rate * 100).toFixed(0)}% · schema-valid {(s.aggregate.schema_valid_rate * 100).toFixed(0)}% · field accuracy {(s.aggregate.extraction_accuracy * 100).toFixed(0)}%{Object.keys(s.aggregate.violations_by_keyword ?? {}).length ? ` · violations: ${Object.entries<number>(s.aggregate.violations_by_keyword).map(([k, v]) => `${k}×${v}`).join(", ")}` : ""}</div>}
              {name === "retrieval_needle" && s.aggregate.grid && (
                <div className="mt-2 overflow-x-auto"><table className="text-xs tabular-nums"><thead className="text-zinc-500"><tr><th className="pr-2 text-left font-normal">context \ fact position</th>{Object.keys(Object.values<any>(s.aggregate.grid)[0]).map((p) => <th key={p} className="px-2 font-normal">{(Number(p) * 100).toFixed(0)}%</th>)}</tr></thead>
                  <tbody>{Object.entries<any>(s.aggregate.grid).map(([L, ps]) => <tr key={L}><td className="pr-2 text-zinc-400">{Number(L).toLocaleString()} tok</td>{Object.entries<any>(ps).map(([p, c]) => <td key={p} className="px-0.5"><div className="rounded px-2 py-1 text-center text-white" style={{ background: heat(c.rate) }} title={`${c.k}/${c.n}`}>{(c.rate * 100).toFixed(0)}%<div className="text-[10px] opacity-70">{c.k}/{c.n}</div></div></td>)}</tr>)}</tbody></table>
                  <div className="mt-1 text-zinc-500">Actual prompt size (server-reported) averaged {Math.round(s.aggregate.prompt_tokens_mean)} tokens{s.length_error != null ? `; mean deviation from target ${(s.length_error * 100).toFixed(1)}%` : ""}.</div></div>)}
            </div>))}</div></details>); })}</div>
    </Card>
  );
}

function Failures({ rows }: { rows: ConfigResultRec[] }) {
  const bad = rows.filter((r) => ["failed", "skipped", "cancelled"].includes(r.status));
  const withEval = rows.filter((r) => r.results?.quality?.suites);
  const [sel, setSel] = useState<string>("");
  const [suite, setSuite] = useState("");
  const [logFor, setLogFor] = useState<number | null>(null);
  const cur = withEval.find((r) => r.key === sel) ?? withEval[0];
  const curSuite = suite && cur?.results?.quality?.suites?.[suite] ? suite : Object.keys(cur?.results?.quality?.suites ?? {})[0];
  const arts = useQuery({ queryKey: ["cases-art", cur?.job_id, cur?.key], enabled: !!cur?.results?.artifacts?.cases && !!cur.job_id,
    queryFn: async () => { const list: any[] = await api(`/jobs/${cur!.job_id}/artifacts`); const a = list.find((x) => x.name === cur!.results!.artifacts.cases); return a ? (await fetch(`${API_URL}/api/artifacts/${a.id}/content`)).json() : null; } });
  const server = useQuery({ queryKey: ["srv-log", logFor], enabled: logFor != null, queryFn: () => api<ManagedServerRec>(`/managed-servers/${logFor}`) });
  const failing = ((arts.data?.suites?.[curSuite ?? ""] ?? []) as any[]).filter((c) => !c.passed).slice(0, 25);
  const modes = new Set<string>(); const table: Record<string, Record<string, number>> = {};
  for (const r of withEval) for (const [n, s] of Object.entries<any>(r.results!.quality.suites)) for (const [k, v] of Object.entries<number>(s.aggregate.failures_by_type ?? {})) { const col = `${n.split("_")[0]}:${k}`; modes.add(col); (table[r.key] ??= {})[col] = v; }
  const cols = [...modes];
  if (bad.length === 0 && cols.length === 0) return <Card title="Failure analysis"><div className="text-sm text-emerald-300">No configuration failed and no evaluation case failed.</div></Card>;
  return (
    <Card title="Failure analysis">
      <div className="space-y-5">
        {bad.length > 0 && <div><div className="mb-1 text-sm font-medium text-zinc-300">Configurations without results</div>
          <table className="w-full text-xs"><tbody className="divide-y divide-zinc-800">{bad.map((r) => (
            <tr key={r.key}><td className="py-1.5 pr-3 align-top">{r.label}</td><td className="pr-3 align-top"><Pill tone={STATUS_TONE[r.status]}>{r.status}</Pill></td><td className="pr-3 align-top text-zinc-400">{r.failure_kind}</td>
              <td className="py-1.5 text-zinc-500"><div className="max-h-16 overflow-auto whitespace-pre-wrap">{r.error}</div>{r.results?.partial && hasData(r) && <div className="text-amber-300">partial data was kept for this configuration (see the comparison only if it completed)</div>}
                {r.managed_server_id && <button type="button" className="text-blue-400 hover:underline" onClick={() => setLogFor(logFor === r.managed_server_id ? null : r.managed_server_id)}>{logFor === r.managed_server_id ? "hide" : "show"} server log</button>}</td></tr>))}</tbody></table>
          {logFor != null && server.data?.log_tail && <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-zinc-950 p-3 text-[11px] text-zinc-400">{server.data.log_tail.split("\n").slice(-120).join("\n")}</pre>}</div>}
        {cols.length > 0 && <div className="overflow-x-auto"><div className="mb-1 text-sm font-medium text-zinc-300">Evaluation failures by type</div>
          <table className="w-full min-w-[520px] text-xs tabular-nums"><thead className="text-left text-zinc-500"><tr><th className="py-1 pr-3 font-medium">Configuration</th>{cols.map((c) => <th key={c} className="pr-3 font-medium">{c}</th>)}</tr></thead>
            <tbody className="divide-y divide-zinc-800">{withEval.map((r) => <tr key={r.key}><td className="py-1 pr-3">{short(r.label)}</td>{cols.map((c) => <td key={c} className="pr-3">{table[r.key]?.[c] ?? 0}</td>)}</tr>)}</tbody></table></div>}
        {cur && <div><div className="mb-1 flex flex-wrap items-center gap-2 text-sm font-medium text-zinc-300">Failing cases
          <select className="rounded border border-zinc-700 bg-zinc-900 px-1 text-xs font-normal" value={cur.key} onChange={(e) => setSel(e.target.value)}>{withEval.map((r) => <option key={r.key} value={r.key}>{short(r.label)}</option>)}</select>
          <select className="rounded border border-zinc-700 bg-zinc-900 px-1 text-xs font-normal" value={curSuite} onChange={(e) => setSuite(e.target.value)}>{Object.keys(cur.results!.quality.suites).map((s) => <option key={s}>{s}</option>)}</select></div>
          {arts.isLoading ? <div className="text-xs text-zinc-500">Loading…</div> : failing.length === 0 ? <div className="text-xs text-emerald-300">No failing cases for this selection.</div> : (
            <div className="max-h-80 space-y-2 overflow-auto">{failing.map((c) => <div key={c.case_id} className="rounded border border-zinc-800 p-2 text-xs"><div className="font-mono text-zinc-400">{c.case_id} <Pill tone="red">{c.failure_type}</Pill></div>
              {c.detail && <div className="mt-0.5 text-zinc-500">{c.detail}</div>}<pre className="mt-1 max-h-24 overflow-auto whitespace-pre-wrap text-zinc-500">{c.response}</pre></div>)}</div>)}</div>}
      </div>
    </Card>
  );
}
