"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type ExperimentDetail, type GpuPreflight, type ModelRec, type ValidationReport } from "@/lib/api";
import { fmtBytes, fmtMb } from "@/lib/format";
import { btn, ErrorBox, Pill } from "@/components/ui";

const input = "mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm";
const SUITES = [
  { id: "structured_json", label: "Structured JSON", hint: "18 held-out cases: parse + schema compliance" },
  { id: "coding_python", label: "Python coding", hint: "13 held-out tasks, hidden tests in a sandbox container" },
  { id: "retrieval_needle", label: "Long-context retrieval", hint: "facts at 5 positions × context lengths" },
];
const ints = (s: string) => s.split(/[,\s]+/).filter(Boolean).map(Number);
const okInts = (a: number[], lo = 1, hi = 1_000_000) => a.length > 0 && a.every((n) => Number.isInteger(n) && n >= lo && n <= hi);

function Check({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: React.ReactNode }) {
  return <label className="flex items-start gap-2 text-sm"><input type="checkbox" className="mt-0.5" checked={checked} onChange={(e) => onChange(e.target.checked)} /><span>{children}</span></label>;
}
function Group({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return <fieldset className="space-y-2 border-t border-zinc-800 pt-3"><legend className="text-sm font-medium text-zinc-300">{title}{hint && <span className="ml-2 text-xs font-normal text-zinc-500">{hint}</span>}</legend>{children}</fieldset>;
}
const toggle = <T,>(arr: T[], v: T) => (arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);

export default function MatrixForm() {
  const router = useRouter();
  const qc = useQueryClient();
  const [models, setModels] = useState<number[]>([]);
  const [ctx, setCtx] = useState("8192");
  const [quant, setQuant] = useState<string[]>(["none"]);
  const [kv, setKv] = useState<string[]>(["auto"]);
  const [util, setUtil] = useState(0.3);
  const [perf, setPerf] = useState(true);
  const [levels, setLevels] = useState("1, 4, 16");
  const [io, setIo] = useState("128/128");
  const [reqs, setReqs] = useState(32);
  const [iters, setIters] = useState(3);
  const [evals, setEvals] = useState<string[]>(SUITES.map((s) => s.id));
  const [split, setSplit] = useState("heldout");
  const [retLens, setRetLens] = useState("2048, 4096");
  const [maxCfg, setMaxCfg] = useState(8);
  const [budgetMin, setBudgetMin] = useState(90);
  const [allow, setAllow] = useState(false);

  const all = useQuery({ queryKey: ["models"], queryFn: () => api<ModelRec[]>("/models") });
  const hf = (all.data ?? []).filter((m) => m.available && m.source === "hf_cache");
  const pre = useQuery({ queryKey: ["gpu-preflight", util], refetchInterval: 5000, queryFn: () => api<GpuPreflight>(`/gpu/preflight?utilization=${util}`) });

  const ioPairs = io.split(/[,\s]+/).filter(Boolean).map((p) => p.split("/").map(Number));
  const ioOk = ioPairs.length > 0 && ioPairs.every((p) => p.length === 2 && p.every((n) => Number.isInteger(n) && n > 0));
  const ctxL = ints(ctx), lvl = ints(levels), rl = ints(retLens);
  const valid = models.length > 0 && okInts(ctxL, 256) && quant.length > 0 && kv.length > 0 && (!perf || (okInts(lvl, 1, 512) && ioOk)) && (perf || evals.length > 0) &&
    (!evals.includes("retrieval_needle") || okInts(rl, 256));
  const config = {
    registry_model_ids: models, max_model_lens: valid ? ctxL : [8192], quantizations: quant, kv_cache_dtypes: kv, gpu_memory_utilization: util,
    perf_enabled: perf, ...(perf && valid ? { concurrency_levels: lvl, io_pairs: ioPairs.map(([i, o]) => ({ input_tokens: i, output_tokens: o })) } : {}),
    requests_per_level: reqs, iterations: iters, evals, eval_split: split, ...(valid && rl.length ? { retrieval_lengths: rl } : {}),
    max_configurations: maxCfg, max_runtime_s: budgetMin * 60, allow_contention: allow,
  };
  const body = JSON.stringify({ type: "optimization_matrix", config });

  const [result, setResult] = useState<{ forBody: string; report: ValidationReport } | null>(null);
  useEffect(() => {
    if (!valid) return;
    const t = setTimeout(async () => {
      let report: ValidationReport;
      try { report = await api<ValidationReport>("/experiments/validate", { method: "POST", body }); }
      catch (e) { report = { ok: false, errors: [e instanceof Error ? e.message : String(e)], warnings: [], info: null }; }
      setResult({ forBody: body, report });
    }, 700);
    return () => clearTimeout(t);
  }, [body, valid]);
  const report = valid && result?.forBody === body ? result.report : null;
  const checking = valid && report === null;
  const create = useMutation({
    mutationFn: () => api<ExperimentDetail>("/experiments", { method: "POST", body }),
    onSuccess: (e) => { qc.invalidateQueries({ queryKey: ["experiments"] }); router.push(`/optimization/${e.id}`); },
  });
  const cfgs: any[] = report?.info?.configs ?? [];

  return (
    <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
      <Group title="Models" hint="Hugging Face cache, vLLM format">
        <div className="max-h-44 space-y-1 overflow-auto rounded-lg border border-zinc-800 p-2">
          {hf.length === 0 && <div className="text-xs text-amber-300">No servable models. Scan on the Models page first.</div>}
          {hf.map((m) => <Check key={m.id} checked={models.includes(m.id)} onChange={() => setModels(toggle(models, m.id))}>{m.name} <span className="text-xs text-zinc-500">{fmtBytes(m.size_bytes)}{m.quantization ? ` · ${m.quantization}` : ""}</span></Check>)}
        </div>
      </Group>
      <Group title="Server settings (each combination = one launch)">
        <label className="block text-sm"><span className="text-zinc-300">Context lengths</span><span className="ml-2 text-xs text-zinc-500">comma separated</span>
          <input className={input} value={ctx} onChange={(e) => setCtx(e.target.value)} /></label>
        <div className="flex flex-wrap gap-x-6 gap-y-1"><span className="text-sm text-zinc-400">Weights:</span>
          {["none", "fp8", "awq", "gptq"].map((q) => <Check key={q} checked={quant.includes(q)} onChange={() => setQuant(toggle(quant, q))}>{q === "none" ? "as stored" : q}</Check>)}</div>
        <div className="flex flex-wrap gap-x-6 gap-y-1"><span className="text-sm text-zinc-400">KV cache:</span>
          {["auto", "fp8"].map((q) => <Check key={q} checked={kv.includes(q)} onChange={() => setKv(toggle(kv, q))}>{q}</Check>)}</div>
        <label className="block text-sm"><span className="text-zinc-300">GPU memory share</span>
          <input type="number" step={0.05} min={0.05} max={0.95} className={input} value={util} onChange={(e) => setUtil(Number(e.target.value))} /></label>
      </Group>
      <Group title="Performance (synthetic prompts)" hint="measured inside each launch">
        <Check checked={perf} onChange={setPerf}>Measure throughput and latency</Check>
        {perf && <div className="grid grid-cols-2 gap-3">
          <label className="text-sm"><span className="text-zinc-300">Concurrency</span><input className={input} value={levels} onChange={(e) => setLevels(e.target.value)} /></label>
          <label className="text-sm"><span className="text-zinc-300">Input/output tokens</span><input className={input} value={io} onChange={(e) => setIo(e.target.value)} placeholder="128/128, 1024/128" /></label>
          <label className="text-sm"><span className="text-zinc-300">Requests per level</span><input type="number" className={input} value={reqs} onChange={(e) => setReqs(Number(e.target.value))} /></label>
          <label className="text-sm"><span className="text-zinc-300">Iterations</span><span className="ml-1 text-xs text-zinc-500">≥3 for CIs</span><input type="number" className={input} value={iters} onChange={(e) => setIters(Number(e.target.value))} /></label>
        </div>}
      </Group>
      <Group title="Quality (real tasks, deterministic scoring)">
        {SUITES.map((s) => <Check key={s.id} checked={evals.includes(s.id)} onChange={() => setEvals(toggle(evals, s.id))}>{s.label} <span className="text-xs text-zinc-500">{s.hint}</span></Check>)}
        {evals.includes("retrieval_needle") && <label className="block text-sm"><span className="text-zinc-300">Retrieval context lengths</span><input className={input} value={retLens} onChange={(e) => setRetLens(e.target.value)} /></label>}
        <label className="block text-sm"><span className="text-zinc-300">Case split</span>
          <select className={input} value={split} onChange={(e) => setSplit(e.target.value)}><option value="heldout">held-out (use this to compare configurations)</option><option value="dev">dev (for iterating on prompts)</option></select></label>
        {split === "dev" && <div className="text-xs text-amber-300">Dev-split results are flagged in the analysis. Compare configurations on held-out cases.</div>}
      </Group>
      <Group title="Limits">
        <div className="grid grid-cols-2 gap-3">
          <label className="text-sm"><span className="text-zinc-300">Max configurations</span><input type="number" className={input} value={maxCfg} onChange={(e) => setMaxCfg(Number(e.target.value))} /></label>
          <label className="text-sm"><span className="text-zinc-300">Runtime budget (min)</span><input type="number" className={input} value={budgetMin} onChange={(e) => setBudgetMin(Number(e.target.value))} /></label>
        </div>
        <div className={`rounded-lg border p-3 text-xs ${pre.data?.contended ? "border-amber-800 bg-amber-950/30" : "border-zinc-800 bg-zinc-900/50"}`}>
          <div className="mb-1 flex items-center gap-2 text-sm text-zinc-300">GPU right now {pre.data && (pre.data.contended ? <Pill tone="amber">contended</Pill> : <Pill tone="green">idle</Pill>)}</div>
          {pre.data && <div className="text-zinc-400">{fmtMb(pre.data.free_mb)} free of {fmtMb(pre.data.total_mb)} · util {pre.data.utilization_pct?.toFixed(0) ?? "?"}%</div>}
          {pre.data?.other_processes.map((p) => <div key={p.pid} className="text-amber-300">• {p.name.split("/").pop()} using {fmtMb(p.memory_mb)}</div>)}
          {pre.data?.contended && <Check checked={allow} onChange={setAllow}><span className="text-amber-200">Run anyway (nothing is stopped; results are marked contended)</span></Check>}
        </div>
      </Group>

      {valid && (
        <div className="space-y-2 text-sm">
          {checking && <div className="text-zinc-500">Validating every configuration…</div>}
          {report && !report.ok && report.errors.map((e) => <div key={e} className="text-red-300">✗ {e}</div>)}
          {report?.ok && <div className="text-emerald-300">✓ {report.info?.launches} server launch(es) planned, {cfgs.length} configuration(s) in the matrix.</div>}
          {report?.warnings.map((w) => <div key={w} className="text-amber-300">⚠ {w}</div>)}
          {cfgs.length > 0 && (
            <div className="max-h-52 overflow-auto rounded-lg border border-zinc-800">
              <table className="w-full text-xs"><tbody className="divide-y divide-zinc-800">{cfgs.map((c) => (
                <tr key={c.key}><td className="px-2 py-1">{c.label}</td><td className="px-2 py-1">{c.status === "feasible" ? <Pill tone="green">feasible</Pill> : <Pill tone="red">skipped</Pill>}</td>
                  <td className="px-2 py-1 text-zinc-500">{c.status === "feasible" ? `~${c.estimates?.kv_tokens?.toLocaleString() ?? "?"} KV tokens` : c.errors[0]}</td></tr>))}</tbody></table>
            </div>
          )}
        </div>
      )}
      {create.isError && <ErrorBox error={create.error} />}
      <button className={btn} disabled={create.isPending || !valid || checking || !report?.ok}>{create.isPending ? "Launching…" : "Launch matrix"}</button>
    </form>
  );
}
