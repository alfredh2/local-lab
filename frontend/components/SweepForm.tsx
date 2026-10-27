"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api, type ExperimentDetail, type GpuPreflight, type ModelRec, type ProbeResult, type RuntimeConfigRec, type ValidationReport,
} from "@/lib/api";
import { fmtBytes, fmtMb } from "@/lib/format";
import { btn, btnGhost, ErrorBox, Pill } from "@/components/ui";

const input = "mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm";

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="text-zinc-300">{label}</span>
      {hint && <span className="ml-2 text-xs text-zinc-500">{hint}</span>}
      {children}
    </label>
  );
}
const num = (v: string): number | "" => (v === "" ? "" : Number(v));

export default function SweepForm() {
  const router = useRouter();
  const qc = useQueryClient();
  const [mode, setMode] = useState<"managed" | "external">("managed");
  const [modelId, setModelId] = useState<number | "">("");
  const [rt, setRt] = useState<Record<string, any>>({
    max_model_len: 4096, gpu_memory_utilization: 0.3, dtype: "auto", quantization: "none", kv_cache_dtype: "auto",
    max_num_seqs: 64, startup_timeout_s: 600, enforce_eager: false,
  });
  const [allowContention, setAllowContention] = useState(false);
  const [levels, setLevels] = useState("1, 2, 4, 8, 16");
  const [w, setW] = useState<Record<string, any>>({
    input_tokens: 128, output_tokens: 128, requests_per_level: 32, min_requests_per_slot: 4, warmup_requests: 4,
    iterations: 3, seed: 0, temperature: 0, request_timeout_s: 300,
  });
  const [serverUrl, setServerUrl] = useState("http://127.0.0.1:8100");
  const [extModel, setExtModel] = useState("");
  const [savedId, setSavedId] = useState<number | "">("");
  const [saveName, setSaveName] = useState("");

  const models = useQuery({ queryKey: ["models"], queryFn: () => api<ModelRec[]>("/models") });
  const saved = useQuery({ queryKey: ["runtime-configs"], queryFn: () => api<RuntimeConfigRec[]>("/runtime-configs") });
  const hf = models.data?.filter((m) => m.available && m.source === "hf_cache") ?? [];
  const util = Number(rt.gpu_memory_utilization) || 0.3;
  const pre = useQuery({
    queryKey: ["gpu-preflight", util], enabled: mode === "managed", refetchInterval: 5000,
    queryFn: () => api<GpuPreflight>(`/gpu/preflight?utilization=${util}`),
  });
  const probe = useMutation({
    mutationFn: () => api<ProbeResult>(`/runtimes/vllm/probe?url=${encodeURIComponent(serverUrl)}`),
    onSuccess: (r) => setExtModel((m) => (r.models.some((x) => x.id === m) ? m : r.models[0]?.id ?? "")),
  });

  const parsedLevels = levels.split(/[,\s]+/).filter(Boolean).map(Number);
  const levelsOk = parsedLevels.length > 0 && parsedLevels.every((n) => Number.isInteger(n) && n >= 1 && n <= 512);
  const clean = (o: Record<string, any>) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== ""));
  const config = {
    mode, concurrency_levels: levelsOk ? parsedLevels : [1], ...clean(w),
    ...(mode === "managed"
      ? { registry_model_id: modelId === "" ? null : modelId, runtime: { ...clean(rt), allow_contention: allowContention } }
      : { server_url: serverUrl, model: extModel }),
  };
  const body = JSON.stringify({ type: "concurrency_sweep", config });
  const ready = levelsOk && (mode === "managed" ? modelId !== "" : extModel !== "");

  const [result, setResult] = useState<{ forBody: string; report: ValidationReport } | null>(null);
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(async () => {
      let report: ValidationReport;
      try { report = await api<ValidationReport>("/experiments/validate", { method: "POST", body }); }
      catch (e) { report = { ok: false, errors: [e instanceof Error ? e.message : String(e)], warnings: [], info: null }; }
      setResult({ forBody: body, report });
    }, 600);
    return () => clearTimeout(t);
  }, [body, ready]);
  const report = ready && result?.forBody === body ? result.report : null;
  const checking = ready && report === null;

  const create = useMutation({
    mutationFn: () => api<ExperimentDetail>("/experiments", { method: "POST", body }),
    onSuccess: (e) => { qc.invalidateQueries({ queryKey: ["experiments"] }); router.push(`/experiments/${e.id}`); },
  });
  const save = useMutation({
    mutationFn: () => api("/runtime-configs", { method: "POST", body: JSON.stringify({ name: saveName, registry_model_id: modelId === "" ? null : modelId, config: clean(rt) }) }),
    onSuccess: () => { setSaveName(""); qc.invalidateQueries({ queryKey: ["runtime-configs"] }); },
  });
  const applySaved = (id: number | "") => {
    setSavedId(id);
    const c = saved.data?.find((x) => x.id === id);
    if (c) { setRt((p) => ({ ...p, ...c.config })); setModelId(c.registry_model_id ?? ""); }
  };
  const setR = (k: string, v: any) => { setRt((p) => ({ ...p, [k]: v })); setSavedId(""); };
  const info = report?.info;
  const needsOverride = mode === "managed" && pre.data?.contended;

  return (
    <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
      <div className="flex gap-1.5">
        {(["managed", "external"] as const).map((m) => (
          <button type="button" key={m} onClick={() => setMode(m)}
            className={`rounded-lg border px-2.5 py-1 text-xs ${mode === m ? "border-blue-500 bg-blue-500/15 text-blue-200" : "border-zinc-700 text-zinc-400 hover:bg-zinc-800"}`}>
            {m === "managed" ? "Managed (LocalLab runs vLLM)" : "External server"}
          </button>
        ))}
      </div>

      {mode === "managed" ? (
        <>
          <Field label="Saved configuration">
            <select className={input} value={savedId} onChange={(e) => applySaved(e.target.value === "" ? "" : Number(e.target.value))}>
              <option value="">— none —</option>
              {saved.data?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </Field>
          <Field label="Model" hint="Hugging Face cache, vLLM format">
            <select className={input} value={modelId} onChange={(e) => { setModelId(e.target.value === "" ? "" : Number(e.target.value)); setSavedId(""); }}>
              <option value="">Select a model…</option>
              {hf.map((m) => <option key={m.id} value={m.id}>{m.name} · {fmtBytes(m.size_bytes)}{m.quantization ? ` · ${m.quantization}` : ""}</option>)}
            </select>
          </Field>
          {models.data && hf.length === 0 && <div className="text-xs text-amber-300">No servable models registered. Scan the Models page first.</div>}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Context length"><input type="number" className={input} value={rt.max_model_len} onChange={(e) => setR("max_model_len", num(e.target.value))} /></Field>
            <Field label="GPU memory share" hint="0.05–0.95"><input type="number" step={0.05} min={0.05} max={0.95} className={input} value={rt.gpu_memory_utilization} onChange={(e) => setR("gpu_memory_utilization", num(e.target.value))} /></Field>
            <Field label="dtype"><select className={input} value={rt.dtype} onChange={(e) => setR("dtype", e.target.value)}>{["auto", "bfloat16", "float16", "float32"].map((d) => <option key={d}>{d}</option>)}</select></Field>
            <Field label="Quantization"><select className={input} value={rt.quantization} onChange={(e) => setR("quantization", e.target.value)}>{["none", "awq", "gptq", "fp8"].map((d) => <option key={d}>{d}</option>)}</select></Field>
            <Field label="KV cache dtype"><select className={input} value={rt.kv_cache_dtype} onChange={(e) => setR("kv_cache_dtype", e.target.value)}>{["auto", "fp8"].map((d) => <option key={d}>{d}</option>)}</select></Field>
            <Field label="Max sequences"><input type="number" className={input} value={rt.max_num_seqs} onChange={(e) => setR("max_num_seqs", num(e.target.value))} /></Field>
          </div>

          <div className={`rounded-lg border p-3 text-xs ${pre.data?.contended ? "border-amber-800 bg-amber-950/30" : "border-zinc-800 bg-zinc-900/50"}`}>
            <div className="mb-1 flex items-center gap-2 text-sm text-zinc-300">GPU right now {pre.data && (pre.data.contended ? <Pill tone="amber">contended</Pill> : <Pill tone="green">idle</Pill>)}</div>
            {pre.isError ? <span className="text-red-300">Cannot read GPU state.</span> : pre.data ? (
              <>
                <div className="text-zinc-400">{pre.data.name} · {fmtMb(pre.data.free_mb)} free of {fmtMb(pre.data.total_mb)} · requested {fmtMb(pre.data.requested_mb)} · util {pre.data.utilization_pct?.toFixed(0) ?? "?"}%</div>
                {pre.data.other_processes.map((p) => <div key={p.pid} className="text-amber-300">• {p.name.split("/").pop()} (pid {p.pid}) using {fmtMb(p.memory_mb)}</div>)}
              </>
            ) : <span className="text-zinc-500">Checking…</span>}
            {needsOverride && (
              <label className="mt-2 flex items-start gap-2 text-amber-200">
                <input type="checkbox" className="mt-0.5" checked={allowContention} onChange={(e) => setAllowContention(e.target.checked)} />
                <span>Run anyway. Nothing will be stopped, but the other work will slow this benchmark (and vice versa), results are marked contended, and vLLM can fail at startup if the other job&apos;s memory changes.</span>
              </label>
            )}
          </div>
        </>
      ) : (
        <div className="space-y-2">
          <Field label="vLLM server">
            <div className="mt-1 flex gap-2">
              <input className={input + " !mt-0"} value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} />
              <button type="button" className={btnGhost} onClick={() => probe.mutate()} disabled={probe.isPending}>{probe.isPending ? "…" : "Connect"}</button>
            </div>
          </Field>
          {probe.isError && <ErrorBox error={probe.error} />}
          {probe.data && (
            <Field label="Served model">
              <select className={input} value={extModel} onChange={(e) => setExtModel(e.target.value)}>
                {probe.data.models.map((m) => <option key={m.id} value={m.id}>{m.id}{m.max_model_len ? ` (ctx ${m.max_model_len})` : ""}</option>)}
              </select>
            </Field>
          )}
        </div>
      )}

      <div className="space-y-3 border-t border-zinc-800 pt-3">
        <Field label="Concurrency levels" hint="comma separated, 1–512">
          <input className={input} value={levels} onChange={(e) => setLevels(e.target.value)} />
        </Field>
        {!levelsOk && <div className="text-xs text-red-300">Enter integers between 1 and 512.</div>}
        <div className="grid grid-cols-2 gap-3">
          <Field label="Input tokens"><input type="number" className={input} value={w.input_tokens} onChange={(e) => setW({ ...w, input_tokens: num(e.target.value) })} /></Field>
          <Field label="Output tokens"><input type="number" className={input} value={w.output_tokens} onChange={(e) => setW({ ...w, output_tokens: num(e.target.value) })} /></Field>
          <Field label="Requests / level"><input type="number" className={input} value={w.requests_per_level} onChange={(e) => setW({ ...w, requests_per_level: num(e.target.value) })} /></Field>
          <Field label="Iterations" hint="≥3 for CIs"><input type="number" className={input} value={w.iterations} onChange={(e) => setW({ ...w, iterations: num(e.target.value) })} /></Field>
          <Field label="Warmup"><input type="number" className={input} value={w.warmup_requests} onChange={(e) => setW({ ...w, warmup_requests: num(e.target.value) })} /></Field>
          <Field label="Seed"><input type="number" className={input} value={w.seed} onChange={(e) => setW({ ...w, seed: num(e.target.value) })} /></Field>
        </div>
      </div>

      {ready && (
        <div className="space-y-1 text-sm">
          {checking && <div className="text-zinc-500">Checking feasibility…</div>}
          {report?.ok && <div className="text-emerald-300">✓ Configuration is valid.</div>}
          {report?.errors.map((e) => <div key={e} className="text-red-300">✗ {e}</div>)}
          {report?.warnings.map((x) => <div key={x} className="text-amber-300">⚠ {x}</div>)}
          {report?.ok && info?.estimates && (
            <div className="text-xs text-zinc-500">
              Estimate: weights {info.estimates.weights_mb} MB, KV room ≈ {info.estimates.kv_tokens?.toLocaleString() ?? "?"} tokens (actual size is read from the server).
              {" "}{info.total_requests} requests in total.
            </div>
          )}
        </div>
      )}
      {create.isError && <ErrorBox error={create.error} />}
      <button className={btn} disabled={create.isPending || !ready || checking || !report?.ok}>
        {create.isPending ? "Launching…" : mode === "managed" ? "Launch managed sweep" : "Launch sweep"}
      </button>

      {mode === "managed" && (
        <div className="flex gap-2 border-t border-zinc-800 pt-3">
          <input className={input + " !mt-0"} placeholder="Save runtime settings as…" value={saveName} onChange={(e) => setSaveName(e.target.value)} />
          <button type="button" className={btnGhost} disabled={!saveName.trim() || save.isPending} onClick={() => save.mutate()}>Save</button>
        </div>
      )}
      {save.isError && <ErrorBox error={save.error} />}
    </form>
  );
}
