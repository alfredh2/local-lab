"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type ExperimentDetail, type ExperimentType, type ModelRec, type ProbeResult, type ValidationReport } from "@/lib/api";
import { btn, btnGhost, ErrorBox } from "@/components/ui";

const PROFILES: Record<string, string> = {
  short_chat: "Short chat", long_prefill: "Long prefill", long_decode: "Long decode", concurrent: "Concurrent", custom: "Custom",
};
const input = "mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm";

function Num({ label, hint, value, onChange, min, max, step }: {
  label: string; hint?: string; value: number | ""; onChange: (v: number | "") => void; min?: number; max?: number; step?: number;
}) {
  return (
    <label className="block text-sm">
      <span className="text-zinc-300">{label}</span>
      {hint && <span className="ml-2 text-xs text-zinc-500">{hint}</span>}
      <input type="number" className={input} value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} />
    </label>
  );
}

export default function BenchmarkForm({ type }: { type: ExperimentType }) {
  const router = useRouter();
  const qc = useQueryClient();
  const [serverUrl, setServerUrl] = useState("http://127.0.0.1:8100");
  const [model, setModel] = useState("");
  const [registryId, setRegistryId] = useState<number | "">("");
  const [profile, setProfile] = useState("short_chat");
  const [f, setF] = useState<Record<string, any>>({
    input_tokens: 128, output_tokens: 256, num_requests: 32, concurrency: 1, warmup_requests: 4, iterations: 1,
    seed: 0, temperature: 0, top_p: 1, request_timeout_s: 300, prompt_mode: "token_ids",
    ...type.presets.short_chat,
  });
  const set = (k: string, v: any) => { setF((p) => ({ ...p, [k]: v })); setProfile("custom"); };

  const probe = useMutation({
    mutationFn: () => api<ProbeResult>(`/runtimes/vllm/probe?url=${encodeURIComponent(serverUrl)}`),
    onSuccess: (r) => setModel((m) => (r.models.some((x) => x.id === m) ? m : r.models[0]?.id ?? "")),
  });
  const models = useQuery({ queryKey: ["models"], queryFn: () => api<ModelRec[]>("/models") });

  const config = {
    runtime: "vllm", server_url: serverUrl, model, profile,
    registry_model_id: registryId === "" ? null : registryId,
    ...Object.fromEntries(Object.entries(f).filter(([, v]) => v !== "")),
  };
  const body = JSON.stringify({ type: type.type, config });

  // A validation report is only trusted for the exact config it was computed for.
  const [result, setResult] = useState<{ forBody: string; report: ValidationReport } | null>(null);
  useEffect(() => {
    if (!model) return;
    const t = setTimeout(async () => {
      let report: ValidationReport;
      try {
        report = await api<ValidationReport>("/experiments/validate", { method: "POST", body });
      } catch (e) {
        report = { ok: false, errors: [e instanceof Error ? e.message : String(e)], warnings: [], info: null };
      }
      setResult({ forBody: body, report });
    }, 500);
    return () => clearTimeout(t);
  }, [body, model]);
  const report = result?.forBody === body ? result.report : null;
  const checking = !!model && report === null;

  const create = useMutation({
    mutationFn: () => api<ExperimentDetail>("/experiments", { method: "POST", body }),
    onSuccess: (e) => { qc.invalidateQueries({ queryKey: ["experiments"] }); router.push(`/experiments/${e.id}`); },
  });
  const served = probe.data?.models.find((m) => m.id === model);

  return (
    <form className="space-y-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
      <div className="space-y-2">
        <label className="block text-sm"><span className="text-zinc-300">vLLM server</span>
          <div className="mt-1 flex gap-2">
            <input className={input + " !mt-0"} value={serverUrl} onChange={(e) => setServerUrl(e.target.value)} />
            <button type="button" className={btnGhost} onClick={() => probe.mutate()} disabled={probe.isPending}>{probe.isPending ? "…" : "Connect"}</button>
          </div>
        </label>
        {probe.isError && <ErrorBox error={probe.error} />}
        {probe.data && (
          <div className="text-xs text-zinc-500">vLLM {probe.data.version ?? "?"} · KV cache {probe.data.kv_cache_tokens?.toLocaleString() ?? "unknown"} tokens</div>
        )}
        {probe.data && (
          <label className="block text-sm"><span className="text-zinc-300">Served model</span>
            <select className={input} value={model} onChange={(e) => setModel(e.target.value)}>
              {probe.data.models.map((m) => <option key={m.id} value={m.id}>{m.id}{m.max_model_len ? ` (ctx ${m.max_model_len})` : ""}</option>)}
            </select>
            {served?.root && <span className="text-xs text-zinc-500">{served.root}</span>}
          </label>
        )}
        {probe.data && (
          <label className="block text-sm"><span className="text-zinc-300">Registry model</span><span className="ml-2 text-xs text-zinc-500">optional, recorded with results</span>
            <select className={input} value={registryId} onChange={(e) => setRegistryId(e.target.value === "" ? "" : Number(e.target.value))}>
              <option value="">— none —</option>
              {models.data?.filter((m) => m.available).map((m) => <option key={m.id} value={m.id}>{m.name} ({m.source})</option>)}
            </select>
          </label>
        )}
      </div>

      <div>
        <div className="mb-1 text-sm text-zinc-300">Workload profile</div>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(PROFILES).map(([k, label]) => (
            <button type="button" key={k} onClick={() => { setProfile(k); if (type.presets[k]) setF((p) => ({ ...p, ...type.presets[k] })); }}
              className={`rounded-lg border px-2.5 py-1 text-xs ${profile === k ? "border-blue-500 bg-blue-500/15 text-blue-200" : "border-zinc-700 text-zinc-400 hover:bg-zinc-800"}`}>{label}</button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Num label="Input tokens" value={f.input_tokens} min={1} onChange={(v) => set("input_tokens", v)} />
        <Num label="Output tokens" value={f.output_tokens} min={1} onChange={(v) => set("output_tokens", v)} />
        <Num label="Requests" hint="measured" value={f.num_requests} min={1} onChange={(v) => set("num_requests", v)} />
        <Num label="Concurrency" value={f.concurrency} min={1} onChange={(v) => set("concurrency", v)} />
        <Num label="Warmup" hint="requests" value={f.warmup_requests} min={0} onChange={(v) => set("warmup_requests", v)} />
        <Num label="Iterations" value={f.iterations} min={1} max={20} onChange={(v) => set("iterations", v)} />
      </div>
      <details className="text-sm">
        <summary className="cursor-pointer text-zinc-400">Decoding & advanced</summary>
        <div className="mt-3 grid grid-cols-2 gap-3">
          <Num label="Seed" value={f.seed} onChange={(v) => set("seed", v)} />
          <Num label="Temperature" value={f.temperature} min={0} max={2} step={0.1} onChange={(v) => set("temperature", v)} />
          <Num label="Top-p" value={f.top_p} min={0.01} max={1} step={0.05} onChange={(v) => set("top_p", v)} />
          <Num label="Timeout (s)" value={f.request_timeout_s} min={5} onChange={(v) => set("request_timeout_s", v)} />
          <label className="col-span-2 block text-sm"><span className="text-zinc-300">Prompt mode</span>
            <select className={input} value={f.prompt_mode} onChange={(e) => set("prompt_mode", e.target.value)}>
              <option value="token_ids">Token IDs (exact input length, server-verified)</option>
              <option value="text">Text (approximate input length)</option>
            </select>
          </label>
        </div>
      </details>

      {model && (
        <div className="space-y-1 text-sm">
          {checking && <div className="text-zinc-500">Checking feasibility…</div>}
          {!checking && report?.ok && <div className="text-emerald-300">✓ Workload is feasible on this server.</div>}
          {!checking && report?.errors.map((e) => <div key={e} className="text-red-300">✗ {e}</div>)}
          {!checking && report?.warnings.map((w) => <div key={w} className="text-amber-300">⚠ {w}</div>)}
        </div>
      )}
      {create.isError && <ErrorBox error={create.error} />}
      <button className={btn} disabled={create.isPending || !model || checking || !report?.ok}>{create.isPending ? "Launching…" : "Launch benchmark"}</button>
    </form>
  );
}
