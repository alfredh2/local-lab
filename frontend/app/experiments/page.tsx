"use client";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type Experiment, type ExperimentDetail, type ExperimentType } from "@/lib/api";
import { fmtDuration, fmtTime } from "@/lib/format";
import BenchmarkForm from "@/components/BenchmarkForm";
import { btn, btnGhost, Card, Empty, ErrorBox, StatusBadge } from "@/components/ui";

function NewExperiment({ types }: { types: ExperimentType[] }) {
  const router = useRouter();
  const qc = useQueryClient();
  const [type, setType] = useState((types.find((x) => x.type === "inference_benchmark") ?? types[0])?.type);
  const [values, setValues] = useState<Record<string, any>>({});
  const t = types.find((x) => x.type === type);
  const props: Record<string, any> = t?.config_schema.properties ?? {};
  const create = useMutation({
    mutationFn: () => {
      const config: Record<string, any> = {};
      for (const [k, spec] of Object.entries(props)) {
        const v = values[k] ?? spec.default;
        if (v !== undefined && v !== "") config[k] = v;
      }
      return api<ExperimentDetail>("/experiments", { method: "POST", body: JSON.stringify({ type, config }) });
    },
    onSuccess: (e) => { qc.invalidateQueries({ queryKey: ["experiments"] }); router.push(`/experiments/${e.id}`); },
  });
  if (!t) return null;
  return (
    <Card title="New experiment">
      <select value={type} onChange={(e) => { setType(e.target.value); setValues({}); }}
        className="mb-3 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm">
        {types.map((x) => <option key={x.type} value={x.type}>{x.title}</option>)}
      </select>
      <p className="mb-3 text-xs text-zinc-500">{t.description}</p>
      {type === "inference_benchmark" ? <BenchmarkForm key={type} type={t} /> : (
      <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
        {Object.entries(props).map(([k, spec]) => (
          <label key={k} className="block text-sm">
            <span className="text-zinc-300">{spec.title ?? k}</span>
            {spec.description && <span className="ml-2 text-xs text-zinc-500">{spec.description}</span>}
            {spec.type === "boolean" ? (
              <input type="checkbox" className="ml-3 align-middle" checked={values[k] ?? spec.default ?? false}
                onChange={(e) => setValues({ ...values, [k]: e.target.checked })} />
            ) : (
              <input type="number" min={spec.minimum} max={spec.maximum} value={values[k] ?? spec.default ?? ""}
                onChange={(e) => setValues({ ...values, [k]: e.target.value === "" ? "" : Number(e.target.value) })}
                className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-sm" />
            )}
          </label>
        ))}
        {create.isError && <ErrorBox error={create.error} />}
        <button className={btn} disabled={create.isPending}>{create.isPending ? "Launching…" : "Launch"}</button>
      </form>
      )}
    </Card>
  );
}

export default function ExperimentsPage() {
  const [picked, setPicked] = useState<number[]>([]);
  const toggle = (id: number) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));
  const exps = useQuery({ queryKey: ["experiments"], queryFn: () => api<Experiment[]>("/experiments"), refetchInterval: 2000 });
  const types = useQuery({ queryKey: ["experiment-types"], queryFn: () => api<ExperimentType[]>("/experiment-types") });
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Experiments</h1>
      <div className="grid gap-6 xl:grid-cols-[380px_1fr]">
        <div>{types.isError ? <ErrorBox error={types.error} /> : types.data && <NewExperiment types={types.data} />}</div>
        <div>
          {exps.isError ? <ErrorBox error={exps.error} /> : exps.data?.length === 0 ? (
            <Empty title="No experiments yet">Launch a system diagnostic to see the full lifecycle: queued → running → completed, with real GPU telemetry recorded.</Empty>
          ) : (
            <Card className="overflow-x-auto p-0">
              <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2 text-xs text-zinc-500">
                <span>Select completed benchmarks to compare them side by side.</span>
                {picked.length >= 2 ? <Link className={btn} href={`/compare?ids=${picked.join(",")}`}>Compare {picked.length}</Link>
                  : <span className={btnGhost + " cursor-not-allowed opacity-50"}>Compare</span>}
              </div>
              <table className="w-full min-w-[720px] text-sm">
                <thead className="text-left text-xs text-zinc-500"><tr className="border-b border-zinc-800">
                  {["", "Name", "Status", "Result", "Duration", "Created"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
                <tbody className="divide-y divide-zinc-800">
                  {exps.data?.map((e) => (
                    <tr key={e.id} className="hover:bg-zinc-900">
                      <td className="pl-4">{e.type === "inference_benchmark" && e.latest_job?.status === "completed" &&
                        <input type="checkbox" aria-label={`Select ${e.name}`} checked={picked.includes(e.id)} onChange={() => toggle(e.id)} />}</td>
                      <td className="min-w-[200px] px-4 py-2"><Link href={`/experiments/${e.id}`} className="font-medium hover:underline">{e.name}</Link>
                        <div className="text-xs text-zinc-500">{e.type.replace("_", " ")}</div></td>
                      <td className="px-4 py-2">{e.latest_job && <StatusBadge status={e.latest_job.status} />}</td>
                      <td className="whitespace-nowrap px-4 py-2 text-xs text-zinc-400">{e.type === "inference_benchmark" && e.latest_job?.results ?
                        `${Number(e.latest_job.results.summary.throughput.aggregate_output_tok_s).toFixed(0)} tok/s · ${String(e.config.model)}` : ""}</td>
                      <td className="whitespace-nowrap px-4 py-2 tabular-nums">{fmtDuration(e.latest_job?.started_at, e.latest_job?.finished_at)}</td>
                      <td className="whitespace-nowrap px-4 py-2 text-xs text-zinc-500">{fmtTime(e.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
