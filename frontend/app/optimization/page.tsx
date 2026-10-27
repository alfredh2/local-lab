"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api, type Experiment } from "@/lib/api";
import { fmtDuration, fmtTime } from "@/lib/format";
import MatrixForm from "@/components/MatrixForm";
import { Card, Empty, ErrorBox, StatusBadge } from "@/components/ui";

export default function OptimizationLab() {
  const exps = useQuery({ queryKey: ["experiments", "optimization"], queryFn: () => api<Experiment[]>("/experiments?type=optimization_matrix"), refetchInterval: 3000 });
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Optimization Lab</h1>
        <p className="mt-1 max-w-3xl text-sm text-zinc-500">Compare model and runtime configurations on speed <em>and</em> task quality, then see which are worth considering. Every number comes from a recorded run on this machine.</p>
      </div>
      <div className="grid gap-6 xl:grid-cols-[430px_1fr]">
        <Card title="New matrix"><MatrixForm /></Card>
        <div>
          {exps.isError ? <ErrorBox error={exps.error} /> : exps.data?.length === 0 ? (
            <Empty title="No optimization runs yet">Pick models and the settings to vary on the left. LocalLab validates every combination, launches one managed vLLM server per configuration, runs performance and quality tests against that same server, and keeps each configuration&apos;s result even if another fails.</Empty>
          ) : (
            <Card title="Runs" className="overflow-x-auto p-0">
              <table className="w-full min-w-[560px] text-sm"><thead className="text-left text-xs text-zinc-500"><tr className="border-b border-zinc-800">
                {["Name", "Status", "Configurations", "Duration", "Created"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}</tr></thead>
                <tbody className="divide-y divide-zinc-800">{exps.data?.map((e) => { const r = e.latest_job?.results; return (
                  <tr key={e.id} className="hover:bg-zinc-900">
                    <td className="px-4 py-2"><Link href={`/optimization/${e.id}`} className="font-medium hover:underline">{e.name}</Link></td>
                    <td className="px-4 py-2">{e.latest_job && <StatusBadge status={e.latest_job.status} />}</td>
                    <td className="px-4 py-2 text-xs text-zinc-400">{r ? Object.entries<number>(r.counts).map(([k, v]) => `${v} ${k}`).join(" · ") : e.latest_job?.progress?.matrix ? `${e.latest_job.progress.matrix.config_index + 1} of ${e.latest_job.progress.matrix.configs_total}` : ""}</td>
                    <td className="whitespace-nowrap px-4 py-2 tabular-nums">{fmtDuration(e.latest_job?.started_at, e.latest_job?.finished_at)}</td>
                    <td className="whitespace-nowrap px-4 py-2 text-xs text-zinc-500">{fmtTime(e.created_at)}</td></tr>); })}</tbody></table>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
