"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, API_URL, type Artifact, type ExperimentDetail, type JobLog } from "@/lib/api";
import { fmtDuration, fmtNum, fmtTime } from "@/lib/format";
import LineChartCard from "@/components/GpuChart";
import { btnGhost, Card, Empty, ErrorBox, Stat, StatusBadge } from "@/components/ui";

type Summ = { min: number; mean: number; max: number } | null;
const range = (s: Summ, d = 0, u = "") => (s ? `${s.min.toFixed(d)} / ${s.mean.toFixed(d)} / ${s.max.toFixed(d)}${u}` : "—");

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const qc = useQueryClient();
  const exp = useQuery({
    queryKey: ["experiment", id], queryFn: () => api<ExperimentDetail>(`/experiments/${id}`),
    refetchInterval: (q) => { const s = q.state.data?.jobs.at(-1)?.status; return s && ["completed", "failed", "cancelled"].includes(s) ? false : 1500; },
  });
  const job = exp.data?.jobs.at(-1);
  const active = job && !["completed", "failed", "cancelled"].includes(job.status);
  const logs = useQuery({
    queryKey: ["logs", job?.id], enabled: !!job, queryFn: () => api<JobLog[]>(`/jobs/${job!.id}/logs`),
    refetchInterval: active ? 1500 : false,
  });
  const arts = useQuery({
    queryKey: ["artifacts", job?.id, job?.status], enabled: !!job && !active,
    queryFn: () => api<Artifact[]>(`/jobs/${job!.id}/artifacts`),
  });
  const telemetryId = arts.data?.find((a) => a.name === "telemetry.json")?.id;
  const series = useQuery({
    queryKey: ["telemetry", telemetryId], enabled: !!telemetryId,
    queryFn: async () => (await fetch(`${API_URL}/api/artifacts/${telemetryId}/content`)).json(),
  });
  const cancel = useMutation({
    mutationFn: () => api(`/jobs/${job!.id}/cancel`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["experiment", id] }),
  });

  if (exp.isError) return <ErrorBox error={exp.error} />;
  if (!exp.data || !job) return <div className="h-24 animate-pulse rounded-xl bg-zinc-900" />;
  const e = exp.data;
  const r = job.results;
  const chart = (series.data?.samples ?? []).map((s: any) => ({
    t: s.t, mem: s.gpus[0]?.memory_used_mb / 1024, util: s.gpus[0]?.utilization_gpu_pct, power: s.gpus[0]?.power_w,
    temp: s.gpus[0]?.temperature_c, cpu: s.cpu_pct,
  }));
  return (
    <div className="space-y-6">
      <div>
        <Link href="/experiments" className="text-xs text-zinc-500 hover:underline">← Experiments</Link>
        <div className="mt-1 flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-semibold">{e.name}</h1>
          <StatusBadge status={job.status} />
          {active && (
            <button className={btnGhost} disabled={cancel.isPending || job.cancel_requested} onClick={() => cancel.mutate()}>
              {job.cancel_requested ? "Cancelling…" : "Cancel"}
            </button>
          )}
        </div>
        <p className="mt-1 text-xs text-zinc-500">Experiment #{e.id} · job #{job.id} · {e.type} · config {e.config_hash.slice(0, 12)}</p>
      </div>
      {cancel.isError && <ErrorBox error={cancel.error} />}
      {job.status === "failed" && (
        <Card title="Error"><pre className="max-h-64 overflow-auto whitespace-pre-wrap text-xs text-red-300">{job.error}</pre></Card>
      )}
      <div className="grid gap-4 md:grid-cols-4">
        <Card><Stat label="Queued" value={<span className="text-sm">{fmtTime(job.queued_at)}</span>} /></Card>
        <Card><Stat label="Started" value={<span className="text-sm">{fmtTime(job.started_at)}</span>} /></Card>
        <Card><Stat label="Finished" value={<span className="text-sm">{fmtTime(job.finished_at)}</span>} /></Card>
        <Card><Stat label="Run time" value={fmtDuration(job.started_at, job.finished_at)} /></Card>
      </div>

      {job.status === "completed" && r?.kind === "diagnostic" ? (
        <>
          <Card title="Recorded results (min / mean / max)">
            <div className="grid gap-4 md:grid-cols-3">
              <Stat label="Samples" value={r.sample_count} sub={`${fmtNum(r.actual_duration_s, 1)}s measured (requested ${r.requested_duration_s}s)`} />
              {Object.entries<any>(r.gpus).map(([i, g]) => (
                <div key={i} className="md:col-span-2 grid grid-cols-2 gap-4">
                  <Stat label={`GPU ${i} memory (MB)`} value={<span className="text-base">{range(g.memory_used_mb)}</span>} />
                  <Stat label={`GPU ${i} util (%)`} value={<span className="text-base">{range(g.utilization_gpu_pct)}</span>} />
                  <Stat label={`GPU ${i} temp (°C)`} value={<span className="text-base">{range(g.temperature_c)}</span>} />
                  <Stat label={`GPU ${i} power (W)`} value={<span className="text-base">{range(g.power_w)}</span>} />
                </div>
              ))}
              {r.cpu_pct && <Stat label="CPU (%)" value={<span className="text-base">{range(r.cpu_pct, 1)}</span>} />}
              {r.ram_used_gb && <Stat label="RAM used (GB)" value={<span className="text-base">{range(r.ram_used_gb, 1)}</span>} />}
            </div>
          </Card>
          {chart.length > 0 && (
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="GPU 0 memory (GB) & utilization (%)">
                <LineChartCard data={chart} xLabel={(v) => `${v}s`} series={[{ key: "mem", label: "Memory GB", color: "#60a5fa" }, { key: "util", label: "Util %", color: "#34d399", axis: "right" }]} />
              </Card>
              <Card title="GPU 0 power (W) & temperature (°C)">
                <LineChartCard data={chart} xLabel={(v) => `${v}s`} series={[{ key: "power", label: "Power W", color: "#fbbf24" }, { key: "temp", label: "Temp °C", color: "#f87171", axis: "right" }]} />
              </Card>
            </div>
          )}
        </>
      ) : !job.finished_at ? (
        <Empty title={job.status === "queued" ? "Waiting for the worker" : "Running…"}>
          {job.status === "queued" ? "Results appear here once a worker picks up this job. Start one with `make worker`." : "Results are recorded when the run completes."}
        </Empty>
      ) : job.status === "cancelled" ? <Empty title="Cancelled">No results were recorded.</Empty> : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Job log">
          <div className="max-h-72 space-y-0.5 overflow-auto font-mono text-xs">
            {logs.data?.map((l) => (
              <div key={l.id} className={l.level === "ERROR" ? "text-red-300" : "text-zinc-400"}>
                <span className="text-zinc-600">{new Date(l.ts + "Z").toLocaleTimeString()}</span> {l.level} {l.message}
              </div>
            ))}
          </div>
        </Card>
        <Card title="Configuration & environment">
          <pre className="max-h-72 overflow-auto text-xs text-zinc-400">{JSON.stringify({ config: e.config, seed: job.seed, environment: e.environment }, null, 2)}</pre>
        </Card>
      </div>
    </div>
  );
}
