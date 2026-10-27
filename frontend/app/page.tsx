"use client";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api, type Experiment, type ModelRec, type SystemStatus } from "@/lib/api";
import { fmtMb, fmtNum, fmtTime } from "@/lib/format";
import { Bar, Card, Empty, ErrorBox, Stat, StatusBadge } from "@/components/ui";

export default function Overview() {
  const sys = useQuery({ queryKey: ["system"], queryFn: () => api<SystemStatus>("/system/status"), refetchInterval: 3000 });
  const models = useQuery({ queryKey: ["models"], queryFn: () => api<ModelRec[]>("/models") });
  const exps = useQuery({ queryKey: ["experiments"], queryFn: () => api<Experiment[]>("/experiments?limit=5"), refetchInterval: 3000 });

  if (sys.isError) return <ErrorBox error={sys.error} />;
  const s = sys.data;
  const gpu = s?.gpu.gpus[0];
  const available = models.data?.filter((m) => m.available).length ?? 0;
  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Overview</h1>
      <div className="grid gap-4 md:grid-cols-3">
        <Card title="GPU">
          {!s ? <Skeleton /> : !s.gpu.available ? (
            <div className="text-sm text-red-300">NVML unavailable: {s.gpu.error}</div>
          ) : gpu && (
            <div className="space-y-2">
              <div className="text-sm font-medium">{gpu.name}</div>
              <Bar pct={(gpu.memory_used_mb / gpu.memory_total_mb) * 100} />
              <div className="flex justify-between text-xs text-zinc-400">
                <span>{fmtMb(gpu.memory_used_mb)} / {fmtMb(gpu.memory_total_mb)}</span>
                <span>{fmtNum(gpu.utilization_gpu_pct, 0, "%")} · {fmtNum(gpu.temperature_c, 0, "°C")} · {fmtNum(gpu.power_w, 0, " W")}</span>
              </div>
              <Link href="/gpu" className="text-xs text-blue-400 hover:underline">Details →</Link>
            </div>
          )}
        </Card>
        <Card title="System">
          {!s ? <Skeleton /> : (
            <div className="space-y-1 text-sm text-zinc-300">
              <div className="truncate">{s.cpu_model ?? "CPU"} ({s.cpu_threads} threads)</div>
              <div>RAM {fmtNum(s.memory_used_gb, 0)} / {fmtNum(s.memory_total_gb, 0)} GB</div>
              <div>Disk free {fmtNum(s.disk_free_gb, 0)} GB</div>
              <div className="text-xs text-zinc-500">Driver {s.gpu.driver_version ?? "—"} · CUDA {s.gpu.cuda_driver_version ?? "—"}</div>
            </div>
          )}
        </Card>
        <Card title="Platform">
          <div className="grid grid-cols-2 gap-4">
            <Stat label="Models" value={models.data ? available : "…"} sub={models.data ? `${models.data.length} registered` : undefined} />
            <Stat label="Worker" value={s ? (s.worker.alive ? "Online" : "Offline") : "…"}
              sub={s?.worker.last_heartbeat ? `seen ${fmtTime(s.worker.last_heartbeat)}` : "never seen"} />
          </div>
        </Card>
      </div>
      <Card title="Recent experiments" action={<Link href="/experiments" className="text-xs text-blue-400 hover:underline">All →</Link>}>
        {exps.isError ? <ErrorBox error={exps.error} /> : exps.data?.length === 0 ? (
          <Empty title="No experiments yet">Run a system diagnostic to exercise the full job lifecycle with real GPU telemetry.
            <div className="mt-3"><Link href="/experiments" className="text-blue-400 hover:underline">Create an experiment</Link></div></Empty>
        ) : (
          <ul className="divide-y divide-zinc-800">
            {exps.data?.map((e) => (
              <li key={e.id} className="flex items-center justify-between py-2 text-sm">
                <Link href={`/experiments/${e.id}`} className="hover:underline">{e.name}</Link>
                <span className="flex items-center gap-3 text-xs text-zinc-500">{fmtTime(e.created_at)}
                  {e.latest_job && <StatusBadge status={e.latest_job.status} />}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}

const Skeleton = () => <div className="h-14 animate-pulse rounded bg-zinc-800/60" />;
