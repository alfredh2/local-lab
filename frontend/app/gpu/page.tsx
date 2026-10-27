"use client";
import { useQuery } from "@tanstack/react-query";
import { api, type GpuPoint, type GpuSnapshot } from "@/lib/api";
import { fmtMb, fmtNum, parseUtc } from "@/lib/format";
import LineChartCard from "@/components/GpuChart";
import { Bar, Card, Empty, ErrorBox, Stat } from "@/components/ui";

export default function GpuPage() {
  const snap = useQuery({ queryKey: ["gpu"], queryFn: () => api<GpuSnapshot>("/gpu"), refetchInterval: 2000 });
  const hist = useQuery({ queryKey: ["gpu-history"], queryFn: () => api<GpuPoint[]>("/gpu/history"), refetchInterval: 2000 });
  if (snap.isError) return <ErrorBox error={snap.error} />;
  if (snap.data && !snap.data.available) return <ErrorBox error={`GPU telemetry unavailable: ${snap.data.error}`} />;
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">GPU</h1>
        {snap.data && <p className="text-sm text-zinc-500">Driver {snap.data.driver_version} · CUDA {snap.data.cuda_driver_version} · refreshes every 2s</p>}
      </div>
      {snap.data?.gpus.map((g) => {
        const points = (hist.data ?? []).filter((p) => p.index === g.index).map((p) => ({
          t: parseUtc(p.timestamp).getTime(), mem: p.memory_used_mb / 1024, util: p.utilization_gpu_pct, power: p.power_w, temp: p.temperature_c,
        }));
        return (
          <div key={g.uuid} className="space-y-4">
            <Card title={`GPU ${g.index} — ${g.name}`}>
              <Bar pct={(g.memory_used_mb / g.memory_total_mb) * 100} />
              <div className="mt-1 text-xs text-zinc-400">{fmtMb(g.memory_used_mb)} used of {fmtMb(g.memory_total_mb)} ({fmtMb(g.memory_free_mb)} free)</div>
              <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-5">
                <Stat label="Utilization" value={fmtNum(g.utilization_gpu_pct, 0, "%")} />
                <Stat label="Temperature" value={fmtNum(g.temperature_c, 0, "°C")} />
                <Stat label="Power" value={fmtNum(g.power_w, 0, " W")} sub={g.power_limit_w ? `limit ${fmtNum(g.power_limit_w, 0)} W` : undefined} />
                <Stat label="SM clock" value={fmtNum(g.clock_sm_mhz, 0, " MHz")} />
                <Stat label="Mem clock" value={fmtNum(g.clock_mem_mhz, 0, " MHz")} />
              </div>
            </Card>
            <div className="grid gap-4 lg:grid-cols-2">
              <Card title="Memory (GB) and utilization (%) — since API start">
                <LineChartCard data={points} xKey="t" xLabel={(v) => new Date(v).toLocaleTimeString()}
                  series={[{ key: "mem", label: "Memory GB", color: "#60a5fa" }, { key: "util", label: "Util %", color: "#34d399", axis: "right" }]} />
              </Card>
              <Card title="Power (W) and temperature (°C)">
                <LineChartCard data={points} xKey="t" xLabel={(v) => new Date(v).toLocaleTimeString()}
                  series={[{ key: "power", label: "Power W", color: "#fbbf24" }, { key: "temp", label: "Temp °C", color: "#f87171", axis: "right" }]} />
              </Card>
            </div>
            <Card title="Processes using this GPU">
              {g.processes.length === 0 ? <Empty title="No processes" /> : (
                <table className="w-full text-sm"><thead className="text-left text-xs text-zinc-500">
                  <tr><th className="py-1">PID</th><th>Name</th><th className="text-right">Memory</th></tr></thead>
                  <tbody className="divide-y divide-zinc-800">{g.processes.map((p) => (
                    <tr key={p.pid}><td className="py-1.5 tabular-nums">{p.pid}</td><td className="truncate">{p.name ?? "—"}</td>
                      <td className="text-right tabular-nums">{p.used_memory_mb == null ? "—" : fmtMb(p.used_memory_mb)}</td></tr>
                  ))}</tbody></table>
              )}
            </Card>
          </div>
        );
      })}
    </div>
  );
}
