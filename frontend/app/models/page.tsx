"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type ModelRec, type ScanResult } from "@/lib/api";
import { fmtBytes, fmtTime } from "@/lib/format";
import { btn, Card, Empty, ErrorBox, Pill } from "@/components/ui";

export default function ModelsPage() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: () => api<ModelRec[]>("/models") });
  const scan = useMutation({
    mutationFn: () => api<ScanResult>("/models/scan", { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Models</h1>
        <button className={btn} onClick={() => scan.mutate()} disabled={scan.isPending}>{scan.isPending ? "Scanning…" : "Scan for models"}</button>
      </div>
      {scan.isError && <ErrorBox error={scan.error} />}
      {scan.data && (
        <div className="rounded-lg border border-zinc-800 p-3 text-sm text-zinc-300">
          Found {scan.data.discovered}: {scan.data.added} new, {scan.data.updated} updated, {scan.data.marked_unavailable} no longer available.
          {scan.data.warnings.map((w) => <div key={w} className="text-amber-300">⚠ {w}</div>)}
        </div>
      )}
      {models.isError ? <ErrorBox error={models.error} /> : models.data?.length === 0 ? (
        <Empty title="No models registered">Scan the Hugging Face cache and Ollama for locally installed models.</Empty>
      ) : (
        <Card className="overflow-x-auto p-0">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="text-left text-xs text-zinc-500"><tr className="border-b border-zinc-800">
              {["Name", "Source", "Format", "Quantization", "Size", "Status", "Last seen"].map((h) => <th key={h} className="px-4 py-2 font-medium">{h}</th>)}
            </tr></thead>
            <tbody className="divide-y divide-zinc-800">
              {models.data?.map((m) => (
                <tr key={m.id} className={m.available ? "" : "opacity-60"}>
                  <td className="px-4 py-2"><div className="font-medium">{m.name}</div>
                    <div className="max-w-xs truncate text-xs text-zinc-500" title={m.ref}>{m.revision ? `${m.revision.slice(0, 8)} · ` : ""}{m.ref}</div></td>
                  <td className="px-4 py-2"><Pill>{m.source}</Pill></td>
                  <td className="px-4 py-2">{m.format ?? "—"}</td>
                  <td className="px-4 py-2">{m.quantization ?? "—"}</td>
                  <td className="px-4 py-2 tabular-nums">{fmtBytes(m.size_bytes)}</td>
                  <td className="px-4 py-2">{m.available ? <Pill tone="green">available</Pill>
                    : <Pill tone="amber" >{(m.meta.note as string) ?? "unavailable"}</Pill>}</td>
                  <td className="px-4 py-2 text-xs text-zinc-500">{fmtTime(m.last_seen_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
