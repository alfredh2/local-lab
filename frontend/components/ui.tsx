import type { JobStatus } from "@/lib/api";

export function Card({ title, action, children, className = "" }: {
  title?: string; action?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-zinc-800 bg-zinc-900/60 p-4 ${className}`}>
      {(title || action) && (
        <header className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium text-zinc-300">{title}</h2>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

const STATUS: Record<JobStatus, string> = {
  queued: "bg-zinc-700/50 text-zinc-300",
  preparing: "bg-sky-500/15 text-sky-300",
  running: "bg-blue-500/15 text-blue-300 animate-pulse",
  completed: "bg-emerald-500/15 text-emerald-300",
  failed: "bg-red-500/15 text-red-300",
  cancelled: "bg-amber-500/15 text-amber-300",
};
export function StatusBadge({ status }: { status: JobStatus }) {
  return <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${STATUS[status]}`}>{status}</span>;
}

export function Pill({ children, tone = "zinc" }: { children: React.ReactNode; tone?: "zinc" | "green" | "red" | "amber" }) {
  const t = { zinc: "bg-zinc-800 text-zinc-300", green: "bg-emerald-500/15 text-emerald-300",
    red: "bg-red-500/15 text-red-300", amber: "bg-amber-500/15 text-amber-300" }[tone];
  return <span className={`rounded-md px-1.5 py-0.5 text-xs ${t}`}>{children}</span>;
}

export function Bar({ pct, warn = 85 }: { pct: number; warn?: number }) {
  const p = Math.min(100, Math.max(0, pct));
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-800">
      <div className={`h-full rounded-full ${p >= warn ? "bg-amber-400" : "bg-blue-500"}`} style={{ width: `${p}%` }} />
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center">
      <div className="font-medium text-zinc-300">{title}</div>
      {children && <div className="mx-auto mt-1 max-w-md text-sm text-zinc-500">{children}</div>}
    </div>
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  return (
    <div className="rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-300">
      {error instanceof Error ? error.message : String(error)}
    </div>
  );
}

export const btn = "rounded-lg bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50";
export const btnGhost = "rounded-lg border border-zinc-700 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-800 disabled:opacity-50";
