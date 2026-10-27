"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api, type SystemStatus } from "@/lib/api";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/gpu", label: "GPU" },
  { href: "/models", label: "Models" },
  { href: "/experiments", label: "Experiments" },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const status = useQuery({ queryKey: ["system"], queryFn: () => api<SystemStatus>("/system/status"), refetchInterval: 5000 });
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  const worker = status.data?.worker;
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <aside className="border-b border-zinc-800 p-4 md:w-56 md:border-b-0 md:border-r">
        <div className="mb-4 text-lg font-semibold tracking-tight">LocalLab</div>
        <nav className="flex gap-1 md:flex-col">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href}
              className={`rounded-lg px-3 py-1.5 text-sm ${active(n.href) ? "bg-zinc-800 text-white" : "text-zinc-400 hover:bg-zinc-900"}`}>
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="mt-4 hidden text-xs text-zinc-500 md:block">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${status.isError ? "bg-red-500" : worker?.alive ? "bg-emerald-500" : "bg-amber-500"}`} />
            {status.isError ? "API offline" : worker?.alive ? "Worker online" : "Worker offline"}
          </div>
          {!status.isError && !worker?.alive && <div className="mt-1">Jobs will queue until a worker starts.</div>}
        </div>
      </aside>
      <main className="min-w-0 flex-1 p-4 md:p-8">{children}</main>
    </div>
  );
}
