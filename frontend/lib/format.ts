// Backend timestamps are naive UTC; make that explicit before parsing.
export const parseUtc = (s: string) => new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : s + "Z");
export const fmtTime = (s?: string | null) => (s ? parseUtc(s).toLocaleString() : "—");
export const fmtBytes = (n?: number | null) => {
  if (n == null) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i > 1 ? 1 : 0)} ${u[i]}`;
};
export const fmtMb = (mb: number) => (mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`);
export const fmtNum = (n: number | null | undefined, d = 1, unit = "") => (n == null ? "—" : `${n.toFixed(d)}${unit}`);
export function fmtDuration(a?: string | null, b?: string | null) {
  if (!a) return "—";
  const ms = (b ? parseUtc(b) : new Date()).getTime() - parseUtc(a).getTime();
  const s = Math.max(0, ms / 1000);
  return s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}
export const fmtMs = (v: number | null | undefined) => (v == null ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${v.toFixed(1)} ms`);
export const fmtInt = (v: number | null | undefined) => (v == null ? "—" : Math.round(v).toLocaleString());
