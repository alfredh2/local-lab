// Same-origin by default (proxied by Next). Set NEXT_PUBLIC_API_URL to call an API directly.
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export type Gpu = {
  index: number; name: string; uuid: string;
  memory_total_mb: number; memory_used_mb: number; memory_free_mb: number;
  utilization_gpu_pct: number | null; utilization_mem_pct: number | null;
  temperature_c: number | null; power_w: number | null; power_limit_w: number | null;
  clock_sm_mhz: number | null; clock_mem_mhz: number | null;
  processes: { pid: number; name: string | null; used_memory_mb: number | null }[];
};
export type GpuSnapshot = {
  available: boolean; error: string | null; timestamp: string;
  driver_version: string | null; cuda_driver_version: string | null; gpus: Gpu[];
};
export type GpuPoint = {
  timestamp: string; index: number; memory_used_mb: number;
  utilization_gpu_pct: number | null; temperature_c: number | null; power_w: number | null;
};
export type SystemStatus = {
  hostname: string; os: string; kernel: string; cpu_model: string | null; cpu_threads: number;
  memory_total_gb: number; memory_used_gb: number; disk_total_gb: number; disk_free_gb: number;
  python_version: string; locallab_version: string; packages: Record<string, string>;
  gpu: GpuSnapshot; worker: { alive: boolean; worker_id: string | null; last_heartbeat: string | null };
};
export type ModelRec = {
  id: number; name: string; source: string; ref: string; revision: string | null;
  format: string | null; quantization: string | null; size_bytes: number | null;
  meta: Record<string, unknown>; available: boolean; last_seen_at: string;
};
export type ScanResult = { discovered: number; added: number; updated: number; marked_unavailable: number; warnings: string[] };
export type JobStatus = "queued" | "preparing" | "running" | "completed" | "failed" | "cancelled";
export type Job = {
  id: number; experiment_id: number; status: JobStatus; cancel_requested: boolean;
  worker_id: string | null; error: string | null; seed: number | null;
  results: Record<string, any> | null;
  progress: Record<string, any> | null;
  queued_at: string; started_at: string | null; finished_at: string | null;
};
export type Experiment = {
  id: number; type: string; name: string; config: Record<string, unknown>; config_hash: string;
  created_at: string; latest_job: Job | null;
};
export type ExperimentDetail = Experiment & { environment: Record<string, any>; jobs: Job[] };
export type JobLog = { id: number; ts: string; level: string; message: string };
export type Artifact = { id: number; job_id: number; name: string; kind: string; size_bytes: number; sha256: string };
export type ExperimentType = {
  type: string; title: string; description: string; config_schema: any; presets: Record<string, Record<string, any>>;
};
export type ValidationReport = { ok: boolean; errors: string[]; warnings: string[]; info: Record<string, any> | null };
export type ProbeResult = {
  runtime: string; base_url: string; version: string | null; kv_cache_tokens: number | null;
  models: { id: string; root: string | null; max_model_len: number | null }[];
};

export class ApiError extends Error {}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/api${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(`Cannot reach the LocalLab API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    throw new ApiError(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}
