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

export type GpuPreflight = {
  gpu_index: number; name: string | null; total_mb: number; free_mb: number; requested_mb: number;
  compute_capability: [number, number] | null; utilization_pct: number | null;
  other_processes: { pid: number; name: string; memory_mb: number }[];
  contended: boolean; override: boolean; errors: string[]; warnings: string[];
};
export type RuntimeConfigRec = { id: number; name: string; runtime: string; registry_model_id: number | null; config: Record<string, any>; created_at: string };
export type ManagedServerRec = {
  id: number; job_id: number | null; container_id: string | null; container_name: string; status: string;
  image_ref: string; image_id: string | null; image_digest: string | null; host_port: number | null;
  config: Record<string, any>; command: string[] | null; model: Record<string, any> | null;
  launched_at: string; ready_at: string | null; startup_s: number | null; stopped_at: string | null;
  stop_reason: string | null; exit_code: number | null; oom_killed: boolean | null; log_tail: string | null;
};

export type ConfigResultRec = {
  id: number; experiment_id: number; job_id: number | null; key: string; label: string; config: Record<string, any>;
  status: "pending" | "running" | "completed" | "failed" | "skipped" | "cancelled"; failure_kind: string | null; error: string | null;
  attempts: number; managed_server_id: number | null; results: Record<string, any> | null; started_at: string | null; finished_at: string | null;
};
export type MetricInfo = { id: string; label: string; unit: string; direction: "max" | "min"; kind: string; percent: boolean };
export type ParetoPoint = { id: string; label: string; status: string; contended: boolean; config: Record<string, any>;
  values: Record<string, { value: number; display: number; lo: number | null; hi: number | null; n: number | null }> };
export type ParetoResponse = {
  cell: { io_index: number; concurrency: number | null; common_levels: number[]; default_concurrency: number | null; io_pairs: { input_tokens: number; output_tokens: number }[] };
  metrics: MetricInfo[]; points: ParetoPoint[]; not_analysed: { id: string; label: string; status: string; reason: string }[]; notes: string[];
  analysis: null | {
    objectives: { metric: string; direction: string }[]; constraints: any[]; feasible: string[]; front: string[]; warnings: string[];
    excluded: { id: string; reason: string; detail: string }[]; dominated: { id: string; dominated_by: { by: string; support: string }[] }[];
    weighted?: { weights: Record<string, number>; formula: string; caveat: string; ranking: { id: string; score: number; normalized: Record<string, number> }[] };
  };
};
