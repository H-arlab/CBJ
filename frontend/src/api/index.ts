// Typed fetch wrappers for the H-Walker backend.
// Contract mirrors backend/routers/{datasets,analyze,compute,stats,graphs,claude}.py.

import type { Dataset } from '../store/page';

const BASE = '';

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ============================================================
// Datasets
// ============================================================

export async function uploadDataset(file: File): Promise<Dataset> {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(BASE + '/api/datasets/upload', { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const listDatasets = () => json<Dataset[]>('/api/datasets');
export const getDataset = (id: string) => json<Dataset & { sample: unknown[] }>(`/api/datasets/${id}`);
export const deleteDataset = (id: string) => fetch(BASE + `/api/datasets/${id}`, { method: 'DELETE' });

export const saveMapping = (id: string, columns: Record<string, string>) =>
  json<{ updated: number }>(`/api/datasets/${id}/mapping`, {
    method: 'POST',
    body: JSON.stringify({ columns }),
  });

// Phase 2 · study-level tags
export const updateDatasetMeta = (
  id: string,
  meta: Partial<Pick<Dataset, 'subject_id' | 'condition' | 'group' | 'date'>>,
) => json<Dataset>(`/api/datasets/${id}/meta`, {
  method: 'PATCH',
  body: JSON.stringify(meta),
});

// ============================================================
// Analyze (GET /api/analyze/{ds_id})
// ============================================================

export interface SideStride {
  n_strides: number;
  stride_time_mean: number;
  stride_time_std: number;
  stride_time_cv: number;
  stride_length_mean: number;
  stride_length_std: number;
  cadence: number;
  stance_pct_mean: number;
  stance_pct_std: number;
  swing_pct_mean: number;
  swing_pct_std: number;
  force_tracking: { rmse: number; mae: number; peak_error: number };
  stride_times_list: number[];
  stride_lengths_list: number[];
}

export interface AnalysisPayload {
  mode: 'hwalker';
  filename: string;
  n_samples: number;
  duration_s: number;
  sample_rate: number;
  symmetry: { stride_time: number; stride_length: number; force: number; stance: number };
  fatigue: { left_pct_change: number; right_pct_change: number };
  left: SideStride;
  right: SideStride;
  profiles: {
    left:  { available: boolean; n_points?: number; mean?: number[]; std?: number[]; des_mean?: number[]; des_std?: number[] };
    right: { available: boolean; n_points?: number; mean?: number[]; std?: number[]; des_mean?: number[]; des_std?: number[] };
  };
}

export interface GenericAnalysisPayload {
  fallback_mode: 'generic';
  filename: string;
  n_samples: number;
  n_columns: number;
  columns: string[];
  descriptive: Record<string, { n: number; mean: number; std: number; min: number; max: number; median: number }>;
  note: string;
}

export type AnalyzeResponse = AnalysisPayload | GenericAnalysisPayload;

export const analyzeDataset = (dsId: string) =>
  json<AnalyzeResponse>(`/api/analyze/${dsId}`);

// ============================================================
// Compute (POST /api/compute)
// ============================================================

export type ComputeMetricKey =
  | 'per_stride' | 'impulse' | 'loading_rate' | 'rom' | 'cadence' | 'target_dev';

export interface ComputeRequest {
  dataset_id: string;
  metric: ComputeMetricKey;
  options?: Record<string, unknown>;
}

export interface ComputeResponse {
  label: string;
  cols: string[];
  rows: string[][];
  summary: { mean: string[] };
  meta?: Record<string, unknown>;
}

export const computeMetric = (req: ComputeRequest) =>
  json<ComputeResponse>('/api/compute', { method: 'POST', body: JSON.stringify(req) });

// ============================================================
// Stats (POST /api/stats)
// ============================================================

export type StatOpKey =
  | 'ttest_paired' | 'ttest_welch' | 'anova1' | 'pearson' | 'cohens_d' | 'shapiro';

export interface StatsRequest {
  op: StatOpKey;
  // Raw series
  a?: number[];
  b?: number[];
  groups?: number[][];
  paired?: boolean;
  // Dataset-backed convenience (single dataset + columns)
  dataset_id?: string;
  a_col?: string;
  b_col?: string;
  groups_cols?: string[];
  // Phase 3 cross-dataset
  datasets_a?: Array<{ id: string; metric: string }>;
  datasets_b?: Array<{ id: string; metric: string }>;
  datasets_groups?: Array<Array<{ id: string; metric: string }>>;
}

export interface MetricDescriptor {
  key: string;
  label: string;
  unit: string;
  side: string;
  kind: string;
}

export const listStatMetrics = () =>
  json<MetricDescriptor[]>('/api/stats/metrics');

export interface StatsResponse {
  op: string;
  name: string;
  stat: number;
  stat_name: string;
  p: number | null;
  df: number | number[] | null;
  effect_size: { name: string; value: number; label?: string } | null;
  ci95: [number, number] | null;
  n: number | number[];
  assumption: { name: string; p: number; passed: boolean } | null;
  fallback_used: boolean;
  warning?: string | null;
  summary: string;
}

export const runStats = (req: StatsRequest) =>
  json<StatsResponse>('/api/stats', { method: 'POST', body: JSON.stringify(req) });

// ============================================================
// Graph render (POST /api/graphs/render) — returns Blob
// ============================================================

export interface RenderGraphRequest {
  template: string;
  preset?: string;
  variant?: 'col1' | 'col2' | 'onehalf';
  format?: 'svg' | 'pdf' | 'eps' | 'png' | 'tiff';
  dpi?: number;
  stride_avg?: boolean;
  colorblind_safe?: boolean;
  keep_palette?: boolean;
  dataset_id?: string;   // single-dataset path (legacy)
  datasets?: Array<{ id: string; label?: string; color?: string }>;  // Phase 1 overlay
  title?: string;
}

export const renderGraph = async (req: RenderGraphRequest): Promise<Blob> => {
  const res = await fetch(BASE + '/api/graphs/render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.blob();
};

// ============================================================
// Bundle export (POST /api/graphs/bundle) — returns Blob
// ============================================================

export interface BundleRequest {
  preset: string;
  variant?: 'col1' | 'col2' | 'onehalf';
  format?: 'svg' | 'pdf' | 'eps' | 'png' | 'tiff';
  dpi?: number;
  cells: Array<{ id: string; template: string; stride_avg?: boolean; preset?: string; variant?: string }>;
  include_readme?: boolean;
}

export const exportBundle = async (req: BundleRequest): Promise<Blob> => {
  const res = await fetch(BASE + '/api/graphs/bundle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.blob();
};

// ============================================================
// Paper bundle (POST /api/paper/bundle) — Phase 5 RUN PAPER
// ============================================================

export interface PaperBundleCell {
  id: string;
  type: 'graph' | 'stat' | 'compute';
  title?: string;
  graph?: string;
  stride_avg?: boolean;
  dataset_id?: string;
  datasets?: Array<{ id: string; label?: string; color?: string }>;
  op?: string;
  a_col?: string;
  b_col?: string;
  datasets_a?: Array<{ id: string; metric: string }>;
  datasets_b?: Array<{ id: string; metric: string }>;
  metric?: string;
}

export interface PaperBundleRequest {
  preset: string;
  variant?: 'col1' | 'col2' | 'onehalf';
  format?: 'pdf' | 'svg' | 'eps' | 'png' | 'tiff';
  dpi?: number;
  paper_title?: string;
  cells: PaperBundleCell[];
  colorblind_safe?: boolean;
}

export const paperBundle = async (req: PaperBundleRequest): Promise<Blob> => {
  const res = await fetch(BASE + '/api/paper/bundle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return res.blob();
};

// ============================================================
// Claude (POST /api/claude/complete)
// ============================================================

export interface ToolUseBlock {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface ClaudeCompleteRequest {
  prompt: string;
  context: {
    cells: unknown[];
    active_dataset_id: string | null;
    history?: Array<{ role: 'user' | 'assistant'; content: string }>;
    datasets?: unknown[];   // Phase 2I · analysis summary per dataset for LLM diagnosis
  };
}

export interface ClaudeCompleteResponse {
  reply: string;
  tool_uses?: ToolUseBlock[];
  suggested_cells?: unknown[];   // legacy
}

export const claudeComplete = (req: ClaudeCompleteRequest) =>
  json<ClaudeCompleteResponse>('/api/claude/complete', {
    method: 'POST',
    body: JSON.stringify(req),
  });

export const claudeHealth = () =>
  json<{ provider: string; model: string; key_present: boolean }>('/api/claude/health');


// ============================================================
// Inspector — per-sync zoom / pan over raw CSV columns
// ============================================================

export interface SyncBoundary {
  index: number;
  t_start: number;
  t_end: number;
  duration: number;
}

export interface SyncsResponse {
  column: string | null;
  n_samples: number;
  sample_rate_hz: number | null;
  cycles: SyncBoundary[];
}

export const listSyncs = (dsId: string) =>
  json<SyncsResponse>(`/api/inspector/${dsId}/syncs`);

export interface WindowRequest {
  columns: string[];
  t_start: number;
  t_end: number;
  max_points?: number;
}

export interface WindowSeries { name: string; y: number[]; }

export interface WindowResponse {
  t: number[];
  series: WindowSeries[];
  n_total: number;
  n_returned: number;
  columns_missing: string[];
}

export const fetchWindow = (dsId: string, req: WindowRequest) =>
  json<WindowResponse>(`/api/inspector/${dsId}/window`, {
    method: "POST",
    body: JSON.stringify(req),
  });



// ============================================================
// Sync alignment — multi-source experiments (Robot + Motion + Loadcell)
// ============================================================

export interface SyncEdgeResponse {
  dataset_id: string;
  sync_column: string | null;
  first_falling_t_s: number | null;
}

export const fetchSyncEdge = (dsId: string) =>
  json<SyncEdgeResponse>(`/api/sync/${dsId}/edge`);

export interface AlignRequest {
  dataset_ids: string[];
  target_fs_hz?: number;
  columns_per_source?: Record<string, string[]>;
}

export interface AlignResponse {
  target_fs_hz: number;
  t_min_s: number;
  t_max_s: number;
  n_grid_samples: number;
  t_aligned_s: number[];
  series: Record<string, Record<string, number[]>>;
  sources: Array<{ dataset_id: string; columns: string[] }>;
  sync_offsets_s: Record<string, number | null>;
  warnings: string[];
}

export const alignSources = (req: AlignRequest) =>
  json<AlignResponse>("/api/sync/align", {
    method: "POST",
    body: JSON.stringify(req),
  });
