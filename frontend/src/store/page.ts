// Zustand workspace store — per HANDOFF §1
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { SEED_CELLS } from '../data/seedCells';
import { SEED_DATASETS } from '../data/seedDatasets';
import { CANONICAL_RECIPES } from '../data/canonicalRecipes';
import {
  analyzeDataset, computeMetric, renderGraph, runStats, updateDatasetMeta,
  type ComputeMetricKey, type StatOpKey, type StatsResponse, type ComputeResponse,
  type AnalyzeResponse, type AnalysisPayload,
} from '../api';

export type CellType = 'graph' | 'stat' | 'compute' | 'llm' | 'inspector';

export interface Cell {
  id: string;
  type: CellType;
  graph?: string;
  op?: string;
  metric?: string;
  dsIds: string[];
  preset?: string;
  strideAvg?: boolean;
  fmt?: 'apa' | 'ieee' | 'csv';
  inputs?: { a: string; b: string };
  title?: string;
  // LLM cell extras
  prompt?: string;
  refs?: string[];
  answer?: { text: string[]; spawns?: Array<{ label: string; action: string }> };
  // Inspector cell extras
  inspectorChannels?: string[];   // CSV columns currently displayed
  inspectorSyncIdx?: number | null;  // -1 / null = full trial
  // Phase 2B: live backend state
  loading?: boolean;
  error?: string;
  computeData?: ComputeResponse;
  statData?: StatsResponse;
  previewBlobUrl?: string;
  previewVariant?: 'col1' | 'col2' | 'onehalf';
  // Phase 1: multi-dataset overlay. When `series` has ≥ 2 entries, the
  // graph renders as a multi-subject overlay with one colored trace per
  // entry. dsIds mirrors `series[].dsId` for backwards compat + filtering.
  series?: Array<{ dsId: string; label?: string; color?: string }>;
  // Phase 3: cross-file stat cell — if set, overrides a_col/b_col mode.
  statDatasetsA?: Array<{ id: string; metric: string }>;
  statDatasetsB?: Array<{ id: string; metric: string }>;
  statMetric?: string;   // default metric when the user picks datasets
}

export interface DatasetColumn {
  name: string;
  unit: string;
  mapped?: string;
  mappedManual?: boolean;
}

export type SourceKind = 'robot' | 'loadcell' | 'motion' | 'unknown';

export interface Dataset {
  id: string;
  name: string;
  tag: string;
  // Legacy column-based grouping (drives recipe picker)
  kind: 'force' | 'imu' | 'cop' | 'emg' | 'trials' | 'mixed';
  // Phase 3: experimental source category. Filename-first (Robot* /
  // Loadcell* / Motion*) with column-signature fallback.
  source_kind?: SourceKind;
  source_confidence?: number;       // 0..1
  source_cues?: string[];           // human-readable matched cues
  rows: number;
  dur: string;
  hz: string;
  cols: DatasetColumn[];
  active: boolean;
  recipeState: Record<string, boolean>;
  // Phase 2: study-level tags (auto-parsed from filename, user-editable)
  subject_id?: string;
  condition?: string;
  group?: string;
  date?: string;
  // Phase 2B: analysis state
  analysis?: AnalyzeResponse;
  analyzing?: boolean;
  analyzeError?: string;
}

export type DrawerKind = null | 'history' | 'exports' | 'stats' | 'settings';

export interface HistoryEntry {
  id: string;
  ts: number;                // unix ms
  kind: 'upload' | 'apply' | 'add' | 'remove' | 'run' | 'preset' | 'chat' | 'tool';
  label: string;
  actor: 'you' | 'Claude';
  meta?: Record<string, unknown>;
}

interface PageState {
  cells: Cell[];
  datasets: Dataset[];
  currentPreset: string;
  globalPreset: string;
  pageTitle: string;
  history: HistoryEntry[];
  drawer: DrawerKind;
  focusCellId: string | null;
  cmdkOpen: boolean;
  mapperOpen: boolean;
  mapperDsId: string | null;
  helpOpen: boolean;
  toast: { msg: string; id: number } | null;
  runAllBusy: boolean;

  addCell: (c: Cell, at?: number) => void;
  updateCell: (id: string, patch: Partial<Cell>) => void;
  removeCell: (id: string) => void;
  duplicateCell: (id: string) => void;
  reorderCells: (from: number, to: number) => void;

  addDataset: (d: Dataset) => void;
  removeDataset: (id: string) => void;
  setActiveDataset: (id: string) => void;
  toggleRecipe: (dsId: string, recipeId: string) => void;
  applyRecipes: (dsId: string) => Promise<void>;
  analyzeIfNeeded: (dsId: string) => Promise<AnalyzeResponse | undefined>;
  setDatasetMeta: (dsId: string, meta: Partial<Pick<Dataset, 'subject_id' | 'condition' | 'group' | 'date'>>) => Promise<void>;

  runCompute: (cellId: string) => Promise<void>;
  runStat: (cellId: string) => Promise<void>;
  runPreview: (cellId: string, variant?: 'col1' | 'col2' | 'onehalf') => Promise<void>;
  runCell: (cellId: string) => Promise<void>;
  runAll: () => Promise<void>;
  compareDatasets: () => Promise<void>;

  setCurrentPreset: (p: string) => void;
  setGlobalPreset: (p: string) => void;
  setPageTitle: (t: string) => void;

  openDrawer: (k: DrawerKind) => void;
  closeDrawer: () => void;
  toggleCmdK: (v?: boolean) => void;
  openMapper: (dsId: string) => void;
  closeMapper: () => void;
  toggleHelp: (v?: boolean) => void;
  focusCell: (id: string | null) => void;
  showToast: (msg: string) => void;

  logHistory: (entry: Omit<HistoryEntry, 'id' | 'ts'>) => void;
  clearHistory: () => void;
}

let _toastSeq = 0;
let _cellSeq = 0;
const nextCellId = () => {
  _cellSeq = (_cellSeq + 1) % 10000;
  return `c${Date.now().toString(36)}${_cellSeq.toString(36)}`;
};

export const usePage = create<PageState>()(
  persist(
    (set, get) => ({
      cells: SEED_CELLS,
      datasets: SEED_DATASETS,
      currentPreset: 'ieee',
      globalPreset: 'ieee',
      pageTitle: 'Pilot subject 03 · Treadmill 0.8 m/s',
      history: [],
      drawer: null,
      focusCellId: null,
      cmdkOpen: false,
      mapperOpen: false,
      mapperDsId: null,
      helpOpen: false,
      toast: null,
      runAllBusy: false,

      logHistory: (entry) => set((s) => {
        const next: HistoryEntry = {
          id: 'h' + Date.now().toString(36) + Math.random().toString(36).slice(2, 5),
          ts: Date.now(),
          ...entry,
        };
        // Keep the most recent 200 entries
        const history = [next, ...s.history].slice(0, 200);
        return { history };
      }),
      clearHistory: () => set({ history: [] }),

      addCell: (c, at) => set((s) => {
        const cells = [...s.cells];
        if (at == null) cells.push(c);
        else cells.splice(at, 0, c);
        return { cells };
      }),
      updateCell: (id, patch) => set((s) => ({
        cells: s.cells.map((c) => (c.id === id ? { ...c, ...patch } : c)),
      })),
      removeCell: (id) => set((s) => {
        const cell = s.cells.find((c) => c.id === id);
        if (cell?.previewBlobUrl) URL.revokeObjectURL(cell.previewBlobUrl);
        return { cells: s.cells.filter((c) => c.id !== id) };
      }),
      duplicateCell: (id) => set((s) => {
        const idx = s.cells.findIndex((c) => c.id === id);
        if (idx < 0) return {};
        const src = s.cells[idx];
        // Don't copy live state/blob URLs — re-run on demand
        const { loading: _l, error: _e, computeData: _cd, statData: _sd,
                previewBlobUrl: _pb, ...rest } = src;
        void _l; void _e; void _cd; void _sd; void _pb;
        const copy: Cell = { ...rest, id: nextCellId() };
        const cells = [...s.cells];
        cells.splice(idx + 1, 0, copy);
        return { cells };
      }),
      reorderCells: (from, to) => set((s) => {
        const cells = [...s.cells];
        const [m] = cells.splice(from, 1);
        cells.splice(to, 0, m);
        return { cells };
      }),

      addDataset: (d) => {
        // Dedup by id — backend's SHA256 hash returns the existing ds_id
        // on repeat uploads, so if this id is already in the store we
        // just refresh the record in place instead of appending a duplicate
        // card. Also defensively dedup by _content-derived (name, rows) for
        // any legacy state that pre-dates the backend dedup.
        const existing = get().datasets.findIndex((x) => x.id === d.id);
        if (existing >= 0) {
          set((s) => ({
            datasets: s.datasets.map((x, i) =>
              i === existing ? { ...x, ...d, active: true } : { ...x, active: false },
            ),
          }));
          get().showToast(`${d.name} already loaded — reusing existing`);
          return;
        }
        set((s) => ({
          datasets: [
            ...s.datasets.map((x) => ({ ...x, active: false })),
            { ...d, active: true },
          ],
        }));
        get().logHistory({
          kind: 'upload', actor: 'you',
          label: `Imported ${d.name} (${d.rows.toLocaleString()} rows, ${d.kind})`,
          meta: { dsId: d.id, kind: d.kind },
        });
      },
      removeDataset: (id) => {
        const d = get().datasets.find((x) => x.id === id);
        set((s) => ({ datasets: s.datasets.filter((x) => x.id !== id) }));
        if (d) get().logHistory({
          kind: 'remove', actor: 'you',
          label: `Removed dataset ${d.name}`,
        });
      },
      setActiveDataset: (id) => set((s) => ({
        datasets: s.datasets.map((d) => ({ ...d, active: d.id === id })),
      })),
      toggleRecipe: (dsId, recipeId) => set((s) => ({
        datasets: s.datasets.map((d) =>
          d.id === dsId ? { ...d, recipeState: { ...d.recipeState, [recipeId]: !d.recipeState[recipeId] } } : d,
        ),
      })),
      setDatasetMeta: async (dsId, meta) => {
        // Optimistic update
        set((s) => ({
          datasets: s.datasets.map((d) =>
            d.id === dsId ? { ...d, ...meta } : d,
          ),
        }));
        try {
          await updateDatasetMeta(dsId, meta);
          get().logHistory({
            kind: 'tool', actor: 'you',
            label: `Tagged ${get().datasets.find((d) => d.id === dsId)?.name}: ` +
              Object.entries(meta).map(([k, v]) => `${k}=${v}`).join(', '),
          });
        } catch (e) {
          get().showToast(`Tag update failed: ${(e as Error).message}`);
        }
      },

      analyzeIfNeeded: async (dsId) => {
        const ds = get().datasets.find((d) => d.id === dsId);
        if (!ds) return undefined;
        if (ds.analysis) return ds.analysis;
        set((s) => ({
          datasets: s.datasets.map((d) =>
            d.id === dsId ? { ...d, analyzing: true, analyzeError: undefined } : d,
          ),
        }));
        try {
          const payload = await analyzeDataset(dsId);
          set((s) => ({
            datasets: s.datasets.map((d) =>
              d.id === dsId ? { ...d, analysis: payload, analyzing: false } : d,
            ),
          }));
          return payload;
        } catch (e) {
          const msg = (e as Error).message;
          set((s) => ({
            datasets: s.datasets.map((d) =>
              d.id === dsId ? { ...d, analyzing: false, analyzeError: msg } : d,
            ),
          }));
          get().showToast(`Analyze failed: ${msg}`);
          return undefined;
        }
      },

      applyRecipes: async (dsId) => {
        const ds = get().datasets.find((d) => d.id === dsId);
        if (!ds) return;
        const recipes = CANONICAL_RECIPES[ds.kind] || CANONICAL_RECIPES.force;
        // Phase 2I · compute metrics ALWAYS run (they're numbers, no choice
        // to make). For graphs, respect the user's toggle state (default:true
        // if untouched, user can flip extras in the AutoRecipes chip strip).
        const chosen = recipes.filter((r) =>
          r.type === 'compute'
            ? true
            : (ds.recipeState[r.id] ?? r.default),
        );
        if (chosen.length === 0) {
          get().showToast('No recipes selected');
          return;
        }

        // Phase 2H · dedup — skip recipes whose cell already exists for this
        // dataset (prevents the same figure being regenerated every time
        // the user drops the same CSV or toggles the panel).
        const existing = get().cells;
        const isDuplicate = (r: typeof chosen[number]): boolean => {
          if (r.type === 'graph' && r.graph) {
            return existing.some((c) =>
              c.type === 'graph' && c.graph === r.graph && c.dsIds.includes(dsId),
            );
          }
          if (r.type === 'compute' && r.compute) {
            return existing.some((c) =>
              c.type === 'compute' && c.metric === r.compute && c.dsIds.includes(dsId),
            );
          }
          return false;
        };
        const toCreate = chosen.filter((r) => !isDuplicate(r));
        if (toCreate.length === 0) {
          get().showToast(`All recipes already present for ${ds.name}`);
          return;
        }

        // Kick off analyzer cache warming (non-blocking for graph cells)
        const analyzePromise = get().analyzeIfNeeded(dsId);

        const newCells: Cell[] = [];
        for (const r of toCreate) {
          const id = nextCellId();
          if (r.type === 'graph' && r.graph) {
            newCells.push({
              id, type: 'graph', graph: r.graph,
              dsIds: [dsId],
              strideAvg: r.graph === 'force_avg',
              loading: true,
            });
          } else if (r.type === 'compute' && r.compute) {
            newCells.push({
              id, type: 'compute', metric: r.compute,
              dsIds: [dsId],
              loading: true,
            });
          }
        }
        set((s) => ({ cells: [...s.cells, ...newCells] }));
        get().logHistory({
          kind: 'apply', actor: 'you',
          label: `Applied ${newCells.length} recipe cells for ${ds.name}`,
          meta: { dsId, count: newCells.length },
        });

        await analyzePromise;

        // Fire requests in parallel
        await Promise.all(newCells.map((c) => get().runCell(c.id)));
        get().showToast(`Applied ${newCells.length} cells for ${ds.name}` +
          (toCreate.length < chosen.length
            ? ` (${chosen.length - toCreate.length} already present)`
            : ''));
      },

      runCell: async (cellId) => {
        const cell = get().cells.find((c) => c.id === cellId);
        if (!cell) return;
        if (cell.type === 'compute') return get().runCompute(cellId);
        if (cell.type === 'stat') return get().runStat(cellId);
        if (cell.type === 'graph') return get().runPreview(cellId);
      },

      runCompute: async (cellId) => {
        const cell = get().cells.find((c) => c.id === cellId);
        if (!cell || cell.type !== 'compute' || !cell.metric || !cell.dsIds[0]) return;
        get().updateCell(cellId, { loading: true, error: undefined });
        try {
          const data = await computeMetric({
            dataset_id: cell.dsIds[0],
            metric: cell.metric as ComputeMetricKey,
          });
          get().updateCell(cellId, { loading: false, computeData: data });
        } catch (e) {
          get().updateCell(cellId, { loading: false, error: (e as Error).message });
        }
      },

      runStat: async (cellId) => {
        const cell = get().cells.find((c) => c.id === cellId);
        if (!cell || cell.type !== 'stat' || !cell.op) return;
        get().updateCell(cellId, { loading: true, error: undefined });
        try {
          // Phase 3 · cross-file path
          if ((cell.statDatasetsA?.length || 0) > 0 || (cell.statDatasetsB?.length || 0) > 0) {
            const data = await runStats({
              op: cell.op as StatOpKey,
              datasets_a: cell.statDatasetsA,
              datasets_b: cell.statDatasetsB,
            });
            get().updateCell(cellId, { loading: false, statData: data });
            return;
          }
          // Legacy · single dataset + columns
          const dsId = cell.dsIds[0];
          const a_col = cell.inputs?.a?.trim();
          const b_col = cell.inputs?.b?.trim();
          if (!dsId || !a_col) {
            throw new Error('dataset_id + a_col required (or switch to cross-file mode)');
          }
          const needsB = cell.op !== 'shapiro' && cell.op !== 'anova1';
          const data = await runStats({
            op: cell.op as StatOpKey,
            dataset_id: dsId,
            a_col,
            b_col: needsB ? b_col : undefined,
          });
          get().updateCell(cellId, { loading: false, statData: data });
        } catch (e) {
          get().updateCell(cellId, { loading: false, error: (e as Error).message });
        }
      },

      runPreview: async (cellId, variant) => {
        const cell = get().cells.find((c) => c.id === cellId);
        if (!cell || cell.type !== 'graph' || !cell.graph) return;
        const preset = cell.preset || get().globalPreset;
        const v = variant || cell.previewVariant || 'col2';
        get().updateCell(cellId, { loading: true, error: undefined });
        try {
          // Phase 1: multi-dataset overlay when series present
          const seriesList: Array<{ dsId: string; label?: string; color?: string }> =
            (cell.series && cell.series.length > 0)
              ? cell.series
              : cell.dsIds.map((id) => ({ dsId: id }));
          const datasets = seriesList.length >= 2
            ? seriesList.map((s) => ({ id: s.dsId, label: s.label, color: s.color }))
            : undefined;
          const blob = await renderGraph({
            template: cell.graph,
            preset,
            variant: v,
            format: 'svg',
            dataset_id: datasets ? undefined : seriesList[0]?.dsId,
            datasets,
            stride_avg: !!cell.strideAvg,
            title: cell.title || '',
          });
          // Revoke old preview URL
          if (cell.previewBlobUrl) URL.revokeObjectURL(cell.previewBlobUrl);
          const url = URL.createObjectURL(blob);
          get().updateCell(cellId, {
            loading: false,
            previewBlobUrl: url,
            previewVariant: v,
          });
        } catch (e) {
          get().updateCell(cellId, { loading: false, error: (e as Error).message });
        }
      },

      compareDatasets: async () => {
        // Phase 2I · one-click multi-file comparison.
        // Walks every dataset, groups by `condition` (Pre/Post/Control).
        // Spawns overlay graph cells + cross-file stat cells so the
        // user doesn't have to wire each one manually.
        const ds = get().datasets.filter((d) => d.kind !== 'trials');
        if (ds.length < 2) {
          get().showToast('Upload at least 2 datasets to compare');
          return;
        }

        // Warm analyzer for every dataset
        await Promise.all(ds.map((d) => get().analyzeIfNeeded(d.id)));

        // Color palette — alternating for two groups, else rotating
        const palette = ['#3B82C4', '#D35454', '#F09708', '#00FFB2',
                         '#A78BFA', '#1E5F9E', '#9E3838', '#FFB347'];

        const series = ds.map((d, i) => ({
          dsId: d.id,
          label: (d.subject_id ? d.subject_id.toUpperCase() : d.name.replace(/\.csv$/i, ''))
            + (d.condition ? ` · ${d.condition}` : ''),
          color: palette[i % palette.length],
        }));

        const newCells: Cell[] = [];
        const overlayTemplates: Array<{ graph: string; strideAvg?: boolean }> = [
          { graph: 'force_avg', strideAvg: true },
          { graph: 'imu_avg' },
          { graph: 'stride_time_trend' },
          { graph: 'asymmetry' },
        ];
        for (const tpl of overlayTemplates) {
          const id = nextCellId();
          newCells.push({
            id, type: 'graph', graph: tpl.graph,
            dsIds: ds.map((d) => d.id),
            series,
            strideAvg: !!tpl.strideAvg,
            title: `Compare ${ds.length} datasets · ${tpl.graph}`,
            loading: true,
          });
        }

        // Cross-file stats: group by `condition` when ≥ 2 conditions present
        const conds = Array.from(new Set(ds.map((d) => d.condition).filter(Boolean)));
        if (conds.length >= 2) {
          const groupA = ds.filter((d) => d.condition === conds[0]);
          const groupB = ds.filter((d) => d.condition === conds[1]);
          const metricsToTest = ['peak_force_L', 'peak_force_R', 'cadence_L',
                                  'stride_time_mean_L', 'stride_length_mean_L'];
          for (const metric of metricsToTest) {
            const id = nextCellId();
            newCells.push({
              id, type: 'stat', op: 'ttest_welch',
              statMetric: metric,
              statDatasetsA: groupA.map((d) => ({ id: d.id, metric })),
              statDatasetsB: groupB.map((d) => ({ id: d.id, metric })),
              dsIds: ds.map((d) => d.id),
              fmt: 'apa',
              title: `${conds[0]} vs ${conds[1]} · ${metric}`,
              loading: true,
            });
          }
          get().showToast(
            `Comparing ${ds.length} files across ${conds.length} conditions ` +
            `(${conds.join(' vs ')}) · ${newCells.length} cells`,
          );
        } else {
          get().showToast(
            `Overlaid ${ds.length} datasets · ${newCells.length} graph cells ` +
            `(tag conditions to also get cross-file stats)`,
          );
        }

        set((s) => ({ cells: [...s.cells, ...newCells] }));
        get().logHistory({
          kind: 'apply', actor: 'you',
          label: `Compared ${ds.length} datasets → ${newCells.length} cells`,
          meta: { n_datasets: ds.length, conditions: conds },
        });

        // Fire in parallel
        await Promise.all(newCells.map((c) => get().runCell(c.id)));
      },

      runAll: async () => {
        if (get().runAllBusy) return;
        set({ runAllBusy: true });
        try {
          const cells = get().cells.filter((c) => c.type !== 'llm' && c.dsIds[0]);
          const dsIds = Array.from(new Set(cells.flatMap((c) => c.dsIds))).filter(Boolean);
          await Promise.all(dsIds.map((id) => get().analyzeIfNeeded(id)));
          await Promise.all(cells.map((c) => get().runCell(c.id)));
          get().logHistory({
            kind: 'run', actor: 'you',
            label: `Ran ALL · ${cells.length} cells across ${dsIds.length} datasets`,
          });
          get().showToast(`Ran ${cells.length} cells across ${dsIds.length} datasets`);
        } finally {
          set({ runAllBusy: false });
        }
      },

      setCurrentPreset: (p) => set({ currentPreset: p }),
      setGlobalPreset: (p) => {
        const prev = get().globalPreset;
        set({ globalPreset: p, currentPreset: p });
        if (prev !== p) {
          get().logHistory({
            kind: 'preset', actor: 'you',
            label: `Journal preset → ${p.toUpperCase()}`,
            meta: { from: prev, to: p },
          });
        }
      },
      setPageTitle: (t) => set({ pageTitle: t }),

      openDrawer: (k) => set({ drawer: k }),
      closeDrawer: () => set({ drawer: null }),
      toggleCmdK: (v) => set((s) => ({ cmdkOpen: v ?? !s.cmdkOpen })),
      openMapper: (dsId) => set({ mapperOpen: true, mapperDsId: dsId }),
      closeMapper: () => set({ mapperOpen: false, mapperDsId: null }),
      toggleHelp: (v) => set((s) => ({ helpOpen: v ?? !s.helpOpen })),
      focusCell: (id) => set({ focusCellId: id }),
      showToast: (msg) => set({ toast: { msg, id: ++_toastSeq } }),
    }),
    {
      name: 'hw_page_v1',
      // Bump this whenever the persisted shape changes in a way that
      // can crash on rehydrate, or when stale cells from a prior
      // session would mis-fire under new code. v3 drops the entire
      // cells array because the LLM-tooling refactor introduced an
      // 'inspector' cell type and removed the 'llm' / mock cells —
      // and stale cells from v2 were auto-firing /api/compute calls
      // on H-Walker datasets that aren't in firmware format
      // (resulting in 409 Conflict spam).
      version: 3,
      migrate: (persisted: unknown, fromVersion: number): Partial<PageState> => {
        if (!persisted || typeof persisted !== 'object') return {};
        try {
          const p = persisted as Record<string, unknown>;
          const safe: Partial<PageState> = {};
          // From v3 onward we keep cells. Anything older gets a
          // clean canvas — the user re-clicks what they want from
          // the Library bookshelf.
          if (fromVersion >= 3 && Array.isArray(p.cells)) {
            safe.cells = p.cells as PageState['cells'];
          }
          if (Array.isArray(p.datasets)) safe.datasets = p.datasets as PageState['datasets'];
          if (typeof p.currentPreset === 'string') safe.currentPreset = p.currentPreset as PageState['currentPreset'];
          if (typeof p.globalPreset === 'string') safe.globalPreset = p.globalPreset as PageState['globalPreset'];
          if (typeof p.pageTitle === 'string') safe.pageTitle = p.pageTitle as PageState['pageTitle'];
          if (Array.isArray(p.history)) safe.history = p.history as PageState['history'];
          if (fromVersion < 3) {
            console.info(
              `[hw_page] migrated persisted state from v${fromVersion} → v3 ` +
              '(cells cleared — old auto-fire cells were producing 409s on ' +
              'non-firmware CSVs; pick fresh ones from the Library)',
            );
          }
          return safe;
        } catch (e) {
          console.warn('[hw_page] persisted state unreadable, starting fresh:', e);
          return {};
        }
      },
      // Strip live (non-serializable) state before persisting
      partialize: (s) => ({
        cells: s.cells.map((c) => {
          const { loading: _l, error: _e, computeData: _cd, statData: _sd,
                  previewBlobUrl: _p, ...rest } = c;
          void _l; void _e; void _cd; void _sd; void _p;
          return rest;
        }),
        datasets: s.datasets.map((d) => {
          const { analyzing: _a, analyzeError: _ae, analysis: _an, ...rest } = d;
          void _a; void _ae; void _an;
          return rest;
        }),
        currentPreset: s.currentPreset,
        globalPreset: s.globalPreset,
        pageTitle: s.pageTitle,
        history: s.history,
      }),
      // One-shot rehydration pass: dedup any dataset entries that old
      // code paths left in localStorage (so the user doesn't need to
      // clear storage manually). Wrapped in try/catch so a corrupt
      // entry can't blank-screen the app.
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        try {
          const seen = new Set<string>();
          const uniq: Dataset[] = [];
          for (const d of state.datasets || []) {
            if (!d || typeof d !== 'object' || !d.id) continue;
            if (seen.has(d.id)) continue;
            seen.add(d.id);
            uniq.push(d);
          }
          if (uniq.length !== (state.datasets || []).length) {
            state.datasets = uniq;
            console.info(`[hw_page] deduplicated datasets on rehydrate`);
          }
        } catch (e) {
          console.warn('[hw_page] rehydrate post-process failed, ignoring:', e);
          state.datasets = [];
          state.cells = [];
        }
      },
    },
  ),
);

// Re-export for components that need payload types
export type { AnalysisPayload };
