import { useRef, useState } from 'react';
import { Trash2 } from 'lucide-react';
import { usePage } from '../store/page';
import { CANONICAL_RECIPES } from '../data/canonicalRecipes';
import { uploadDataset, deleteDataset as apiDeleteDataset } from '../api';

export default function DatasetPanel() {
  const datasets = usePage((s) => s.datasets);
  const setActive = usePage((s) => s.setActiveDataset);
  const applyRecipes = usePage((s) => s.applyRecipes);
  const toggleRecipe = usePage((s) => s.toggleRecipe);
  const addDataset = usePage((s) => s.addDataset);
  const removeDataset = usePage((s) => s.removeDataset);
  const setDatasetMeta = usePage((s) => s.setDatasetMeta);
  const showToast = usePage((s) => s.showToast);
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const active = datasets.find((d) => d.active) || datasets[0];
  const recipes = active ? CANONICAL_RECIPES[active.kind] || CANONICAL_RECIPES.force : [];

  async function handleFiles(files: FileList | null) {
    if (!files || !files.length) return;
    let accepted = 0;
    let skipped: string[] = [];
    for (const f of Array.from(files)) {
      // Accept anything whose name ends with .csv (case-insensitive) OR
      // has no recognisable extension (e.g. `foo` — try upload, backend
      // will 422 if it's not a CSV). This catches `.CSV`, `foo.csv (1)`,
      // and Finder-rename quirks that dropped the extension.
      const n = f.name.toLowerCase();
      const looksCsv = /\.csv(\b|$|\s|\.)/.test(n) || !/\.\w+$/.test(n);
      if (!looksCsv) {
        skipped.push(f.name);
        continue;
      }
      try {
        const ds = await uploadDataset(f);
        addDataset(ds);
        setActive(ds.id);
        showToast(`Uploaded ${f.name} · running default recipes…`);
        // Auto-apply default recipes — the "one-click" flow
        applyRecipes(ds.id).catch((e) => showToast(`Auto-run failed: ${(e as Error).message}`));
        accepted += 1;
      } catch (e) {
        showToast(`Upload failed (${f.name}): ${(e as Error).message}`);
      }
    }
    if (accepted === 0 && skipped.length) {
      showToast(`Not CSV files: ${skipped.slice(0, 3).join(', ')}${skipped.length > 3 ? '…' : ''}`);
    }
  }

  return (
    <div className="ds-panel">
      <div className="ds-head">
        <span className="ey">DATA SOURCES</span>
        <span className="label">Datasets</span>
        <div className="rest">
          <button className="ds-btn plain" onClick={() => fileRef.current?.click()}>＋ Add CSV</button>
        </div>
      </div>

      {datasets.length === 0 && (
        <div className="ds-empty">
          <div className="ds-empty-title">No datasets yet</div>
          <div className="ds-empty-sub">
            Drop a CSV below to get started — default recipe auto-runs and
            fills the canvas with graphs + metrics for the active journal
            preset.
          </div>
        </div>
      )}

      <div className="ds-grid">
        {datasets.map((d) => (
          <div
            key={d.id}
            className={`ds-card${d.active ? ' active' : ''}`}
            onClick={() => setActive(d.id)}
          >
            <div className="ds-row1">
              {d.source_kind && d.source_kind !== 'unknown' && (
                <span
                  className={`tag src-${d.source_kind}`}
                  title={[
                    `Detected as ${d.source_kind.toUpperCase()} `
                      + `(confidence ${(d.source_confidence ?? 0).toFixed(2)})`,
                    ...(d.source_cues || []).map((c) => `· ${c}`),
                  ].join('\n')}
                  style={{
                    background:
                      d.source_kind === 'robot'    ? 'rgba(240,151,8,.18)'
                    : d.source_kind === 'motion'   ? 'rgba(167,139,250,.18)'
                    : d.source_kind === 'loadcell' ? 'rgba(127,181,228,.18)'
                    : undefined,
                    color:
                      d.source_kind === 'robot'    ? '#F09708'
                    : d.source_kind === 'motion'   ? '#A78BFA'
                    : d.source_kind === 'loadcell' ? '#7FB5E4'
                    : undefined,
                    border: '1px solid currentColor',
                    fontWeight: 700,
                  }}
                >
                  {d.source_kind.toUpperCase()}
                </span>
              )}
              <span className={`tag ${d.tag}`}>{d.tag}</span>
              <span className="name" title={d.name}>{d.name}</span>
              <button
                className="ds-del"
                title="Delete dataset"
                onClick={(e) => {
                  e.stopPropagation();
                  if (!confirm(`Delete ${d.name}? This also removes linked cells' bindings.`)) return;
                  apiDeleteDataset(d.id)
                    .catch(() => { /* proceed with local removal anyway */ })
                    .finally(() => {
                      removeDataset(d.id);
                      showToast(`Deleted ${d.name}`);
                    });
                }}
              ><Trash2 size={12} /></button>
            </div>
            <div className="ds-row2">
              <span><b>{d.rows.toLocaleString()}</b> rows</span>
              <span><b>{d.dur}</b></span>
              <span><b>{d.hz}</b></span>
              {d.analyzing && <span style={{ color: '#F09708' }}>· analyzing…</span>}
              {d.analysis && 'mode' in d.analysis && d.analysis.mode === 'hwalker' && (
                <span style={{ color: '#00FFB2' }}>
                  · {d.analysis.left.n_strides}L / {d.analysis.right.n_strides}R strides
                </span>
              )}
              {d.analysis && 'fallback_mode' in d.analysis && (
                <span style={{ color: '#7FB5E4' }}>· generic mode</span>
              )}
              {d.analyzeError && (
                <span
                  style={{ color: '#f87171', cursor: 'help' }}
                  title={d.analyzeError}
                >
                  · err: {d.analyzeError.length > 60
                    ? d.analyzeError.slice(0, 60) + '…'
                    : d.analyzeError}
                </span>
              )}
            </div>
            <div className="ds-tags" onClick={(e) => e.stopPropagation()}>
              <DatasetTag
                label="subj"
                value={d.subject_id || ''}
                placeholder="s01"
                onChange={(v) => setDatasetMeta(d.id, { subject_id: v })}
              />
              <DatasetTag
                label="cond"
                value={d.condition || ''}
                placeholder="Pre / Post / Control"
                onChange={(v) => setDatasetMeta(d.id, { condition: v, group: v })}
              />
              {d.group && d.group !== d.condition && (
                <DatasetTag
                  label="group"
                  value={d.group}
                  placeholder="—"
                  onChange={(v) => setDatasetMeta(d.id, { group: v })}
                />
              )}
            </div>
            <div className="ds-cols">
              {d.cols.slice(0, 5).map((c, i) => (
                <span key={i} className={`ds-col${c.mapped && c.mapped !== '—' ? ' mapped' : ''}`}>
                  {c.name}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div
        className={`ds-upload${dragOver ? ' drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <div className="ds-drop" onClick={() => fileRef.current?.click()}>
          <b>Drop CSV</b>&nbsp;files here, or click to browse
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".csv"
          multiple
          hidden
          onChange={(e) => { handleFiles(e.target.files); e.target.value = ''; }}
        />
      </div>

      {recipes.length > 0 && active && (
        <AutoRecipes
          recipes={recipes}
          active={active}
          onToggle={(id) => toggleRecipe(active.id, id)}
          onApply={() => applyRecipes(active.id)}
        />
      )}
    </div>
  );
}

/** Phase 2 · inline tag editor for subject/condition/group. */
function DatasetTag({ label, value, placeholder, onChange }: {
  label: string;
  value: string;
  placeholder: string;
  onChange: (v: string) => void;
}) {
  return (
    <span className={`ds-tag-ed${value ? ' set' : ''}`}>
      <span className="ds-tag-lbl">{label}</span>
      <input
        className="ds-tag-inp"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </span>
  );
}

/**
 * AutoRecipes · Phase 2I
 *
 * Replaces the old checkbox grid. Compute metrics auto-run (they're just
 * numbers, no user choice needed). Graphs show only the currently-enabled
 * set in a compact chip strip; the user can toggle extras from a
 * "+ more graphs" expandable section. "Apply" button re-runs the active
 * set (skip duplicates thanks to dedup in page.applyRecipes).
 */
function AutoRecipes({ recipes, active, onToggle, onApply }: {
  recipes: Array<{ id: string; label: string; default: boolean; type: 'graph' | 'compute'; hint?: string }>;
  active: { id: string; kind: string; recipeState: Record<string, boolean>; analyzing?: boolean };
  onToggle: (id: string) => void;
  onApply: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const graphs = recipes.filter((r) => r.type === 'graph');
  const computes = recipes.filter((r) => r.type === 'compute');
  const enabledGraph = graphs.filter((r) => active.recipeState[r.id] ?? r.default);
  const disabledGraph = graphs.filter((r) => !(active.recipeState[r.id] ?? r.default));

  return (
    <div className="recipes" style={{ margin: '0 12px 12px' }}>
      <div className="recipes-head">
        <h4>Auto-analysis · {active.kind}</h4>
        <span className="sub">
          {computes.length} metrics run automatically · {enabledGraph.length} graphs enabled
        </span>
        <button
          className="apply"
          onClick={onApply}
          disabled={active.analyzing}
        >
          {active.analyzing ? 'Analyzing…' : 'Run on this dataset'}
        </button>
      </div>

      <div className="auto-summary">
        <span className="auto-lbl">Graphs</span>
        {enabledGraph.map((r) => (
          <button
            key={r.id}
            className="auto-chip on"
            title={r.hint || r.label}
            onClick={() => onToggle(r.id)}
          >
            {r.label}
            <span className="auto-x">×</span>
          </button>
        ))}
        {disabledGraph.length > 0 && (
          <button
            className="auto-more"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? '− hide' : `+ ${disabledGraph.length} more`}
          </button>
        )}
      </div>

      {expanded && (
        <div className="auto-advanced">
          {disabledGraph.map((r) => (
            <button
              key={r.id}
              className="auto-chip off"
              title={r.hint || r.label}
              onClick={() => onToggle(r.id)}
            >
              + {r.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
