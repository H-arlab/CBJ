import { useEffect, useState } from 'react';
import type { Cell } from '../../store/page';
import { usePage } from '../../store/page';
import { GRAPH_TPLS, type GraphTemplate } from '../../data/graphTemplates';
import { JOURNAL_PRESETS } from '../../data/journalPresets';
import { renderGraph } from '../../api';

interface Props { cell: Cell; }

export default function GraphCell({ cell }: Props) {
  const globalPreset = usePage((s) => s.globalPreset);
  const updateCell = usePage((s) => s.updateCell);
  const showToast = usePage((s) => s.showToast);
  const runPreview = usePage((s) => s.runPreview);
  const allDatasets = usePage((s) => s.datasets);
  const [menuOpen, setMenuOpen] = useState(false);
  const [overlayOpen, setOverlayOpen] = useState(false);
  const [svgInline, setSvgInline] = useState<string | null>(null);

  const activeKey =
    cell.strideAvg && cell.graph === 'force' && GRAPH_TPLS.force_avg ? 'force_avg' : cell.graph;
  const tpl = GRAPH_TPLS[activeKey || 'force'];
  const preset = globalPreset;
  const P = JOURNAL_PRESETS[preset];
  const canToggleAvg = cell.graph === 'force' || cell.graph === 'force_avg';
  const hasDataset = !!cell.dsIds[0];

  // Auto-trigger backend preview when dataset is bound and no preview exists yet.
  useEffect(() => {
    if (hasDataset && !cell.previewBlobUrl && !cell.loading && !cell.error) {
      runPreview(cell.id);
    }
  }, [hasDataset, cell.previewBlobUrl, cell.loading, cell.error, cell.id, runPreview]);

  // Re-run when the graph template, global preset, strideAvg, title, or
  // the multi-dataset overlay series list changes.
  useEffect(() => {
    if (!hasDataset) return;
    const h = setTimeout(() => { runPreview(cell.id); }, 180);
    return () => clearTimeout(h);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cell.graph, cell.strideAvg, cell.title, globalPreset,
      JSON.stringify(cell.series)]);

  useEffect(() => {
    if (!cell.previewBlobUrl) { setSvgInline(null); return; }
    let cancelled = false;
    fetch(cell.previewBlobUrl)
      .then((r) => r.text())
      .then((txt) => { if (!cancelled) setSvgInline(txt); })
      .catch(() => { if (!cancelled) setSvgInline(null); });
    return () => { cancelled = true; };
  }, [cell.previewBlobUrl]);

  async function backendRender(fmt: 'svg' | 'pdf' | 'png' | 'eps' | 'tiff', variant: 'col1' | 'col2' | 'onehalf' = 'col2') {
    try {
      const blob = await renderGraph({
        template: activeKey || 'force',
        preset,
        variant,
        format: fmt,
        stride_avg: !!cell.strideAvg,
        dataset_id: cell.dsIds[0],
        title: cell.title || '',
      });
      const w = variant === 'col1' ? P?.col1.w : (variant === 'onehalf' ? P?.onehalf?.w ?? P?.col2.w : P?.col2.w);
      downloadBlob(blob, `${cell.id}_${activeKey}_${preset}_${Math.round(w || 0)}mm.${fmt}`);
      showToast(`${fmt.toUpperCase()} · ${preset} · ${Math.round(w || 0)}mm`);
    } catch (e) {
      showToast(`Export failed: ${(e as Error).message}`);
    }
  }

  function exportSvgClient() {
    const svgEl = document.getElementById(`plot-${cell.id}`);
    if (!svgEl) return;
    const xml = new XMLSerializer().serializeToString(svgEl);
    downloadBlob(new Blob([xml], { type: 'image/svg+xml' }), `${cell.id}_preview.svg`);
  }

  return (
    <>
      <div className="graph-toolbar">
        <div className="gt-field">
          <span className="gt-lbl">Graph</span>
          <select
            className="gt-sel"
            value={cell.graph || ''}
            onChange={(e) => updateCell(cell.id, { graph: e.target.value })}
          >
            {Object.keys(GRAPH_TPLS).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
        </div>
        {canToggleAvg && (
          <label className="gt-toggle">
            <input
              type="checkbox"
              checked={!!cell.strideAvg}
              onChange={(e) => updateCell(cell.id, { strideAvg: e.target.checked })}
            />
            Stride avg (mean±SD)
          </label>
        )}
        <div className="gt-spacer" />
        {hasDataset && (
          <button
            className="gt-btn"
            title="Re-render from dataset"
            onClick={() => runPreview(cell.id)}
            disabled={!!cell.loading}
          >
            {cell.loading ? '…' : '⟳ Run'}
          </button>
        )}
        {hasDataset && allDatasets.length > 1 && (
          <div style={{ position: 'relative' }}>
            <button
              className="gt-btn"
              title="Add another dataset as an overlay"
              onClick={() => setOverlayOpen((v) => !v)}
            >
              + Overlay ({(cell.series?.length || cell.dsIds.length) || 1})
            </button>
            {overlayOpen && (
              <div className="gt-menu" onMouseLeave={() => setOverlayOpen(false)}
                   style={{ minWidth: 240, padding: '6px' }}>
                <div style={{ padding: '4px 8px', font: '600 9px/1 JetBrains Mono, monospace',
                              color: '#6B7280', letterSpacing: '.18em', textTransform: 'uppercase' }}>
                  Datasets in this plot
                </div>
                {allDatasets.map((d) => {
                  const active = (cell.series || cell.dsIds.map((id) => ({ dsId: id })))
                    .some((s) => ('dsId' in s ? s.dsId : s) === d.id);
                  return (
                    <label key={d.id} className="gt-mitem"
                           style={{ display: 'flex', gap: 8, alignItems: 'center', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={active}
                        onChange={() => {
                          const cur: Array<{ dsId: string; label?: string; color?: string }> =
                            cell.series && cell.series.length
                              ? [...cell.series]
                              : cell.dsIds.map((id) => ({ dsId: id }));
                          const idx = cur.findIndex((s) => s.dsId === d.id);
                          if (idx >= 0) cur.splice(idx, 1);
                          else cur.push({ dsId: d.id, label: d.name });
                          updateCell(cell.id, {
                            series: cur,
                            dsIds: cur.map((s) => s.dsId),
                          });
                        }}
                      />
                      <span className={`tag ${d.tag}`} style={{ padding: '1px 5px', fontSize: 9 }}>{d.tag}</span>
                      <span style={{ fontSize: 11, color: '#E2E8F0', flex: 1 }}>{d.name}</span>
                    </label>
                  );
                })}
                <div style={{ padding: '6px 8px 2px', font: '500 9.5px/1.3 Pretendard,sans-serif',
                              color: '#6B7280', borderTop: '1px solid rgba(255,255,255,.06)', marginTop: 4 }}>
                  {(cell.series?.length || cell.dsIds.length) >= 2
                    ? 'Overlay mode · one colored line per dataset'
                    : 'Add ≥ 2 datasets to overlay them on the same figure'}
                </div>
              </div>
            )}
          </div>
        )}
        <span className="gt-meta">
          {P?.name} · {P?.col2.w}mm · {P?.dpi}dpi · {P?.sizes.body}pt {P?.font}
        </span>
        <div className="gt-export">
          <button className="gt-btn" onClick={() => setMenuOpen((v) => !v)}>Export ▾</button>
          {menuOpen && (
            <div className="gt-menu" onMouseLeave={() => setMenuOpen(false)}>
              <div style={{ padding: '4px 10px', font: '600 8.5px/1 JetBrains Mono, monospace', color: '#6B7280', letterSpacing: '.14em' }}>
                {P?.name} · {Math.round(P?.col2.w || 0)}mm · {P?.dpi}dpi
              </div>
              <button className="gt-mitem" onClick={() => { backendRender('svg', 'col2'); setMenuOpen(false); }}>SVG · 2-col</button>
              <button className="gt-mitem" onClick={() => { backendRender('svg', 'col1'); setMenuOpen(false); }}>SVG · 1-col</button>
              <button className="gt-mitem" onClick={() => { backendRender('pdf', 'col2'); setMenuOpen(false); }}>PDF · 2-col</button>
              <button className="gt-mitem" onClick={() => { backendRender('png', 'col2'); setMenuOpen(false); }}>PNG · 2-col</button>
              <button className="gt-mitem" onClick={() => { backendRender('eps', 'col2'); setMenuOpen(false); }}>EPS · 2-col</button>
              <button className="gt-mitem" onClick={() => { backendRender('tiff', 'col2'); setMenuOpen(false); }}>TIFF · 2-col</button>
              <div style={{ padding: '4px 10px', borderTop: '1px solid rgba(255,255,255,.06)', marginTop: 2 }}>
                <button className="gt-mitem" onClick={() => { exportSvgClient(); setMenuOpen(false); }}>Preview SVG</button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Figure caption — prominent, journal-style, above the plot.
          Empty value means no in-plot title in the exported file either. */}
      <div className="fig-caption">
        <span className="fig-cap-prefix">Figure</span>
        <input
          className="fig-cap-input"
          value={cell.title || ''}
          placeholder="Figure caption — one sentence describing what this figure shows (leave empty to omit the inline title)"
          onChange={(e) => updateCell(cell.id, { title: e.target.value })}
        />
      </div>

      <div
        className={`plot preset-pub${cell.loading ? ' plot-loading' : ''}`}
        style={{
          ['--pub-font' as never]: `'${P?.font}', ${P?.fontFallback}`,
          ['--pub-stroke' as never]: `${P?.stroke}`,
          ['--pub-grid' as never]: `${P?.gridColor}`,
          ['--pub-axis-pt' as never]: `${P?.sizes.axis}`,
          ['--pub-body-pt' as never]: `${P?.sizes.body}`,
        } as React.CSSProperties}
      >
        {cell.error ? (
          <div className="plot-error">
            <b>Render failed</b>
            <div>{cell.error}</div>
            <button onClick={() => runPreview(cell.id)}>Retry</button>
          </div>
        ) : hasDataset && svgInline ? (
          <div className="plot-real" dangerouslySetInnerHTML={{ __html: svgInline }} />
        ) : hasDataset && cell.loading ? (
          <div className="plot-skeleton">
            <div className="ps-bar" /><div className="ps-bar" /><div className="ps-bar" />
            <div className="ps-label">Rendering…</div>
          </div>
        ) : (
          <EmptyPlot tpl={tpl} hasDataset={hasDataset} />
        )}
        <div className="pub-rule-ruler">
          {P?.name} · {P?.col2.w}mm · {P?.font} {P?.sizes.body}pt · {P?.dpi}dpi
        </div>
      </div>

      {/* Legend + meta come from the rendered figure (SVG <g class="legend">
          inside the backend response) — we no longer fabricate fake series
          labels or "n=14" summaries from the template object. */}
      <div className="cell-meta">
        {hasDataset && cell.previewBlobUrl && (
          <span style={{ color: '#00FFB2' }}>● live</span>
        )}
        <span style={{ color: '#6B7280' }}>
          {tpl.ey} · {tpl.xUnit}
        </span>
      </div>
    </>
  );
}

function EmptyPlot({ tpl, hasDataset }: { tpl: GraphTemplate; hasDataset: boolean }) {
  // Pure empty state. NO mock SVG paths — the cell shows axes + units +
  // a clear "bind a CSV" message until the backend returns a real figure.
  // Past versions rendered fake bezier curves and fake "n strides=14"
  // summaries here, which leaked into screenshots and demos.
  const vb = '0 0 456 210';
  return (
    <svg id={`empty-${tpl.title}`} viewBox={vb} preserveAspectRatio="none">
      {tpl.yTicks.map((_, i) => (
        <line key={'gy' + i} className="grid-line" x1={44} x2={448} y1={23 + i * 40} y2={23 + i * 40} />
      ))}
      {tpl.xTicks.map((_, i) => (
        <line key={'gx' + i} className="grid-line" x1={48 + i * 90} x2={48 + i * 90} y1={20} y2={182} />
      ))}
      <line className="axis-line" x1={44} x2={448} y1={182} y2={182} />
      <line className="axis-line" x1={44} x2={44} y1={20} y2={182} />
      {tpl.yTicks.map((v, i) => (
        <text key={'ty' + i} className="tick-label" x={40} y={25 + i * 40} textAnchor="end">{v}</text>
      ))}
      {tpl.xTicks.map((v, i) => (
        <text key={'tx' + i} className="tick-label" x={48 + i * 90} y={196} textAnchor="middle">{v}</text>
      ))}
      <text className="axis-title" x={246} y={208} textAnchor="middle">{tpl.xUnit}</text>
      <text className="axis-title" x={14} y={100} textAnchor="middle" transform="rotate(-90 14,100)">{tpl.yUnit}</text>
      <text x={246} y={96} textAnchor="middle"
            style={{ fill: '#6B7280', fontSize: 11, fontWeight: 500 }}>
        {hasDataset ? '· press Render ·' : '· bind a CSV to plot ·'}
      </text>
      <text x={246} y={114} textAnchor="middle"
            style={{ fill: '#4B5563', fontSize: 9 }}>
        no mock data
      </text>
    </svg>
  );
}

function downloadBlob(blob: Blob, name: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
