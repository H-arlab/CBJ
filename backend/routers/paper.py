"""
/api/paper/bundle — Phase 5 · one-click manuscript artifact.

Takes the current workspace (dataset registry + user-selected cell
list) and emits a single ZIP containing:

  paper_<timestamp>.zip
  ├── figures/
  │   ├── Fig1_<template>.pdf   (journal preset, col2 by default)
  │   ├── Fig1_<template>.svg
  │   ├── Fig2_<template>.pdf
  │   └── …
  ├── tables/
  │   ├── Table1_<metric>.csv
  │   ├── Table2_<op>.csv
  │   └── stats_apa.tex         (LaTeX tabular snippets, ready to \input)
  ├── captions.txt              (figure/table captions, auto-drafted)
  ├── main.tex                  (minimal compile-ready skeleton)
  └── README.txt                (file-→-artifact traceability)

This is an assistive template. The user edits captions + main body,
but the mechanical work (render at exact journal size, interpolate
stats into APA strings, track provenance) is automated.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Any, Optional, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


router = APIRouter(prefix="/api/paper", tags=["paper"])


class CellSpec(BaseModel):
    """Minimal description of a workspace cell, posted by the frontend."""
    id: str
    type: Literal["graph", "stat", "compute"]
    title: Optional[str] = None
    # graph fields
    graph: Optional[str] = None
    stride_avg: Optional[bool] = None
    dataset_id: Optional[str] = None
    datasets: Optional[list[dict[str, Any]]] = None
    # stat fields
    op: Optional[str] = None
    a_col: Optional[str] = None
    b_col: Optional[str] = None
    datasets_a: Optional[list[dict[str, Any]]] = None
    datasets_b: Optional[list[dict[str, Any]]] = None
    # compute fields
    metric: Optional[str] = None


class BundleRequest(BaseModel):
    preset: str = "ieee"
    variant: Literal["col1", "col2", "onehalf"] = "col2"
    format: Literal["pdf", "svg", "eps", "png", "tiff"] = "pdf"
    dpi: Optional[int] = None
    paper_title: Optional[str] = None
    cells: list[CellSpec]
    colorblind_safe: Optional[bool] = None


# ============================================================
# Helpers
# ============================================================


def _caption_for_graph(idx: int, cell: CellSpec) -> str:
    """Draft caption for a figure cell — user should edit before submission."""
    if cell.title:
        return f"Figure {idx}. {cell.title}"
    tpl = cell.graph or "graph"
    base = {
        "force":             "Ground reaction force (L vs R) over the gait cycle",
        "force_avg":         "Group-averaged GRF waveform (mean ± SD)",
        "asymmetry":         "Asymmetry index across strides",
        "cop":               "Center-of-pressure trajectory",
        "trials":            "Trial overlay (normalized force)",
        "cv_bar":            "Coefficient of variation across trials",
        "imu_avg":           "Joint-angle mean ± SD over the gait cycle (per channel)",
        "cyclogram":         "Shank-vs-thigh pitch cyclogram",
        "stride_time_trend": "Stride time across strides with linear fit",
        "stance_swing_bar":  "Stance and swing percentages (L vs R)",
        "rom_bar":           "Range of motion by joint and plane",
        "symmetry_radar":    "Multi-axis symmetry summary",
    }.get(tpl, tpl.replace("_", " ").title())
    n_series = len(cell.datasets) if cell.datasets else (1 if cell.dataset_id else 0)
    if n_series > 1:
        return f"Figure {idx}. {base} (N = {n_series} datasets)."
    return f"Figure {idx}. {base}."


def _caption_for_stat(idx: int, cell: CellSpec, result: dict[str, Any]) -> str:
    """APA-ish caption pulling numbers from the live stat result."""
    op_name = result.get("name", cell.op or "stat")
    stat_name = result.get("stat_name", "stat")
    stat_val = result.get("stat", 0)
    p = result.get("p")
    p_str = ("<.001" if isinstance(p, (int, float)) and p < 0.001 else
             f"= {p:.3f}" if p is not None else "= –")
    df = result.get("df")
    if isinstance(df, list):
        df_str = f"({df[0]}, {df[1]})"
    elif df is not None:
        df_str = f"({df:.1f})" if isinstance(df, float) else f"({df})"
    else:
        df_str = ""
    eff = result.get("effect_size")
    eff_str = (f", {eff['name']} = {eff['value']:.3f}"
               if isinstance(eff, dict) and "value" in eff else "")
    return (f"Table {idx}. {op_name}: {stat_name}{df_str} = "
            f"{stat_val:.2f}, p {p_str}{eff_str}.")


def _render_cell(cell: CellSpec, req: BundleRequest) -> tuple[bytes, str] | None:
    """Invoke the render endpoint's real-data path for a single cell."""
    from backend.routers.graphs import (
        RenderRequest, _render_multi_dataset, _render_real_data
    )
    from backend.services.publication_engine import render as render_mock

    r = RenderRequest(
        template=cell.graph or "force",
        preset=req.preset,
        variant=req.variant,
        format=req.format,
        dpi=req.dpi,
        stride_avg=bool(cell.stride_avg),
        dataset_id=cell.dataset_id,
        datasets=cell.datasets or [],
        title=cell.title or "",
        colorblind_safe=req.colorblind_safe,
    )
    try:
        multi = _render_multi_dataset(r)
        if multi is not None:
            return multi
        real = _render_real_data(r)
        if real is not None:
            return real
        data, mime = render_mock(
            template=r.template, preset=r.preset, variant=r.variant,
            format=r.format, dpi=r.dpi, stride_avg=r.stride_avg,
            colorblind_safe=r.colorblind_safe, title_override=r.title,
        )
        return data, mime
    except Exception:
        return None


def _run_compute(cell: CellSpec) -> dict[str, Any] | None:
    if not cell.metric or not cell.dataset_id:
        return None
    from backend.routers.analyze import analyze_cached
    from backend.services import compute_engine
    from backend.services.dataset_registry import get_path
    import pandas as pd
    try:
        res, _ = analyze_cached(cell.dataset_id)
        if res is None:
            return None
        path = get_path(cell.dataset_id)
        if not path:
            return None
        df = pd.read_csv(path)
        return compute_engine.compute(cell.metric, df, res)
    except Exception:
        return None


def _run_stat(cell: CellSpec) -> dict[str, Any] | None:
    from backend.services import stats_engine
    from backend.routers.stats import _extract_from_datasets, DatasetMetricRef

    if not cell.op:
        return None
    try:
        payload: dict[str, Any] = {}
        if cell.datasets_a or cell.datasets_b:
            refs_a = [DatasetMetricRef(**d) for d in (cell.datasets_a or [])]
            refs_b = [DatasetMetricRef(**d) for d in (cell.datasets_b or [])]
            if cell.op == "shapiro":
                payload["a"] = _extract_from_datasets(refs_a)
            else:
                payload["a"] = _extract_from_datasets(refs_a)
                payload["b"] = _extract_from_datasets(refs_b)
            return stats_engine.run(cell.op, payload)
        # Column-based (pull values from dataset CSV)
        if cell.dataset_id:
            import pandas as pd
            from backend.services.dataset_registry import get_path
            path = get_path(cell.dataset_id)
            if not path:
                return None
            df = pd.read_csv(path)
            a = df[cell.a_col].dropna().tolist() if cell.a_col and cell.a_col in df.columns else []
            b = df[cell.b_col].dropna().tolist() if cell.b_col and cell.b_col in df.columns else []
            if cell.op == "shapiro":
                return stats_engine.run("shapiro", {"a": a})
            return stats_engine.run(cell.op, {"a": a, "b": b})
    except Exception:
        return None
    return None


def _compute_table(compute_data: dict[str, Any]) -> str:
    """Render a compute result as CSV text."""
    lines = [",".join(compute_data.get("cols", []))]
    for row in compute_data.get("rows", []):
        lines.append(",".join(str(c) for c in row))
    return "\n".join(lines)


def _stat_apa_latex(idx: int, cell: CellSpec, r: dict[str, Any]) -> str:
    """Write a minimal APA-style tabular LaTeX snippet."""
    rows = [
        ("Test",  r.get("name", "")),
        ("n",     str(r.get("n", ""))),
        (r.get("stat_name", "stat"), f"{r.get('stat', 0):.3f}"),
        ("df",    str(r.get("df", ""))),
        ("p",     ("<.001" if isinstance(r.get("p"), float) and r["p"] < 0.001
                    else f"{r['p']:.3f}" if r.get('p') is not None else "–")),
    ]
    eff = r.get("effect_size")
    if isinstance(eff, dict) and "value" in eff:
        rows.append((eff.get("name", "effect"), f"{eff['value']:.3f}"))
    if r.get("ci95"):
        ci = r["ci95"]
        rows.append(("95% CI", f"[{ci[0]:.2f}, {ci[1]:.2f}]"))
    body = " \\\\\n".join(f"  {k} & {v}" for k, v in rows)
    return (
        f"% Table {idx} — {r.get('name', cell.op)}\n"
        f"\\begin{{tabular}}{{ll}}\n"
        f"\\hline\n"
        f"{body} \\\\\n"
        f"\\hline\n"
        f"\\end{{tabular}}\n"
    )


def _main_tex(req: BundleRequest, figure_files: list[str], table_files: list[str]) -> str:
    """Minimal compile-ready skeleton. Uses article + graphicx + booktabs.
    Figures go at the actual journal width via \\includegraphics."""
    title = req.paper_title or "Untitled study"
    body_figs = "\n".join(
        f"\\begin{{figure}}[!t]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=\\columnwidth]{{figures/{name}}}\n"
        f"  \\caption{{Replace with caption from captions.txt}}\n"
        f"  \\label{{fig:{name.split('.', 1)[0]}}}\n"
        f"\\end{{figure}}"
        for name in figure_files
    )
    body_tabs = "\n".join(
        f"\\begin{{table}}[!t]\n"
        f"  \\centering\n"
        f"  \\input{{tables/{name}}}\n"
        f"  \\caption{{Replace with caption from captions.txt}}\n"
        f"  \\label{{tab:{name.split('.', 1)[0]}}}\n"
        f"\\end{{table}}"
        for name in table_files if name.endswith(".tex")
    )
    return f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}

\\title{{{title}}}
\\author{{Your Name}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
Replace with 150–250 word abstract.
\\end{{abstract}}

\\section{{Introduction}}
Motivation and prior work.

\\section{{Methods}}
Participants, data acquisition (H-Walker @ {{fs}} Hz), signal processing
(heel-strike detection, stride filtering IQR × 2), and analysis (Shapiro-
Wilk normality screening before t-tests, non-parametric fall-back, effect
size reported).

\\section{{Results}}
{body_figs}

{body_tabs}

\\section{{Discussion}}
Interpretation + limitations.

\\bibliographystyle{{IEEEtran}}
\\bibliography{{refs}}

\\end{{document}}
"""


def _readme(req: BundleRequest, manifest: list[dict[str, Any]]) -> str:
    lines = [
        "H-Walker CORE · paper bundle",
        "============================",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Journal preset: {req.preset.upper()}",
        f"Figure width: {req.variant}",
        f"Figure format: {req.format.upper()}",
        f"DPI: {req.dpi or 'preset default'}",
        f"Paper title: {req.paper_title or '(none)'}",
        "",
        "Artifact provenance",
        "-------------------",
    ]
    for m in manifest:
        src = m.get("source", "—")
        lines.append(f"  {m['filename']:36s} ← {m['type']:7s} cell {m['cell_id']:12s} · {src}")
    lines += [
        "",
        "Next steps",
        "----------",
        "  1. Open main.tex and replace the skeleton sections with your write-up.",
        "  2. Pull figure captions from captions.txt into \\caption{} calls.",
        "  3. Check statistical test assumptions (Shapiro result in tables/*_apa.tex).",
        "  4. Run `pdflatex main.tex` (twice) → PDF.",
        "",
        "Regenerate this bundle any time by clicking RUN PAPER in the workspace.",
    ]
    return "\n".join(lines)


# ============================================================
# Endpoint
# ============================================================


@router.post("/bundle")
def bundle(req: BundleRequest):
    from backend.services.dataset_registry import _REGISTRY

    if not req.cells:
        raise HTTPException(status_code=400, detail="no cells provided")

    buf = io.BytesIO()
    fig_files: list[str] = []
    table_files: list[str] = []
    captions: list[str] = []
    manifest: list[dict[str, Any]] = []
    errors: list[str] = []

    fig_idx = 1
    tbl_idx = 1

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for cell in req.cells:
            src_desc = _dataset_desc(cell, _REGISTRY)

            if cell.type == "graph":
                rendered = _render_cell(cell, req)
                if rendered is None:
                    errors.append(f"  figure cell {cell.id}: render failed")
                    continue
                data, _mime = rendered
                fname = f"Fig{fig_idx}_{cell.graph or 'graph'}.{req.format}"
                zf.writestr(f"figures/{fname}", data)
                fig_files.append(fname)
                captions.append(_caption_for_graph(fig_idx, cell))
                manifest.append({
                    "filename": f"figures/{fname}",
                    "type": "figure", "cell_id": cell.id, "source": src_desc,
                })
                # Also drop an SVG companion for editing / presentations
                if req.format != "svg":
                    svg_req = req.copy(update={"format": "svg"})
                    svg_rendered = _render_cell(cell, svg_req)
                    if svg_rendered:
                        svg_name = f"Fig{fig_idx}_{cell.graph or 'graph'}.svg"
                        zf.writestr(f"figures/{svg_name}", svg_rendered[0])
                        manifest.append({
                            "filename": f"figures/{svg_name}",
                            "type": "figure", "cell_id": cell.id, "source": src_desc,
                        })
                fig_idx += 1

            elif cell.type == "compute":
                result = _run_compute(cell)
                if result is None:
                    errors.append(f"  compute cell {cell.id}: compute failed")
                    continue
                csv_name = f"Table{tbl_idx}_{cell.metric or 'metric'}.csv"
                zf.writestr(f"tables/{csv_name}", _compute_table(result))
                table_files.append(csv_name)
                captions.append(
                    f"Table {tbl_idx}. {result.get('label', cell.metric)} "
                    f"(source: {src_desc})."
                )
                manifest.append({
                    "filename": f"tables/{csv_name}",
                    "type": "table", "cell_id": cell.id, "source": src_desc,
                })
                tbl_idx += 1

            elif cell.type == "stat":
                result = _run_stat(cell)
                if result is None:
                    errors.append(f"  stat cell {cell.id}: run failed")
                    continue
                tex_name = f"Table{tbl_idx}_{cell.op or 'stat'}_apa.tex"
                zf.writestr(f"tables/{tex_name}", _stat_apa_latex(tbl_idx, cell, result))
                table_files.append(tex_name)
                # JSON sidecar so users can reparse without re-running
                zf.writestr(
                    f"tables/Table{tbl_idx}_{cell.op}.json",
                    json.dumps(result, indent=2, default=str),
                )
                captions.append(_caption_for_stat(tbl_idx, cell, result))
                manifest.append({
                    "filename": f"tables/{tex_name}",
                    "type": "stat", "cell_id": cell.id, "source": src_desc,
                })
                tbl_idx += 1

        # captions.txt — one block per figure/table in order
        zf.writestr("captions.txt", "\n\n".join(captions) or "(no captions)")
        # main.tex skeleton
        zf.writestr("main.tex", _main_tex(req, fig_files, table_files))
        # README with full provenance trail
        zf.writestr("README.txt", _readme(req, manifest))
        # Errors sidecar, if any
        if errors:
            zf.writestr("ERRORS.txt", "\n".join(errors))

    buf.seek(0)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"hwalker_paper_{req.preset}_{stamp}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _dataset_desc(cell: CellSpec, reg: dict[str, Any]) -> str:
    ids = []
    if cell.dataset_id:
        ids.append(cell.dataset_id)
    for d in (cell.datasets or []):
        if isinstance(d, dict) and d.get("id"):
            ids.append(d["id"])
    for d in (cell.datasets_a or []) + (cell.datasets_b or []):
        if isinstance(d, dict) and d.get("id"):
            ids.append(d["id"])
    if not ids:
        return "(no dataset bound)"
    names = []
    for i in ids:
        ds = reg.get(i, {})
        tag = ds.get("condition") or ds.get("tag") or ""
        names.append(f"{ds.get('name', i)}{' [' + tag + ']' if tag else ''}")
    return " + ".join(names)
