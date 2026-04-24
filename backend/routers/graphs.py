"""
/api/graphs/* — Phase 2 publication export.

Endpoints:
  GET  /api/graphs/templates         → list of GRAPH_TPLS keys
  GET  /api/graphs/presets           → list of JOURNAL_PRESETS with submission specs
  POST /api/graphs/render            → single figure at exact journal size
  POST /api/graphs/bundle            → ZIP of SVGs for multiple cells
  POST /api/graphs/multi_panel       → 2–4 cells composed as a single figure
                                        with (a)(b)(c)(d) sub-panels
"""
from __future__ import annotations

import io
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from backend.services.publication_engine import (
    JOURNAL_PRESETS, GRAPH_SPECS, render, list_templates, list_presets,
    render_from_traces, Trace,
)


router = APIRouter(prefix="/api/graphs", tags=["graphs"])


@router.get("/templates")
def templates() -> list[str]:
    return list_templates()


@router.get("/presets")
def presets() -> list[dict]:
    return list_presets()


class DatasetSeries(BaseModel):
    """Phase 1 multi-dataset — one dataset contributes one colored trace."""
    id: str
    label: Optional[str] = None
    color: Optional[str] = None


class RenderRequest(BaseModel):
    template: str
    preset: str = "ieee"
    variant: Literal["col1", "col2", "onehalf"] = "col2"
    format: Literal["svg", "pdf", "eps", "png", "tiff"] = "svg"
    dpi: Optional[int] = None
    stride_avg: bool = False
    colorblind_safe: Optional[bool] = None
    keep_palette: bool = False
    # Single-dataset legacy path
    dataset_id: Optional[str] = None
    # Phase 1: multi-dataset overlay. When non-empty, takes precedence
    # over `dataset_id`. Each entry contributes one trace labeled+colored
    # per the series config (color falls back to the preset palette).
    datasets: list[DatasetSeries] = []
    # Phase 2E: optional user-provided title.
    title: Optional[str] = None
    # L/R/both side filter — 'both' keeps current behavior (show both limbs).
    side: Literal["L", "R", "both"] = "both"


def _suggest_filename(req: RenderRequest, P) -> str:
    dim = f"{int(P.col2[0] if req.variant == 'col2' else P.col1[0])}mm"
    return f"hwalker_{req.template}_{req.preset}_{dim}.{req.format}"


REAL_DATA_TEMPLATES = {
    # Force / kinetic
    "force", "force_avg", "force_lr_subplot", "asymmetry", "peak_box", "trials",
    # Motion / kinematic
    "imu", "imu_avg", "cyclogram", "stride_time_trend",
    "stance_swing_bar", "rom_bar", "symmetry_radar",
    # New analysis templates
    "kinematics_ensemble", "spatiotemporal_bar", "force_tracking", "mos_trajectory",
    # Debug
    "debug_ts",
}

MULTI_DATASET_TEMPLATES = {
    "force_avg",
    "imu_avg",
    "kinematics_ensemble",  # overlay per-subject angle profiles
    "stride_time_trend",
    "asymmetry",
    "cyclogram",
}

_DEFAULT_SERIES_PALETTE = [
    "#3B82C4", "#D35454", "#F09708", "#00FFB2", "#A78BFA",
    "#1E5F9E", "#9E3838", "#FFB347", "#56B4E9", "#009E73",
    "#CC79A7", "#F0E442", "#0072B2", "#E69F00", "#7FB5E4",
]


def _auto_sync_if_needed(req: RenderRequest) -> None:
    """Phase 3 · Auto-align cross-source datasets.

    If any dataset in the overlay has a different sample rate from the
    others, transparently run /api/sync/align and swap the ids on the
    request to point at the freshly synced datasets. This is what makes
    Robot × MoCap × Force-plate comparisons "just work" without the user
    having to click anything when Hz mismatches are present.
    """
    if not req.datasets or len(req.datasets) < 2:
        return
    from backend.routers.datasets import _REGISTRY

    rates: set[float] = set()
    source_types: set[str] = set()
    has_unsynced = False
    for s in req.datasets:
        d = _REGISTRY.get(s.id)
        if d is None:
            continue
        if d.get("synced_from"):
            continue  # already synced
        has_unsynced = True
        try:
            rates.add(float(str(d.get("hz", "100")).replace("Hz", "").strip()))
        except ValueError:
            pass
        source_types.add(d.get("source_type", "unknown"))

    # Only auto-sync when fs differs OR the sources are mixed (e.g.
    # robot + mocap at the same nominal fs but temporally offset).
    cross_source = len({s for s in source_types if s != "unknown"}) >= 2
    fs_mismatch = len(rates) >= 2
    if not has_unsynced or not (cross_source or fs_mismatch):
        return

    from backend.routers.sync import AlignRequest, align as _do_align
    try:
        out = _do_align(AlignRequest(
            dataset_ids=[s.id for s in req.datasets],
            crop_to_a7=True,
        ))
    except Exception:
        return  # fall back to raw rendering silently

    # Swap IDs on the request to the synced outputs
    id_map = {a.original_id: a.new_id for a in out.aligned}
    for s in req.datasets:
        if s.id in id_map:
            s.id = id_map[s.id]


def _render_multi_dataset(req: RenderRequest) -> tuple[bytes, str] | None:
    """Phase 1 · Multi-dataset overlay (with Phase 3 auto-sync).

    Returns None if this path isn't applicable (template doesn't support
    overlay, no datasets list, etc.) — caller falls through to single-
    dataset rendering.
    """
    if not req.datasets or len(req.datasets) < 2:
        return None
    if req.template not in MULTI_DATASET_TEMPLATES:
        return None

    # If Hz mismatches or sources are mixed, sync transparently first.
    _auto_sync_if_needed(req)

    from backend.routers.analyze import analyze_cached
    from backend.routers.datasets import get_path, _REGISTRY
    from backend.services.compute_engine import resample_column
    import numpy as _np
    import pandas as _pd

    # Color assignment — respect user-provided colors, fall back to palette
    def color_for(i: int, series: DatasetSeries) -> str:
        return series.color or _DEFAULT_SERIES_PALETTE[i % len(_DEFAULT_SERIES_PALETTE)]

    def label_for(series: DatasetSeries) -> str:
        if series.label:
            return series.label
        reg = _REGISTRY.get(series.id)
        return (reg or {}).get("name", series.id)

    traces: list[Trace] = []
    gcp_axis = list(_np.linspace(0, 100, 101))

    # ── force_avg · GRF mean±SD per subject (L side, overlay)
    if req.template == "force_avg":
        for i, s in enumerate(req.datasets):
            try:
                res, _ = analyze_cached(s.id)
            except Exception:
                continue
            if res is None or res.left_force_profile.mean is None:
                continue
            mean = res.left_force_profile.mean
            std = res.left_force_profile.std
            c = color_for(i, s)
            if std is not None:
                traces.append(Trace(kind="band", name="",
                                    x=gcp_axis, y=list(mean),
                                    y_upper=list(mean + std), y_lower=list(mean - std),
                                    color=c, opacity=0.15))
            traces.append(Trace(kind="line", name=label_for(s),
                                x=gcp_axis, y=list(mean),
                                color=c, width=1.8))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Gait cycle (%)", y_label="Force (N)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # ── imu_avg · joint angle mean per subject
    if req.template == "imu_avg":
        for i, s in enumerate(req.datasets):
            try:
                res, _ = analyze_cached(s.id)
            except Exception:
                continue
            if res is None:
                continue
            path = get_path(s.id)
            if not path:
                continue
            try:
                df = _pd.read_csv(path)
            except Exception:
                continue
            ls = res.left_stride
            result = resample_column(df, "L_Pitch", ls.hs_indices, ls.valid_mask)
            if result is None:
                continue
            mean, std = result
            c = color_for(i, s)
            traces.append(Trace(kind="band", name="",
                                x=gcp_axis, y=list(mean),
                                y_upper=list(mean + std), y_lower=list(mean - std),
                                color=c, opacity=0.12))
            traces.append(Trace(kind="line", name=label_for(s),
                                x=gcp_axis, y=list(mean),
                                color=c, width=1.8))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Gait cycle (%)", y_label="Pitch (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # ── stride_time_trend · per-subject stride-time series + fit
    if req.template == "stride_time_trend":
        for i, s in enumerate(req.datasets):
            try:
                res, _ = analyze_cached(s.id)
            except Exception:
                continue
            if res is None or len(res.left_stride.stride_times) < 2:
                continue
            times = res.left_stride.stride_times
            xs = list(range(1, len(times) + 1))
            c = color_for(i, s)
            traces.append(Trace(kind="scatter", name=label_for(s),
                                x=xs, y=list(times),
                                color=c, width=1.8, opacity=0.7))
            if len(times) >= 3:
                coef = _np.polyfit(xs, times, 1)
                fit_y = list(_np.polyval(coef, xs))
                traces.append(Trace(kind="line", name="",
                                    x=xs, y=fit_y,
                                    color=c, width=1.1, dash=True))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Stride #", y_label="Stride time (s)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # ── asymmetry · per-subject asymmetry series
    if req.template == "asymmetry":
        from backend.services.compute_engine import _asym_idx
        for i, s in enumerate(req.datasets):
            try:
                res, _ = analyze_cached(s.id)
            except Exception:
                continue
            if res is None:
                continue
            lfp, rfp = res.left_force_profile, res.right_force_profile
            if lfp.individual is None or rfp.individual is None:
                continue
            n = min(lfp.individual.shape[0], rfp.individual.shape[0])
            peaks_l = lfp.individual[:n].max(axis=1)
            peaks_r = rfp.individual[:n].max(axis=1)
            asym = [_asym_idx(float(pl), float(pr)) for pl, pr in zip(peaks_l, peaks_r)]
            xs = list(range(1, n + 1))
            c = color_for(i, s)
            traces.append(Trace(kind="line", name=label_for(s),
                                x=xs, y=asym,
                                color=c, width=1.8))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Stride #", y_label="Asymmetry (%)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # ── cyclogram · multi-subject phase portraits
    if req.template == "cyclogram":
        for i, s in enumerate(req.datasets):
            try:
                res, _ = analyze_cached(s.id)
            except Exception:
                continue
            if res is None:
                continue
            path = get_path(s.id)
            if not path:
                continue
            try:
                df = _pd.read_csv(path)
            except Exception:
                continue
            if "L_Pitch" not in df.columns or "R_Pitch" not in df.columns:
                continue
            ls = res.left_stride
            xr = resample_column(df, "L_Pitch", ls.hs_indices, ls.valid_mask)
            yr = resample_column(df, "R_Pitch", ls.hs_indices, ls.valid_mask)
            if xr is None or yr is None:
                continue
            c = color_for(i, s)
            traces.append(Trace(kind="line", name=label_for(s),
                                x=list(xr[0]), y=list(yr[0]),
                                color=c, width=1.6))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="L_Pitch (°)", y_label="R_Pitch (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # ── kinematics_ensemble · per-subject angle profile overlay
    if req.template == "kinematics_ensemble":
        _ANGLE_PRIORITY = ("L_Hip_Flex", "L_HipFlexion", "L_Knee_Flex", "L_KneeFlexion",
                            "L_Ankle_Flex", "L_AnkleFlexion", "L_Pitch")
        for i, s in enumerate(req.datasets):
            try:
                res, _ = analyze_cached(s.id)
            except Exception:
                continue
            if res is None:
                continue
            path = get_path(s.id)
            if not path:
                continue
            try:
                df = _pd.read_csv(path)
            except Exception:
                continue
            ls = res.left_stride
            angle_col = next((c for c in _ANGLE_PRIORITY if c in df.columns), None)
            if angle_col is None:
                continue
            result = resample_column(df, angle_col, ls.hs_indices, ls.valid_mask)
            if result is None:
                continue
            mean, std = result
            c = color_for(i, s)
            traces.append(Trace(kind="band", name="",
                                x=gcp_axis, y=list(mean),
                                y_upper=list(mean + std), y_lower=list(mean - std),
                                color=c, opacity=0.12))
            traces.append(Trace(kind="line", name=label_for(s),
                                x=gcp_axis, y=list(mean),
                                color=c, width=1.8))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Gait cycle (%)", y_label="Angle (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    return None


def _render_generic_csv(req: "RenderRequest", df) -> tuple[bytes, str] | None:
    """Fallback renderer for CSVs that don't match H-Walker schema.

    The analyzer couldn't produce stride-aware profiles, but the user still
    has real data they want to inspect. We pick a reasonable signal from the
    DataFrame and render a simple time-series or (if we can find footfall
    events) a per-stride overlay — so plots aren't silently replaced by
    mock bezier curves.

    Covers the most common "just show me my data" templates:
      - force / force_avg  → any force-like column vs time (or gait cycle)
      - debug_ts           → every numeric column, small multiples
      - trials             → same as debug_ts
    Returns None for templates that genuinely need per-stride analysis.
    """
    if df is None or df.empty:
        return None

    import numpy as _np
    from backend.services.sync_engine import find_force_column
    from backend.services.publication_engine import (
        JOURNAL_PRESETS as _JP, _compose_rc, _emit,
    )
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt

    # Guess sample rate from registry
    from backend.routers.datasets import _REGISTRY as _REG
    hz_str = _REG.get(req.dataset_id, {}).get("hz", "100Hz") if req.dataset_id else "100Hz"
    try:
        fs = float(str(hz_str).replace("Hz", "").strip()) or 100.0
    except Exception:
        fs = 100.0

    numeric_cols = df.select_dtypes(include=[_np.number]).columns.tolist()
    # Drop obvious time columns from the pickable set
    time_like = {c for c in numeric_cols
                  if c.lower() in ("time", "time_s", "t", "timestamp", "sample")}
    plottable = [c for c in numeric_cols if c not in time_like]

    if not plottable:
        return None

    t = _np.arange(len(df)) / fs
    P = _JP[req.preset]

    # Figure size per journal preset
    if req.variant == "col1":
        w_mm, h_mm = P.col1
    elif req.variant == "onehalf" and P.onehalf:
        w_mm, h_mm = P.onehalf
    else:
        w_mm, h_mm = P.col2
    dpi_val = req.dpi or P.dpi
    inch_w, inch_h = w_mm / 25.4, h_mm / 25.4

    def _emit_fig(fig):
        return _emit(fig, req.format, dpi_val)

    # Apply the cell's L/R/both side filter here too — generic CSVs
    # need to honor the toggle just like the H-Walker-schema path.
    inc_L = req.side in ("L", "both")
    inc_R = req.side in ("R", "both")

    def _side_of(col: str) -> str | None:
        """Return 'L' / 'R' / None based on a heuristic on the column name."""
        cl = col.lower()
        if cl.startswith("l_") or "_l_" in cl or cl.startswith("left") or cl.endswith("_l"):
            return "L"
        if cl.startswith("r_") or "_r_" in cl or cl.startswith("right") or cl.endswith("_r"):
            return "R"
        return None

    def _keep(col: str) -> bool:
        s = _side_of(col)
        if s == "L":
            return inc_L
        if s == "R":
            return inc_R
        # No side token — always keep (user may have single-channel data)
        return True

    with _mpl.rc_context(_compose_rc(P)):
        if req.template in ("force", "force_avg"):
            # Find force-like columns; fallback to the first 1-2 plottable columns.
            force_col = find_force_column(df)
            picks: list[str]
            if force_col:
                # Try to also find opposite side
                lower = force_col.lower()
                pair = None
                if "l_" in lower or lower.startswith("l"):
                    for c in plottable:
                        cl = c.lower()
                        if c != force_col and ("r_" in cl or cl.startswith("r")) and "force" in cl:
                            pair = c
                            break
                elif "r_" in lower or lower.startswith("r"):
                    for c in plottable:
                        cl = c.lower()
                        if c != force_col and ("l_" in cl or cl.startswith("l")) and "force" in cl:
                            pair = c
                            break
                picks = [force_col] + ([pair] if pair else [])
            else:
                picks = plottable[:2]

            # Respect the L/R side toggle
            picks = [c for c in picks if _keep(c)]
            if not picks:
                raise HTTPException(
                    status_code=400,
                    detail=f"Side filter '{req.side}' left no force columns to plot. "
                           f"Detected columns: {[c for c in plottable if 'force' in c.lower()] or plottable[:4]}."
                )

            fig, ax = _plt.subplots(figsize=(inch_w, inch_h), dpi=dpi_val)
            colors = ["#1E5F9E", "#9E3838", "#F09708"]
            for c, color in zip(picks, colors):
                ax.plot(t, df[c].to_numpy(dtype=float), color=color, linewidth=1.2, label=c)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(picks[0] if len(picks) == 1 else "Value")
            if req.title:
                ax.set_title(req.title)
            if len(picks) >= 2:
                ax.legend(loc="best", frameon=False)
            fig.tight_layout()
            return _emit_fig(fig)

        if req.template in ("debug_ts", "trials"):
            # Small multiples: up to 4 rows — respect L/R filter
            rows = [c for c in plottable if _keep(c)][:4]
            n = len(rows)
            fig, axes = _plt.subplots(n, 1, figsize=(inch_w, inch_h * max(1, n / 2)),
                                        dpi=dpi_val, sharex=True)
            if n == 1:
                axes = [axes]
            for ax, c in zip(axes, rows):
                ax.plot(t, df[c].to_numpy(dtype=float), linewidth=1.0, color="#1E5F9E")
                ax.set_ylabel(c, fontsize=max(6, P.body_pt - 2))
                ax.tick_params(labelsize=max(6, P.axis_pt - 1))
            axes[-1].set_xlabel("Time (s)")
            if req.title:
                fig.suptitle(req.title, fontsize=P.body_pt)
            fig.tight_layout()
            return _emit_fig(fig)

    return None


def _render_real_data(req: RenderRequest) -> tuple[bytes, str] | None:
    """If dataset_id is set and the template supports real-data binding, build
    traces from the analyzer result and render. Returns None if not applicable.
    """
    if not req.dataset_id:
        return None

    if req.template not in REAL_DATA_TEMPLATES:
        return None

    from backend.routers.analyze import analyze_cached  # avoid cycle at import time

    try:
        res, _payload = analyze_cached(req.dataset_id)
    except HTTPException:
        raise
    except Exception:
        return None

    import numpy as _np
    # Load the raw df for columns the analyzer doesn't pre-resample (joint angles, etc.)
    from backend.routers.datasets import get_path
    df_path = get_path(req.dataset_id)
    import pandas as _pd
    df = _pd.read_csv(df_path) if df_path else None

    if res is None:
        # Generic-CSV fallback: the CSV doesn't match H-Walker schema, but we
        # still want the user to see their real data instead of a mock bezier.
        return _render_generic_csv(req, df)

    gcp_axis = list(_np.linspace(0, 100, 101))
    traces: list[Trace] = []
    inc_L = req.side in ("L", "both")
    inc_R = req.side in ("R", "both")

    if req.template in ("force", "force_avg"):
        lfp = res.left_force_profile
        rfp = res.right_force_profile
        if req.template == "force_avg" or req.stride_avg:
            if inc_L and lfp.mean is not None and lfp.std is not None:
                traces.append(Trace(kind="band", name="L ± SD",
                                    x=gcp_axis,
                                    y=list(lfp.mean), y_upper=list(lfp.mean + lfp.std),
                                    y_lower=list(lfp.mean - lfp.std),
                                    color="#3B82C4", opacity=0.2))
                traces.append(Trace(kind="line", name="L mean",
                                    x=gcp_axis, y=list(lfp.mean),
                                    color="#1E5F9E", width=2.0))
            if inc_R and rfp.mean is not None and rfp.std is not None:
                traces.append(Trace(kind="band", name="R ± SD",
                                    x=gcp_axis,
                                    y=list(rfp.mean), y_upper=list(rfp.mean + rfp.std),
                                    y_lower=list(rfp.mean - rfp.std),
                                    color="#D35454", opacity=0.2))
                traces.append(Trace(kind="line", name="R mean",
                                    x=gcp_axis, y=list(rfp.mean),
                                    color="#9E3838", width=2.0))
        else:
            # Instantaneous L/R mean + desired overlay
            if inc_L and lfp.mean is not None:
                traces.append(Trace(kind="line", name="L Actual",
                                    x=gcp_axis, y=list(lfp.mean),
                                    color="#3B82C4", width=2.0))
            if inc_L and lfp.des_mean is not None:
                traces.append(Trace(kind="line", name="L Desired",
                                    x=gcp_axis, y=list(lfp.des_mean),
                                    color="#7FB5E4", width=1.3, dash=True))
            if inc_R and rfp.mean is not None:
                traces.append(Trace(kind="line", name="R Actual",
                                    x=gcp_axis, y=list(rfp.mean),
                                    color="#D35454", width=2.0))
            if inc_R and rfp.des_mean is not None:
                traces.append(Trace(kind="line", name="R Desired",
                                    x=gcp_axis, y=list(rfp.des_mean),
                                    color="#E89B9B", width=1.3, dash=True))
        # Journal convention: no in-figure title. Caption goes below in the
        # manuscript. Only draw a title if the user explicitly provided one.
        return render_from_traces(
            traces, title=req.title or "", x_label="Gait cycle (%)", y_label="Force (N)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    if req.template == "asymmetry":
        # Per-stride asymmetry of peak force: use analyzer's per-stride data
        lfp = res.left_force_profile
        rfp = res.right_force_profile
        if lfp.individual is None or rfp.individual is None:
            return None
        n = min(lfp.individual.shape[0], rfp.individual.shape[0])
        peaks_l = lfp.individual[:n].max(axis=1)
        peaks_r = rfp.individual[:n].max(axis=1)
        denom = (peaks_l + peaks_r) / 2.0
        asym = _np.where(_np.abs(denom) > 1e-9,
                         _np.abs(peaks_l - peaks_r) / _np.abs(denom) * 100.0,
                         0.0)
        xs = list(range(1, n + 1))
        traces.append(Trace(kind="line", name="asym_idx",
                            x=xs, y=list(asym),
                            color="#F09708", width=1.8))
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Stride #", y_label="Asymmetry (%)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=False,
        )

    if req.template == "peak_box":
        lfp = res.left_force_profile
        rfp = res.right_force_profile
        if lfp.individual is None and rfp.individual is None:
            return None
        box_traces: list[Trace] = []
        if inc_L and lfp.individual is not None:
            peaks_l = lfp.individual.max(axis=1).tolist()
            box_traces.append(Trace(kind="box", name="L", x=["L"], y=peaks_l,
                                    color="#3B82C4"))
        if inc_R and rfp.individual is not None:
            peaks_r = rfp.individual.max(axis=1).tolist()
            box_traces.append(Trace(kind="box", name="R", x=["R"], y=peaks_r,
                                    color="#D35454"))
        return render_from_traces(
            box_traces, title=req.title or "",
            x_label="", y_label="Peak GRF (N)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=False,
        )

    if req.template == "trials":
        # Overlay individual stride profiles (up to 5 for readability)
        # side selects which limb's strides to show (default: L)
        if inc_L and not inc_R:
            fp_side = res.left_force_profile
        elif inc_R and not inc_L:
            fp_side = res.right_force_profile
        else:
            fp_side = res.left_force_profile  # 'both' → show L (cleaner for trials)
        if fp_side.individual is None:
            return None
        n = min(5, fp_side.individual.shape[0])
        colors = ["#7FB5E4", "#3B82C4", "#E89B9B", "#D35454", "#F09708"]
        side_label = "R" if (inc_R and not inc_L) else "L"
        for i in range(n):
            traces.append(Trace(kind="line", name=f"{side_label} Stride {i + 1}",
                                x=gcp_axis, y=fp_side.individual[i].tolist(),
                                color=colors[i % len(colors)], width=1.4))
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Gait cycle (%)", y_label="Force (N)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # =====================================================
    # Phase 0 · Motion / kinematic templates
    # =====================================================

    if req.template == "imu":
        # Pitch/roll/yaw time series (raw, first 8 seconds). Labels follow
        # the actual column name — no hard-coded "shank" / "thigh" assumption.
        if df is None:
            return None
        fs = res.sample_rate
        n_max = int(min(len(df), fs * 8.0))
        t = list(_np.arange(n_max) / fs)
        all_imu = [c for c in ("L_Pitch", "R_Pitch", "L_Roll", "R_Roll",
                                "L_Yaw", "R_Yaw") if c in df.columns]
        candidates = [c for c in all_imu
                      if (inc_L and c.startswith("L_")) or (inc_R and c.startswith("R_"))][:4]
        colors = ["#3B82C4", "#D35454", "#7FB5E4", "#E89B9B"]
        for i, col in enumerate(candidates):
            traces.append(Trace(kind="line", name=col,
                                x=t, y=df[col].iloc[:n_max].tolist(),
                                color=colors[i % len(colors)], width=1.6))
        if not traces:
            raise HTTPException(
                status_code=400,
                detail=f"IMU time-series needs at least one of "
                       f"L_Pitch/R_Pitch/L_Roll/R_Roll/L_Yaw/R_Yaw in the CSV. "
                       f"Available columns: {list(df.columns)[:20]}"
                       + (" …" if len(df.columns) > 20 else "") + "."
            )
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Time (s)", y_label="Pitch (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    if req.template == "imu_avg":
        # Pitch-angle mean ± SD over 0–100% gait cycle. Labels reflect the
        # actual column name — no hard-coded "shank" / "thigh" assumption.
        if df is None:
            return None
        from backend.services.compute_engine import resample_column
        ls, rs = res.left_stride, res.right_stride
        available = [c for c in ("L_Pitch", "R_Pitch") if c in df.columns]
        if not available:
            raise HTTPException(
                status_code=400,
                detail=f"imu_avg needs L_Pitch and/or R_Pitch. "
                       f"Neither found in this CSV. "
                       f"Available pitch-like columns: "
                       f"{[c for c in df.columns if 'Pitch' in c or 'pitch' in c] or 'none'}."
            )
        imu_avg_cols = []
        if inc_L:
            imu_avg_cols.append(("L_Pitch", "#1E5F9E", "#3B82C4"))
        if inc_R:
            imu_avg_cols.append(("R_Pitch", "#9E3838", "#D35454"))
        for col, color_line, color_band in imu_avg_cols:
            if col not in df.columns:
                continue
            side_stride = ls if col.startswith("L_") else rs
            result = resample_column(df, col, side_stride.hs_indices,
                                      side_stride.valid_mask)
            if result is None:
                continue
            mean, std = result
            traces.append(Trace(kind="band", name=f"{col} ± SD",
                                x=gcp_axis, y=list(mean),
                                y_upper=list(mean + std),
                                y_lower=list(mean - std),
                                color=color_band, opacity=0.2))
            traces.append(Trace(kind="line", name=f"{col} mean",
                                x=gcp_axis, y=list(mean),
                                color=color_line, width=2.0))
        if not traces:
            # Column(s) existed but had no valid stride data
            raise HTTPException(
                status_code=400,
                detail=f"imu_avg: {', '.join(available)} present but no valid "
                       f"strides found to average. Check heel-strike detection "
                       f"(debug_ts figure) and make sure force data is also valid."
            )
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Gait cycle (%)", y_label="Pitch (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    if req.template == "cyclogram":
        # Phase plot: L pitch vs R pitch over the gait cycle
        if df is None:
            return None
        from backend.services.compute_engine import resample_column
        ls = res.left_stride

        x_col = "L_Pitch" if "L_Pitch" in df.columns else None
        y_col = "R_Pitch" if "R_Pitch" in df.columns else None
        if not x_col or not y_col:
            missing = [c for c, have in [("L_Pitch", x_col), ("R_Pitch", y_col)] if not have]
            raise HTTPException(
                status_code=400,
                detail=f"cyclogram needs both L_Pitch and R_Pitch columns. "
                       f"Missing: {', '.join(missing)}. "
                       f"Available pitch-like columns: "
                       f"{[c for c in df.columns if 'Pitch' in c or 'pitch' in c] or 'none'}."
            )

        x_res = resample_column(df, x_col, ls.hs_indices, ls.valid_mask)
        y_res = resample_column(df, y_col, ls.hs_indices, ls.valid_mask)
        if x_res is None or y_res is None:
            return None
        x_mean, _ = x_res
        y_mean, _ = y_res
        traces.append(Trace(kind="line", name="Cycle avg",
                            x=list(x_mean), y=list(y_mean),
                            color="#F09708", width=1.8))
        # Mark 0 / 25 / 50 / 75 / 100 % gait-cycle points
        for pct, label in [(0, "HS"), (25, "25%"), (50, "TO"), (75, "75%")]:
            idx = int(pct * 1.01)
            traces.append(Trace(kind="scatter", name=label,
                                x=[float(x_mean[idx])], y=[float(y_mean[idx])],
                                color="#00FFB2", width=4.0, opacity=1.0))
        return render_from_traces(
            traces, title=req.title or "",
            x_label=f"{x_col} (°)", y_label=f"{y_col} (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    if req.template == "stride_time_trend":
        # Stride time across stride # + linear-regression fit
        ls, rs = res.left_stride, res.right_stride
        stride_sides = []
        if inc_L:
            stride_sides.append((ls.stride_times, "L", "#3B82C4"))
        if inc_R:
            stride_sides.append((rs.stride_times, "R", "#D35454"))
        for times, label, color in stride_sides:
            if len(times) < 2:
                continue
            xs = list(range(1, len(times) + 1))
            traces.append(Trace(kind="scatter", name=f"{label} strides",
                                x=xs, y=list(times),
                                color=color, width=2.0, opacity=0.7))
            # Linear fit
            if len(times) >= 3:
                coef = _np.polyfit(xs, times, 1)
                fit_y = list(_np.polyval(coef, xs))
                traces.append(Trace(kind="line",
                                    name=f"{label} trend (slope={coef[0]*1000:.1f} ms/stride)",
                                    x=xs, y=fit_y,
                                    color=color, width=1.2, dash=True))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Stride #", y_label="Stride time (s)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    if req.template == "force_lr_subplot":
        # Side-by-side L / R panels (both = default). If only one side is
        # selected, render a single panel at the same journal size.
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        from backend.services.publication_engine import (
            JOURNAL_PRESETS as _JP, _compose_rc, _emit,
        )
        lfp, rfp = res.left_force_profile, res.right_force_profile
        P = _JP[req.preset]
        if req.variant == "col1":
            w_mm, h_mm = P.col1
        elif req.variant == "onehalf" and P.onehalf:
            w_mm, h_mm = P.onehalf
        else:
            w_mm, h_mm = P.col2
        dpi_val = req.dpi or P.dpi
        inch_w, inch_h = w_mm / 25.4, h_mm / 25.4
        x = _np.linspace(0, 100, 101)
        rc = _compose_rc(P)

        # Single-side: render as a compact single panel
        if not (inc_L and inc_R):
            fp = lfp if inc_L else rfp
            side_label = "Left" if inc_L else "Right"
            color_line = "#1E5F9E" if inc_L else "#9E3838"
            color_band = "#3B82C4" if inc_L else "#D35454"
            color_des = "#7FB5E4" if inc_L else "#E89B9B"
            act_label = "L actual" if inc_L else "R actual"
            des_label = "L desired" if inc_L else "R desired"
            if fp.mean is None:
                return None
            with _mpl.rc_context(rc):
                fig, ax = _plt.subplots(figsize=(inch_w, inch_h))
                if fp.std is not None:
                    ax.fill_between(x, fp.mean - fp.std, fp.mean + fp.std,
                                    color=color_band, alpha=0.18, linewidth=0)
                ax.plot(x, fp.mean, color=color_line,
                        linewidth=P.stroke_pt * 1.8, label=act_label)
                if fp.des_mean is not None:
                    ax.plot(x, fp.des_mean, color=color_des,
                            linewidth=P.stroke_pt * 1.2, linestyle="--",
                            dashes=(3, 2), label=des_label)
                ax.set_xlabel("Gait cycle (%)")
                ax.set_ylabel("Force (N)")
                ax.set_title(side_label, fontsize=P.title_pt)
                ax.grid(True, linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)
                ax.legend(loc="best", frameon=False, fontsize=P.legend_pt)
                if req.title:
                    fig.suptitle(req.title, fontsize=P.title_pt, y=0.998)
                fig.tight_layout(pad=0.4)
                fig.set_size_inches(inch_w, inch_h)
                return _emit(fig, req.format, dpi_val, preset_name=P.name)

        # Both sides: classic side-by-side subplot
        if lfp.mean is None or rfp.mean is None:
            return None
        with _mpl.rc_context(rc):
            fig, (axL, axR) = _plt.subplots(
                1, 2, figsize=(inch_w, inch_h), sharey=True,
            )
            # Left panel
            if lfp.std is not None:
                axL.fill_between(x, lfp.mean - lfp.std, lfp.mean + lfp.std,
                                  color="#3B82C4", alpha=0.18, linewidth=0)
            axL.plot(x, lfp.mean, color="#1E5F9E",
                     linewidth=P.stroke_pt * 1.8, label="L actual")
            if lfp.des_mean is not None:
                axL.plot(x, lfp.des_mean, color="#7FB5E4",
                         linewidth=P.stroke_pt * 1.2, linestyle="--",
                         dashes=(3, 2), label="L desired")
            axL.set_xlabel("Gait cycle (%)")
            axL.set_ylabel("Force (N)")
            axL.set_title("Left", fontsize=P.title_pt)
            axL.grid(True, linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)
            axL.legend(loc="best", frameon=False, fontsize=P.legend_pt)
            # Right panel
            if rfp.std is not None:
                axR.fill_between(x, rfp.mean - rfp.std, rfp.mean + rfp.std,
                                  color="#D35454", alpha=0.18, linewidth=0)
            axR.plot(x, rfp.mean, color="#9E3838",
                     linewidth=P.stroke_pt * 1.8, label="R actual")
            if rfp.des_mean is not None:
                axR.plot(x, rfp.des_mean, color="#E89B9B",
                         linewidth=P.stroke_pt * 1.2, linestyle="--",
                         dashes=(3, 2), label="R desired")
            axR.set_xlabel("Gait cycle (%)")
            axR.set_title("Right", fontsize=P.title_pt)
            axR.grid(True, linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)
            axR.legend(loc="best", frameon=False, fontsize=P.legend_pt)
            if req.title:
                fig.suptitle(req.title, fontsize=P.title_pt, y=0.995)
            fig.tight_layout(pad=0.4)
            fig.set_size_inches(inch_w, inch_h)
            return _emit(fig, req.format, dpi_val, preset_name=P.name)

    if req.template == "stance_swing_bar":
        ls, rs = res.left_stride, res.right_stride
        bar_items = []
        if inc_L:
            bar_items += [("L stance", ls.stance_pct_mean, "#3B82C4"),
                          ("L swing",  ls.swing_pct_mean,  "#7FB5E4")]
        if inc_R:
            bar_items += [("R stance", rs.stance_pct_mean, "#D35454"),
                          ("R swing",  rs.swing_pct_mean,  "#E89B9B")]
        for lab, h, c in bar_items:
            traces.append(Trace(kind="bar", name=lab,
                                x=[lab], y=[float(h)], color=c))
        return render_from_traces(
            traces, title=req.title or "",
            x_label="", y_label="% of gait cycle",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=False,
            y_range=(0, 100),
        )

    if req.template == "rom_bar":
        # ROM per joint (pitch / roll / yaw) for L and R
        if df is None:
            return None
        ls = res.left_stride
        joints = [("Pitch", "sagittal"), ("Roll", "frontal"), ("Yaw", "transverse")]
        rom_sides = []
        if inc_L:
            rom_sides.append(("L", ls.hs_indices, ls.valid_mask, "#3B82C4"))
        if inc_R:
            rom_sides.append(("R", res.right_stride.hs_indices, res.right_stride.valid_mask, "#D35454"))
        for side, hs, valid, color in rom_sides:
            for joint, plane in joints:
                col = f"{side}_{joint}"
                if col not in df.columns:
                    continue
                roms = []
                for i in range(len(valid)):
                    if not valid[i] or i + 1 >= len(hs):
                        continue
                    chunk = df[col].iloc[hs[i]:hs[i+1]].to_numpy(dtype=float)
                    chunk = chunk[_np.isfinite(chunk)]
                    if len(chunk) >= 2:
                        roms.append(float(_np.max(chunk) - _np.min(chunk)))
                if roms:
                    traces.append(Trace(kind="bar", name=f"{side} {plane}",
                                        x=[f"{side} {plane}"], y=[float(_np.mean(roms))],
                                        color=color))
        if not traces:
            return None
        return render_from_traces(
            traces, title=req.title or "",
            x_label="", y_label="ROM (°)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=False,
        )

    if req.template == "debug_ts":
        # Full-duration raw time-series, small multiples. Intended for
        # visual debugging — where did the signal go weird? Heel-strike
        # events are overlaid as vertical dotted lines so the user can
        # orient themselves against the gait cycle.
        if df is None:
            return None
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        from backend.services.publication_engine import (
            JOURNAL_PRESETS as _JP, _compose_rc, _emit,
        )
        P = _JP[req.preset]
        # Taller figure for stacked panels
        if req.variant == "col1":
            w_mm, h_mm = P.col1[0], P.col1[1] * 2.0
        elif req.variant == "onehalf" and P.onehalf:
            w_mm, h_mm = P.onehalf[0], P.onehalf[1] * 2.0
        else:
            w_mm, h_mm = P.col2[0], P.col2[1] * 1.8
        dpi_val = req.dpi or P.dpi
        inch_w, inch_h = w_mm / 25.4, h_mm / 25.4
        fs = res.sample_rate
        t = _np.arange(len(df)) / fs

        # Pick the signals that exist
        rows = []
        for cols, ylabel, colors in [
            (["L_ActForce_N", "R_ActForce_N"], "Force (N)", ["#1E5F9E", "#9E3838"]),
            (["L_Pitch", "R_Pitch"],           "Pitch (°)", ["#3B82C4", "#D35454"]),
            (["L_ActVel_mps", "R_ActVel_mps"], "Vel (m/s)", ["#F09708", "#FFB347"]),
        ]:
            present = [c for c in cols if c in df.columns]
            if present:
                rows.append((present, ylabel, colors))
        if not rows:
            return None

        rc = _compose_rc(P)
        with _mpl.rc_context(rc):
            fig, axes = _plt.subplots(
                len(rows), 1, figsize=(inch_w, inch_h), sharex=True,
            )
            if len(rows) == 1:
                axes = [axes]

            # Heel-strike markers (L + R)
            hs_L = res.left_stride.hs_indices
            hs_R = res.right_stride.hs_indices

            for ax, (cols, ylabel, colors) in zip(axes, rows):
                for col, c in zip(cols, colors):
                    ax.plot(t, df[col].to_numpy(dtype=float),
                            color=c, linewidth=P.stroke_pt * 1.2, label=col)
                # HS markers
                for idx in hs_L:
                    if 0 <= idx < len(t):
                        ax.axvline(t[idx], color="#3B82C4",
                                   linestyle=":", linewidth=P.grid_pt * 1.5, alpha=0.5)
                for idx in hs_R:
                    if 0 <= idx < len(t):
                        ax.axvline(t[idx], color="#D35454",
                                   linestyle=":", linewidth=P.grid_pt * 1.5, alpha=0.5)
                ax.set_ylabel(ylabel)
                ax.grid(True, linewidth=P.grid_pt, color=P.grid_color, alpha=0.5)
                ax.legend(loc="upper right", frameon=False,
                          fontsize=P.legend_pt, ncol=len(cols))
            axes[-1].set_xlabel("Time (s)  ·  dotted lines = heel-strikes (blue L · red R)")
            if req.title:
                fig.suptitle(req.title, fontsize=P.title_pt, y=0.998)
            fig.tight_layout(pad=0.4)
            fig.set_size_inches(inch_w, inch_h)
            return _emit(fig, req.format, dpi_val, preset_name=P.name)

    # ── kinematics_ensemble · GCP-normalized joint angle, detect available joints
    if req.template == "kinematics_ensemble":
        if df is None:
            return None
        from backend.services.compute_engine import resample_column as _resample
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        from backend.services.publication_engine import JOURNAL_PRESETS as _JP, _compose_rc, _emit

        # Detect joints in priority order (hip/knee/ankle for MoCap; shank pitch for H-Walker)
        _JOINT_SPECS = [
            ("Hip",       "Hip flex (°)",   [("L_Hip_Flex",   "R_Hip_Flex"),   ("L_HipFlexion",  "R_HipFlexion")]),
            ("Knee",      "Knee flex (°)",  [("L_Knee_Flex",  "R_Knee_Flex"),  ("L_KneeFlexion", "R_KneeFlexion")]),
            ("Ankle",     "Ankle (°)",      [("L_Ankle_Flex", "R_Ankle_Flex"), ("L_AnkleFlexion","R_AnkleFlexion"), ("L_AnkleDorsi","R_AnkleDorsi")]),
            ("Shank IMU", "Pitch (°)",      [("L_Pitch",      "R_Pitch")]),
        ]
        ls, rs = res.left_stride, res.right_stride
        panels = []
        for jname, ylabel, cands in _JOINT_SPECS:
            l_col = next((lc for lc, rc in cands if lc in df.columns), None)
            r_col = next((rc for lc, rc in cands if rc in df.columns), None)
            if l_col or r_col:
                panels.append((jname, ylabel, l_col, r_col))

        if not panels:
            return None

        P = _JP[req.preset]
        dpi_val = req.dpi or P.dpi
        if req.variant == "col1":
            w_mm, h_mm = P.col1
        elif req.variant == "onehalf" and P.onehalf:
            w_mm, h_mm = P.onehalf
        else:
            w_mm, h_mm = P.col2
        n_panels = len(panels)
        inch_w = w_mm / 25.4
        inch_h = (h_mm / 25.4) * (0.6 + 0.55 * n_panels)
        x = _np.linspace(0, 100, 101)

        rc = _compose_rc(P)
        with _mpl.rc_context(rc):
            fig, axes = _plt.subplots(n_panels, 1, figsize=(inch_w, inch_h), sharex=True)
            if n_panels == 1:
                axes = [axes]
            for ax, (jname, ylabel, l_col, r_col) in zip(axes, panels):
                if inc_L and l_col:
                    lr = _resample(df, l_col, ls.hs_indices, ls.valid_mask)
                    if lr is not None:
                        m, s = lr
                        ax.fill_between(x, m - s, m + s, color="#3B82C4", alpha=0.18, linewidth=0)
                        ax.plot(x, m, color="#1E5F9E", linewidth=P.stroke_pt * 1.8, label="L")
                if inc_R and r_col:
                    rr = _resample(df, r_col, rs.hs_indices, rs.valid_mask)
                    if rr is not None:
                        m, s = rr
                        ax.fill_between(x, m - s, m + s, color="#D35454", alpha=0.18, linewidth=0)
                        ax.plot(x, m, color="#9E3838", linewidth=P.stroke_pt * 1.8, label="R")
                ax.set_ylabel(ylabel)
                ax.set_title(jname, fontsize=P.title_pt, loc="left", pad=2)
                ax.grid(True, linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)
                ax.legend(loc="best", frameon=False, fontsize=P.legend_pt, ncol=2)
            axes[-1].set_xlabel("Gait cycle (%)")
            if req.title:
                fig.suptitle(req.title, fontsize=P.title_pt)
            fig.tight_layout(pad=0.35)
            fig.set_size_inches(inch_w, inch_h)
            return _emit(fig, req.format, dpi_val, preset_name=P.name)

    # ── spatiotemporal_bar · cadence / stride length / symmetry, 3 panels
    if req.template == "spatiotemporal_bar":
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        from backend.services.publication_engine import JOURNAL_PRESETS as _JP, _compose_rc, _emit

        ls, rs = res.left_stride, res.right_stride
        P = _JP[req.preset]
        dpi_val = req.dpi or P.dpi
        if req.variant == "col1":
            w_mm, h_mm = P.col1
        elif req.variant == "onehalf" and P.onehalf:
            w_mm, h_mm = P.onehalf
        else:
            w_mm, h_mm = P.col2
        inch_w, inch_h = w_mm / 25.4, h_mm / 25.4
        bar_w = 0.5

        rc = _compose_rc(P)
        with _mpl.rc_context(rc):
            fig, (ax1, ax2, ax3) = _plt.subplots(1, 3, figsize=(inch_w, inch_h))

            # Cadence
            cad_x, cad_y, cad_c = [], [], []
            if inc_L and ls.cadence > 0:
                cad_x.append("L"); cad_y.append(ls.cadence); cad_c.append("#1E5F9E")
            if inc_R and rs.cadence > 0:
                cad_x.append("R"); cad_y.append(rs.cadence); cad_c.append("#9E3838")
            if cad_y:
                ax1.bar(cad_x, cad_y, color=cad_c, width=bar_w)
            ax1.set_ylabel("Cadence (steps/min)")
            ax1.set_title("Cadence", fontsize=P.title_pt)
            ax1.grid(axis="y", linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)

            # Stride length ± SD
            sl_x, sl_y, sl_e, sl_c = [], [], [], []
            if inc_L and ls.stride_length_mean > 0:
                sl_x.append("L"); sl_y.append(ls.stride_length_mean)
                sl_e.append(ls.stride_length_std); sl_c.append("#1E5F9E")
            if inc_R and rs.stride_length_mean > 0:
                sl_x.append("R"); sl_y.append(rs.stride_length_mean)
                sl_e.append(rs.stride_length_std); sl_c.append("#9E3838")
            if sl_y:
                ax2.bar(sl_x, sl_y, color=sl_c, width=bar_w,
                        yerr=sl_e if any(e > 0 for e in sl_e) else None,
                        capsize=3, error_kw={"linewidth": P.stroke_pt, "ecolor": "#6B7280"})
            ax2.set_ylabel("Stride length (m)")
            ax2.set_title("Stride Length", fontsize=P.title_pt)
            ax2.grid(axis="y", linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)

            # Symmetry indices — red if >5 %
            sym_pairs = [
                ("T",  res.stride_time_symmetry,   "Stride time"),
                ("L",  res.stride_length_symmetry, "Stride length"),
                ("St", res.stance_symmetry,         "Stance %"),
                ("F",  res.force_symmetry,          "Force RMSE"),
            ]
            sym_x = [p[0] for p in sym_pairs if p[1] >= 0]
            sym_y = [max(p[1], 0.0) for p in sym_pairs if p[1] >= 0]
            sym_c = ["#D35454" if v > 5.0 else "#F09708" for v in sym_y]
            if sym_y:
                ax3.bar(sym_x, sym_y, color=sym_c, width=0.5)
                ax3.axhline(5.0, color="#6B7280", linestyle="--",
                            linewidth=P.grid_pt * 1.5, label="5% threshold")
                ax3.legend(frameon=False, fontsize=P.legend_pt)
            ax3.set_ylabel("Asymmetry (%)")
            ax3.set_title("Symmetry", fontsize=P.title_pt)
            ax3.grid(axis="y", linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)

            if req.title:
                fig.suptitle(req.title, fontsize=P.title_pt)
            fig.tight_layout(pad=0.35)
            fig.set_size_inches(inch_w, inch_h)
            return _emit(fig, req.format, dpi_val, preset_name=P.name)

    # ── force_tracking · per-stride RMSE convergence (ILC)
    if req.template == "force_tracking":
        lft = res.left_force_tracking
        rft = res.right_force_tracking
        has_l = inc_L and len(lft.rmse_per_stride) > 0
        has_r = inc_R and len(rft.rmse_per_stride) > 0
        if not has_l and not has_r:
            return None
        if has_l:
            n_l = len(lft.rmse_per_stride)
            traces.append(Trace(kind="line", name=f"L  (mean {lft.rmse:.2f} N)",
                                x=list(range(1, n_l + 1)),
                                y=list(lft.rmse_per_stride),
                                color="#3B82C4", width=1.8))
            if lft.rmse > 0:
                traces.append(Trace(kind="line", name="",
                                    x=[1, n_l], y=[lft.rmse, lft.rmse],
                                    color="#7FB5E4", width=1.0, dash=True))
        if has_r:
            n_r = len(rft.rmse_per_stride)
            traces.append(Trace(kind="line", name=f"R  (mean {rft.rmse:.2f} N)",
                                x=list(range(1, n_r + 1)),
                                y=list(rft.rmse_per_stride),
                                color="#D35454", width=1.8))
            if rft.rmse > 0:
                traces.append(Trace(kind="line", name="",
                                    x=[1, n_r], y=[rft.rmse, rft.rmse],
                                    color="#E89B9B", width=1.0, dash=True))
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Stride #", y_label="Force RMSE (N)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    # ── mos_trajectory · XCoM / MoS time series (requires XCoM columns in CSV)
    if req.template == "mos_trajectory":
        if df is None:
            return None
        _MOS_CANDIDATES = [
            ("XCoM_AP", "XCoM AP (m)"),
            ("MoS_AP",  "MoS AP (m)"),
            ("XCoM_ML", "XCoM ML (m)"),
            ("MoS_ML",  "MoS ML (m)"),
            ("CoM_AP",  "CoM AP (m)"),
            ("BOS_ant", "BOS anterior (m)"),
        ]
        available_mos = [(col, lbl) for col, lbl in _MOS_CANDIDATES if col in df.columns]
        if not available_mos:
            return None
        fs = res.sample_rate
        n = min(len(df), int(fs * 30))
        t = list(_np.arange(n) / fs)
        _mos_colors = ["#00FFB2", "#A78BFA", "#F09708", "#7FB5E4"]
        for i, (col, lbl) in enumerate(available_mos[:4]):
            traces.append(Trace(kind="line", name=lbl,
                                x=t, y=df[col].iloc[:n].tolist(),
                                color=_mos_colors[i % len(_mos_colors)], width=1.6))
        return render_from_traces(
            traces, title=req.title or "",
            x_label="Time (s)", y_label="Distance (m)",
            preset=req.preset, variant=req.variant, format=req.format,
            dpi=req.dpi, colorblind_safe=req.colorblind_safe, legend=True,
        )

    if req.template == "symmetry_radar":
        # Multi-axis symmetry polar chart — rendered as polar bars for
        # journal compatibility (vector-friendly).
        import matplotlib as _mpl
        import matplotlib.pyplot as _plt
        from backend.services.publication_engine import JOURNAL_PRESETS, _compose_rc, _emit
        P = JOURNAL_PRESETS[req.preset]
        if req.variant == "col1":
            w_mm, h_mm = P.col1
        elif req.variant == "onehalf" and P.onehalf:
            w_mm, h_mm = P.onehalf
        else:
            w_mm, h_mm = P.col2
        dpi_val = req.dpi or P.dpi
        inch_w, inch_h = w_mm / 25.4, h_mm / 25.4

        axes = [
            ("Stride T", res.stride_time_symmetry),
            ("Stride L", res.stride_length_symmetry if res.stride_length_symmetry >= 0 else 0),
            ("Stance %", res.stance_symmetry if res.stance_symmetry >= 0 else 0),
            ("Force RMSE", res.force_symmetry),
        ]
        # Add peak-GRF asymmetry if available
        lfp = res.left_force_profile; rfp = res.right_force_profile
        if lfp.mean is not None and rfp.mean is not None:
            from backend.services.compute_engine import _asym_idx
            axes.append(("Peak GRF",
                         _asym_idx(float(_np.max(lfp.mean)), float(_np.max(rfp.mean)))))
        # Close the polygon
        labels = [a[0] for a in axes]
        values = [max(float(a[1]), 0.0) for a in axes]
        angles = _np.linspace(0, 2 * _np.pi, len(values), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        rc = _compose_rc(P)
        with _mpl.rc_context(rc):
            fig, ax = _plt.subplots(figsize=(inch_w, inch_h),
                                    subplot_kw={"projection": "polar"})
            ax.plot(angles, values, color="#F09708", linewidth=P.stroke_pt * 1.4)
            ax.fill(angles, values, color="#F09708", alpha=0.18)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_rlabel_position(30)
            ax.grid(True, linewidth=P.grid_pt, color=P.grid_color, alpha=0.6)
            max_v = max(values) if values else 10
            ax.set_ylim(0, max(max_v * 1.15, 5))
            if req.title:
                ax.set_title(req.title, fontsize=P.title_pt, pad=18)
            fig.set_size_inches(inch_w, inch_h)
            return _emit(fig, req.format, dpi_val, preset_name=P.name)

    return None


@router.post("/render")
def render_endpoint(req: RenderRequest):
    if req.template not in GRAPH_SPECS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown template '{req.template}'. Known: {sorted(GRAPH_SPECS.keys())}",
        )
    if req.preset not in JOURNAL_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown preset '{req.preset}'. Known: {sorted(JOURNAL_PRESETS.keys())}",
        )
    # Phase 1 · multi-dataset overlay first (datasets[] ≥ 2)
    try:
        multi = _render_multi_dataset(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"multi-dataset render failed: {exc}") from exc

    # Real-data single-dataset path
    real = None
    if multi is None:
        # If datasets[] has exactly one entry, treat its id as dataset_id
        if req.datasets and not req.dataset_id:
            req = req.copy(update={"dataset_id": req.datasets[0].id})
        try:
            real = _render_real_data(req)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"real-data render failed: {exc}") from exc

    if multi is not None:
        data, mime = multi
    elif real is not None:
        data, mime = real
    else:
        try:
            data, mime = render(
                template=req.template,
                preset=req.preset,
                variant=req.variant,
                format=req.format,
                dpi=req.dpi,
                stride_avg=req.stride_avg,
                colorblind_safe=req.colorblind_safe,
                keep_palette=req.keep_palette,
                title_override=req.title,
            )
        except Exception as exc:  # noqa: BLE001 — surface full error to client
            raise HTTPException(status_code=500, detail=f"render failed: {exc}") from exc

    P = JOURNAL_PRESETS[req.preset]
    fname = _suggest_filename(req, P)
    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


class BundleRequest(BaseModel):
    preset: str = "ieee"
    variant: Literal["col1", "col2", "onehalf"] = "col2"
    format: Literal["svg", "pdf", "eps", "png", "tiff"] = "svg"
    dpi: Optional[int] = None
    cells: list[dict]  # [{id, template, stride_avg?, preset?}]
    include_readme: bool = True


@router.post("/bundle")
def bundle_endpoint(req: BundleRequest):
    if req.preset not in JOURNAL_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{req.preset}'")
    P = JOURNAL_PRESETS[req.preset]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, cell in enumerate(req.cells):
            tpl = cell.get("template") or cell.get("graph")
            if not tpl or tpl not in GRAPH_SPECS:
                continue
            cell_preset = cell.get("preset") or req.preset
            cell_variant = cell.get("variant") or req.variant
            stride_avg = bool(cell.get("stride_avg", False))
            ds_id = cell.get("dataset_id") or cell.get("dataset_id")
            datasets_raw = cell.get("datasets") or []
            try:
                real: tuple[bytes, str] | None = None
                if ds_id or len(datasets_raw) >= 2:
                    cell_req = RenderRequest(
                        template=tpl, preset=cell_preset, variant=cell_variant,
                        format=req.format, dpi=req.dpi, stride_avg=stride_avg,
                        dataset_id=ds_id,
                        datasets=[DatasetSeries(**d) for d in datasets_raw],
                        title=cell.get("title") or "",
                    )
                    try:
                        if len(datasets_raw) >= 2:
                            real = _render_multi_dataset(cell_req)
                        if real is None and ds_id:
                            real = _render_real_data(cell_req)
                    except Exception:
                        pass
                if real is not None:
                    data, _ = real
                else:
                    data, _ = render(
                        template=tpl, preset=cell_preset, variant=cell_variant,
                        format=req.format, dpi=req.dpi, stride_avg=stride_avg,
                    )
            except Exception as exc:  # noqa: BLE001
                zf.writestr(
                    f"ERRORS/{cell.get('id', i)}.txt",
                    f"render failed: {exc}\n",
                )
                continue
            name = f"{cell.get('id', f'cell{i}')}_{tpl}.{req.format}"
            zf.writestr(name, data)

        if req.include_readme:
            readme = _bundle_readme(req, P)
            zf.writestr("README.txt", readme)

    buf.seek(0)
    dim = int(P.col2[0] if req.variant == "col2" else P.col1[0])
    fname = f"hwalker_bundle_{req.preset}_{dim}mm.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


def _bundle_readme(req: BundleRequest, P) -> str:
    col = "col2" if req.variant == "col2" else ("col1" if req.variant == "col1" else "onehalf")
    w_mm = P.col2[0] if col == "col2" else (P.col1[0] if col == "col1" else (P.onehalf[0] if P.onehalf else P.col2[0]))
    h_mm = P.col2[1] if col == "col2" else (P.col1[1] if col == "col1" else (P.onehalf[1] if P.onehalf else P.col2[1]))
    return (
        f"H-Walker CORE · publication bundle\n"
        f"====================================\n"
        f"Journal preset : {P.full} ({P.name})\n"
        f"Width variant  : {col} = {w_mm} × {h_mm} mm\n"
        f"Format         : {req.format.upper()}\n"
        f"DPI            : {req.dpi or P.dpi}\n"
        f"Font           : {P.font} ({P.body_pt}pt body, {P.axis_pt}pt axis, {P.legend_pt}pt legend)\n"
        f"Stroke         : {P.stroke_pt}pt\n"
        f"Grid           : {P.grid_pt}pt, {P.grid_color}\n"
        f"Colorblind     : {'yes' if P.colorblind_safe else 'no'}\n"
        f"Accepted fmts  : {', '.join(P.formats)}\n"
        f"\n"
        f"Files are named <cell_id>_<template>.{req.format}.\n"
        f"Any render errors are recorded under ERRORS/.\n"
        f"\n"
        f"Journal notes  : {P.notes}\n"
    )


# ============================================================
# Multi-panel composition — Fig 1(a)(b)(c)(d)
# ============================================================

class PanelCell(BaseModel):
    template: str
    dataset_id: Optional[str] = None
    datasets: list[dict] = []         # overlay form — [{id,label,color}]
    stride_avg: bool = False
    title: Optional[str] = None       # panel caption (below or beside)


class MultiPanelRequest(BaseModel):
    panels: list[PanelCell]            # 2–4 panels
    preset: str = "ieee"
    variant: Literal["col1", "col2", "onehalf"] = "col2"
    format: Literal["svg", "pdf", "png"] = "pdf"
    layout: Literal["auto", "1x2", "2x1", "2x2", "1x3", "3x1"] = "auto"
    dpi: Optional[int] = None


_LAYOUT_MAP = {
    2: {"auto": (1, 2), "1x2": (1, 2), "2x1": (2, 1)},
    3: {"auto": (1, 3), "1x3": (1, 3), "3x1": (3, 1)},
    4: {"auto": (2, 2), "2x2": (2, 2), "1x2": (2, 2)},
}


def _render_panel_svg(panel: PanelCell, preset: str, variant: str, dpi: Optional[int]) -> bytes:
    """Render one panel to SVG bytes via the existing /render code path."""
    req = RenderRequest(
        template=panel.template,
        preset=preset,
        variant=variant,
        format="svg",
        dpi=dpi,
        stride_avg=panel.stride_avg,
        dataset_id=panel.dataset_id,
        datasets=[DatasetSeries(**d) for d in panel.datasets] if panel.datasets else [],
        title=panel.title or "",
    )
    multi = _render_multi_dataset(req) if req.datasets else None
    if multi:
        return multi[0]
    # Fall through to single-dataset or mockup path
    from backend.routers.graphs import _render_real_data
    real = _render_real_data(req)
    if real:
        return real[0]
    data, _ = render(
        template=req.template, preset=req.preset, variant=req.variant, format="svg",
        dpi=req.dpi, stride_avg=req.stride_avg,
    )
    return data


@router.post("/multi_panel")
def multi_panel(req: MultiPanelRequest):
    """Compose 2–4 panel SVGs into a single multi-panel figure with
    (a)(b)(c)(d) labels, correctly sized to the journal preset.

    Uses matplotlib's figure grid as the composition canvas and embeds
    each panel via `imshow` of its rasterized bitmap (kept hi-dpi) OR
    via `inset_axes` for vector-preserving SVG. PDF output is vector.
    """
    n = len(req.panels)
    if n < 2 or n > 4:
        raise HTTPException(status_code=400, detail=f"multi_panel needs 2-4 panels (got {n})")
    if req.preset not in JOURNAL_PRESETS:
        raise HTTPException(status_code=400, detail=f"unknown preset '{req.preset}'")
    P = JOURNAL_PRESETS[req.preset]
    rows, cols = _LAYOUT_MAP[n].get(req.layout, _LAYOUT_MAP[n]["auto"])

    # Render each panel once (SVG strings kept for vector quality).
    panel_svgs: list[bytes] = []
    for p in req.panels:
        try:
            panel_svgs.append(_render_panel_svg(p, req.preset, req.variant, req.dpi))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"panel {p.template} failed: {exc}") from exc

    # Compose via matplotlib — each panel is a subplot, its rendered PNG
    # is read back as the axis content. This preserves pixel quality at
    # the target DPI and yields a proper multi-panel PDF/SVG/PNG.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from PIL import Image

    # Figure canvas sized for the journal variant (each panel is a fraction)
    w_mm, h_mm = (P.col2 if req.variant == "col2" else
                  (P.col1 if req.variant == "col1" else (P.onehalf or P.col2)))
    # 다중 panel 은 col2 폭 유지, 행 개수만큼 세로 확장
    inch_w, inch_h = w_mm / 25.4, (h_mm * rows) / 25.4 * 0.95

    with mpl.rc_context({
        "font.family": [P.font] + P.font_fallback,
        "font.size": P.body_pt,
        "figure.facecolor": P.bg,
        "savefig.facecolor": P.bg,
    }):
        fig, axes = plt.subplots(rows, cols, figsize=(inch_w, inch_h))
        axes_flat = list(axes.flat) if hasattr(axes, 'flat') else [axes]

        LABELS = "abcdefgh"
        for i, (ax, panel, svg_bytes) in enumerate(zip(axes_flat, req.panels, panel_svgs)):
            # Convert panel SVG → PNG @ 2× DPI for crisp embedding
            tmp_svg = Path(tempfile.mkstemp(suffix=".svg")[1])
            tmp_png = Path(tempfile.mkstemp(suffix=".png")[1])
            tmp_svg.write_bytes(svg_bytes)
            try:
                # Try cairosvg first (high fidelity)
                import cairosvg
                cairosvg.svg2png(
                    bytestring=svg_bytes,
                    write_to=str(tmp_png),
                    output_width=int(inch_w / cols * 300),
                    output_height=int((inch_h / rows) * 300),
                )
            except Exception:
                # Fall back to rsvg-convert if available on the system
                try:
                    subprocess.run(
                        ["rsvg-convert", "-a", "-w", str(int(inch_w / cols * 300)),
                         str(tmp_svg), "-o", str(tmp_png)],
                        check=True, capture_output=True, timeout=10,
                    )
                except Exception:
                    # Final fallback: blank panel with error text
                    ax.text(0.5, 0.5, f"(panel {i+1}: SVG→PNG failed)",
                            ha='center', va='center', transform=ax.transAxes,
                            color='red', fontsize=P.legend_pt)
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values(): sp.set_visible(False)
                    continue
            img = Image.open(tmp_png)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            # (a)/(b)/(c)/(d) label, top-left
            ax.text(-0.04, 1.02, f"({LABELS[i]})", transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=P.title_pt, fontweight='bold')
            if panel.title:
                ax.set_title(panel.title, fontsize=P.axis_pt, pad=4)
            try:
                tmp_svg.unlink(); tmp_png.unlink()
            except OSError:
                pass

        fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)

        # Emit in requested format
        buf = io.BytesIO()
        try:
            if req.format == "pdf":
                fig.savefig(buf, format="pdf", dpi=req.dpi or P.dpi, pad_inches=0.05)
                mime = "application/pdf"
            elif req.format == "png":
                fig.savefig(buf, format="png", dpi=req.dpi or P.dpi, pad_inches=0.05)
                mime = "image/png"
            else:
                fig.savefig(buf, format="svg", dpi=req.dpi or P.dpi, pad_inches=0.05)
                mime = "image/svg+xml"
        finally:
            plt.close(fig)

    fname = f"hwalker_panel_{len(req.panels)}panel_{req.preset}.{req.format}"
    return Response(
        content=buf.getvalue(), media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ============================================================
# LLM Codegen — Haiku가 실데이터 보고 matplotlib 코드 생성
# ============================================================

_CODEGEN_SYSTEM = """\
You are a biomechanics matplotlib expert for the H-Walker rehabilitation robot.

You receive a pandas DataFrame (`df`) and a user request. Generate ONLY a matplotlib drawing block.

Rules:
- `df`, `np`, `mpl`, `plt`, `fig`, `ax` are already available in scope
- Do NOT call plt.subplots(), plt.figure(), plt.show(), plt.savefig(), plt.close()
- Draw directly on `ax` (or create subplots with fig.subplots() if multi-panel is needed)
- Use: L side = #3B82C4, R side = #D35454, accent = #F09708
- Des (desired) columns → dashed; Act (actual) → solid
- If L/R not specified, plot both
- Add legend, axis labels. No title (journal convention).
- Output ONLY Python code, no explanation, no markdown fences."""


class CodegenRequest(BaseModel):
    dataset_id: str
    prompt: str
    preset: str = "ieee"
    variant: Literal["col1", "col2", "onehalf"] = "col2"
    format: Literal["svg", "png", "pdf"] = "svg"
    dpi: Optional[int] = None


@router.post("/codegen")
async def codegen_endpoint(req: CodegenRequest):
    """LLM-generated custom matplotlib figure from real CSV data.

    Haiku sees the DataFrame schema + user prompt, generates matplotlib code,
    which runs on the backend against real data and returns the figure.
    """
    from backend.routers.datasets import get_path
    import pandas as _pd
    import numpy as _np
    import matplotlib as _mpl
    import matplotlib.pyplot as _plt
    from backend.services.publication_engine import JOURNAL_PRESETS, _compose_rc, _emit
    from backend.services.config import ANTHROPIC_API_KEY

    csv_path = get_path(req.dataset_id)
    if not csv_path:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        df = _pd.read_csv(csv_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV load failed: {exc}")

    # Build compact schema for Haiku
    col_lines = []
    for col in df.columns[:50]:
        s = df[col].dropna().head(3).tolist()
        col_lines.append(f"  {col}: {df[col].dtype} sample={s}")
    schema = "\n".join(col_lines)

    user_msg = (
        f"DataFrame ({len(df)} rows, {len(df.columns)} cols):\n{schema}\n\n"
        f"Request: {req.prompt}"
    )

    # Call Haiku
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=_CODEGEN_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        code_text: str = resp.content[0].text.strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    # Strip markdown fences if present
    if code_text.startswith("```"):
        lines = code_text.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        code_text = "\n".join(lines[start:end])

    # Journal sizing
    P = JOURNAL_PRESETS.get(req.preset, JOURNAL_PRESETS["ieee"])
    if req.variant == "col1":
        w_mm, h_mm = P.col1
    elif req.variant == "onehalf" and P.onehalf:
        w_mm, h_mm = P.onehalf
    else:
        w_mm, h_mm = P.col2
    dpi_val = req.dpi or P.dpi
    inch_w, inch_h = w_mm / 25.4, h_mm / 25.4

    rc = _compose_rc(P)
    safe_builtins = {k: getattr(__builtins__, k, None) for k in (
        "range", "len", "list", "dict", "str", "int", "float", "bool",
        "enumerate", "zip", "min", "max", "sum", "abs", "round",
        "isinstance", "hasattr", "getattr", "print", "type",
        "ValueError", "TypeError", "KeyError", "IndexError",
    ) if getattr(__builtins__, k, None) is not None}

    with _mpl.rc_context(rc):
        _fig, _ax = _plt.subplots(figsize=(inch_w, inch_h))
        ns = {
            "df": df, "np": _np, "mpl": _mpl, "plt": _plt,
            "fig": _fig, "ax": _ax,
            "__builtins__": safe_builtins,
        }
        try:
            exec(compile(code_text, "<llm_codegen>", "exec"), ns)
        except Exception as exec_err:
            _plt.close(_fig)
            raise HTTPException(
                status_code=422,
                detail=f"Generated code failed: {exec_err}\n\nCode:\n{code_text[:800]}",
            )
        _fig.set_size_inches(inch_w, inch_h)
        try:
            data, mime = _emit(_fig, req.format, dpi_val)
        finally:
            _plt.close(_fig)

    fname = f"hwalker_codegen.{req.format}"
    return Response(content=data, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})
