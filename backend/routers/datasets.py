"""
/api/datasets/* — Phase 2 frontend contract (HANDOFF §2.1).

Wraps the existing tmp-file upload with a richer Dataset shape:
    {id, name, kind, rows, dur, hz, cols:[{name, unit, mapped, mappedManual}], active}

Phase 2H · persistent storage + SHA256 dedup:
  - Uploaded CSVs land in ~/.hw_graph/uploads/<hash>.csv instead of tmp/
  - On startup the registry is rebuilt from `~/.hw_graph/uploads/registry.json`,
    so re-opening the app resurrects previously uploaded datasets.
  - If the same content is dropped twice, we reuse the existing ds_id
    (return 200 with the old record) instead of creating a duplicate.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import Response

from backend.services.knowledge_loader import register_csv_columns


router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# ----- Persistent upload storage ---------------------------------------

_UPLOAD_DIR = Path(os.path.expanduser("~/.hw_graph/uploads"))
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_REGISTRY_PATH = _UPLOAD_DIR / "registry.json"
# In-memory mirror (rebuilt from disk on startup)
_REGISTRY: dict[str, dict[str, Any]] = {}
# SHA256 → ds_id for O(1) dedup on upload
_HASH_INDEX: dict[str, str] = {}


def _save_registry() -> None:
    """Persist the in-memory registry to disk (strip private _ keys only
    from network responses; keep them on disk for restart recovery)."""
    try:
        payload = {
            dsid: {k: v for k, v in d.items() if k != '_tmp_path_removed'}
            for dsid, d in _REGISTRY.items()
        }
        with open(_REGISTRY_PATH, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as exc:
        print(f"[datasets] registry save failed: {exc}")


def _load_registry() -> None:
    """Rebuild the in-memory registry from disk at startup. Entries whose
    underlying CSV no longer exists are dropped."""
    if not _REGISTRY_PATH.exists():
        return
    try:
        with open(_REGISTRY_PATH) as f:
            disk = json.load(f)
        for dsid, d in disk.items():
            p = d.get('_path')
            if p and os.path.isfile(p):
                _REGISTRY[dsid] = d
                h = d.get('_content_hash')
                if h:
                    _HASH_INDEX[h] = dsid
    except Exception as exc:
        print(f"[datasets] registry load failed: {exc}")


# Rebuild at import time (once per process)
_load_registry()


def _content_hash(data: bytes) -> str:
    """SHA256 of the file content — stable dedup key regardless of name."""
    return hashlib.sha256(data).hexdigest()[:32]


# Phase 2: filename auto-parsing regex list. Tries each pattern in order;
# first match wins. Named groups drive the Dataset.{group, subject_id,
# condition} fields. Keep these case-insensitive and tolerant of
# separators (_, -, .).
import re
_FILENAME_PATTERNS = [
    # explicit labels: subject=s01 · condition=pre · date=2024-05-01
    re.compile(r'(?i)(?:subj(?:ect)?[_-]?)?(?P<subject_id>s\d+)[_\-.]+'
               r'(?P<condition>pre|post|control|experimental|baseline|'
               r'treatment|treadmill|overground|\w{2,12})'
               r'(?:[_\-.]+(?P<date>\d{4}[_\-.]?\d{2}[_\-.]?\d{2}))?'),
    # condition first: pre_s01_… / control-subj02-…
    re.compile(r'(?i)^(?P<condition>pre|post|control|experimental|baseline|'
               r'treatment|fast|slow|natural|\w{3,10})'
               r'[_\-.]+(?:subj(?:ect)?[_-]?)?(?P<subject_id>s\d+|\d{1,3})'),
    # numeric only: 001_pre_…
    re.compile(r'(?i)^(?P<subject_id>\d{1,3})[_\-.]+(?P<condition>[a-z]{3,15})'),
    # trial_N style (keep existing recipe behavior, not really a group)
    re.compile(r'(?i)(?P<subject_id>trial[_-]?\d+)'),
]

_CONDITION_CANONICAL = {
    'pre': 'Pre', 'post': 'Post', 'control': 'Control',
    'experimental': 'Experimental', 'baseline': 'Baseline',
    'treatment': 'Treatment', 'treadmill': 'Treadmill',
    'overground': 'Overground', 'fast': 'Fast', 'slow': 'Slow',
    'natural': 'Natural',
}


def _parse_filename(fname: str) -> dict[str, str]:
    """Extract {subject_id, condition, group, date} from a filename.

    Falls back to empty dict for unrecognized patterns. The user can
    always override via PATCH /api/datasets/{id}/meta.
    """
    import os
    stem = os.path.splitext(os.path.basename(fname))[0]
    for pat in _FILENAME_PATTERNS:
        m = pat.search(stem)
        if not m:
            continue
        out: dict[str, str] = {}
        gd = m.groupdict()
        if gd.get('subject_id'):
            out['subject_id'] = gd['subject_id'].lower()
        if gd.get('condition'):
            cond = gd['condition'].lower()
            out['condition'] = _CONDITION_CANONICAL.get(cond, cond.title())
            # Group defaults to condition (user can override)
            out['group'] = out['condition']
        if gd.get('date'):
            out['date'] = gd['date']
        if out:
            return out
    return {}


_UNIT_HINTS = [
    ('force', 'N'), ('_n', 'N'),
    ('pitch', '°'), ('roll', '°'), ('yaw', '°'),
    ('vel', 'm/s'), ('pos', 'mm'), ('current', 'A'),
    ('time', 's'), ('t_', 's'),
    ('cop', 'mm'),
    ('gcp', '%'),
]
_KIND_HINTS = {
    'force': 'force', 'grf': 'force', 'cop': 'cop',
    'imu': 'imu', 'pitch': 'imu', 'gyro': 'imu', 'acc': 'imu',
    'emg': 'emg', 'trial': 'trials',
}
_ROLE_HINTS = [
    ('time', 'time'), ('t_', 'time'),
    ('l_force', 'L force'), ('l_grf', 'L force'),
    ('r_force', 'R force'), ('r_grf', 'R force'),
    ('shank', 'shank'), ('thigh', 'thigh'),
    ('trial', 'group'),
]


def _guess_unit(col: str) -> str:
    lc = col.lower()
    for hint, unit in _UNIT_HINTS:
        if hint in lc:
            return unit
    return '—'


def _guess_role(col: str) -> tuple[str, float]:
    lc = col.lower()
    for hint, role in _ROLE_HINTS:
        if hint in lc:
            return role, 0.85
    return '—', 0.0


def _guess_kind(cols: list[str]) -> str:
    joined = ' '.join(cols).lower()
    # Phase 0: if the CSV carries BOTH force and IMU-family signals, flag
    # it as 'mixed' so the canonical recipe set covers both sides.
    has_force = any(h in joined for h in ('force', 'grf'))
    has_imu = any(h in joined for h in ('pitch', 'gyro', 'acc', '_ax', '_ay', '_az',
                                         '_gx', '_gy', '_gz', 'roll', 'yaw', 'imu'))
    if has_force and has_imu:
        return 'mixed'
    for hint, kind in _KIND_HINTS.items():
        if hint in joined:
            return kind
    return 'force'


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=422, detail="filename missing")

    # Read the entire file upfront so we can hash for dedup
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="file is empty")

    chash = _content_hash(content)

    # Dedup — if the same bytes were uploaded before, reuse that dataset
    if chash in _HASH_INDEX:
        existing_id = _HASH_INDEX[chash]
        ds = _REGISTRY.get(existing_id)
        if ds and os.path.isfile(ds.get('_path', '')):
            return {k: v for k, v in ds.items() if not k.startswith('_')}

    # Persist to ~/.hw_graph/uploads/<hash>.csv
    save_path = _UPLOAD_DIR / f"{chash}.csv"
    try:
        with open(save_path, 'wb') as f:
            f.write(content)
        df = pd.read_csv(save_path, nrows=500)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV parse failed: {exc}") from exc

    col_names = [str(c).strip() for c in df.columns.tolist()]
    ds_id = 'ds_' + uuid.uuid4().hex[:8]
    hz = 100
    try:
        if 'time' in col_names[0].lower() or 't' == col_names[0].lower():
            dt = df.iloc[:, 0].diff().dropna().median()
            if dt and dt > 0:
                hz = int(round(1.0 / float(dt)))
    except Exception:
        pass
    kind = _guess_kind(col_names)

    # Phase 3 (multi-source experiment): also detect Robot / Loadcell /
    # Motion source category from filename + column signature. The
    # legacy `kind` (force/imu/cop/...) drives the recipe picker; the
    # new `source_kind` drives source-specific compute / alignment.
    from backend.services.source_kind import detect_source_kind
    detection = detect_source_kind(file.filename, col_names)

    rows = len(df) if len(df) < 500 else None
    if rows is None:
        try:
            with open(save_path) as fh:
                rows = sum(1 for _ in fh) - 1
        except Exception:
            rows = len(df)

    cols = []
    for name in col_names:
        role, conf = _guess_role(name)
        cols.append({
            'name': name,
            'unit': _guess_unit(name),
            'mapped': role,
            'inferred_role': role,
            'confidence': conf,
            'preview': df[name].head(5).astype(str).tolist(),
        })

    try:
        register_csv_columns(str(save_path), col_names)
    except Exception:
        pass

    # Phase 2: auto-parse filename for subject/condition/group
    parsed = _parse_filename(file.filename)

    ds = {
        'id': ds_id,
        'name': file.filename,
        'tag': kind,
        'kind': kind,
        'source_kind': detection.kind,
        'source_confidence': detection.confidence,
        'source_cues': detection.matched_cues,
        'rows': rows,
        'dur': f"{rows / hz:.1f}s" if hz else "—",
        'hz': f"{hz}Hz",
        'cols': cols,
        'active': False,
        'recipeState': {},
        'subject_id': parsed.get('subject_id', ''),
        'condition': parsed.get('condition', ''),
        'group': parsed.get('group', ''),
        'date': parsed.get('date', ''),
        '_path': str(save_path),
        '_content_hash': chash,
    }
    _REGISTRY[ds_id] = ds
    _HASH_INDEX[chash] = ds_id
    _save_registry()
    return {k: v for k, v in ds.items() if not k.startswith('_')}


@router.get("")
def list_datasets() -> list[dict[str, Any]]:
    return [{k: v for k, v in d.items() if not k.startswith('_')} for d in _REGISTRY.values()]


@router.get("/{ds_id}")
def get_dataset(ds_id: str) -> dict[str, Any]:
    if ds_id not in _REGISTRY:
        raise HTTPException(status_code=404, detail="dataset not found")
    d = _REGISTRY[ds_id]
    try:
        df = pd.read_csv(d['_path'], nrows=500)
        sample = df.head(500).to_dict(orient='records')
    except Exception:
        sample = []
    return {**{k: v for k, v in d.items() if not k.startswith('_')}, 'sample': sample}


@router.delete("/{ds_id}")
def delete_dataset(ds_id: str) -> Response:
    d = _REGISTRY.pop(ds_id, None)
    if d:
        # Remove hash-index entry
        h = d.get('_content_hash')
        if h and _HASH_INDEX.get(h) == ds_id:
            _HASH_INDEX.pop(h, None)
        # Remove disk file
        try:
            os.unlink(d['_path'])
        except Exception:
            pass
        # Invalidate analyzer cache
        try:
            from backend.routers.analyze import invalidate_cache
            invalidate_cache(ds_id)
        except Exception:
            pass
        _save_registry()
    return Response(status_code=204)


@router.patch("/{ds_id}/meta")
def update_meta(ds_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Phase 2 · Update dataset's group/subject/condition tags."""
    if ds_id not in _REGISTRY:
        raise HTTPException(status_code=404, detail="dataset not found")
    d = _REGISTRY[ds_id]
    for key in ('subject_id', 'condition', 'group', 'date'):
        if key in payload:
            d[key] = str(payload[key] or '')
    _save_registry()
    return {k: v for k, v in d.items() if not k.startswith('_')}


@router.post("/{ds_id}/mapping")
def save_mapping(ds_id: str, payload: dict[str, Any]) -> dict[str, int]:
    if ds_id not in _REGISTRY:
        raise HTTPException(status_code=404, detail="dataset not found")
    columns = payload.get('columns', {}) or {}
    d = _REGISTRY[ds_id]
    updated = 0
    for c in d['cols']:
        new_role = columns.get(c['name'])
        if new_role and new_role != c['mapped']:
            c['mapped'] = new_role
            c['mappedManual'] = True
            updated += 1
    return {'updated': updated}


def get_path(ds_id: str) -> str | None:
    return _REGISTRY.get(ds_id, {}).get('_path')
