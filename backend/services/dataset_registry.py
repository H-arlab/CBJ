"""Tiny shared dataset-registry helpers.

Lives outside `backend/routers/datasets.py` so the registry can be
imported without dragging FastAPI's Form decorator (and therefore
the optional python-multipart dep) into the import graph. Anything
that just needs `path = get_path(ds_id)` should pull from here.

The router still owns mutation (upload/delete) and disk persistence;
this module only exposes the in-memory mirror plus the read helpers.
"""
from __future__ import annotations

from typing import Any


# Mutated by backend/routers/datasets.py on upload / delete / startup.
_REGISTRY: dict[str, dict[str, Any]] = {}
_HASH_INDEX: dict[str, str] = {}


def get_path(ds_id: str) -> str | None:
    return _REGISTRY.get(ds_id, {}).get("_path")


def get_meta(ds_id: str) -> dict[str, Any] | None:
    return _REGISTRY.get(ds_id)
