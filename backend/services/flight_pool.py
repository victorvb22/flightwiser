"""Pool of real flights replacing the live OpenSky calls the deployed
backend can no longer make (blocked at the network level on
Render/Railway/Cloudflare Workers — cf. the README's Limitations section).

Each entry contains ONLY what a live OpenSky call would have returned
(identifiant, statut, trajectoire — cf. scripts/collect_opensky_pool.py):
no precomputed scores. pipeline.py computes anomalie/ecart/directness at the
moment a flight is drawn from here, exactly as it would have for a fresh
flight — a model fix therefore applies immediately to the whole pool, with
no new collection needed. Assembled by scripts/build_flight_pool.py
(data/reference/flight_pool.jsonl, bundled with the deployment like
aircraft_database_trimmed.parquet/airports_trimmed.parquet).

Loaded into memory once, on first access (same pattern as
services/aircraft_database.py): a few MB, not something to re-read on every
request.
"""

import json
import random
from pathlib import Path
from typing import Any

from services.aircraft_database import get_registration

BACKEND_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BACKEND_DIR.parent / "data" / "reference"
POOL_PATH = REFERENCE_DIR / "flight_pool.jsonl"

_pool: list[dict[str, Any]] | None = None


def _load_pool() -> list[dict[str, Any]]:
    if not POOL_PATH.exists():
        return []
    entries = []
    with open(POOL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _get_pool() -> list[dict[str, Any]]:
    global _pool
    if _pool is None:
        _pool = _load_pool()
    return _pool


def find_by_identifiant(identifiant: str) -> dict[str, Any] | None:
    """Exact (case/whitespace-insensitive) lookup by displayed identifier
    (callsign), registration, or icao24 — the same three ways of naming a
    flight the old live-fetch resolution used to accept, before it was
    replaced by the pool (cf. pipeline.py's own docstring). Registration
    isn't stored in the pool (derivable from the icao24, as
    pipeline._flight_metadata already does) — looked up here on the fly
    rather than risking going stale if the aircraft database is
    regenerated."""
    normalized = identifiant.strip().upper()
    for entry in _get_pool():
        icao24 = entry["_icao24"]
        if entry["identifiant"].strip().upper() == normalized:
            return entry
        if icao24.strip().upper() == normalized:
            return entry
        registration = get_registration(icao24)
        if registration and registration.strip().upper() == normalized:
            return entry
    return None


def draw_random(want_on_ground: bool | None, excluded_icao24: set[str]) -> dict[str, Any] | None:
    """Draws a random flight among those never served yet (excluded_icao24,
    cf. services/pool_tracking.py) — "once picked, a flight can't be picked
    again" (a product decision). Filtered on status if want_on_ground is
    given (True -> landed, False -> airborne). None if the pool is exhausted
    for this filter."""
    statut_filter = None
    if want_on_ground is True:
        statut_filter = "atterri"
    elif want_on_ground is False:
        statut_filter = "en_vol"

    candidates = [
        entry
        for entry in _get_pool()
        if entry["_icao24"] not in excluded_icao24 and (statut_filter is None or entry["statut"] == statut_filter)
    ]
    if not candidates:
        return None
    return random.choice(candidates)
