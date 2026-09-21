"""ONLY access point to Supabase (brief section 6) — no other module should
touch the DB directly.

Direct REST access (PostgREST) rather than an SDK — consistent with
services/opensky_client.py, and plenty for a get/upsert by key on a single
table. The cache is an optimization, not a correctness dependency: any
network/HTTP error is logged and treated as a cache miss (read) or ignored
(write) rather than failing a search whose result has already been
computed.

Schema: db/schema.sql (run once in the Supabase SQL Editor).
"""

import logging
from datetime import datetime, timezone
from typing import Any

import requests

from config import settings
from db.models_orm import FlightCacheRow

logger = logging.getLogger("cache")

_TABLE_URL = f"{settings.supabase_url}/rest/v1/flights_cache"
_HEADERS = {
    "apikey": settings.supabase_key,
    "Authorization": f"Bearer {settings.supabase_key}",
    "Content-Type": "application/json",
}

_CACHED_FIELDS = ("statut", "trajectoire", "ecart_trajectoire", "anomalie", "directness", "kpi_bonus")
_PAGE_SIZE = 1000  # same PostgREST cap as services/historical.py — paginate with Range rather than trust limit alone.


def build_cache_id(icao24: str) -> str:
    """icao24 + today's UTC date — a flight searched live always corresponds
    to "today" for that aircraft, never derived from the trajectory itself
    (which would already require an OpenSky call, exactly what the cache is
    meant to avoid)."""
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{icao24}_{today}"


def get_cached_flight(cache_id: str) -> dict[str, Any] | None:
    """Returns the relevant `flights_cache` fields for this key, or None if
    absent or on a network/HTTP error (silent cache miss)."""
    try:
        response = requests.get(
            _TABLE_URL,
            params={"id": f"eq.{cache_id}", "select": ",".join(_CACHED_FIELDS)},
            headers=_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Could not read cache for %s, treated as a miss", cache_id, exc_info=True)
        return None

    rows = response.json()
    return rows[0] if rows else None


def get_all_cached_flights() -> list[dict[str, Any]]:
    """Every flight ever searched (and therefore cached), across all aircraft
    and dates — the source for a search history that survives the session/
    browser, unlike purely client-side state. Most recent first. Empty list
    on a network/HTTP error (same "silent miss" policy as get_cached_flight)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    try:
        while True:
            response = requests.get(
                _TABLE_URL,
                params={"select": f"id,{','.join(_CACHED_FIELDS)},calcule_le", "order": "calcule_le.desc"},
                headers={**_HEADERS, "Range-Unit": "items", "Range": f"{offset}-{offset + _PAGE_SIZE - 1}"},
                timeout=30,
            )
            response.raise_for_status()
            page = response.json()
            rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
    except requests.RequestException:
        logger.warning("Could not read the full history, returning an empty list", exc_info=True)
        return []
    return rows


def touch_flight(cache_id: str) -> None:
    """Bumps calcule_le to now, without touching anything else -- for a
    cache HIT (pipeline.py: the flight was already computed earlier today),
    where store_flight is deliberately never called again (no point paying
    for a full re-write of an unchanged trajectory/scores payload just to
    serve the same result a second time). Without this, calcule_le stays
    frozen at whenever the row was first computed, and the Aggregate view
    shows it under a column literally labelled "Searched" (VueAgregee.tsx)
    -- a flight searched again hours or days later would silently sit
    wherever its first computation left it in that recency sort, instead of
    where a second search actually belongs: found and fixed from a real
    report (searched today, not visible near the top of the history).
    Same best-effort policy as store_flight: failure logged, never
    surfaced."""
    try:
        response = requests.patch(
            _TABLE_URL,
            params={"id": f"eq.{cache_id}"},
            json={"calcule_le": datetime.now(timezone.utc).isoformat()},
            headers=_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Could not touch cache for %s", cache_id, exc_info=True)


def store_flight(cache_id: str, result: dict[str, Any]) -> None:
    """Upserts the computed result for this flight. Failure logged, never
    surfaced: the user already has their response, don't fail it because the
    cache write failed."""
    row: FlightCacheRow = {
        "id": cache_id,
        "statut": result["statut"],
        "trajectoire": result["trajectoire"],
        "ecart_trajectoire": result["ecart_trajectoire"],
        "anomalie": result["anomalie"],
        "directness": result["directness"],
        "kpi_bonus": result["kpi_bonus"],
        "source": result["source"],
        "calcule_le": datetime.now(timezone.utc).isoformat(),
    }
    try:
        response = requests.post(
            _TABLE_URL,
            params={"on_conflict": "id"},
            json=row,
            headers={**_HEADERS, "Prefer": "resolution=merge-duplicates"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Could not write cache for %s", cache_id, exc_info=True)
