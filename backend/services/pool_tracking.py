"""Tracks which pool flights (services/flight_pool.py) have already been
served — persistent via Supabase, like services/cache.py, because Render's
disk is ephemeral between redeploys: without this table, a redeploy would
make every pool flight look "never served" again, and a random draw could
serve an already-seen flight a second time.

Direct REST access (PostgREST), same policy as cache.py: a network/HTTP
error is logged and treated as "nothing has been served yet" (read) or
ignored (write) — must never fail a search/draw whose result is already
determined.

Schema: db/schema.sql.
"""

import logging
from datetime import datetime, timezone

import requests

from config import settings

logger = logging.getLogger("pool_tracking")

_TABLE_URL = f"{settings.supabase_url}/rest/v1/pool_served"
_HEADERS = {
    "apikey": settings.supabase_key,
    "Authorization": f"Bearer {settings.supabase_key}",
    "Content-Type": "application/json",
}
_PAGE_SIZE = 1000  # same PostgREST cap as services/cache.py — paginate with Range rather than trust limit alone.


def get_served_icao24s() -> set[str]:
    """Every icao24 already served from the pool — empty set on a network/
    HTTP error (better to risk an occasional duplicate than block every
    random draw)."""
    icao24s: set[str] = set()
    offset = 0
    try:
        while True:
            response = requests.get(
                _TABLE_URL,
                params={"select": "icao24"},
                headers={**_HEADERS, "Range-Unit": "items", "Range": f"{offset}-{offset + _PAGE_SIZE - 1}"},
                timeout=15,
            )
            response.raise_for_status()
            page = response.json()
            icao24s.update(row["icao24"] for row in page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
    except requests.RequestException:
        logger.warning("Could not read pool_served, returning an empty set", exc_info=True)
        return set()
    return icao24s


def mark_served(icao24: str) -> None:
    """Marks this flight as served. Failure logged, never surfaced: the user
    already has their response, don't fail it because this write failed (at
    worst, a possible duplicate later on)."""
    try:
        response = requests.post(
            _TABLE_URL,
            params={"on_conflict": "icao24"},
            json={"icao24": icao24, "servi_le": datetime.now(timezone.utc).isoformat()},
            headers={**_HEADERS, "Prefer": "resolution=merge-duplicates"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Could not write pool_served for %s", icao24, exc_info=True)
