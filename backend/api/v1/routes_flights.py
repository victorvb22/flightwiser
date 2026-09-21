"""Endpoints only (brief section 6) — no business logic here, everything
goes through pipeline.py and services/cache.py.
"""

from fastapi import APIRouter, HTTPException

import pipeline
from services import cache, flight_pool, pool_tracking
from services.aircraft_category import categorize
from services.aircraft_database import get_registration, get_typecode

router = APIRouter(prefix="/api/v1/flights", tags=["flights"])


@router.get("/example")
def get_example_identifiant():
    """A real, not-yet-served pool identifier — backs the Search page's
    placeholder (RechercheVol.tsx), so "e.g. ..." always names a flight
    that actually exists in the pool rather than a made-up one. Registered
    ahead of /{identifiant} for the same reason /history is. Deliberately
    lighter than a real search: only draws from the pool (services.
    flight_pool.draw_random, the same "never already served" pick the
    Random flight button uses) and returns the bare identifier — no model
    scoring, no cache write, no pool_served write, so fetching an example
    never itself counts as serving one."""
    served = pool_tracking.get_served_icao24s()
    entry = flight_pool.draw_random(None, served)
    if entry is None:
        raise HTTPException(status_code=404, detail="No example available right now")
    return {"identifiant": entry["identifiant"]}


@router.get("/history")
def get_search_history():
    """Every flight ever searched (and therefore cached) — backs the
    Aggregate view's chart/table with a history that survives a reload or a
    different browser, not just this one session's local state. Registered
    ahead of /{identifiant} so "history" is never swallowed as an identifier.
    """
    rows = cache.get_all_cached_flights()
    history = []
    for row in rows:
        icao24, _, _date = row["id"].partition("_")
        typecode = get_typecode(icao24)
        history.append(
            {
                "identifiant": get_registration(icao24) or icao24,
                "typecode": typecode,
                # Same category the anomaly model actually scored this flight
                # against (services.aircraft_category.categorize) — not a
                # separate guess, so it explains a surprising score rather
                # than risking disagreeing with it. None when the typecode
                # itself is unknown, or unidentified in the category table.
                "categorie": categorize(typecode) if typecode is not None else None,
                "statut": row["statut"],
                "trajectoire": row["trajectoire"],
                "ecart_trajectoire": row["ecart_trajectoire"],
                "anomalie": row["anomalie"],
                "directness": row["directness"],
                "calcule_le": row["calcule_le"],
            }
        )
    return history


@router.get("/{identifiant}")
def get_flight(identifiant: str):
    """Search by registration or callsign — response contract, brief section 8."""
    result = pipeline.get_or_compute_flight(identifiant)
    if result is None:
        raise HTTPException(status_code=404, detail="Flight not found")
    return result


@router.get("/random/search")
def get_random_flight(statut: str | None = None):
    """Random flight drawn from the pool (cf. pipeline.py, the deployed
    backend can't call OpenSky live) — no identifier needed, handy for a demo.
    `statut` optionally filters to "en_vol" or "atterri"."""
    want_on_ground = {"atterri": True, "en_vol": False}.get(statut) if statut else None
    result = pipeline.get_random_flight(want_on_ground)
    if result is None:
        raise HTTPException(status_code=404, detail="No flight available right now")
    return result
