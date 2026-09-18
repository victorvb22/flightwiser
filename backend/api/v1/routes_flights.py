"""Endpoints uniquement (brief section 6) — aucune logique métier ici, tout
passe par pipeline.py et services/cache.py.
"""

from fastapi import APIRouter, HTTPException

import pipeline
from models._anomalie_features import categorize
from services import cache
from services.aircraft_database import get_registration, get_typecode

router = APIRouter(prefix="/api/v1/flights", tags=["flights"])


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
                # against (models._anomalie_features.categorize) — not a
                # separate guess, so it explains a surprising score rather
                # than risking disagreeing with it. None when the typecode
                # itself is unknown (no category is computed without one,
                # cf. pipeline.py — nothing to categorize).
                "categorie": categorize(typecode, icao24) if typecode is not None else None,
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
    """Random flight from the current live ADS-B feed — no identifier needed, handy for a demo.
    `statut` optionally filters to "en_vol" or "atterri"."""
    want_on_ground = {"atterri": True, "en_vol": False}.get(statut) if statut else None
    result = pipeline.get_random_flight(want_on_ground)
    if result is None:
        raise HTTPException(status_code=404, detail="No flight available right now")
    return result
