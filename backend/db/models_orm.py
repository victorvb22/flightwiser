"""Typed shapes for the Supabase/Postgres tables (brief section 9). Real
schema lives in db/schema.sql (paste it once into the Supabase SQL Editor).

No SQLAlchemy: services/cache.py only needs a get/upsert by key on a single
table, covered directly by REST requests (PostgREST) — an ORM wouldn't add
anything here. These TypedDicts only exist to type the exchanges in
cache.py.
"""

from typing import Any, TypedDict


class FlightCacheRow(TypedDict):
    id: str  # icao24 + UTC date, e.g. "39de4e_2026-09-16"
    statut: str
    trajectoire: list[dict[str, Any]]
    ecart_trajectoire: dict[str, Any] | None
    anomalie: dict[str, Any] | None
    directness: dict[str, Any] | None
    kpi_bonus: Any | None
    source: str
    calcule_le: str
