"""Formes typées des tables Supabase/Postgres (brief section 9). Schéma réel
dans db/schema.sql (à coller une fois dans le SQL Editor Supabase).

Pas de SQLAlchemy : services/cache.py n'a besoin que d'un get/upsert par clé
sur une seule table, couvert directement par des requêtes REST (PostgREST) —
un ORM n'apporterait rien ici. Ces TypedDict servent uniquement à typer les
échanges dans cache.py.
"""

from typing import Any, TypedDict


class FlightCacheRow(TypedDict):
    id: str  # icao24 + date UTC, ex. "39de4e_2026-09-16"
    statut: str
    trajectoire: list[dict[str, Any]]
    ecart_trajectoire: dict[str, Any] | None
    anomalie: dict[str, Any] | None
    directness: dict[str, Any] | None
    kpi_bonus: Any | None
    source: str
    calcule_le: str
