"""SEUL point d'accès à Supabase (brief section 6) — aucun autre module ne doit
toucher la DB directement.

Accès REST direct (PostgREST) plutôt qu'un SDK — cohérent avec
services/opensky_client.py, et largement suffisant pour un get/upsert par clé
sur une seule table. Le cache est une optimisation, pas une dépendance de
correction : toute erreur réseau/HTTP est journalisée et traitée comme un
cache miss (lecture) ou ignorée (écriture) plutôt que de faire échouer une
recherche dont le résultat a déjà été calculé.

Schéma : db/schema.sql (à exécuter une fois dans le SQL Editor Supabase).
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
_PAGE_SIZE = 1000  # même plafond PostgREST que services/historical.py — paginer avec Range plutôt que se fier à limit seul.


def build_cache_id(icao24: str) -> str:
    """icao24 + date UTC du jour — un vol recherché en direct correspond
    toujours à "aujourd'hui" pour cet appareil, jamais dérivé de la
    trajectoire elle-même (qui nécessiterait déjà un appel OpenSky, ce que
    le cache doit précisément permettre d'éviter)."""
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{icao24}_{today}"


def get_cached_flight(cache_id: str) -> dict[str, Any] | None:
    """Retourne les champs pertinents de `flights_cache` pour cette clé, ou
    None si absente ou en cas d'erreur réseau/HTTP (cache miss silencieux)."""
    try:
        response = requests.get(
            _TABLE_URL,
            params={"id": f"eq.{cache_id}", "select": ",".join(_CACHED_FIELDS)},
            headers=_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("Lecture cache impossible pour %s, traité comme un miss", cache_id, exc_info=True)
        return None

    rows = response.json()
    return rows[0] if rows else None


def get_all_cached_flights() -> list[dict[str, Any]]:
    """Tous les vols jamais recherchés (donc mis en cache), tous appareils et
    toutes dates confondus — la source d'un historique de recherche qui
    survit à la session/au navigateur, contrairement à un état purement
    client. Le plus récent en premier. Liste vide en cas d'erreur réseau/HTTP
    (même politique "miss silencieux" que get_cached_flight)."""
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
        logger.warning("Lecture de l'historique complet impossible, liste vide retournée", exc_info=True)
        return []
    return rows


def store_flight(cache_id: str, result: dict[str, Any]) -> None:
    """Upsert du résultat calculé pour ce vol. Échec journalisé, jamais
    remonté : l'utilisateur a déjà sa réponse, ne pas la faire échouer
    parce que l'écriture cache a échoué."""
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
        logger.warning("Écriture cache impossible pour %s", cache_id, exc_info=True)
