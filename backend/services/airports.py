"""Référentiel aéroports (OurAirports) — chargement et résolution du plus
proche par proximité géographique. Partagé entre
scripts/preprocess_historical.py et pipeline.py (résolution origine/
destination pour l'affichage, brief section 8) — évite une 2e copie de la
même logique de recherche.

Chargé une fois en mémoire au premier accès (même motif que
services/aircraft_database.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from services.aircraft_database import download_if_missing
from services.trajectory_cleaning import haversine_km

BACKEND_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BACKEND_DIR.parent / "data" / "reference"

AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
AIRPORTS_CACHE = REFERENCE_DIR / "airports.csv"

DEFAULT_MAX_KM = 5.0

_airports: pd.DataFrame | None = None


def load_airports() -> pd.DataFrame:
    path = download_if_missing(AIRPORTS_URL, AIRPORTS_CACHE)
    airports = pd.read_csv(
        path,
        usecols=[
            "ident",
            "icao_code",
            "iso_country",
            "scheduled_service",
            "latitude_deg",
            "longitude_deg",
            "type",
            "municipality",
            "name",
        ],
    )
    # Exclut les héliports/aérodromes fermés/points non pertinents pour un vol commercial.
    airports = airports[~airports["type"].isin(["closed", "heliport", "balloonport"])]
    # Préfère le vrai code ICAO (icao_code) ; à défaut, retombe sur "ident" (souvent
    # identique pour les aéroports significatifs, mais pas garanti pour les petits terrains).
    airports["icao"] = airports["icao_code"].fillna(airports["ident"])
    return airports.dropna(subset=["latitude_deg", "longitude_deg", "icao"]).reset_index(drop=True)


def _get_airports() -> pd.DataFrame:
    global _airports
    if _airports is None:
        _airports = load_airports()
    return _airports


def find_nearest_airport_info(
    lat: float, lon: float, airports: pd.DataFrame | None = None, max_km: float = DEFAULT_MAX_KM
) -> dict[str, str | None] | None:
    """Aéroport le plus proche (code ICAO + ville + nom), ou None si aucun à
    moins de `max_km`. `airports` par défaut au référentiel mondial complet
    (chargé paresseusement) ; un appelant peut passer un sous-ensemble déjà
    filtré (ex. preprocess_historical.py)."""
    df = airports if airports is not None else _get_airports()
    distances = haversine_km(lat, lon, df["latitude_deg"].to_numpy(), df["longitude_deg"].to_numpy())
    idx = np.argmin(distances)
    if distances[idx] > max_km:
        return None
    row = df.iloc[idx]
    return {
        "icao": row["icao"],
        "ville": row["municipality"] if pd.notna(row["municipality"]) else None,
        "nom": row["name"] if pd.notna(row["name"]) else None,
    }


def find_nearest_airport(lat: float, lon: float, airports: pd.DataFrame | None = None, max_km: float = DEFAULT_MAX_KM) -> str | None:
    """Code ICAO seul — miroir léger de find_nearest_airport_info pour les
    appelants qui n'ont pas besoin de la ville (scripts/*.py, qui ne
    l'affichent jamais)."""
    info = find_nearest_airport_info(lat, lon, airports, max_km)
    return info["icao"] if info else None
