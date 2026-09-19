"""Airport reference data (OurAirports) — loading and nearest-neighbor
lookup by geographic proximity. Shared between
scripts/preprocess_historical.py and pipeline.py (origin/destination
resolution for display, brief section 8) — avoids a second copy of the same
lookup logic.

Loaded into memory once, on first access (same pattern as
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
# Already-filtered/processed snapshot, committed to the repo — same reason
# as aircraft_database.BUNDLED_PATH: find_nearest_airport_info runs on every
# search result, a live network fetch (or timeout) has no place on that
# path.
BUNDLED_PATH = REFERENCE_DIR / "airports_trimmed.parquet"

DEFAULT_MAX_KM = 5.0

_airports: pd.DataFrame | None = None


def load_airports() -> pd.DataFrame:
    if BUNDLED_PATH.exists():
        return pd.read_parquet(BUNDLED_PATH)
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
    # Excludes heliports/closed airfields/points not relevant to a commercial flight.
    airports = airports[~airports["type"].isin(["closed", "heliport", "balloonport"])]
    # Prefers the real ICAO code (icao_code); falls back to "ident" otherwise (often
    # identical for significant airports, but not guaranteed for small airfields).
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
    """Nearest airport (ICAO code + city + name), or None if none within
    `max_km`. `airports` defaults to the full worldwide reference data
    (lazily loaded); a caller can pass an already-filtered subset instead
    (e.g. preprocess_historical.py)."""
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
    """ICAO code only — a lightweight mirror of find_nearest_airport_info for
    callers that don't need the city (scripts/*.py, which never display
    it)."""
    info = find_nearest_airport_info(lat, lon, airports, max_km)
    return info["icao"] if info else None
