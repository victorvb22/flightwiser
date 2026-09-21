"""OpenSky's public aircraft database (icao24 -> registration/typecode/
icaoaircrafttype), used by the batch preprocessing
(scripts/preprocess_historical.py), the pool collection
(scripts/collect_opensky_pool.py), and building the categorization table
(scripts/build_aircraft_categories.py).

Loaded into memory once, on first access (module-level singleton), the same
way models/anomalie.py loads its artifact once — this is a ~94 MB file, not
something to re-read on every request.

BUNDLED_PATH (a snapshot already trimmed to the 4 useful columns, committed
to the repo) is tried first, before any download: serving a request depends
on it directly (get_typecode/get_registration), so waiting on a 94 MB
fetch — or failing on one — in the middle of an HTTP response isn't
acceptable. Observed in prod (Render): connection timeout to
opensky-network.org, 500 on the first call touching the database. The live
download stays as the fallback for a dev machine that hasn't generated the
snapshot yet (see its comment in .gitignore)."""

from pathlib import Path

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()

BACKEND_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BACKEND_DIR.parent / "data" / "reference"

AIRCRAFT_DB_URL = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
AIRCRAFT_DB_CACHE = REFERENCE_DIR / "aircraft_database.csv"
BUNDLED_PATH = REFERENCE_DIR / "aircraft_database_trimmed.parquet"

_aircraft_db: pd.DataFrame | None = None


def download_if_missing(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        print(f"Downloading {url} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return cache_path


def load_aircraft_database() -> pd.DataFrame:
    if BUNDLED_PATH.exists():
        return pd.read_parquet(BUNDLED_PATH)
    path = download_if_missing(AIRCRAFT_DB_URL, AIRCRAFT_DB_CACHE)
    aircraft = pd.read_csv(
        path,
        usecols=["icao24", "registration", "typecode", "icaoaircrafttype"],
        dtype=str,
        low_memory=False,
    )
    aircraft["icao24"] = aircraft["icao24"].str.strip().str.lower()
    aircraft["typecode"] = aircraft["typecode"].str.strip()
    aircraft["registration"] = aircraft["registration"].str.strip()
    aircraft["icaoaircrafttype"] = aircraft["icaoaircrafttype"].str.strip()
    aircraft = aircraft[aircraft["icao24"].fillna("") != ""]
    return aircraft.drop_duplicates(subset=["icao24"])


def _get_aircraft_db() -> pd.DataFrame:
    global _aircraft_db
    if _aircraft_db is None:
        _aircraft_db = load_aircraft_database()
    return _aircraft_db


def resolve_icao24_by_registration(registration: str) -> str | None:
    """Exact (case/whitespace-insensitive) lookup of an icao24 from a
    registration, e.g. "F-GKXA". None if no match."""
    normalized = registration.strip().upper()
    db = _get_aircraft_db()
    matches = db[db["registration"].str.upper() == normalized]
    if matches.empty:
        return None
    return matches.iloc[0]["icao24"]


def get_typecode(icao24: str) -> str | None:
    """Known typecode for this icao24, or None if absent from the database."""
    db = _get_aircraft_db()
    matches = db[db["icao24"] == icao24.strip().lower()]
    if matches.empty or pd.isna(matches.iloc[0]["typecode"]) or matches.iloc[0]["typecode"] == "":
        return None
    return matches.iloc[0]["typecode"]


def get_registration(icao24: str) -> str | None:
    """Known registration for this icao24, or None if absent from the
    database (mirrors get_typecode — used to show a readable identifier
    instead of the raw icao24, e.g. in the Aggregate view's search history)."""
    db = _get_aircraft_db()
    matches = db[db["icao24"] == icao24.strip().lower()]
    if matches.empty or pd.isna(matches.iloc[0]["registration"]) or matches.iloc[0]["registration"] == "":
        return None
    return matches.iloc[0]["registration"]
