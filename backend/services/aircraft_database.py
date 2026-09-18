"""Base aéronefs publique OpenSky (icao24 -> immatriculation/typecode),
utilisée à la fois par le prétraitement batch (scripts/preprocess_historical.py)
et par la résolution d'identifiant en direct (pipeline.py).

Chargée une fois en mémoire au premier accès (singleton module), comme
models/anomalie.py charge son artefact une fois — c'est un fichier de ~94 Mo,
pas quelque chose à relire à chaque requête.
"""

from pathlib import Path

import pandas as pd
import requests
import truststore

truststore.inject_into_ssl()

BACKEND_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BACKEND_DIR.parent / "data" / "reference"

AIRCRAFT_DB_URL = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"
AIRCRAFT_DB_CACHE = REFERENCE_DIR / "aircraft_database.csv"

_aircraft_db: pd.DataFrame | None = None


def download_if_missing(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        print(f"Téléchargement de {url} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        cache_path.write_bytes(response.content)
    return cache_path


def load_aircraft_database() -> pd.DataFrame:
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
    """Recherche exacte (insensible à la casse/espaces) d'un icao24 à partir
    d'une immatriculation, ex. "F-GKXA". None si aucune correspondance."""
    normalized = registration.strip().upper()
    db = _get_aircraft_db()
    matches = db[db["registration"].str.upper() == normalized]
    if matches.empty:
        return None
    return matches.iloc[0]["icao24"]


def get_typecode(icao24: str) -> str | None:
    """Typecode connu pour cet icao24, ou None si absent de la base."""
    db = _get_aircraft_db()
    matches = db[db["icao24"] == icao24.strip().lower()]
    if matches.empty or pd.isna(matches.iloc[0]["typecode"]) or matches.iloc[0]["typecode"] == "":
        return None
    return matches.iloc[0]["typecode"]


def get_registration(icao24: str) -> str | None:
    """Immatriculation connue pour cet icao24, ou None si absente de la base
    (miroir de get_typecode — utilisé pour afficher un identifiant lisible
    plutôt que l'icao24 brut, ex. Vue agrégée par date)."""
    db = _get_aircraft_db()
    matches = db[db["icao24"] == icao24.strip().lower()]
    if matches.empty or pd.isna(matches.iloc[0]["registration"]) or matches.iloc[0]["registration"] == "":
        return None
    return matches.iloc[0]["registration"]


def is_helicopter(icao24: str) -> bool:
    """True si `icaoaircrafttype` (désignateur ICAO Doc 8643 : premier
    caractère = catégorie — H = hélicoptère/giravion) commence par "H" pour
    cet icao24. False si l'icao24 est absent de la base ou que le champ est
    vide — pas d'hypothèse par défaut, cohérent avec get_typecode/
    get_registration qui renvoient None plutôt que de deviner."""
    db = _get_aircraft_db()
    matches = db[db["icao24"] == icao24.strip().lower()]
    if matches.empty or pd.isna(matches.iloc[0]["icaoaircrafttype"]):
        return False
    return matches.iloc[0]["icaoaircrafttype"].startswith("H")
