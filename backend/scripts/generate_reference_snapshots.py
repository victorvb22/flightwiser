"""Régénère les snapshots bundlés avec le déploiement
(data/reference/aircraft_database_trimmed.parquet, airports_trimmed.parquet)
— à relancer quand la source amont (OpenSky, OurAirports) change et qu'on
veut rafraîchir la base embarquée. Lit les CSV bruts en cache local
(les télécharge si absents, via la même fonction que services/
aircraft_database.download_if_missing), applique exactement le même
traitement que load_aircraft_database()/load_airports() (colonnes utiles
uniquement, filtrage héliports/fermés pour les aéroports), écrit le
résultat déjà nettoyé en parquet — services/*.py le lit tel quel, sans
retraitement, à l'exécution.

Usage : python scripts/generate_reference_snapshots.py
"""

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.aircraft_database import AIRCRAFT_DB_CACHE, AIRCRAFT_DB_URL, BUNDLED_PATH as AIRCRAFT_BUNDLED_PATH, download_if_missing  # noqa: E402
from services.airports import AIRPORTS_CACHE, AIRPORTS_URL, BUNDLED_PATH as AIRPORTS_BUNDLED_PATH  # noqa: E402


def generate_aircraft_snapshot() -> None:
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
    aircraft = aircraft.drop_duplicates(subset=["icao24"])
    aircraft.to_parquet(AIRCRAFT_BUNDLED_PATH, index=False)
    print(f"{AIRCRAFT_BUNDLED_PATH} : {len(aircraft)} lignes")


def generate_airports_snapshot() -> None:
    path = download_if_missing(AIRPORTS_URL, AIRPORTS_CACHE)
    airports = pd.read_csv(
        path,
        usecols=["ident", "icao_code", "iso_country", "scheduled_service", "latitude_deg", "longitude_deg", "type", "municipality", "name"],
    )
    airports = airports[~airports["type"].isin(["closed", "heliport", "balloonport"])]
    airports["icao"] = airports["icao_code"].fillna(airports["ident"])
    airports = airports.dropna(subset=["latitude_deg", "longitude_deg", "icao"]).reset_index(drop=True)
    airports.to_parquet(AIRPORTS_BUNDLED_PATH, index=False)
    print(f"{AIRPORTS_BUNDLED_PATH} : {len(airports)} lignes")


if __name__ == "__main__":
    generate_aircraft_snapshot()
    generate_airports_snapshot()
