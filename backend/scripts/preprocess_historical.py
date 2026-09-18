"""Prétraitement du dataset historique brut (brief section 10).

Applique, dans l'ordre :
  1. dédupliquer les points consécutifs quasi identiques par vol
  2. recalculer la vitesse (absente du CSV brut)
  3. déduire l'aéroport de destination (proximité géographique)
  4. enrichir le type d'appareil (jointure sur icao24)

Entrée  : data/raw/flights_raw.csv (une ligne par vol, colonnes
          icao24, callsign, departure_airport, takeoff_time, landing_time,
          duration_min, n_waypoints, still_airborne, waypoints — waypoints
          étant une liste de tuples (timestamp, lat, lon, altitude, cap,
          vitesse_verticale, au_sol) sérialisée en texte).
Sortie  : data/processed/flights_clean.parquet (une ligne par vol ; waypoints
          sérialisé en JSON, avec vitesse ajoutée par point, plus les
          colonnes destination_airport, typecode, type_confidence).

Usage : python scripts/preprocess_historical.py
"""

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import truststore

truststore.inject_into_ssl()

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Nécessaire pour "python scripts/preprocess_historical.py" (sys.path[0] est
# alors scripts/, pas backend/) ; sans effet si déjà lancé via -m.
sys.path.insert(0, str(BACKEND_DIR))

from services.aircraft_database import load_aircraft_database  # noqa: E402
from services.airports import find_nearest_airport, load_airports  # noqa: E402
from services.trajectory_cleaning import (  # noqa: E402
    clip_altitude,
    clip_vertical_rate,
    compute_speeds,
    dedupe_waypoints,
    drop_missing_position,
    drop_position_teleports,
)

DATA_DIR = BACKEND_DIR.parent / "data"
RAW_CSV = DATA_DIR / "raw" / "flights_raw.csv"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_PARQUET = PROCESSED_DIR / "flights_clean.parquet"

DEFAULT_TYPECODE = "A320"

_BARE_NAN_RE = re.compile(r"\bnan\b")


def parse_waypoints(raw: str) -> list[tuple]:
    # Le CSV contient des NaN Python non quotés (repr(float('nan')) == "nan"),
    # qui ne sont pas un littéral valide pour ast.literal_eval — on les
    # convertit en None (valeur ADS-B manquante : altitude/cap/vitesse
    # verticale, ou lat+lon ensemble quand la position n'a pas été reçue).
    return ast.literal_eval(_BARE_NAN_RE.sub("None", raw))


def process_flight(row: pd.Series, airports: pd.DataFrame) -> dict:
    points = parse_waypoints(row["waypoints"])
    points = drop_missing_position(points)
    points = dedupe_waypoints(points)
    points = drop_position_teleports(points)
    points = clip_altitude(points)
    points = clip_vertical_rate(points)
    speeds = compute_speeds(points)

    waypoints_clean = [
        {
            "timestamp": p[0],
            "lat": p[1],
            "lon": p[2],
            "altitude": p[3],
            "cap": p[4],
            "vitesse_verticale": p[5],
            "au_sol": p[6],
            "vitesse": speeds[i],
        }
        for i, p in enumerate(points)
    ]

    destination_airport = None
    if not row["still_airborne"] and waypoints_clean:
        last = waypoints_clean[-1]
        destination_airport = find_nearest_airport(last["lat"], last["lon"], airports)

    return {
        "icao24": row["icao24"],
        "callsign": row["callsign"],
        "departure_airport": row["departure_airport"],
        "destination_airport": destination_airport,
        "takeoff_time": row["takeoff_time"],
        "landing_time": row["landing_time"],
        "duration_min": row["duration_min"],
        "still_airborne": row["still_airborne"],
        "n_waypoints": len(waypoints_clean),
        "waypoints": json.dumps(waypoints_clean),
    }


def main():
    print(f"Lecture de {RAW_CSV} ...")
    raw = pd.read_csv(RAW_CSV)
    print(f"{len(raw)} vols chargés.")

    airports = load_airports()
    print(f"{len(airports)} aéroports de référence chargés.")

    cleaned = [process_flight(row, airports) for _, row in raw.iterrows()]
    clean_df = pd.DataFrame(cleaned)

    aircraft_db = load_aircraft_database()
    clean_df["icao24"] = clean_df["icao24"].str.strip().str.lower()
    clean_df = clean_df.merge(aircraft_db, on="icao24", how="left")

    matched_mask = clean_df["typecode"].notna() & (clean_df["typecode"] != "")
    clean_df["type_confidence"] = np.where(matched_mask, "opensky_db", "defaut_faible_confiance")
    clean_df["typecode"] = clean_df["typecode"].where(matched_mask, DEFAULT_TYPECODE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(OUTPUT_PARQUET, index=False)

    n_total = len(clean_df)
    n_with_destination = clean_df["destination_airport"].notna().sum()
    n_typecode_matched = matched_mask.sum()
    print(f"Écrit {OUTPUT_PARQUET} ({n_total} vols).")
    print(f"Destination déduite : {n_with_destination}/{n_total} ({n_with_destination / n_total:.1%})")
    print(f"Typecode trouvé via OpenSky : {n_typecode_matched}/{n_total} ({n_typecode_matched / n_total:.1%})")


if __name__ == "__main__":
    main()
