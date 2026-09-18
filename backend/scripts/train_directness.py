"""Entraînement offline du score de trajet direct (secondaire, cf.
models/_directness_features.py pour le rationnel) : calcule, par catégorie
d'appareil (services.aircraft_category), les points de percentile
empiriques du ratio distance grand-cercle / distance parcourue sur les vols
atterris de l'ensemble du dataset disponible — mêmes deux sources,
dédoublonnées de la même façon (cf. scripts._training_data), que
scripts/train_anomalie.py (brief : "utiliser le dataset entier") :
  - data/processed/flights_clean.parquet
  - flights_historical_features.parquet (scripts/extract_historical_for_training.py)

Pas de split train/validation ici, contrairement à train_anomalie.py : une
seule feature scalaire, calibrée par ses propres percentiles empiriques,
n'a pas de moyenne/écart-type à sur-ajuster sur un petit échantillon — les
breakpoints de percentile sont calculés directement sur tout l'ensemble
disponible par catégorie, comme le fait déjà train_anomalie.py pour ses
propres breakpoints (calculés sur train, pas sur validation).

Usage : python scripts/train_directness.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Nécessaire pour "python scripts/train_directness.py" (sys.path[0] est
# alors scripts/, pas backend/) ; sans effet si déjà lancé via -m.
sys.path.insert(0, str(BACKEND_DIR))

from models._directness_features import extract_route_directness  # noqa: E402
from scripts._training_data import load_deduplicated_landed_flights  # noqa: E402
from services.aircraft_category import CATEGORIES, categorize  # noqa: E402

DATA_DIR = BACKEND_DIR.parent / "data"
EXTERNAL_DATA_DIR = Path(os.environ.get("IMPORT_DATA_DIR", r"D:\ML_data\flightwiser"))
INPUT_PARQUETS = [
    DATA_DIR / "processed" / "flights_clean.parquet",
    EXTERNAL_DATA_DIR / "processed" / "flights_historical_features.parquet",
]
ARTIFACT_PATH = BACKEND_DIR / "models" / "artifacts" / "directness_params.json"

N_PERCENTILE_BREAKPOINTS = 101  # percentiles 0..100 inclus
N_LOWEST_TO_INSPECT = 10
MIN_FLIGHTS_PER_CATEGORY = 30


def main():
    landed = load_deduplicated_landed_flights(INPUT_PARQUETS)

    records = []
    n_insufficient = 0
    for _, row in landed.iterrows():
        waypoints = json.loads(row["waypoints"])
        ratio = extract_route_directness(waypoints)
        if ratio is None:
            n_insufficient += 1
            continue
        records.append({"icao24": row["icao24"], "ratio": ratio, "categorie": categorize(row["typecode"])})

    print(f"Vols avec ratio exploitable : {len(records)} ({n_insufficient} exclus, distance grand-cercle trop courte ou données insuffisantes)")

    feat_df = pd.DataFrame(records)
    print("\nRépartition par catégorie :")
    print(feat_df["categorie"].value_counts())

    categories = {}
    for name in CATEGORIES:
        subset = feat_df[feat_df["categorie"] == name]
        print(f"\n=== Catégorie : {name} ({len(subset)} vols) ===")
        if len(subset) < MIN_FLIGHTS_PER_CATEGORY:
            print(f"ATTENTION : {len(subset)} < {MIN_FLIGHTS_PER_CATEGORY} vols, distribution peu fiable pour cette catégorie.")
        print(subset["ratio"].describe())

        n_lowest = min(N_LOWEST_TO_INSPECT, len(subset))
        print(f"{n_lowest} vols les moins directs (à inspecter manuellement) :")
        print(subset.nsmallest(n_lowest, "ratio")[["icao24", "ratio"]].to_string(index=False))

        percentiles = np.linspace(0, 100, N_PERCENTILE_BREAKPOINTS)
        breakpoints = np.percentile(subset["ratio"], percentiles)
        categories[name] = {"percentile_breakpoints": breakpoints.tolist(), "n": len(subset)}

    artifact = {"categories": categories}
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"\nParamètres écrits dans {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
