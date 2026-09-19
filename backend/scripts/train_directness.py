"""Offline training of the route-directness score (secondary, cf.
models/_directness_features.py for the rationale): computes, per aircraft
category (services.aircraft_category), the empirical percentile points of
the great-circle-distance / distance-flown ratio over landed flights from
the whole available dataset — the same two sources, deduplicated the same
way (cf. scripts._training_data), as scripts/train_anomalie.py (brief: "use
the entire dataset"):
  - data/processed/flights_clean.parquet
  - flights_historical_features.parquet (scripts/extract_historical_for_training.py)

No train/validation split here, unlike train_anomalie.py: a single scalar
feature, calibrated by its own empirical percentiles, has no mean/std to
overfit on a small sample — the percentile breakpoints are computed
directly on the whole available set per category, the same way
train_anomalie.py already computes its own breakpoints (on train, not
validation).

Usage: python scripts/train_directness.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Needed for "python scripts/train_directness.py" (sys.path[0] is then
# scripts/, not backend/); no effect if already run via -m.
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

N_PERCENTILE_BREAKPOINTS = 101  # percentiles 0..100 inclusive
N_LOWEST_TO_INSPECT = 10
MIN_FLIGHTS_PER_CATEGORY = 30


def main():
    landed = load_deduplicated_landed_flights(INPUT_PARQUETS)

    records = []
    n_insufficient = 0
    # itertuples(), not iterrows(): cf. the same comment in
    # scripts/train_anomalie.py (pandas/pyarrow crashes on iterrows() here).
    for row in landed.itertuples(index=False):
        waypoints = json.loads(row.waypoints)
        ratio = extract_route_directness(waypoints)
        if ratio is None:
            n_insufficient += 1
            continue
        records.append({"icao24": row.icao24, "ratio": ratio, "categorie": categorize(row.typecode)})

    print(f"Flights with a usable ratio: {len(records)} ({n_insufficient} excluded, great-circle distance too short or insufficient data)")

    feat_df = pd.DataFrame(records)
    print("\nBreakdown by category:")
    print(feat_df["categorie"].value_counts())

    categories = {}
    for name in CATEGORIES:
        subset = feat_df[feat_df["categorie"] == name]
        print(f"\n=== Category: {name} ({len(subset)} flights) ===")
        if len(subset) < MIN_FLIGHTS_PER_CATEGORY:
            print(f"WARNING: {len(subset)} < {MIN_FLIGHTS_PER_CATEGORY} flights, unreliable distribution for this category.")
        print(subset["ratio"].describe())

        n_lowest = min(N_LOWEST_TO_INSPECT, len(subset))
        print(f"{n_lowest} least direct flights (for manual inspection):")
        print(subset.nsmallest(n_lowest, "ratio")[["icao24", "ratio"]].to_string(index=False))

        percentiles = np.linspace(0, 100, N_PERCENTILE_BREAKPOINTS)
        breakpoints = np.percentile(subset["ratio"], percentiles)
        categories[name] = {"percentile_breakpoints": breakpoints.tolist(), "n": len(subset)}

    artifact = {"categories": categories}
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"\nParameters written to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
