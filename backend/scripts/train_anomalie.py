"""Offline training of the anomaly model (brief section 11): fits a diagonal
Gaussian per aircraft category (one per summary flight feature, cf.
models/_anomalie_features.py) on landed flights from the whole available
dataset, and persists the parameters to
models/artifacts/anomalie_params.json for the service (models/anomalie.py).

Two sources concatenated (brief: "use the entire dataset") rather than one:
  - data/processed/flights_clean.parquet (3132 flights, 2022-06-27)
  - data/processed/flights_historical_features.parquet, produced by
    scripts/extract_historical_for_training.py (Europe bbox) for the same
    day.

The two sources partially overlap (same day) — deduplicated via
scripts._training_data.load_deduplicated_landed_flights before merging
(otherwise a flight present in both would weigh twice in μ/σ), which keeps
the flights_historical_features.parquet version on a duplicate.

Categorization into avion_ligne / jet_affaire / petit_avion / helicoptere
(cf. models/_anomalie_features.categorize): a single Gaussian over the whole
dataset would systematically penalize minority groups, their speed/altitude/
climb-rate distributions being too different — helicopter isolated from
petit_avion in particular, its profile (hover flight, no classic climb/
cruise/descent phases) sitting outside a fixed-wing aircraft's distribution
even for a normal flight.

No real anomaly labels in this dataset (brief section 11): the ε threshold
is chosen, per category, as the 1st percentile of log-likelihood on a
validation split, then the lowest-scoring flights are printed for a manual
sanity check — a documented heuristic, not a supervised precision/recall
validation. The score exposed by the service is a percentile rank (0-1)
recomputed against this same training distribution, which stays comparable
from one category to another despite raw log-likelihoods that aren't
(different scales per category) — ε is therefore only used, as before, for
the manual inspection below, not for the served score.

Usage: python scripts/train_anomalie.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Needed for "python scripts/train_anomalie.py" (sys.path[0] is then
# scripts/, not backend/); a no-op if already run via -m.
sys.path.insert(0, str(BACKEND_DIR))

from models._anomalie_features import CATEGORIES, FEATURES, LOG1P_FEATURES, categorize, extract_features  # noqa: E402
from scripts._training_data import load_deduplicated_landed_flights  # noqa: E402

DATA_DIR = BACKEND_DIR.parent / "data"
# flights_historical_features.parquet lives outside the repo (cf. the script
# that generates it, scripts/extract_historical_for_training.py) — same
# environment variable as the raw_states/ cache, for consistency.
EXTERNAL_DATA_DIR = Path(os.environ.get("IMPORT_DATA_DIR", r"D:\ML_data\flightwiser"))
INPUT_PARQUETS = [
    DATA_DIR / "processed" / "flights_clean.parquet",
    EXTERNAL_DATA_DIR / "processed" / "flights_historical_features.parquet",
]
ARTIFACT_PATH = BACKEND_DIR / "models" / "artifacts" / "anomalie_params.json"

VALIDATION_FRACTION = 0.10
EPSILON_PERCENTILE = 1.0  # 1st percentile of validation log-likelihood
N_LOWEST_TO_INSPECT = 10
RANDOM_SEED = 0
N_PERCENTILE_BREAKPOINTS = 101  # percentiles 0..100 inclusive
# Below this many flights, a Gaussian (let alone a train/validation split)
# is no longer reliable — better to flag it loudly than to silently persist
# an under-trained artifact.
MIN_FLIGHTS_PER_CATEGORY = 30


def gaussian_log_pdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Log-density of a diagonal Gaussian, summed over features (axis=1)."""
    var = std**2
    return np.sum(-0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var), axis=1)


def fit_category(name: str, feat_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    print(f"\n=== Category: {name} ({len(feat_df)} flights) ===")
    if len(feat_df) < MIN_FLIGHTS_PER_CATEGORY:
        print(f"WARNING: {len(feat_df)} < {MIN_FLIGHTS_PER_CATEGORY} flights, unreliable Gaussian for this category.")

    print("Skew per feature (after transformation):")
    print(feat_df[FEATURES].skew())

    shuffled_idx = rng.permutation(len(feat_df))
    n_val = max(1, int(len(feat_df) * VALIDATION_FRACTION))
    val_idx, train_idx = shuffled_idx[:n_val], shuffled_idx[n_val:]
    train_df, val_df = feat_df.iloc[train_idx], feat_df.iloc[val_idx]
    print(f"Train: {len(train_df)}  Validation: {len(val_df)}")

    mean = train_df[FEATURES].mean()
    std = train_df[FEATURES].std().replace(0, 1e-6)  # avoids a division by 0 if a feature is constant

    train_loglik = gaussian_log_pdf(train_df[FEATURES].to_numpy(), mean.to_numpy(), std.to_numpy())
    val_loglik = gaussian_log_pdf(val_df[FEATURES].to_numpy(), mean.to_numpy(), std.to_numpy())

    epsilon = float(np.percentile(val_loglik, EPSILON_PERCENTILE))
    print(f"ε threshold (1st percentile, validation): {epsilon:.2f}")

    n_lowest = min(N_LOWEST_TO_INSPECT, len(val_df))
    print(f"{n_lowest} lowest-scoring validation flights (for manual inspection):")
    lowest = val_df.assign(log_vraisemblance=val_loglik).nsmallest(n_lowest, "log_vraisemblance")
    print(lowest[["icao24", "callsign", "log_vraisemblance", *FEATURES]].to_string(index=False))

    percentiles = np.linspace(0, 100, N_PERCENTILE_BREAKPOINTS)
    breakpoints = np.percentile(train_loglik, percentiles)

    return {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "epsilon_log_vraisemblance": epsilon,
        "percentile_breakpoints": breakpoints.tolist(),
        "n_train": len(train_df),
        "n_validation": len(val_df),
    }


def main():
    landed = load_deduplicated_landed_flights(INPUT_PARQUETS)

    records = []
    n_insufficient = 0
    # itertuples(), not iterrows(): pandas/pyarrow (Arrow-backed string dtype
    # by default here) crashes trying to homogenize every column into a
    # single array for iterrows() as soon as waypoints (multi-KB JSON
    # strings) is mixed with columns of different dtypes (callsign object,
    # still_airborne bool) — itertuples() doesn't need that single array, so
    # it doesn't trigger the conversion that crashes.
    for row in landed.itertuples(index=False):
        waypoints = json.loads(row.waypoints)
        features = extract_features(waypoints)
        if features is None:
            n_insufficient += 1
            continue
        features["icao24"] = row.icao24
        features["callsign"] = row.callsign
        features["categorie"] = categorize(row.typecode)
        records.append(features)

    print(f"Flights with usable features: {len(records)} ({n_insufficient} excluded, insufficient data)")

    feat_df = pd.DataFrame(records)
    print("\nBreakdown by category:")
    print(feat_df["categorie"].value_counts())

    rng = np.random.default_rng(RANDOM_SEED)
    categories = {name: fit_category(name, feat_df[feat_df["categorie"] == name], rng) for name in CATEGORIES}

    artifact = {
        "features": FEATURES,
        "log1p_features": sorted(LOG1P_FEATURES),
        "categories": categories,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
    print(f"\nParameters written to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
