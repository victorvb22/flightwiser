"""Checks that the anomaly model, trained on Europe-candidate flights (cf.
scripts/extract_historical_for_training.py), still holds up reasonably for a
flight drawn at random anywhere in the world — the app's random draw is no
longer a geographic-proximity search, so a non-Europe flight is a real case,
not a hypothetical one.

For avion_ligne, jet_affaire, and petit_avion (not helicoptere, which has
its own unconditional "out of the training scope" flag, cf.
models/anomalie.py — unrelated to this check's result): extracts a modest
sample of non-Europe candidate flights on the same day (2022-06-27, the
same source as training),
computes their 7 summary features (models/_anomalie_features), and compares
mean/std per feature to those of the trained Europe model
(models/artifacts/anomalie_params.json).

Usage: python scripts/check_geographic_consistency.py --date 2022-06-27 --sample-size 300
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import truststore

truststore.inject_into_ssl()

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from models._anomalie_features import CATEGORIES, FEATURES, apply_transforms, categorize, extract_raw_features  # noqa: E402
from services.aircraft_database import get_typecode  # noqa: E402
from services.opensky_historical import (  # noqa: E402
    EUROPE_BBOX,
    build_trajectoire,
    extract_candidate_points,
    scan_candidates,
    segment_flights,
)

WORLD_BBOX = (-90.0, 90.0, -180.0, 180.0)
ARTIFACT_PATH = BACKEND_DIR / "models" / "artifacts" / "anomalie_params.json"
RANDOM_SEED = 0
# "Notable" divergence: a non-Europe mean more than this many (Europe)
# standard deviations from the Europe mean. 1.0 = roughly half the
# distribution's own typical width — a conservative threshold, meant to
# catch a real population shift, not sample noise.
NOTABLE_Z_THRESHOLD = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--sample-size", type=int, default=300, help="Non-Europe flights sampled (per target category, before filtering)")
    args = parser.parse_args()
    date = args.date

    print(f"Pass 1 - world candidates for {date} ...")
    world_candidates = scan_candidates(date, WORLD_BBOX)
    print(f"Pass 1 - Europe candidates for {date} ...")
    europe_candidates = scan_candidates(date, EUROPE_BBOX)
    non_europe_candidates = world_candidates - europe_candidates
    print(f"{len(world_candidates)} world candidates, {len(europe_candidates)} Europe, {len(non_europe_candidates)} non-Europe.")

    rng = random.Random(RANDOM_SEED)
    sample = set(rng.sample(sorted(non_europe_candidates), min(args.sample_size, len(non_europe_candidates))))
    print(f"Non-Europe sample: {len(sample)} icao24.")

    print("Pass 2 - extracting sample positions ...")
    points_by_icao = extract_candidate_points(date, sample)

    records = []
    for icao24, points in points_by_icao.items():
        for segment in segment_flights(points):
            au_sol_values = {p[6] for p in segment}
            if True not in au_sol_values or False not in au_sol_values:
                continue  # no air/ground transition -> flight not usable (same filter as the training extraction)
            trajectoire = build_trajectoire(segment)
            if trajectoire is None:
                continue
            if trajectoire[-1]["au_sol"] is not True:
                continue  # still airborne -> out of scope (same filter as train_anomalie.py)
            raw = extract_raw_features(trajectoire)
            if raw is None:
                continue
            typecode = get_typecode(icao24) or "A320"
            records.append({"icao24": icao24, "categorie": categorize(typecode), **apply_transforms(raw)})

    df = pd.DataFrame(records)
    print(f"{len(df)} usable non-Europe flights (out of {len(sample)} sampled candidates).")

    europe_params = json.loads(ARTIFACT_PATH.read_text())["categories"]

    for category in CATEGORIES:
        if category == "helicoptere":
            continue  # unconditional flag, independent of this test, cf. the module's docstring
        subset = df[df["categorie"] == category]
        print(f"\n=== {category}: {len(subset)} usable non-Europe flights ===")
        if len(subset) < 10:
            print("Too few non-Europe flights sampled for this category to conclude anything — increase --sample-size.")
            continue

        europe_mean = europe_params[category]["mean"]
        europe_std = europe_params[category]["std"]
        notable = []
        for feature in FEATURES:
            non_europe_mean = float(subset[feature].mean())
            non_europe_std = float(subset[feature].std())
            z = abs(non_europe_mean - europe_mean[feature]) / max(europe_std[feature], 1e-6)
            flag = " <-- NOTABLE" if z > NOTABLE_Z_THRESHOLD else ""
            print(
                f"  {feature:<20} europe mean={europe_mean[feature]:8.3f} std={europe_std[feature]:7.3f}  |  "
                f"non-europe mean={non_europe_mean:8.3f} std={non_europe_std:7.3f}  |  z={z:5.2f}{flag}"
            )
            if z > NOTABLE_Z_THRESHOLD:
                notable.append(feature)

        if notable:
            print(f"  VERDICT: divergent distributions on {notable} — consider an 'out of scope' flag for {category}.")
        else:
            print(f"  VERDICT: comparable distributions — no flag needed for {category}.")


if __name__ == "__main__":
    main()
