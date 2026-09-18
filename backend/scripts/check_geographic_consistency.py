"""Vérifie que le modèle d'anomalie, entraîné sur des vols candidats Europe
(cf. scripts/extract_historical_for_training.py), reste raisonnable pour un
vol tiré au hasard n'importe où dans le monde — le tirage aléatoire de l'app
n'est plus une recherche par proximité géographique, un vol hors Europe est
donc un cas réel, pas hypothétique.

Pour avion_ligne et petit_avion (pas helicoptere, qui a son propre flag
inconditionnel "out of the training scope", cf. models/anomalie.py — sans
lien avec le résultat de cette vérification) : extrait un échantillon modeste
de vols candidats hors Europe le même jour (27/06/2022, même source que
l'entraînement), calcule leurs 7 features résumées (models/_anomalie_features)
et compare moyenne/écart-type par feature à celles du modèle Europe entraîné
(models/artifacts/anomalie_params.json).

Usage : python scripts/check_geographic_consistency.py --date 2022-06-27 --sample-size 300
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
# Divergence "notable" : une moyenne hors-Europe à plus de ce nombre
# d'écarts-types (Europe) de la moyenne Europe. 1.0 = à peu près la moitié de
# la largeur typique de la distribution elle-même — un seuil conservateur,
# pensé pour repérer un vrai décalage de population, pas du bruit d'échantillon.
NOTABLE_Z_THRESHOLD = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--sample-size", type=int, default=300, help="Vols hors-Europe échantillonnés (par catégorie visée, avant filtrage)")
    args = parser.parse_args()
    date = args.date

    print(f"Passe 1 — candidats monde pour {date} ...")
    world_candidates = scan_candidates(date, WORLD_BBOX)
    print(f"Passe 1 — candidats Europe pour {date} ...")
    europe_candidates = scan_candidates(date, EUROPE_BBOX)
    non_europe_candidates = world_candidates - europe_candidates
    print(f"{len(world_candidates)} candidats monde, {len(europe_candidates)} Europe, {len(non_europe_candidates)} hors-Europe.")

    rng = random.Random(RANDOM_SEED)
    sample = set(rng.sample(sorted(non_europe_candidates), min(args.sample_size, len(non_europe_candidates))))
    print(f"Échantillon hors-Europe : {len(sample)} icao24.")

    print("Passe 2 — extraction des positions de l'échantillon ...")
    points_by_icao = extract_candidate_points(date, sample)

    records = []
    for icao24, points in points_by_icao.items():
        for segment in segment_flights(points):
            au_sol_values = {p[6] for p in segment}
            if True not in au_sol_values or False not in au_sol_values:
                continue  # pas de transition air/sol -> vol pas exploitable (même filtre que l'extraction d'entraînement)
            trajectoire = build_trajectoire(segment)
            if trajectoire is None:
                continue
            if trajectoire[-1]["au_sol"] is not True:
                continue  # encore en vol -> hors périmètre (même filtre que train_anomalie.py)
            raw = extract_raw_features(trajectoire)
            if raw is None:
                continue
            typecode = get_typecode(icao24) or "A320"
            records.append({"icao24": icao24, "categorie": categorize(typecode, icao24), **apply_transforms(raw)})

    df = pd.DataFrame(records)
    print(f"{len(df)} vols hors-Europe exploitables (sur {len(sample)} candidats échantillonnés).")

    europe_params = json.loads(ARTIFACT_PATH.read_text())["categories"]

    for category in CATEGORIES:
        if category == "helicoptere":
            continue  # flag inconditionnel indépendant de ce test, cf. docstring du module
        subset = df[df["categorie"] == category]
        print(f"\n=== {category} : {len(subset)} vols hors-Europe exploitables ===")
        if len(subset) < 10:
            print("Trop peu de vols hors-Europe échantillonnés pour cette catégorie pour conclure — augmenter --sample-size.")
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
            print(f"  VERDICT : distributions divergentes sur {notable} — envisager un flag 'hors périmètre' pour {category}.")
        else:
            print(f"  VERDICT : distributions comparables — pas de flag nécessaire pour {category}.")


if __name__ == "__main__":
    main()
