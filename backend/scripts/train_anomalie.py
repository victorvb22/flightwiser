"""Entraînement offline du modèle d'anomalie (brief section 11) : ajuste une
gaussienne diagonale par catégorie d'appareil (une par feature résumée de
vol, cf. models/_anomalie_features.py) sur les vols atterris de l'ensemble
du dataset disponible, et persiste les paramètres dans
models/artifacts/anomalie_params.json pour le service (models/anomalie.py).

Deux sources concaténées (brief : "utiliser le dataset entier") plutôt
qu'une seule :
  - data/processed/flights_clean.parquet (3132 vols, 27/06/2022)
  - data/processed/flights_historical_features.parquet, produit par
    scripts/extract_historical_for_training.py (bbox Europe) pour la même
    journée.

Les deux sources se recouvrent partiellement (même jour) — dédoublonnées via
scripts._training_data.load_deduplicated_landed_flights avant fusion (sinon
un vol présent dans les deux pèserait deux fois dans μ/σ), qui garde la
version flights_historical_features.parquet en cas de doublon.

Catégorisation avion_ligne / petit_avion / helicoptere (cf.
models/_anomalie_features.categorize) : une gaussienne unique sur tout le
dataset pénaliserait systématiquement les groupes minoritaires, leurs
distributions de vitesse/altitude/taux de montée étant trop différentes —
hélicoptère isolé de petit_avion en particulier, son profil (vol
stationnaire, pas de phases montée/croisière/descente classiques) étant hors
de la distribution d'un avion à voilure fixe même pour un vol normal.

Pas de vraies étiquettes d'anomalie dans ce dataset (brief section 11) : le
seuil ε est choisi, par catégorie, comme le 1er percentile de la
log-vraisemblance sur un split de validation, puis les vols les plus bas
classés sont affichés pour une inspection manuelle de cohérence — une
heuristique documentée, pas une validation supervisée précision/rappel.
Le score exposé côté service est un rang percentile (0-1) recalculé par
rapport à cette même distribution d'entraînement, ce qui reste comparable
d'une catégorie à l'autre malgré des log-vraisemblances brutes non
comparables (échelles différentes par catégorie) — ε ne sert donc, comme
avant, qu'à l'inspection manuelle ci-dessous, pas au score servi.

Usage : python scripts/train_anomalie.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Nécessaire pour "python scripts/train_anomalie.py" (sys.path[0] est alors
# scripts/, pas backend/) ; sans effet si déjà lancé via -m.
sys.path.insert(0, str(BACKEND_DIR))

from models._anomalie_features import CATEGORIES, FEATURES, LOG1P_FEATURES, categorize, extract_features  # noqa: E402
from scripts._training_data import load_deduplicated_landed_flights  # noqa: E402

DATA_DIR = BACKEND_DIR.parent / "data"
# flights_historical_features.parquet vit hors du dépôt (cf. son script de
# génération, scripts/extract_historical_for_training.py) — même variable
# d'environnement que le cache raw_states/ pour rester cohérent.
EXTERNAL_DATA_DIR = Path(os.environ.get("IMPORT_DATA_DIR", r"D:\ML_data\flightwiser"))
INPUT_PARQUETS = [
    DATA_DIR / "processed" / "flights_clean.parquet",
    EXTERNAL_DATA_DIR / "processed" / "flights_historical_features.parquet",
]
ARTIFACT_PATH = BACKEND_DIR / "models" / "artifacts" / "anomalie_params.json"

VALIDATION_FRACTION = 0.10
EPSILON_PERCENTILE = 1.0  # 1er percentile de la log-vraisemblance de validation
N_LOWEST_TO_INSPECT = 10
RANDOM_SEED = 0
N_PERCENTILE_BREAKPOINTS = 101  # percentiles 0..100 inclus
# En dessous de ce nombre de vols, une gaussienne (et a fortiori un split
# train/validation) n'est plus fiable — mieux vaut le signaler bruyamment
# que de persister un artefact silencieusement sous-entraîné.
MIN_FLIGHTS_PER_CATEGORY = 30


def gaussian_log_pdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Log-densité d'une gaussienne diagonale, sommée sur les features (axis=1)."""
    var = std**2
    return np.sum(-0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var), axis=1)


def fit_category(name: str, feat_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    print(f"\n=== Catégorie : {name} ({len(feat_df)} vols) ===")
    if len(feat_df) < MIN_FLIGHTS_PER_CATEGORY:
        print(f"ATTENTION : {len(feat_df)} < {MIN_FLIGHTS_PER_CATEGORY} vols, gaussienne peu fiable pour cette catégorie.")

    print("Skew par feature (après transformation) :")
    print(feat_df[FEATURES].skew())

    shuffled_idx = rng.permutation(len(feat_df))
    n_val = max(1, int(len(feat_df) * VALIDATION_FRACTION))
    val_idx, train_idx = shuffled_idx[:n_val], shuffled_idx[n_val:]
    train_df, val_df = feat_df.iloc[train_idx], feat_df.iloc[val_idx]
    print(f"Train : {len(train_df)}  Validation : {len(val_df)}")

    mean = train_df[FEATURES].mean()
    std = train_df[FEATURES].std().replace(0, 1e-6)  # évite une division par 0 si une feature est constante

    train_loglik = gaussian_log_pdf(train_df[FEATURES].to_numpy(), mean.to_numpy(), std.to_numpy())
    val_loglik = gaussian_log_pdf(val_df[FEATURES].to_numpy(), mean.to_numpy(), std.to_numpy())

    epsilon = float(np.percentile(val_loglik, EPSILON_PERCENTILE))
    print(f"Seuil ε (1er percentile, validation) : {epsilon:.2f}")

    n_lowest = min(N_LOWEST_TO_INSPECT, len(val_df))
    print(f"{n_lowest} vols de validation les plus bas classés (à inspecter manuellement) :")
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
    for _, row in landed.iterrows():
        waypoints = json.loads(row["waypoints"])
        features = extract_features(waypoints)
        if features is None:
            n_insufficient += 1
            continue
        features["icao24"] = row["icao24"]
        features["callsign"] = row["callsign"]
        features["categorie"] = categorize(row["typecode"], row["icao24"])
        records.append(features)

    print(f"Vols avec features exploitables : {len(records)} ({n_insufficient} exclus, données insuffisantes)")

    feat_df = pd.DataFrame(records)
    print("\nRépartition par catégorie :")
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
    print(f"\nParamètres écrits dans {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
