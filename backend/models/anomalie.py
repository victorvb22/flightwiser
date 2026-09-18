"""Anomalie de trajectoire (brief section 11) — estimation de densité
gaussienne diagonale par catégorie d'appareil (avion_ligne / petit_avion /
helicoptere, cf. _anomalie_features.categorize — une gaussienne unique
pénaliserait systématiquement les groupes minoritaires), une gaussienne par
feature résumée de vol (cf. _anomalie_features.py), entraînée uniquement sur
des trajectoires normales (pas d'injection synthétique d'anomalies).
Paramètres appris hors ligne par scripts/train_anomalie.py et persistés dans
artifacts/anomalie_params.json.

Input  : trajectoire réelle (liste de points timestamp/lat/lon/altitude/cap/
          vitesse_verticale/au_sol/vitesse — le format produit par
          preprocess_historical.py, pas le JSON trimé du contrat API), le
          typecode de l'appareil et son icao24 (les deux servent à choisir
          la catégorie — icao24 pour détecter un hélicoptère, cf.
          services.aircraft_database.is_helicopter).
Output : la portion `anomalie` du JSON de réponse (brief section 8) —
          {"score": float, "features_contributives": [str, ...],
          "features_detail": [{"feature", "valeur", "reference"}, ...]}, où
          un score plus bas signale une anomalie plus marquée. `score` est
          le rang percentile (0-1) de la log-vraisemblance du vol par
          rapport à la distribution d'entraînement DE SA CATÉGORIE —
          monotone avec la vraisemblance brute, mais borné et plus lisible
          côté UI qu'une log-vraisemblance brute (valeurs très négatives),
          et comparable d'une catégorie à l'autre malgré des échelles de
          vraisemblance brute différentes (un rang percentile est toujours
          0-1). `features_detail` (ajout au contrat brief, pas une
          modification : score/features_contributives restent inchangés)
          donne, pour chacune des features_contributives, la valeur brute du
          vol et la valeur de référence (moyenne d'entraînement de sa
          catégorie, dé-transformée) dans la même unité — affiché au survol
          côté frontend plutôt que de sur-remplir la carte en permanence.
Complètement indépendant de ecart_trajectoire.py, qui résout le typecode pour
un tout autre usage (choisir un modèle de performance OpenAP dans
_ecart_features.py), pas pour catégoriser. La frontière avion_ligne/
petit_avion de base vit dans services/aircraft_category.py, partagée avec
models/directness.py — mais _anomalie_features.categorize() y ajoute une
troisième catégorie (helicoptere) propre à ce modèle, cf. son propre
docstring pour le pourquoi.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from services.opensky_historical import EUROPE_BBOX, flight_touches_bbox

from ._anomalie_features import CATEGORIES, FEATURES, apply_transforms, categorize, extract_raw_features, inverse_transform

_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "anomalie_params.json"
_N_CONTRIBUTIVE_FEATURES = 3


class _CategoryModel:
    def __init__(self, params: dict):
        self.mean_by_feature = params["mean"]
        self.mean = np.array([params["mean"][f] for f in FEATURES])
        self.std = np.array([params["std"][f] for f in FEATURES])
        self.breakpoints = np.array(params["percentile_breakpoints"])
        self.percentiles = np.linspace(0, 100, len(self.breakpoints))


_raw_params = json.loads(_ARTIFACT_PATH.read_text())
_models = {name: _CategoryModel(_raw_params["categories"][name]) for name in CATEGORIES}


def compute(trajectoire: list[dict], typecode: str, icao24: str) -> dict[str, Any] | None:
    """None si la trajectoire n'a pas assez de points ou manque totalement
    d'une mesure nécessaire (cf. _anomalie_features.MIN_POINTS) — un score
    ne serait pas fiable, pas une erreur en soi (ex. tout début de vol)."""
    raw_features = extract_raw_features(trajectoire)
    if raw_features is None:
        return None
    features = apply_transforms(raw_features)

    category = categorize(typecode, icao24)
    model = _models[category]
    x = np.array([features[f] for f in FEATURES])
    z = (x - model.mean) / model.std
    var = model.std**2
    log_likelihood = float(np.sum(-0.5 * np.log(2 * np.pi * var) - (x - model.mean) ** 2 / (2 * var)))

    score = float(np.interp(log_likelihood, model.breakpoints, model.percentiles) / 100.0)
    contributive_order = np.argsort(-np.abs(z))
    features_contributives = [FEATURES[i] for i in contributive_order[:_N_CONTRIBUTIVE_FEATURES]]
    features_detail = [
        {
            "feature": f,
            "valeur": raw_features[f],
            "reference": inverse_transform(f, model.mean_by_feature[f]),
        }
        for f in features_contributives
    ]

    # Helicopters vary a lot by mission (offshore, medical, touristic) —
    # unlike fixed-wing flight, which is universal enough that a Europe-
    # trained baseline travels reasonably well worldwide (checked directly
    # against a non-Europe sample, cf. Documentation page). A random
    # helicopter draw can land anywhere in the world (no more geographic
    # proximity search), so this flag is unconditional for helicoptere —
    # not contingent on any statistical check — whenever the trajectory
    # never touches the training region at all.
    out_of_training_scope = category == "helicoptere" and not flight_touches_bbox(trajectoire, EUROPE_BBOX)

    return {
        "score": score,
        "features_contributives": features_contributives,
        "features_detail": features_detail,
        "out_of_training_scope": out_of_training_scope,
    }


def get_category_summaries() -> dict[str, dict[str, Any]]:
    """Read-only snapshot for the Documentation page — real, trained
    percentile breakpoints and epsilon per category, not illustrative
    numbers, so the docs stay honest about what's actually deployed. mean/std
    excluded: at 7 dimensions they don't reduce to a single plottable curve,
    unlike percentile_breakpoints (already a 1-D distribution — of
    log-likelihood — by construction)."""
    return {
        name: {
            "n_train": _raw_params["categories"][name]["n_train"],
            "n_validation": _raw_params["categories"][name]["n_validation"],
            "epsilon_log_vraisemblance": _raw_params["categories"][name]["epsilon_log_vraisemblance"],
            "percentile_breakpoints": _raw_params["categories"][name]["percentile_breakpoints"],
        }
        for name in CATEGORIES
    }
