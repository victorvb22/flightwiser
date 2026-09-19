"""Trajectory anomaly (brief section 11) — diagonal Gaussian density
estimation per aircraft category (avion_ligne / jet_affaire / petit_avion /
helicoptere, cf. services.aircraft_category — a single shared Gaussian would
systematically penalize minority groups), one Gaussian per summary flight
feature (cf. _anomalie_features.py), trained only on normal trajectories (no
synthetic anomaly injection). Parameters learned offline by
scripts/train_anomalie.py and persisted in artifacts/anomalie_params.json.

Input:  real trajectory (list of timestamp/lat/lon/altitude/cap/
        vitesse_verticale/au_sol/vitesse points — the format produced by
        preprocess_historical.py, not the API contract's trimmed JSON) and
        the aircraft's typecode (to pick the category, cf.
        services.aircraft_category.categorize).
Output: the `anomalie` portion of the response JSON (brief section 8) —
        {"score": float, "features_contributives": [str, ...],
        "features_detail": [{"feature", "valeur", "reference"}, ...]}, where
        a lower score signals a more pronounced anomaly. `score` is the
        percentile rank (0-1) of the flight's log-likelihood against its
        CATEGORY's own training distribution — monotone with the raw
        likelihood, but bounded and more readable on the UI side than a raw
        log-likelihood (very negative values), and comparable across
        categories despite different raw likelihood scales (a percentile
        rank is always 0-1). `features_detail` (an addition to the brief's
        contract, not a modification: score/features_contributives stay
        unchanged) gives, for each of the features_contributives, the
        flight's raw value and the reference value (its category's training
        mean, un-transformed) in the same unit — shown on hover on the
        frontend rather than permanently cluttering the card.
Completely independent from ecart_trajectoire.py, which resolves the
typecode for an entirely different purpose (picking an OpenAP performance
model in _ecart_features.py — "can this flight be simulated," not "what
kind of aircraft is this"). None if the typecode is in none of the four
categories (services.aircraft_category.categorize returns None): no score
rather than a guessed default classification.
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


def compute(trajectoire: list[dict], typecode: str) -> dict[str, Any] | None:
    """None if the trajectory doesn't have enough points or is entirely
    missing a required measurement (cf. _anomalie_features.MIN_POINTS) — a
    score wouldn't be reliable, not an error in itself (e.g. very start of a
    flight). Also None if the typecode isn't identified in any of the four
    categories (services.aircraft_category.categorize)."""
    category = categorize(typecode)
    if category is None:
        return None

    raw_features = extract_raw_features(trajectoire)
    if raw_features is None:
        return None
    features = apply_transforms(raw_features)

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
