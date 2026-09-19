"""Extracting per-flight summary features for the anomaly model (brief
section 11). Shared between training (scripts/train_anomalie.py) and
serving (models/anomalie.py) to avoid any train/serve mismatch.

Each flight becomes one example with summary features over its whole
trajectory (not a score per raw point): pooling raw points from every
flight to fit one Gaussian per variable would mix flight phases (climb/
cruise/descent have very different vertical-speed/altitude distributions)
and break the Gaussian assumption.

Features and transforms chosen empirically on
data/processed/flights_clean.parquet (landed flights): only `duration_min`
is strongly skewed (skew 1.73) and clearly benefits from a log1p (skew -0.25
afterward). The other features stay moderately skewed after attempting a
transform, with no real improvement — treated as a deliberate MVP
simplification rather than over-fitting transforms with no measurable gain.
"""

import numpy as np

from services.aircraft_category import CATEGORIES, categorize

MIN_POINTS = 5

FEATURES = [
    "vitesse_moyenne",
    "vitesse_max",
    "vitesse_std",
    "altitude_max",
    "taux_montee_max",
    "taux_descente_max",
    "duration_min",
]

LOG1P_FEATURES = {"duration_min"}


def extract_raw_features(waypoints: list[dict]) -> dict[str, float] | None:
    """Summary features, before transformation. None if the flight doesn't
    have enough points or is entirely missing one of the required
    measurements."""
    if len(waypoints) < MIN_POINTS:
        return None

    speeds = [w["vitesse"] for w in waypoints if w.get("vitesse") is not None]
    altitudes = [w["altitude"] for w in waypoints if w.get("altitude") is not None]
    vertical_rates = [w["vitesse_verticale"] for w in waypoints if w.get("vitesse_verticale") is not None]
    if not speeds or not altitudes or not vertical_rates:
        return None

    timestamps = [w["timestamp"] for w in waypoints]
    duration_min = (max(timestamps) - min(timestamps)) / 60.0
    if duration_min <= 0:
        return None

    return {
        "vitesse_moyenne": float(np.mean(speeds)),
        "vitesse_max": float(np.max(speeds)),
        "vitesse_std": float(np.std(speeds)),
        "altitude_max": float(np.max(altitudes)),
        "taux_montee_max": float(np.max(vertical_rates)),
        "taux_descente_max": float(np.min(vertical_rates)),
        "duration_min": duration_min,
    }


def apply_transforms(raw_features: dict[str, float]) -> dict[str, float]:
    """Applies log1p to the features listed in LOG1P_FEATURES."""
    return {
        name: np.log1p(value) if name in LOG1P_FEATURES else value
        for name, value in raw_features.items()
    }


def inverse_transform(name: str, value: float) -> float:
    """Inverse of apply_transforms for a single feature — converts a learned
    mean (transformed scale) back to a displayable raw unit (e.g. minutes
    rather than log1p(minutes))."""
    return float(np.expm1(value)) if name in LOG1P_FEATURES else float(value)


def extract_features(waypoints: list[dict]) -> dict[str, float] | None:
    """Summary features ready for the model (transformed). None if the
    flight doesn't have enough data to be scored reliably."""
    raw = extract_raw_features(waypoints)
    if raw is None:
        return None
    return apply_transforms(raw)
