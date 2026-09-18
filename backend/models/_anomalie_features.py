"""Extraction des features résumées par vol pour le modèle d'anomalie
(brief section 11). Partagé entre l'entraînement (scripts/train_anomalie.py)
et le service (models/anomalie.py) pour éviter tout écart train/serve.

Chaque vol devient un exemple à features résumées sur l'ensemble de sa
trajectoire (et non un score par point brut) : poolér les points bruts de
tous les vols pour ajuster une gaussienne par variable mélangerait les
phases de vol (montée/croisière/descente ont des distributions de vitesse
verticale/altitude très différentes) et casserait l'hypothèse gaussienne.

Features et transformations choisies empiriquement sur
data/processed/flights_clean.parquet (vols atterris) : seul `duration_min`
est fortement asymétrique (skew 1.73) et bénéficie clairement d'un log1p
(skew -0.25 après). Les autres features restent modérément asymétriques
après tentative de transformation sans réelle amélioration — traité comme
une simplification MVP assumée plutôt que de sur-ajuster des transformations
sans gain mesurable.
"""

import numpy as np

from services.aircraft_category import categorize as _categorize_by_typecode
from services.aircraft_database import is_helicopter

# Anomaly-model-specific: a third category, isolated from petit_avion rather
# than merged into it. A helicopter's baseline profile (hover, no fixed-wing
# climb/cruise/descent phases) sits structurally outside a fixed-wing
# distribution even for a perfectly ordinary flight — pooled with petit_avion,
# it would risk flagging most of that population as anomalous regardless of
# the actual flight, the opposite of what the score is for. Only this model
# splits it out: models/directness.py keeps using
# services.aircraft_category.categorize() directly (helicopters still fall
# under petit_avion there), since that split hasn't been requested for it —
# don't change the shared function itself, or directness's categories change
# along with it.
CATEGORIES = ["avion_ligne", "petit_avion", "helicoptere"]


def categorize(typecode: str, icao24: str) -> str:
    """Same OpenAP-recognition boundary as services.aircraft_category.
    categorize, with a helicopter override checked first (via
    services.aircraft_database.is_helicopter) — see module-level comment."""
    if is_helicopter(icao24):
        return "helicoptere"
    return _categorize_by_typecode(typecode)

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
    """Features résumées, avant transformation. None si le vol n'a pas assez
    de points ou manque totalement d'une des mesures nécessaires."""
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
    """Applique log1p aux features désignées dans LOG1P_FEATURES."""
    return {
        name: np.log1p(value) if name in LOG1P_FEATURES else value
        for name, value in raw_features.items()
    }


def inverse_transform(name: str, value: float) -> float:
    """Inverse de apply_transforms pour une seule feature — reconvertit une
    moyenne apprise (échelle transformée) en unité brute affichable
    (ex. minutes plutôt que log1p(minutes))."""
    return float(np.expm1(value)) if name in LOG1P_FEATURES else float(value)


def extract_features(waypoints: list[dict]) -> dict[str, float] | None:
    """Features résumées prêtes pour le modèle (transformées). None si le
    vol n'a pas assez de données pour être scoré de façon fiable."""
    raw = extract_raw_features(waypoints)
    if raw is None:
        return None
    return apply_transforms(raw)
