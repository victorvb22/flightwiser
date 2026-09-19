"""Type d'anomalie de trajectoire (go-around / holding pattern / emergency
descent) pour un vol avion_ligne déjà flagué anormal par models/anomalie.py —
un projet ML classique distinct ("ML classiques sur données de vol/
classification model", notebooks 01_dataset_construction.ipynb +
02_modelling.ipynb), reconstruit à l'identique par
scripts/train_anomaly_type_classifier.py (cf. sa docstring) et persisté dans
artifacts/anomaly_type_classifier.joblib.

Entraîné UNIQUEMENT sur des anomalies injectées synthétiquement (rebond
d'altitude, circuit d'attente, descente à taux constant — jamais un vrai cas
réel, cf. la docstring du script d'entraînement et le README du projet
source). Un résultat "normal" ici sur un vol que flightwiser a lui-même
flagué anormal est une information honnête — aucun des trois patterns appris
ici ne correspond — pas un échec du modèle : à afficher tel quel côté
frontend plutôt que de forcer une étiquette qui ne voudrait rien dire.

Complètement indépendant de models/anomalie.py (qui décide SI un vol est
anormal) : ce module décide seulement, en aval, QUEL type si applicable —
comme ecart_trajectoire.py et directness.py, il ne se préoccupe pas non plus
de la catégorie/du statut de l'appareil : c'est pipeline.py qui restreint
l'appel à ce cas (vol avion_ligne atterri, cf. sa docstring).

Input : trajectoire réelle, même format interne que ecart_trajectoire.py/
anomalie.py (timestamp/lat/lon/altitude/cap/vitesse_verticale/au_sol/vitesse)
— vitesse_verticale déjà lissée par services/trajectory_cleaning.py, ce qui
joue exactement le rôle que la fenêtre glissante de 60s du notebook source
jouait pour ses propres données OpenSky live (irrégulièrement échantillonnées) :
aucune raison de réimplémenter ce lissage ici, l'entrée est déjà équivalente.
Output : un des 4 libellés du modèle ("go_around", "holding",
"emergency_descent", "normal"), ou None si la trajectoire n'a pas assez de
points exploitables.
"""

from pathlib import Path

import joblib
import numpy as np

from ._ecart_features import haversine_km

_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "anomaly_type_classifier.joblib"
_artifact = joblib.load(_ARTIFACT_PATH)
_model = _artifact["model"]
_FEATURE_COLS = _artifact["feature_cols"]

MIN_ALTITUDES = 10
MIN_VERTRATES = 5
# Cf. section "Real-Time Inference" du notebook source : au-delà de cet écart
# entre deux points dans la zone d'approche, le rebond d'altitude calculé
# reflète la répartition des points plutôt qu'une vraie remontée — mis à 0
# plutôt que de produire un chiffre trompeur.
MAX_APPROACH_GAP_S = 300
# Dernier 30% de la trajectoire — même découpage que le projet source pour
# la zone d'approche (feature alt_rebound_max).
APPROACH_FRACTION = 0.7
# Un point est en "croisière" au-delà de 80% de l'altitude max atteinte —
# même seuil que le projet source (feature cruise_duration_ratio).
CRUISE_ALTITUDE_FRACTION = 0.8
MAX_PLAUSIBLE_ALTITUDE_M = 15000


def _compute_features(trajectoire: list[dict], duration_min: float) -> dict[str, float] | None:
    alts = [w["altitude"] for w in trajectoire if w["altitude"] is not None and 0 <= w["altitude"] <= MAX_PLAUSIBLE_ALTITUDE_M]
    headings = [w["cap"] for w in trajectoire if w["cap"] is not None]
    vertrates = [w["vitesse_verticale"] for w in trajectoire if w["vitesse_verticale"] is not None]
    if len(alts) < MIN_ALTITUDES or len(vertrates) < MIN_VERTRATES:
        return None

    max_alt = max(alts)
    cruise_threshold = max_alt * CRUISE_ALTITUDE_FRACTION

    vrate_descent_max = abs(min(vertrates))

    cruise_vrates = [
        w["vitesse_verticale"]
        for w in trajectoire
        if w["altitude"] is not None
        and w["vitesse_verticale"] is not None
        and 0 <= w["altitude"] <= MAX_PLAUSIBLE_ALTITUDE_M
        and w["altitude"] >= cruise_threshold
    ]
    vrate_cruise_std = float(np.std(cruise_vrates)) if len(cruise_vrates) >= 3 else 0.0

    # Changements de cap > 30° — gère le wrap-around 0/360 (ex. 355° -> 5° =
    # 10° d'écart, pas 350°).
    heading_changes = 0
    for i in range(1, len(headings)):
        delta = abs(headings[i] - headings[i - 1])
        delta = min(delta, 360 - delta)
        if delta > 30:
            heading_changes += 1

    valid_pos = [(w["lat"], w["lon"]) for w in trajectoire if w["lat"] is not None and w["lon"] is not None]
    gc_dist_km = float(haversine_km(*valid_pos[0], *valid_pos[-1])) if len(valid_pos) >= 2 else 0.0
    ratio_dist_duration = gc_dist_km / duration_min if duration_min > 0 else 0.0

    approach = trajectoire[int(len(trajectoire) * APPROACH_FRACTION):]
    approach_alts = [w["altitude"] for w in approach if w["altitude"] is not None and 0 <= w["altitude"] <= MAX_PLAUSIBLE_ALTITUDE_M]
    approach_times = [w["timestamp"] for w in approach if w["altitude"] is not None and 0 <= w["altitude"] <= MAX_PLAUSIBLE_ALTITUDE_M]
    max_gap = max((approach_times[i] - approach_times[i - 1] for i in range(1, len(approach_times))), default=0)
    if approach_alts and max_gap <= MAX_APPROACH_GAP_S:
        min_approach = min(approach_alts)
        min_idx = approach_alts.index(min_approach)
        after_min = approach_alts[min_idx:]
        alt_rebound_max = (max(after_min) - min_approach) if after_min else 0.0
    else:
        alt_rebound_max = 0.0

    n_cruise = sum(
        1
        for w in trajectoire
        if w["altitude"] is not None and 0 <= w["altitude"] <= MAX_PLAUSIBLE_ALTITUDE_M and w["altitude"] >= cruise_threshold
    )
    cruise_duration_ratio = n_cruise / len(trajectoire) if trajectoire else 0.0

    return {
        "vrate_descent_max": vrate_descent_max,
        "vrate_cruise_std": vrate_cruise_std,
        "heading_changes_count": heading_changes,
        "ratio_dist_duration": ratio_dist_duration,
        "alt_rebound_max": alt_rebound_max,
        "cruise_duration_ratio": cruise_duration_ratio,
    }


def compute(trajectoire: list[dict]) -> str | None:
    """Un des 4 libellés du modèle, ou None si la trajectoire n'a pas assez
    de points exploitables (cf. MIN_ALTITUDES/MIN_VERTRATES)."""
    if not trajectoire:
        return None
    timestamps = [w["timestamp"] for w in trajectoire]
    duration_min = (max(timestamps) - min(timestamps)) / 60.0
    features = _compute_features(trajectoire, duration_min)
    if features is None:
        return None
    x = np.array([[features[f] for f in _FEATURE_COLS]])
    return str(_model.predict(x)[0])
