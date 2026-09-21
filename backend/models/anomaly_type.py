"""Trajectory anomaly type (go-around / holding pattern / emergency
descent) for an avion_ligne flight already flagged anomalous by
models/anomalie.py — a separate classical ML project ("ML classiques sur
données de vol/classification model", notebooks
01_dataset_construction.ipynb + 02_modelling.ipynb), reconstructed
identically by scripts/train_anomaly_type_classifier.py (cf. its docstring)
and persisted in artifacts/anomaly_type_classifier.joblib.

Trained ONLY on synthetically injected anomalies (altitude rebound, holding
circuit, constant-rate descent — never a real case, cf. the training
script's docstring and the source project's README). A "normal" result here
on a flight flightwiser itself flagged anomalous is honest information —
none of the three learned patterns match — not a model failure: shown as-is
on the frontend rather than forcing a label that wouldn't mean anything.

Completely independent of models/anomalie.py (which decides WHETHER a
flight is anomalous): this module only decides, downstream, WHICH type if
applicable — like ecart_trajectoire.py and directness.py, it also doesn't
concern itself with the aircraft's category/status: pipeline.py is what
restricts the call to this one case (a landed avion_ligne flight, cf. its
docstring).

Input: the real trajectory, same internal format as ecart_trajectoire.py/
anomalie.py (timestamp/lat/lon/altitude/cap/vitesse_verticale/au_sol/vitesse)
— vitesse_verticale already smoothed by services/trajectory_cleaning.py,
which plays exactly the role the source notebook's own 60s sliding window
played for its own, irregularly sampled live OpenSky data: no reason to
reimplement that smoothing here, the input is already equivalent.
Output: one of the model's 4 labels ("go_around", "holding",
"emergency_descent", "normal"), or None if the trajectory doesn't have
enough usable points.
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
# Cf. the source notebook's "Real-Time Inference" section: beyond this gap
# between two points in the approach zone, the computed altitude rebound
# reflects how the points are spaced rather than a real climb-back — set to
# 0 rather than producing a misleading number.
MAX_APPROACH_GAP_S = 300
# Last 30% of the trajectory — same split as the source project for the
# approach zone (feature alt_rebound_max).
APPROACH_FRACTION = 0.7
# A point is "in cruise" above 80% of the max altitude reached — same
# threshold as the source project (feature cruise_duration_ratio).
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

    # Heading changes > 30° — handles the 0/360 wrap-around (e.g. 355° -> 5°
    # = a 10° gap, not 350°).
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
    """One of the model's 4 labels, or None if the trajectory doesn't have
    enough usable points (cf. MIN_ALTITUDES/MIN_VERTRATES)."""
    if not trajectoire:
        return None
    timestamps = [w["timestamp"] for w in trajectoire]
    duration_min = (max(timestamps) - min(timestamps)) / 60.0
    features = _compute_features(trajectoire, duration_min)
    if features is None:
        return None
    x = np.array([[features[f] for f in _FEATURE_COLS]])
    return str(_model.predict(x)[0])
