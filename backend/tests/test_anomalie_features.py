"""models/_anomalie_features.py -- only the one behaviour that needed a
regression test: a NaN value (reachable via a JSON round trip through
flights_cache, unlike the CSV training path) must be filtered out, not
silently propagate into the computed features."""

import math

from models._anomalie_features import extract_raw_features


def _waypoint(t, vitesse=100.0, altitude=5000.0, vitesse_verticale=0.0):
    return {"timestamp": t, "vitesse": vitesse, "altitude": altitude, "vitesse_verticale": vitesse_verticale}


def test_a_nan_value_is_filtered_not_propagated():
    waypoints = [_waypoint(float(i)) for i in range(6)]
    waypoints[2]["vitesse"] = float("nan")

    features = extract_raw_features(waypoints)

    assert features is not None
    assert not math.isnan(features["vitesse_moyenne"])
    assert not math.isnan(features["vitesse_max"])
    assert not math.isnan(features["vitesse_std"])


def test_normal_trajectory_extracts_features():
    waypoints = [_waypoint(float(i), vitesse=100.0 + i) for i in range(6)]
    features = extract_raw_features(waypoints)
    assert features is not None
    assert features["vitesse_max"] == 105.0
    assert features["duration_min"] > 0
