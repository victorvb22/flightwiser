"""models/anomaly_type.py -- only the one behaviour that needed a
regression test: heading_changes_count skipping a pair of readings too far
apart in time to represent one continuous turn."""

from models.anomaly_type import MAX_APPROACH_GAP_S, _compute_features


def _waypoint(t: float, cap: float | None = None) -> dict:
    # Enough non-None altitude/vertrate points to clear MIN_ALTITUDES(10)/
    # MIN_VERTRATES(5) regardless of how many carry a cap -- only heading is
    # under test here.
    return {"timestamp": t, "altitude": 5000.0, "vitesse_verticale": 0.0, "cap": cap, "lat": 48.0, "lon": 2.0}


def test_a_large_time_gap_between_two_headings_is_not_counted_as_one_turn():
    # Two adjacent (in list order) cap readings spanning more than
    # MAX_APPROACH_GAP_S, with a genuine >30 degree difference -- real ADS-B
    # data drops `cap` for several points in a row often enough that this is
    # a real, not hypothetical, case. Should NOT count as a turn: the two
    # readings aren't temporally adjacent, so the delta reflects a reporting
    # gap, not one continuous manoeuvre.
    trajectoire = [_waypoint(float(i)) for i in range(12)]
    trajectoire[0]["cap"] = 10.0
    trajectoire[1]["cap"] = 200.0
    trajectoire[1]["timestamp"] = MAX_APPROACH_GAP_S + 100.0

    features = _compute_features(trajectoire, duration_min=10.0)

    assert features is not None
    assert features["heading_changes_count"] == 0


def test_a_real_turn_within_the_gap_threshold_is_still_counted():
    trajectoire = [_waypoint(float(i)) for i in range(12)]
    trajectoire[0]["cap"] = 10.0
    trajectoire[1]["timestamp"] = 30.0
    trajectoire[1]["cap"] = 200.0

    features = _compute_features(trajectoire, duration_min=10.0)

    assert features is not None
    assert features["heading_changes_count"] == 1
