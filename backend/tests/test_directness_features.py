"""models/_directness_features.py -- the great-circle / distance-flown ratio
feeding the route-directness model."""

from models._directness_features import MIN_GREAT_CIRCLE_KM, extract_route_directness


def _waypoint(lat, lon):
    return {"lat": lat, "lon": lon}


def test_straight_line_route_scores_close_to_one():
    # Five points on (approximately) a straight line, far enough apart to
    # clear MIN_GREAT_CIRCLE_KM.
    waypoints = [_waypoint(48.0 + i * 0.5, 2.0 + i * 0.5) for i in range(5)]
    ratio = extract_route_directness(waypoints)
    assert ratio is not None
    assert ratio > 0.99


def test_detour_scores_meaningfully_below_a_direct_route():
    # Same endpoints as the straight-line case, but routed through a point
    # well off the direct path -- flown distance grows, great-circle
    # distance (endpoint to endpoint) doesn't.
    waypoints = [
        _waypoint(48.0, 2.0),
        _waypoint(50.0, 8.0),  # detour north
        _waypoint(50.0, -2.0),  # detour back west
        _waypoint(50.0, 8.0),
        _waypoint(50.0, 2.0),
    ]
    ratio = extract_route_directness(waypoints)
    assert ratio is not None
    assert ratio < 0.5


def test_ratio_never_exceeds_one():
    # A dense, near-straight line: summing many short segments' great-circle
    # distances can, with floating-point rounding, land marginally above the
    # true endpoint-to-endpoint distance -- exactly the ADS-B-noise case the
    # 1.0 cap exists for (cf. extract_route_directness's docstring).
    waypoints = [_waypoint(48.0 + i * 0.01, 2.0 + i * 0.01) for i in range(50)]
    ratio = extract_route_directness(waypoints)
    assert ratio is not None
    assert ratio <= 1.0


def test_too_few_points_returns_none():
    assert extract_route_directness([_waypoint(48.0, 2.0), _waypoint(48.1, 2.1)]) is None


def test_great_circle_too_short_returns_none():
    # Endpoints well under MIN_GREAT_CIRCLE_KM apart.
    tiny_offset = (MIN_GREAT_CIRCLE_KM / 111.0) / 10  # a fraction of a km in degrees
    waypoints = [_waypoint(48.0, 2.0 + i * tiny_offset) for i in range(6)]
    assert extract_route_directness(waypoints) is None
