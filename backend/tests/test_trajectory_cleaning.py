"""services/trajectory_cleaning.py -- shared by the batch preprocessing and
the live pipeline, so a bug here silently affects both training data and
every served score at once. All pure functions, no I/O -- cheap to pin down
exactly."""

from services.trajectory_cleaning import (
    clip_altitude,
    clip_vertical_rate,
    compute_speeds,
    dedupe_waypoints,
    drop_missing_position,
    drop_position_teleports,
    haversine_km,
)

# (timestamp, lat, lon, altitude, cap, vitesse_verticale, au_sol)


def test_haversine_known_distance_paris_london():
    # Real-world reference: ~344 km great-circle, Paris CDG to London LHR.
    km = haversine_km(49.0097, 2.5479, 51.4700, -0.4543)
    assert 335 < km < 355


def test_drop_missing_position_removes_points_with_no_lat_or_lon():
    points = [
        (0, 48.0, 2.0, 1000.0, 90.0, 0.0, False),
        (10, None, 2.0, 1000.0, 90.0, 0.0, False),
        (20, 48.1, None, 1000.0, 90.0, 0.0, False),
        (30, 48.2, 2.2, 1000.0, 90.0, 0.0, False),
    ]
    result = drop_missing_position(points)
    assert len(result) == 2
    assert all(p[1] is not None and p[2] is not None for p in result)


def test_clip_altitude_nulls_out_implausible_values_without_dropping_the_point():
    points = [
        (0, 48.0, 2.0, 40000.0, 90.0, 0.0, False),  # above MAX_PLAUSIBLE_ALTITUDE_M
        (10, 48.0, 2.0, 10000.0, 90.0, 0.0, False),  # plausible
    ]
    result = clip_altitude(points)
    assert len(result) == 2  # never drops the point itself
    assert result[0][3] is None
    assert result[1][3] == 10000.0


def test_clip_vertical_rate_nulls_out_implausible_values():
    points = [
        (0, 48.0, 2.0, 1000.0, 90.0, 90.0, False),  # far above MAX_PLAUSIBLE_VERTICAL_RATE_MS
        (10, 48.0, 2.0, 1000.0, 90.0, 5.0, False),  # plausible
    ]
    result = clip_vertical_rate(points)
    assert result[0][5] is None
    assert result[1][5] == 5.0


def test_dedupe_waypoints_drops_identical_and_non_advancing_points():
    points = [
        (0, 48.0, 2.0, 1000.0, 90.0, 0.0, False),
        (0, 48.0, 2.0, 1000.0, 90.0, 0.0, False),  # same coords, same timestamp
        (5, 48.0, 2.0, 1000.0, 90.0, 0.0, False),  # same coords, later timestamp
        (10, 48.1, 2.1, 1000.0, 90.0, 0.0, False),  # genuinely new point
        (8, 48.2, 2.2, 1000.0, 90.0, 0.0, False),  # non-increasing timestamp
    ]
    result = dedupe_waypoints(points)
    assert result == [
        (0, 48.0, 2.0, 1000.0, 90.0, 0.0, False),
        (10, 48.1, 2.1, 1000.0, 90.0, 0.0, False),
    ]


def test_dedupe_waypoints_handles_empty_input():
    assert dedupe_waypoints([]) == []


def test_drop_position_teleports_removes_an_impossible_single_point_jump():
    # A single point 3000 km away from both neighbours, 10s apart on both
    # sides -- far beyond TELEPORT_SPEED_THRESHOLD_MS in both directions.
    points = [
        (0, 48.85, 2.35, 1000.0, 90.0, 0.0, False),   # Paris
        (10, 40.71, -74.00, 1000.0, 90.0, 0.0, False),  # New York -- the bad point
        (20, 48.86, 2.36, 1000.0, 90.0, 0.0, False),   # back near Paris
    ]
    result = drop_position_teleports(points)
    assert len(result) == 2
    assert (10, 40.71, -74.00, 1000.0, 90.0, 0.0, False) not in result


def test_drop_position_teleports_keeps_a_genuinely_fast_but_consistent_segment():
    # Same total distance covered smoothly across several points -- nothing
    # here should look like a one-off corrupted point.
    points = [(i * 10, 48.0 + i * 0.05, 2.0 + i * 0.05, 1000.0, 90.0, 0.0, False) for i in range(5)]
    assert drop_position_teleports(points) == points


def test_compute_speeds_matches_haversine_distance_over_time():
    points = [
        (0, 48.0, 2.0, 1000.0, 90.0, 0.0, False),
        (100, 48.0, 2.2, 1000.0, 90.0, 0.0, False),
    ]
    speeds = compute_speeds(points)
    assert len(speeds) == 2
    expected = haversine_km(48.0, 2.0, 48.0, 2.2) * 1000 / 100  # m/s
    # The first point mirrors the second segment's speed (no earlier
    # segment exists), so both entries should match this two-point case.
    assert speeds[0] == speeds[1]
    assert abs(speeds[0] - expected) < 0.01


def test_compute_speeds_short_input_returns_zeros():
    assert compute_speeds([(0, 48.0, 2.0, 1000.0, 90.0, 0.0, False)]) == [0.0]
    assert compute_speeds([]) == []
