"""ADS-B trajectory cleanup — deduplication, artifact removal (position
teleports, altitude/vertical-speed outside a physically plausible range),
speed recomputation. Shared between the batch preprocessing
(scripts/preprocess_historical.py) and the live pipeline (pipeline.py), to
avoid duplicating logic that's already had to be debugged three times on
this project (teleports, 38,000 m altitude, 90 m/s vertical speed — all real
ADS-B artifacts found on the historical dataset).

Every function operates on 7-element tuples
(timestamp, lat, lon, altitude, cap, vitesse_verticale, au_sol) — the raw
format shared by both data sources (the historical CSV and OpenSky's live
trajectory, once the latter is converted to this same format).
"""

import numpy as np
import pandas as pd

SPEED_SMOOTHING_WINDOW = 5
# A point whose implied speed exceeds this threshold on both sides (the
# incoming AND outgoing segment) is a corrupted ADS-B point (a one-off
# "teleport", e.g. a France -> Pacific jump in 10s observed on this
# dataset), not real motion — dropped rather than letting it distort the
# trajectory.
TELEPORT_SPEED_THRESHOLD_MS = 2000.0
# Defensive physical ceiling on the final (smoothed) speed: no subsonic
# civil aircraft exceeds ~450 m/s (875 kt) over ground, even with an
# exceptional jet stream. Absorbs residual ADS-B noise that survives
# teleport removal.
MAX_PLAUSIBLE_SPEED_MS = 450.0
# Physically plausible barometric altitude range for a commercial flight
# (wide bounds: ceiling ~45,000 ft, floor below sea level for the airports
# concerned). Some points in the historical dataset exceed 38,000 m (the
# world altitude record for crewed flight, clearly an ADS-B artifact), which
# would distort downstream altitude statistics.
MIN_PLAUSIBLE_ALTITUDE_M = -500.0
MAX_PLAUSIBLE_ALTITUDE_M = 13716.0
# Plausible vertical speed for a commercial aircraft (±30 m/s ~ ±5900 ft/min,
# well beyond real climb/descent rates). Same kind of ADS-B artifact as
# altitude: a few points in the historical dataset exceed 90 m/s.
MAX_PLAUSIBLE_VERTICAL_RATE_MS = 30.0

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points (scalars or numpy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def drop_missing_position(points: list[tuple]) -> list[tuple]:
    """Drops points with no position (missing lat/lon) — unusable for the
    trajectory, deduplication, or speed computation."""
    return [p for p in points if p[1] is not None and p[2] is not None]


def clip_altitude(points: list[tuple]) -> list[tuple]:
    """Replaces altitudes outside the physically plausible range with None
    rather than clamping them to the bound, which would create an artificial
    spike of identical values. Treated as a missing altitude, already
    handled downstream."""
    return [
        p if p[3] is None or MIN_PLAUSIBLE_ALTITUDE_M <= p[3] <= MAX_PLAUSIBLE_ALTITUDE_M
        else (p[0], p[1], p[2], None, p[4], p[5], p[6])
        for p in points
    ]


def clip_vertical_rate(points: list[tuple]) -> list[tuple]:
    """Same logic as clip_altitude, for vertical speed."""
    return [
        p if p[5] is None or abs(p[5]) <= MAX_PLAUSIBLE_VERTICAL_RATE_MS
        else (p[0], p[1], p[2], p[3], p[4], None, p[6])
        for p in points
    ]


def dedupe_waypoints(points: list[tuple]) -> list[tuple]:
    """Drops consecutive points with identical coordinates or a
    non-advancing timestamp — real sources of division-by-~0 in the speed
    computation."""
    if not points:
        return points
    deduped = [points[0]]
    for point in points[1:]:
        prev = deduped[-1]
        same_coords = point[1] == prev[1] and point[2] == prev[2]
        non_increasing_time = point[0] <= prev[0]
        if same_coords or non_increasing_time:
            continue
        deduped.append(point)
    return deduped


def drop_position_teleports(points: list[tuple]) -> list[tuple]:
    """Drops corrupted ADS-B points: a single wrong position implies an
    absurd speed both to reach it and to leave it (unlike a genuinely fast
    segment, which is only inconsistent in one direction if it touches a
    valid point). Repeats until stable, since removing one point can expose
    another now isolated between two neighbors that are too far apart (rare,
    but observed)."""
    for _ in range(5):
        n = len(points)
        if n < 3:
            return points
        keep = [True] * n
        for i in range(1, n - 1):
            t0, lat0, lon0 = points[i - 1][0], points[i - 1][1], points[i - 1][2]
            t1, lat1, lon1 = points[i][0], points[i][1], points[i][2]
            t2, lat2, lon2 = points[i + 1][0], points[i + 1][1], points[i + 1][2]
            speed_in = haversine_km(lat0, lon0, lat1, lon1) * 1000 / (t1 - t0)
            speed_out = haversine_km(lat1, lon1, lat2, lon2) * 1000 / (t2 - t1)
            if speed_in > TELEPORT_SPEED_THRESHOLD_MS and speed_out > TELEPORT_SPEED_THRESHOLD_MS:
                keep[i] = False
        if all(keep):
            return points
        points = [p for p, k in zip(points, keep) if k]
    return points


def compute_speeds(points: list[tuple]) -> list[float]:
    """Ground speed (m/s) per point via haversine distance / time delta,
    smoothed with a rolling mean then capped at MAX_PLAUSIBLE_SPEED_MS
    (residual ADS-B noise). The first point reuses the second segment's
    speed (no preceding segment available)."""
    n = len(points)
    if n < 2:
        return [0.0] * n

    raw_speeds = [None] * n
    for i in range(1, n):
        t0, lat0, lon0 = points[i - 1][0], points[i - 1][1], points[i - 1][2]
        t1, lat1, lon1 = points[i][0], points[i][1], points[i][2]
        dt = t1 - t0
        dist_km = haversine_km(lat0, lon0, lat1, lon1)
        raw_speeds[i] = (dist_km * 1000) / dt  # m/s
    raw_speeds[0] = raw_speeds[1]

    speeds = pd.Series(raw_speeds).rolling(
        window=SPEED_SMOOTHING_WINDOW, center=True, min_periods=1
    ).mean().clip(upper=MAX_PLAUSIBLE_SPEED_MS)
    return speeds.tolist()


def compute_vertical_rates(points: list[tuple]) -> list[float | None]:
    """Vertical speed (m/s) per point via altitude delta / time delta,
    smoothed then capped at MAX_PLAUSIBLE_VERTICAL_RATE_MS — mirrors
    compute_speeds, but for a trajectory that doesn't already have a
    measured vertical speed (e.g. live OpenSky tracking, unlike the
    historical CSV which has it directly). None if altitude is missing for
    the point or its neighbor."""
    n = len(points)
    if n < 2:
        return [None] * n

    raw_vr: list[float | None] = [None] * n
    for i in range(1, n):
        t0, alt0 = points[i - 1][0], points[i - 1][3]
        t1, alt1 = points[i][0], points[i][3]
        if alt0 is None or alt1 is None:
            continue
        raw_vr[i] = (alt1 - alt0) / (t1 - t0)
    raw_vr[0] = raw_vr[1]

    vr = (
        pd.Series(raw_vr, dtype=float)
        .rolling(window=SPEED_SMOOTHING_WINDOW, center=True, min_periods=1)
        .mean()
        .clip(lower=-MAX_PLAUSIBLE_VERTICAL_RATE_MS, upper=MAX_PLAUSIBLE_VERTICAL_RATE_MS)
    )
    return [None if pd.isna(v) else float(v) for v in vr]
