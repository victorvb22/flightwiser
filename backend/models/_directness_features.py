"""Extracting the "direct route" feature (brief section 11, an extension):
great-circle distance between the trajectory's real endpoints, divided by
the distance actually flown along the path — a signal neither existing
model sees. The anomaly model (models/_anomalie_features.py) only looks at
scalar summary features (speed/altitude/climb rate), blind to the route's
SHAPE; the deviation model (models/_ecart_features.py) compares the
altitude/speed profile per phase against an OpenAP simulation, not lateral
routing. A flight that circles, deviates widely, or undergoes extended ATC
vectoring can therefore go unnoticed by both, without its speed or altitude
ever looking unusual.

Ratio bounded by construction (a straight line is the shortest possible path
on a sphere): close to 1 for a direct route, lower for a detour. Shared
between training (scripts/train_directness.py) and serving
(models/directness.py) to avoid any train/serve mismatch, same as
_anomalie_features.py does for its own model.
"""

from services.geo import haversine_km
from ._shared import MIN_POINTS

# Below this great-circle distance, the ratio is dominated by ADS-B position
# noise rather than a real detour (a local there-and-back or a touch-and-go
# doesn't really have a "direct route" to measure).
MIN_GREAT_CIRCLE_KM = 5.0


def extract_route_directness(waypoints: list[dict]) -> float | None:
    """None if too few usable points, or if the great-circle distance is too
    short for a reliable ratio (cf. MIN_GREAT_CIRCLE_KM) — not an error in
    itself, just a flight for which this particular signal isn't usable (the
    other models stay independent of this one). Every known caller today
    already runs drop_missing_position first (services/trajectory_cleaning.py),
    so a None lat/lon isn't reachable in practice yet -- filtered here anyway,
    same as models/anomaly_type.py's own feature extraction, rather than
    leaving this function's only defense against one be "every future caller
    happens to remember to filter first"."""
    usable = [w for w in waypoints if w["lat"] is not None and w["lon"] is not None]
    if len(usable) < MIN_POINTS:
        return None

    lats = [w["lat"] for w in usable]
    lons = [w["lon"] for w in usable]

    great_circle_km = float(haversine_km(lats[0], lons[0], lats[-1], lons[-1]))
    if great_circle_km < MIN_GREAT_CIRCLE_KM:
        return None

    flown_km = sum(float(haversine_km(lats[i], lons[i], lats[i + 1], lons[i + 1])) for i in range(len(lats) - 1))
    if flown_km <= 0:
        return None

    # Capped at 1.0: ADS-B position noise can push the ratio marginally
    # above it (a straight line is the *theoretical* shortest path, not a
    # point-by-point guarantee on noisy measurements) — 1.0 remains the best
    # possible value, never meaningfully exceeded in practice.
    return min(1.0, great_circle_km / flown_km)
