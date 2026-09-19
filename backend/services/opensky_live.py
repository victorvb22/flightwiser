"""icao24 + trajectory resolution via the **live** OpenSky API — no longer
used by the deployed backend (pipeline.py now serves a pre-fetched pool, cf.
its docstring and the README's Limitations section: OpenSky blocks requests
coming from Render/Railway/Cloudflare Workers at the network level).

This module only remains for scripts/collect_opensky_pool.py, which has to
run from a machine where OpenSky is actually reachable (locally, not on a
shared cloud host) to feed that pool offline — by reusing these same
functions, a collected record is exactly in the shape a live call would have
produced back when the backend still made them.
"""

from services import trajectory_cleaning
from services.opensky_client import OpenSkyApi

# Persistent client for the script's whole lifetime — a reused TokenManager
# handles its own token refresh (see opensky_client.py); recreating it on
# every call would force a re-authentication each time.
_api = OpenSkyApi.from_settings()


def _fetch_and_clean_trajectory(icao24: str) -> list[dict] | None:
    """Fetches the live/recent trajectory and applies the same cleanup as
    preprocess_historical.py (services/trajectory_cleaning.py). OpenSky's
    trajectory has neither ground speed nor vertical speed — recomputed here
    just like for the historical CSV (plus vertical speed, absent from the
    source there too)."""
    track = _api.get_track_by_aircraft(icao24)
    if track is None or not track.path:
        return None

    raw_points = [
        (wp.time, wp.latitude, wp.longitude, wp.baro_altitude, wp.true_track, None, wp.on_ground)
        for wp in track.path
    ]
    points = trajectory_cleaning.drop_missing_position(raw_points)
    points = trajectory_cleaning.dedupe_waypoints(points)
    points = trajectory_cleaning.drop_position_teleports(points)
    points = trajectory_cleaning.clip_altitude(points)

    vertical_rates = trajectory_cleaning.compute_vertical_rates(points)
    points = [(p[0], p[1], p[2], p[3], p[4], vr, p[6]) for p, vr in zip(points, vertical_rates)]
    points = trajectory_cleaning.clip_vertical_rate(points)
    speeds = trajectory_cleaning.compute_speeds(points)

    return [
        {
            "timestamp": p[0],
            "lat": p[1],
            "lon": p[2],
            "altitude": p[3],
            "cap": p[4],
            "vitesse_verticale": p[5],
            "au_sol": p[6],
            "vitesse": speeds[i],
        }
        for i, p in enumerate(points)
    ]


def _determine_statut(icao24: str, known_on_ground: bool | None) -> str:
    if known_on_ground is not None:
        return "atterri" if known_on_ground else "en_vol"
    states = _api.get_states(icao24=icao24)
    if states is not None and states.states and states.states[0].on_ground is False:
        return "en_vol"
    return "atterri"
