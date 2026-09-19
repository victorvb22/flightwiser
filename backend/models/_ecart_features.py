"""Shared utilities for the trajectory-deviation model (brief section 11):
resolving the aircraft type against OpenAP, great-circle distance of the
real trajectory, and segmenting that trajectory into phases (climb/cruise/
descent) via openap.phase.FlightPhase.

Empirically verified gotcha (OpenAP's own docstring says otherwise):
`FlightGenerator.cruise(range_cr=...)` expects METERS, not kilometers — the
internal default value (WRAP, in km) is converted to meters before feeding
the same `s` accumulator, but a value passed explicitly isn't.
"""

import numpy as np
from openap.gen import FlightGenerator
from openap.phase import FlightPhase

MIN_POINTS = 5
MIN_CRUISE_KM = 10.0  # floor for a flight too short to have a real cruise phase (MVP simplification)
# Physical ceiling: the longest non-stop commercial flight is ~17,000 km;
# beyond that, a distance computed from the trajectory's endpoints is
# necessarily an artifact (an outlier point not filtered upstream), not a
# real flight — better to give up than launch an OpenAP simulation with an
# absurd distance (observed: causes a MemoryError in FlightGenerator.cruise).
MAX_PLAUSIBLE_DISTANCE_KM = 20_000.0

M_TO_FT = 3.280839895
MS_TO_KT = 1.9438444924
MS_TO_FPM = 196.8503937

EARTH_RADIUS_KM = 6371.0

_PHASE_MAP = {"CL": "montee", "CR": "croisiere", "LVL": "croisiere", "DE": "descente"}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def resolve_typecode(typecode: str) -> FlightGenerator | None:
    """OpenAP generator for this typecode, or None if OpenAP doesn't
    recognize it (same synonym table OpenAP itself uses — this overlaps with,
    but is a narrower question than, services.aircraft_category.categorize:
    "can this specific typecode be simulated," not "what category is it").

    No longer falls back to A320: a flight compared against an A320
    simulation it has nothing to do with (helicopters included — OpenAP
    doesn't model rotary wings, no hover, no classic climb/descent) produced
    a normal-looking numeric score with no physical meaning at all. An
    icao24 with an unknown typecode (no match in aircraft_database.csv) is a
    different problem, handled upstream in preprocessing — this function
    assumes a typecode has already been resolved."""
    try:
        return FlightGenerator(ac=typecode.lower(), use_synonym=True)
    except ValueError:
        return None


def trajectory_distance_km(trajectoire: list[dict]) -> float:
    """Great-circle distance between the real trajectory's first and last
    point — used as an approximation of the flight's total distance (OpenAP
    doesn't know about airports)."""
    first, last = trajectoire[0], trajectoire[-1]
    return float(haversine_km(first["lat"], first["lon"], last["lat"], last["lon"]))


def label_real_phases(trajectoire: list[dict]) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Segments the real trajectory into climb/cruise/descent via
    openap.phase.FlightPhase. Points with missing altitude or
    vitesse_verticale are excluded beforehand (same logic as the anomaly
    model).

    Returns (points_by_phase, duration_by_phase). Duration is the sum of
    intervals between CONSECUTIVE points sharing the same phase — not simply
    (last - first) of the point list, which would badly overestimate
    duration if the phase isn't continuous in time (e.g. a single misclassified
    point far from the others, or a step-climb)."""
    usable = [w for w in trajectoire if w.get("altitude") is not None and w.get("vitesse_verticale") is not None]
    points_by_phase: dict[str, list[dict]] = {"montee": [], "croisiere": [], "descente": []}
    duration_by_phase: dict[str, float] = {"montee": 0.0, "croisiere": 0.0, "descente": 0.0}
    if len(usable) < MIN_POINTS:
        return points_by_phase, duration_by_phase

    t0 = usable[0]["timestamp"]
    ts = np.array([w["timestamp"] - t0 for w in usable], dtype=float)
    alt_ft = np.array([w["altitude"] * M_TO_FT for w in usable])
    spd_kt = np.array([w["vitesse"] * MS_TO_KT for w in usable])
    roc_fpm = np.array([w["vitesse_verticale"] * MS_TO_FPM for w in usable])

    fp = FlightPhase()
    fp.set_trajectory(ts, alt_ft, spd_kt, roc_fpm)
    labels = fp.phaselabel()
    phases = [_PHASE_MAP.get(label) for label in labels]

    for i, (point, phase) in enumerate(zip(usable, phases)):
        if phase is None:
            continue
        points_by_phase[phase].append(point)
        if i > 0 and phases[i - 1] == phase:
            duration_by_phase[phase] += point["timestamp"] - usable[i - 1]["timestamp"]

    return points_by_phase, duration_by_phase


def summarize_real_phase(points: list[dict], duree_s: float, phase: str) -> dict[str, float] | None:
    """Summary statistics for a real phase, in SI units (meters, m/s) — same
    units as OpenAP's own `h`/`v`/`vs` raw columns, so no conversion is
    needed for comparison."""
    if not points:
        return None
    if phase == "croisiere":
        return {
            "altitude_moyenne": float(np.mean([p["altitude"] for p in points])),
            "vitesse_moyenne": float(np.mean([p["vitesse"] for p in points])),
        }
    return {
        "vitesse_verticale_moyenne": float(np.mean([p["vitesse_verticale"] for p in points])),
        "duree_s": duree_s,
    }


def summarize_sim_phase(df, phase: str) -> dict[str, float]:
    """Summary statistics for a simulated phase, from the raw columns h (m),
    v (m/s), vs (m/s) — already in SI, no conversion."""
    if phase == "croisiere":
        return {
            "altitude_moyenne": float(df["h"].mean()),
            "vitesse_moyenne": float(df["v"].mean()),
        }
    return {
        "vitesse_verticale_moyenne": float(df["vs"].mean()),
        "duree_s": float(df["t"].max() - df["t"].min()),
    }
