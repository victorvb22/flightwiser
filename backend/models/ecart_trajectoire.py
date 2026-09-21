"""Deviation vs. an OpenAP-simulated optimal trajectory (brief section 11).

Input:  real trajectory (format prepared by preprocess_historical.py —
        timestamp/lat/lon/altitude/cap/vitesse_verticale/au_sol/vitesse)
        + the aircraft's typecode. Unlike the anomaly model, there's neither
        training nor a persisted artifact: OpenAP is a physical/statistical
        simulator called on demand.
Output: the `ecart_trajectoire` portion of the response JSON (brief
        section 8) — score_global + per-phase detail (climb/cruise/
        descent), or None if the trajectory is too short, no phase is
        detectable, or the typecode isn't recognized by OpenAP (this
        currently covers `jet_affaire` and `petit_avion`, helicopters
        included — no fixed-wing performance model exists for them here —
        no fallback simulation, a score compared against a different
        aircraft wouldn't have any physical meaning).

MVP constraint (see the initial implementation plan): for a flight still
`en_vol` (destination unknown), this module must not be called — the
calling route returns `ecart_trajectoire: null` with an explicit status
rather than inventing a destination. Flight distance is therefore never
derived from an airport: it comes directly from the real trajectory's two
endpoints (already complete for a landed flight).
"""

from typing import Any

import numpy as np

from ._ecart_features import (
    MAX_PLAUSIBLE_DISTANCE_KM,
    MIN_CRUISE_KM,
    MIN_POINTS,
    label_real_phases,
    resolve_typecode,
    summarize_real_phase,
    summarize_sim_phase,
    trajectory_distance_km,
)

_PHASE_METRICS = {
    "montee": ["vitesse_verticale_moyenne", "duree_s"],
    "croisiere": ["altitude_moyenne", "vitesse_moyenne"],
    "descente": ["vitesse_verticale_moyenne", "duree_s"],
}

# Labels shown on the frontend (JaugeEcart.tsx) — application text, English
# like the rest of the UI, unlike the internal keys above which stay in
# French throughout the backend code.
_METRIC_LABELS = {
    "vitesse_verticale_moyenne": "avg vertical speed",
    "duree_s": "duration (s)",
    "altitude_moyenne": "avg altitude",
    "vitesse_moyenne": "avg speed",
}


def _relative_deviation(real: float, sim: float, eps: float = 1e-6) -> float:
    return float(min(abs(real - sim) / max(abs(sim), eps), 1.0))


def _phase_result(
    real_summary: dict[str, float] | None, sim_summary: dict[str, float], metrics: list[str]
) -> tuple[float | None, str]:
    if real_summary is None:
        return None, "phase not detected in the real trajectory"
    deviations = [_relative_deviation(real_summary[m], sim_summary[m]) for m in metrics]
    ecart = float(np.mean(deviations))
    detail = "; ".join(f"{_METRIC_LABELS.get(m, m)}: actual {real_summary[m]:.1f} vs optimal {sim_summary[m]:.1f}" for m in metrics)
    return ecart, detail


def compute(trajectoire: list[dict], typecode: str) -> dict[str, Any] | None:
    if len(trajectoire) < MIN_POINTS:
        return None

    distance_km = trajectory_distance_km(trajectoire)
    # np.isfinite also rejects NaN/inf, unlike a direct comparison (NaN > X
    # is always false, so a ceiling alone isn't enough to rule out a NaN
    # distance — observed: a NaN silently passed to OpenAP, whose simulation
    # loop never terminates).
    if not np.isfinite(distance_km) or distance_km <= 0 or distance_km > MAX_PLAUSIBLE_DISTANCE_KM:
        return None

    real_points, real_durations = label_real_phases(trajectoire)
    if not any(real_points.values()):
        return None

    fgen = resolve_typecode(typecode)
    if fgen is None:
        # OpenAP has no performance model to simulate this typecode against
        # — a narrower, purely OpenAP-list question than
        # services.aircraft_category.categorize's category boundary (cf.
        # resolve_typecode's own docstring): always true for petit_avion and
        # helicoptere, and true for most of jet_affaire too, except the
        # couple of business-jet families OpenAP's own ~40-typecode list
        # happens to include. No score rather than a silent, physically
        # meaningless comparison against a substituted A320 (the previous
        # behaviour).
        return None

    sim_climb = fgen.climb(dt=10)
    sim_descent = fgen.descent(dt=10)
    climb_km = sim_climb["s"].max() / 1000
    descent_km = sim_descent["s"].max() / 1000
    cruise_km = max(distance_km - climb_km - descent_km, MIN_CRUISE_KM)
    sim_cruise = fgen.cruise(dt=10, range_cr=cruise_km * 1000)  # meters, cf. _ecart_features' docstring

    sim_summaries = {
        "montee": summarize_sim_phase(sim_climb, "montee"),
        "croisiere": summarize_sim_phase(sim_cruise, "croisiere"),
        "descente": summarize_sim_phase(sim_descent, "descente"),
    }

    par_phase = {}
    ecarts = []
    for phase, metrics in _PHASE_METRICS.items():
        real_summary = summarize_real_phase(real_points[phase], real_durations[phase], phase)
        ecart, detail = _phase_result(real_summary, sim_summaries[phase], metrics)
        par_phase[phase] = {"ecart": ecart, "detail": detail}
        if ecart is not None:
            ecarts.append(ecart)

    return {
        "score_global": float(np.mean(ecarts)),
        "par_phase": par_phase,
    }
