"""Écart vs trajectoire optimale simulée par OpenAP (brief section 11).

Input  : trajectoire réelle (format préparé par preprocess_historical.py —
          timestamp/lat/lon/altitude/cap/vitesse_verticale/au_sol/vitesse)
          + typecode de l'appareil. Contrairement au modèle d'anomalie, il
          n'y a ni entraînement ni artefact persisté : OpenAP est un
          simulateur physique/statistique appelé à la demande.
Output : la portion `ecart_trajectoire` du JSON de réponse (brief section 8)
          — score_global + détail par phase (montée/croisière/descente),
          ou None si la trajectoire est trop courte, qu'aucune phase n'est
          détectable, ou que le typecode n'est pas reconnu par OpenAP (même
          frontière que services.aircraft_category.categorize : ces vols
          sont "petit_avion" précisément parce qu'aucun modèle de
          performance fixed-wing n'existe pour eux ici — pas de simulation
          de repli, un score comparé à un appareil différent n'aurait pas de
          sens physique).

Contrainte MVP (voir plan de mise en place initiale) : pour un vol encore
`en_vol` (destination inconnue), ce module ne doit pas être appelé — la route
appelante renvoie `ecart_trajectoire: null` avec un statut explicite plutôt
que d'inventer une destination. La distance de vol n'est donc jamais déduite
d'un aéroport : elle vient directement des deux extrémités de la trajectoire
réelle (déjà complète pour un vol atterri).
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

# Libellés affichés côté frontend (JaugeEcart.tsx) — texte applicatif, donc en
# anglais comme le reste de l'interface, contrairement aux clés internes
# ci-dessus qui restent en français dans tout le code backend.
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
    # np.isfinite rejette aussi NaN/inf, contrairement à une comparaison
    # directe (NaN > X est toujours faux, donc un plafond seul ne suffit
    # pas à écarter une distance NaN — observé : NaN silencieusement passé
    # à OpenAP, dont la boucle de simulation ne se termine jamais).
    if not np.isfinite(distance_km) or distance_km <= 0 or distance_km > MAX_PLAUSIBLE_DISTANCE_KM:
        return None

    real_points, real_durations = label_real_phases(trajectoire)
    if not any(real_points.values()):
        return None

    fgen = resolve_typecode(typecode)
    if fgen is None:
        # petit_avion (OpenAP doesn't recognize this typecode, by the same
        # boundary services.aircraft_category.categorize uses) — no
        # fixed-wing performance model exists to compare against, so no
        # score rather than a silent, physically meaningless comparison
        # against a substituted A320 (the previous behaviour). This applies
        # to the whole category, not just helicopters: any typecode OpenAP
        # doesn't recognize has the same problem.
        return None

    sim_climb = fgen.climb(dt=10)
    sim_descent = fgen.descent(dt=10)
    climb_km = sim_climb["s"].max() / 1000
    descent_km = sim_descent["s"].max() / 1000
    cruise_km = max(distance_km - climb_km - descent_km, MIN_CRUISE_KM)
    sim_cruise = fgen.cruise(dt=10, range_cr=cruise_km * 1000)  # mètres, cf. docstring de _ecart_features

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
