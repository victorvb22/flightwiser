"""Utilitaires partagés pour le modèle d'écart trajectoire (brief section 11) :
résolution du type d'appareil auprès d'OpenAP, distance orthodromique de la
trajectoire réelle, et segmentation de cette trajectoire en phases
(montée/croisière/descente) via openap.phase.FlightPhase.

Piège vérifié empiriquement (la docstring d'OpenAP dit le contraire) :
`FlightGenerator.cruise(range_cr=...)` attend des MÈTRES, pas des kilomètres —
la valeur par défaut interne (WRAP, en km) est convertie en mètres avant
d'alimenter le même accumulateur `s`, mais une valeur passée explicitement ne
l'est pas.
"""

import numpy as np
from openap.gen import FlightGenerator
from openap.phase import FlightPhase

MIN_POINTS = 5
MIN_CRUISE_KM = 10.0  # plancher pour un vol trop court pour une vraie croisière (simplification MVP)
# Plafond physique : le vol commercial non-stop le plus long fait ~17 000 km ;
# au-delà, la distance calculée à partir des extrémités de la trajectoire est
# forcément un artefact (point aberrant non filtré en amont), pas un vrai
# vol — mieux vaut renoncer que de lancer une simulation OpenAP avec une
# distance absurde (observé : provoque un MemoryError dans FlightGenerator.cruise).
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
    """Générateur OpenAP pour ce typecode, ou None si OpenAP ne le reconnaît
    pas (même table de synonymes que services.aircraft_category.categorize —
    c'est d'ailleurs exactement la frontière qui définit "petit_avion" :
    aucun typecode de cette catégorie n'a de modèle de performance ici).

    Ne retombe plus sur A320 : un vol "petit_avion" comparé à une simulation
    A320 (hélicoptère y compris — OpenAP ne modélise pas la voilure tournante,
    pas de vol stationnaire, pas de montée/descente classique) produisait un
    score numérique d'apparence normale sans aucun sens physique. Le typecode
    inconnu au niveau icao24 (pas de match dans aircraft_database.csv) est un
    problème différent, réglé en amont dans le prétraitement — celui-ci
    suppose un typecode déjà résolu."""
    try:
        return FlightGenerator(ac=typecode.lower(), use_synonym=True)
    except ValueError:
        return None


def trajectory_distance_km(trajectoire: list[dict]) -> float:
    """Distance orthodromique entre le premier et le dernier point de la
    trajectoire réelle — sert d'approximation de la distance totale du vol
    (OpenAP ne connaît pas les aéroports)."""
    first, last = trajectoire[0], trajectoire[-1]
    return float(haversine_km(first["lat"], first["lon"], last["lat"], last["lon"]))


def label_real_phases(trajectoire: list[dict]) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Segmente la trajectoire réelle en montée/croisière/descente via
    openap.phase.FlightPhase. Points avec altitude ou vitesse_verticale
    manquante exclus au préalable (même logique que le modèle d'anomalie).

    Retourne (points_par_phase, duree_par_phase). La durée est la somme des
    intervalles entre points CONSÉCUTIFS partageant la même phase — pas
    simplement (dernier - premier) de la liste de points, qui surestimerait
    largement la durée si la phase n'est pas continue dans le temps (ex. un
    point isolé mal classé loin des autres, ou un step-climb)."""
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
    """Statistiques résumées d'une phase réelle, en unités SI (mètres, m/s) —
    mêmes unités que les colonnes brutes `h`/`v`/`vs` d'OpenAP, donc aucune
    conversion nécessaire pour la comparaison."""
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
    """Statistiques résumées d'une phase simulée, à partir des colonnes
    brutes h (m), v (m/s), vs (m/s) — déjà en SI, pas de conversion."""
    if phase == "croisiere":
        return {
            "altitude_moyenne": float(df["h"].mean()),
            "vitesse_moyenne": float(df["v"].mean()),
        }
    return {
        "vitesse_verticale_moyenne": float(df["vs"].mean()),
        "duree_s": float(df["t"].max() - df["t"].min()),
    }
