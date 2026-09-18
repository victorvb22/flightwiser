"""Nettoyage de trajectoire ADS-B — dédoublonnage, suppression des artefacts
(téléportations de position, altitude/vitesse verticale hors plage physique),
recalcul de vitesse. Partagé entre le prétraitement batch
(scripts/preprocess_historical.py) et le pipeline live (pipeline.py), pour
éviter de dupliquer une logique qu'on a déjà dû déboguer trois fois sur ce
projet (téléportations, altitude à 38 000 m, vitesse verticale à 90 m/s —
tous des artefacts ADS-B réels rencontrés sur le dataset historique).

Toutes les fonctions opèrent sur des tuples 7 éléments
(timestamp, lat, lon, altitude, cap, vitesse_verticale, au_sol) — le format
brut commun aux deux sources de données (CSV historique et trajectoire live
OpenSky, une fois cette dernière convertie dans ce même format).
"""

import numpy as np
import pandas as pd

SPEED_SMOOTHING_WINDOW = 5
# Un point dont la vitesse implicite dépasse ce seuil des deux côtés (segment
# entrant ET sortant) est un point ADS-B corrompu ("téléportation" ponctuelle,
# ex. un saut France -> Pacifique en 10s observé sur ce dataset), pas un vrai
# mouvement — on le supprime plutôt que de fausser la trajectoire.
TELEPORT_SPEED_THRESHOLD_MS = 2000.0
# Plafond physique défensif sur la vitesse finale (lissée) : aucun avion civil
# subsonique ne dépasse ~450 m/s (875 kt) au sol, même avec un jet-stream
# exceptionnel. Absorbe le bruit ADS-B résiduel qui survit à la suppression
# des téléportations.
MAX_PLAUSIBLE_SPEED_MS = 450.0
# Plage d'altitude barométrique physiquement plausible pour un vol commercial
# (bornes larges : plafond ~45 000 ft, plancher sous le niveau de la mer pour
# les aéroports concernés). Certains points du dataset historique dépassent
# 38 000 m (record du monde d'altitude en vol habité, clairement un artefact
# ADS-B), ce qui fausserait les statistiques d'altitude en aval.
MIN_PLAUSIBLE_ALTITUDE_M = -500.0
MAX_PLAUSIBLE_ALTITUDE_M = 13716.0
# Vitesse verticale plausible pour un avion commercial (±30 m/s ~ ±5900 ft/min,
# bien au-delà des taux de montée/descente réels). Même artefact ADS-B que
# l'altitude : quelques points du dataset historique dépassent 90 m/s.
MAX_PLAUSIBLE_VERTICAL_RATE_MS = 30.0

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance haversine en km entre deux points (scalaires ou tableaux numpy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def drop_missing_position(points: list[tuple]) -> list[tuple]:
    """Supprime les points sans position (lat/lon manquants) — inutilisables
    pour la trajectoire, la déduplication ou le calcul de vitesse."""
    return [p for p in points if p[1] is not None and p[2] is not None]


def clip_altitude(points: list[tuple]) -> list[tuple]:
    """Remplace par None les altitudes hors plage physiquement plausible
    plutôt que de les plafonner à la borne, ce qui créerait un pic artificiel
    de valeurs identiques. Traité comme une altitude manquante, déjà géré en
    aval."""
    return [
        p if p[3] is None or MIN_PLAUSIBLE_ALTITUDE_M <= p[3] <= MAX_PLAUSIBLE_ALTITUDE_M
        else (p[0], p[1], p[2], None, p[4], p[5], p[6])
        for p in points
    ]


def clip_vertical_rate(points: list[tuple]) -> list[tuple]:
    """Même logique que clip_altitude, pour la vitesse verticale."""
    return [
        p if p[5] is None or abs(p[5]) <= MAX_PLAUSIBLE_VERTICAL_RATE_MS
        else (p[0], p[1], p[2], p[3], p[4], None, p[6])
        for p in points
    ]


def dedupe_waypoints(points: list[tuple]) -> list[tuple]:
    """Supprime les points consécutifs à coordonnées identiques ou dont le
    timestamp n'avance pas — sources réelles de division par ~0 dans le
    calcul de vitesse."""
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
    """Supprime les points ADS-B corrompus : une position ponctuelle fausse
    implique une vitesse absurde à la fois pour y arriver et pour en repartir
    (contrairement à un vrai segment rapide, qui n'est incohérent que dans un
    sens s'il touche un point valide). Répète jusqu'à stabilité car retirer un
    point peut révéler un autre point désormais isolé entre deux voisins trop
    éloignés (rare, mais observé)."""
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
    """Vitesse sol (m/s) par point via distance haversine / delta de temps,
    lissée par moyenne glissante puis plafonnée à MAX_PLAUSIBLE_SPEED_MS
    (bruit ADS-B résiduel). Le premier point reprend la vitesse du second
    segment (pas de segment précédent disponible)."""
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
    """Vitesse verticale (m/s) par point via delta d'altitude / delta de
    temps, lissée puis plafonnée à MAX_PLAUSIBLE_VERTICAL_RATE_MS — miroir de
    compute_speeds, mais pour une trajectoire qui n'a pas déjà de vitesse
    verticale mesurée (ex. suivi live OpenSky, contrairement au CSV
    historique qui l'a directement). None si l'altitude manque pour le point
    ou son voisin."""
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
