"""Extraction de la feature "trajet direct" (brief section 11, extension) :
distance grand-cercle entre les extrémités réelles de la trajectoire,
divisée par la distance effectivement parcourue le long du chemin — un
signal qu'aucun des deux modèles existants ne voit. Le modèle d'anomalie
(models/_anomalie_features.py) ne regarde que des features résumées
scalaires (vitesse/altitude/taux de montée), aveugle à la FORME du trajet ;
le modèle d'écart (models/_ecart_features.py) compare le profil altitude/
vitesse par phase à une simulation OpenAP, pas le routage latéral. Un vol
qui tourne en rond, dévie largement ou subit un vectorage ATC prolongé peut
donc passer inaperçu des deux, sans que sa vitesse ou son altitude sortent
jamais de l'ordinaire.

Ratio borné par construction (une ligne droite est le plus court chemin
possible sur une sphère) : proche de 1 pour un trajet direct, plus bas pour
un détour. Partagé entre l'entraînement (scripts/train_directness.py) et le
service (models/directness.py) pour éviter tout écart train/serve, comme
_anomalie_features.py pour son propre modèle.
"""

import numpy as np

MIN_POINTS = 5
# En dessous de cette distance grand-cercle, le ratio est dominé par le
# bruit de position ADS-B plutôt que par un vrai détour (un aller-retour
# local ou un touch-and-go n'a pas vraiment de "trajet direct" à mesurer).
MIN_GREAT_CIRCLE_KM = 5.0
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def extract_route_directness(waypoints: list[dict]) -> float | None:
    """None si trop peu de points, ou si la distance grand-cercle est trop
    courte pour un ratio fiable (cf. MIN_GREAT_CIRCLE_KM) — pas une erreur
    en soi, juste un vol pour lequel ce signal particulier n'est pas
    exploitable (les autres modèles restent indépendants de celui-ci)."""
    if len(waypoints) < MIN_POINTS:
        return None

    lats = [w["lat"] for w in waypoints]
    lons = [w["lon"] for w in waypoints]

    great_circle_km = _haversine_km(lats[0], lons[0], lats[-1], lons[-1])
    if great_circle_km < MIN_GREAT_CIRCLE_KM:
        return None

    flown_km = sum(_haversine_km(lats[i], lons[i], lats[i + 1], lons[i + 1]) for i in range(len(lats) - 1))
    if flown_km <= 0:
        return None

    # Plafonné à 1.0 : le bruit de position ADS-B peut pousser le ratio
    # marginalement au-dessus (une ligne droite est le plus court chemin
    # *théorique*, pas une garantie point par point sur des mesures
    # bruitées) — 1.0 reste la meilleure valeur possible, jamais dépassée
    # en pratique de façon significative.
    return min(1.0, great_circle_km / flown_km)
