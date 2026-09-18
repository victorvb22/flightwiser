"""Orchestrateur (brief section 6/12) : résolution identifiant -> icao24 ->
trajectoire OpenSky -> modules de models/ (liste itérable) -> assemblage du
JSON de réponse (brief section 8).

Principe d'extensibilité : ajouter un KPI = ajouter un module à `MODULES`,
sans toucher au reste de ce fichier ni aux modules existants.

Flux (brief section 12) : résoudre l'icao24 -> vérifier le cache (clé
icao24 + date UTC du jour, calculée avant tout appel OpenSky pour qu'un hit
n'en déclenche aucun) -> si absent, trajectoire -> statut -> modèles ->
écrire dans le cache -> répondre.
"""

import random

from models import anomalie, directness, ecart_trajectoire
from services import cache, trajectory_cleaning
from services.aircraft_database import get_registration, get_typecode, resolve_icao24_by_registration
from services.airports import find_nearest_airport_info
from services.opensky_client import OpenSkyApi

MODULES = [ecart_trajectoire, anomalie]

# tracks/all est documenté "purement expérimental" par OpenSky (voir
# opensky_client.py) : l'avion le plus proche n'a pas toujours de trajectoire
# disponible. Essayer plusieurs candidats avant d'abandonner.
NEARBY_MAX_CANDIDATES = 10

# Client persistant pour toute la durée de vie du backend — un TokenManager
# réutilisé gère son propre rafraîchissement de token (voir opensky_client.py),
# le recréer à chaque requête forcerait une ré-authentification à chaque fois.
_api = OpenSkyApi.from_settings()


def _resolve_icao24(identifiant: str) -> tuple[str, bool | None] | tuple[None, None]:
    """Immatriculation d'abord (recherche locale, pas d'appel réseau), sinon
    callsign parmi les états live (aucun endpoint de recherche par callsign
    n'existe dans l'API OpenSky classique).

    Retourne (icao24, on_ground) — on_ground vient gratuitement de l'état déjà
    récupéré pendant la résolution par callsign (None si résolu par
    immatriculation, pas encore interrogé). Important : le client OpenSky a
    son propre rate-limit interne par méthode (5s entre deux appels
    authentifiés à get_states) — appeler get_states() une seconde fois dans
    la même requête pour le statut renverrait silencieusement None. Réutiliser
    l'état déjà obtenu ici évite ce problème plutôt que de le contourner."""
    icao24 = resolve_icao24_by_registration(identifiant)
    if icao24:
        return icao24, None
    return _resolve_icao24_by_callsign(identifiant)


def _resolve_icao24_by_callsign(callsign: str) -> tuple[str, bool | None] | tuple[None, None]:
    normalized = callsign.strip().upper()
    states = _api.get_states()
    if states is None:
        return None, None
    for state in states.states:
        if state.callsign and state.callsign.strip().upper() == normalized:
            return state.icao24, state.on_ground
    return None, None


def _fetch_and_clean_trajectory(icao24: str) -> list[dict] | None:
    """Récupère la trajectoire live/récente et applique le même nettoyage que
    preprocess_historical.py (services/trajectory_cleaning.py). La trajectoire
    OpenSky n'a ni vitesse sol ni vitesse verticale — recalculées ici comme
    pour le CSV historique (vitesse verticale en plus, absente à la source)."""
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


def _trim_trajectoire_externe(trajectoire: list[dict]) -> list[dict]:
    """Réduit aux champs du contrat API (brief section 8) — cap/vitesse_verticale/
    au_sol restent internes, utilisés seulement pour nourrir les modèles."""
    return [
        {
            "lat": w["lat"],
            "lon": w["lon"],
            "altitude": w["altitude"],
            "vitesse": w["vitesse"],
            "timestamp": w["timestamp"],
        }
        for w in trajectoire
    ]


def get_or_compute_flight(identifiant: str) -> dict | None:
    icao24, known_on_ground = _resolve_icao24(identifiant)
    if icao24 is None:
        return None
    return _get_or_compute_for_icao24(icao24, identifiant, known_on_ground)


def _get_or_compute_for_icao24(icao24: str, identifiant: str, known_on_ground: bool | None) -> dict | None:
    cache_id = cache.build_cache_id(icao24)
    cached = cache.get_cached_flight(cache_id)
    if cached is not None:
        result = {"identifiant": identifiant, "source": "cache", **cached}
        return {**result, **_flight_metadata(icao24, result)}

    trajectoire = _fetch_and_clean_trajectory(icao24)
    if trajectoire is None:
        return None

    statut = _determine_statut(icao24, known_on_ground)

    ecart_trajectoire_result = None
    anomalie_result = None
    directness_result = None
    # Vol en_vol : destination/trajectoire incomplètes, les trois modèles
    # restent null (décision prise lors de la conception des modèles).
    # Typecode inconnu (icao24 absent de aircraft_database.csv) : même
    # traitement, pas de repli sur "A320" — un typecode fabriqué ferait
    # comparer ce vol à un modèle de performance qui n'est pas le sien,
    # silencieusement, sur les trois scores à la fois (même problème que
    # l'ancien repli d'ecart_trajectoire, mais une couche plus haut : ici
    # aucun typecode réel n'existe du tout, pas seulement un typecode
    # qu'OpenAP ne reconnaît pas).
    typecode = get_typecode(icao24)
    if statut == "atterri" and typecode is not None:
        ecart_trajectoire_result = ecart_trajectoire.compute(trajectoire, typecode)
        anomalie_result = anomalie.compute(trajectoire, typecode, icao24)
        directness_result = directness.compute(trajectoire, typecode)

    result = {
        "identifiant": identifiant,
        "statut": statut,
        "trajectoire": _trim_trajectoire_externe(trajectoire),
        "ecart_trajectoire": ecart_trajectoire_result,
        "anomalie": anomalie_result,
        "directness": directness_result,
        "kpi_bonus": None,
        "source": "direct",
    }
    cache.store_flight(cache_id, result)
    return {**result, **_flight_metadata(icao24, result)}


def _flight_metadata(icao24: str, result: dict) -> dict:
    """Champs d'affichage façon FlightRadar24 (brief section 8 : "n'hésite
    pas à..."). Jamais mis en cache (db/schema.sql / services/cache.py ne
    connaissent que statut/trajectoire/ecart_trajectoire/anomalie/kpi_bonus) :
    dérivables à coût quasi nul de l'icao24 (base aéronefs déjà en mémoire,
    voir services/aircraft_database.py) et de la trajectoire déjà en cache,
    donc recalculés à chaque réponse plutôt que d'exiger une migration du
    schéma de cache pour de la métadonnée statique/redérivable."""
    trajectoire = result["trajectoire"]
    dernier = trajectoire[-1]
    origine_info = find_nearest_airport_info(trajectoire[0]["lat"], trajectoire[0]["lon"])
    destination_info = None
    if result["statut"] == "atterri":
        # Seulement une fois posé : avant l'atterrissage, l'aéroport le plus
        # proche de la position courante n'est pas la destination réelle.
        destination_info = find_nearest_airport_info(dernier["lat"], dernier["lon"])
    return {
        "typecode": get_typecode(icao24),
        "immatriculation": get_registration(icao24),
        "origine": origine_info["icao"] if origine_info else None,
        "origine_ville": origine_info["ville"] if origine_info else None,
        "destination": destination_info["icao"] if destination_info else None,
        "destination_ville": destination_info["ville"] if destination_info else None,
        "altitude_actuelle": dernier["altitude"],
        "vitesse_actuelle": dernier["vitesse"],
    }


def get_random_flight(want_on_ground: bool | None = None) -> dict | None:
    """Vol pris au hasard parmi les états ADS-B actuellement disponibles —
    pratique pour une démo : pas besoin de connaître un identifiant à
    l'avance, et le vol est nécessairement récent puisqu'il vient du flux
    live. `want_on_ground` filtre sur en vol (False) / posé (True) ; None
    pour n'importe lequel."""
    states = _api.get_states()
    if states is None or not states.states:
        return None

    candidates = list(states.states)
    if want_on_ground is not None:
        candidates = [s for s in candidates if s.on_ground == want_on_ground]
    # Pas de pré-filtrage par typecode connu ici : get_typecode() scanne
    # linéairement les ~520k lignes de aircraft_database.csv à chaque appel
    # (pas d'index sur icao24) — appliqué à toute la liste des états live
    # (potentiellement des milliers), ça rendait la recherche aléatoire
    # nettement plus lente pour un gain mineur. Un typecode inconnu laisse
    # simplement les trois scores null une fois le vol récupéré (cf.
    # _get_or_compute_for_icao24), sans fausser quoi que ce soit.
    random.shuffle(candidates)

    for state in candidates[:NEARBY_MAX_CANDIDATES]:
        identifiant_affiche = (state.callsign or state.icao24).strip()
        result = _get_or_compute_for_icao24(state.icao24, identifiant_affiche, state.on_ground)
        if result is not None:
            return result
    return None
