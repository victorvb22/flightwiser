"""Orchestrator (brief section 6/12): identifier -> flight from the pool ->
models (models/) -> API response assembly (brief section 8).

The deployed backend can no longer query OpenSky live (blocked at the
network level on Render/Railway/Cloudflare Workers, empirically verified —
cf. the README's Limitations section): instead of resolving an identifier
to an icao24 and then fetching its trajectory live, we draw from a pool of
real flights (services/flight_pool.py, fed offline by
scripts/collect_opensky_pool.py + scripts/build_flight_pool.py from a
machine where OpenSky is reachable). Only the "go fetch the trajectory from
OpenSky" step is replaced — the models are computed here, at the moment the
flight is served, exactly as they would have been for a flight freshly
fetched live (not frozen to whatever the code looked like on the day it was
collected, cf. services/flight_pool.py).

The rest of the flow is unchanged (brief section 12): check the cache (key:
icao24 + today's UTC date) -> if absent, serve from the pool -> write to the
cache -> respond. A flight already served (from the pool or a targeted
search) is marked in pool_served so a random draw never picks it twice
(services/pool_tracking.py — persistent, unlike the daily cache, since
"already drawn" must never be forgotten)."""

from models import anomalie, anomaly_type, directness, ecart_trajectoire
from services import cache, flight_pool, pool_tracking
from services.aircraft_category import categorize
from services.aircraft_database import get_registration, get_typecode
from services.airports import find_nearest_airport_info


def get_or_compute_flight(identifiant: str) -> dict | None:
    entry = flight_pool.find_by_identifiant(identifiant)
    if entry is None:
        return None
    return _serve_pool_entry(entry, entry["identifiant"])


def get_random_flight(want_on_ground: bool | None = None) -> dict | None:
    """A random flight from the pool, never served before. `want_on_ground`
    filters on airborne (False) / landed (True); None for either."""
    served = pool_tracking.get_served_icao24s()
    entry = flight_pool.draw_random(want_on_ground, served)
    if entry is None:
        return None
    return _serve_pool_entry(entry, entry["identifiant"])


def _serve_pool_entry(entry: dict, identifiant_affiche: str) -> dict:
    icao24 = entry["_icao24"]
    cache_id = cache.build_cache_id(icao24)
    cached = cache.get_cached_flight(cache_id)
    if cached is not None:
        result = {"identifiant": identifiant_affiche, "source": "cache", **cached}
        return {**result, **_flight_metadata(icao24, result)}

    trajectoire = entry["trajectoire"]
    statut = entry["statut"]
    typecode = get_typecode(icao24)

    ecart_trajectoire_result = None
    anomalie_result = None
    directness_result = None
    # Airborne flight: destination/trajectory incomplete, all three models
    # stay null. Typecode absent from the categorization table (icao24
    # unknown to aircraft_database.csv, or typecode unidentified in
    # services/aircraft_category.py): same treatment — no score rather than
    # a guessed default classification that would compare this flight to
    # the wrong population.
    categorie = categorize(typecode) if typecode is not None else None
    if statut == "atterri" and categorie is not None:
        # OpenAP doesn't model rotary wings — no point attempting the
        # simulation for a helicopter (ecart_trajectoire.compute would reject
        # it anyway via resolve_typecode, but better to make it explicit here
        # than leave it as a coincidence).
        if categorie != "helicoptere":
            ecart_trajectoire_result = ecart_trajectoire.compute(trajectoire, typecode)
        anomalie_result = anomalie.compute(trajectoire, typecode)
        directness_result = directness.compute(trajectoire, typecode)
        # models/anomaly_type.py is a separate ML project (cf. its
        # docstring), trained only on airliner profiles — go-around/holding/
        # emergency descent don't mean anything for a business jet or a
        # small aircraft, hence the restriction to this one category.
        # Always computed when applicable (not only when the score flags an
        # anomaly) — it's up to the frontend to decide when to show it,
        # rather than duplicating a threshold here that only exists on the
        # UI side.
        if categorie == "avion_ligne" and anomalie_result is not None:
            anomalie_result["type_anomalie"] = anomaly_type.compute(trajectoire)

    result = {
        "identifiant": identifiant_affiche,
        "statut": statut,
        "trajectoire": _trim_trajectoire_externe(trajectoire),
        "ecart_trajectoire": ecart_trajectoire_result,
        "anomalie": anomalie_result,
        "directness": directness_result,
        "kpi_bonus": None,
        "source": "vedette",
    }
    cache.store_flight(cache_id, result)
    pool_tracking.mark_served(icao24)
    return {**result, **_flight_metadata(icao24, result)}


def _trim_trajectoire_externe(trajectoire: list[dict]) -> list[dict]:
    """Trims to the API contract's fields (brief section 8) — cap/
    vitesse_verticale/au_sol stay internal, used only to feed the models."""
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


def _flight_metadata(icao24: str, result: dict) -> dict:
    """FlightRadar24-style display fields (brief section 8: "feel free
    to..."). Never cached (db/schema.sql / services/cache.py only know about
    statut/trajectoire/ecart_trajectoire/anomalie/kpi_bonus): derivable at
    near-zero cost from the icao24 (aircraft database already in memory, see
    services/aircraft_database.py — a purely local lookup, no dependency on
    OpenSky) and from the trajectory already in the cache, so recomputed on
    every response instead of requiring a cache schema migration for static/
    re-derivable metadata."""
    trajectoire = result["trajectoire"]
    dernier = trajectoire[-1]
    origine_info = find_nearest_airport_info(trajectoire[0]["lat"], trajectoire[0]["lon"])
    destination_info = None
    if result["statut"] == "atterri":
        # Only once landed: before touchdown, the nearest airport to the
        # current position isn't the real destination.
        destination_info = find_nearest_airport_info(dernier["lat"], dernier["lon"])
    typecode = get_typecode(icao24)
    return {
        "typecode": typecode,
        # Same categorie a cache hit's own scores were computed against
        # (cf. _serve_pool_entry) — re-derived here rather than threaded
        # through the cache row, same reasoning as the rest of this
        # metadata: a cheap local lookup, not worth a schema migration.
        "categorie": categorize(typecode) if typecode is not None else None,
        "immatriculation": get_registration(icao24),
        "origine": origine_info["icao"] if origine_info else None,
        "origine_ville": origine_info["ville"] if origine_info else None,
        "destination": destination_info["icao"] if destination_info else None,
        "destination_ville": destination_info["ville"] if destination_info else None,
        "altitude_actuelle": dernier["altitude"],
        "vitesse_actuelle": dernier["vitesse"],
    }
