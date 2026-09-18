"""Extraction de trajectoires complètes depuis les échantillons hebdomadaires
OpenSky (états mondiaux, un fichier .csv.gz par heure, mis en cache sur
disque — cf. download_hour) — utilitaires génériques, sans dépendance à une
zone géographique ou un usage particulier (import batch vers une table,
génération d'un dataset d'entraînement, etc.), partagés par
scripts/extract_historical_for_training.py.

Un vol au départ d'une zone peut atterrir n'importe où dans le monde (et
inversement) — un filtrage géographique en une seule passe (ne garder que
les points dans une bounding box donnée) tronquerait la trajectoire à la
portion survolant cette zone. D'où deux passes locales (le téléchargement,
lui, ne se fait qu'une fois par heure, mis en cache sur disque) :

  1. repérer les icao24 passés par la bbox donnée ce jour-là (scan_candidates) ;
  2. pour ces seuls icao24, extraire TOUTES leurs positions de la journée,
     n'importe où dans le monde, pour la trajectoire complète
     (extract_candidate_points).

Puis segmentation en vols individuels par écart temporel (segment_flights) et
nettoyage/reconstruction d'une trajectoire exploitable (build_trajectoire,
même nettoyage que services/trajectory_cleaning.py utilisé par le reste du
projet).
"""

import os
import tarfile
from pathlib import Path

import pandas as pd
import requests

from services import trajectory_cleaning

DATA_DIR = Path(os.environ.get("IMPORT_DATA_DIR", r"D:\ML_data\flightwiser"))
RAW_STATES_DIR = DATA_DIR / "raw_states"

STATES_URL_TEMPLATE = "https://s3.opensky-network.org/data-samples/states/{date}/{hour:02d}/states_{date}-{hour:02d}.csv.tar"

FLIGHT_GAP_SECONDS = 30 * 60  # écart temporel au-delà duquel on considère un nouveau vol
MIN_POINTS_PER_FLIGHT = 5
DEFAULT_TYPECODE = "A320"

# Bounding box large (lat_min, lat_max, lon_min, lon_max) utilisée à la fois
# pour la candidature d'extraction (scripts/extract_historical_for_training.py)
# et, à l'exécution, pour signaler qu'un vol tiré au hasard est hors de la
# zone d'entraînement (models/anomalie.py) — une seule définition partagée,
# pas deux copies qui pourraient diverger. Continentale : Islande/Scandinavie
# au nord jusqu'aux Canaries/Chypre au sud, Açores/Irlande à l'ouest jusqu'à
# l'Oural à l'est.
EUROPE_BBOX = (34.0, 71.0, -25.0, 45.0)

STATE_COLUMNS = ["time", "icao24", "lat", "lon", "baroaltitude", "heading", "vertrate", "onground"]
CHUNKSIZE = 2_000_000


def download_hour(date: str, hour: int) -> Path:
    """Télécharge et extrait en streaming le .csv.gz d'une heure donnée (le
    .tar lui-même n'est jamais gardé). Passe si déjà en cache localement."""
    dest_dir = RAW_STATES_DIR / date
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{hour:02d}.csv.gz"
    if dest.exists():
        return dest

    url = STATES_URL_TEMPLATE.format(date=date, hour=hour)
    print(f"Téléchargement {url} ...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    response.raw.decode_content = False

    tmp = dest.with_suffix(".tmp")
    with tarfile.open(fileobj=response.raw, mode="r|") as tf:
        member = tf.next()
        src = tf.extractfile(member)
        with open(tmp, "wb") as out:
            while chunk := src.read(1024 * 1024):
                out.write(chunk)
    tmp.rename(dest)
    return dest


def scan_candidates(date: str, bbox: tuple[float, float, float, float]) -> set[str]:
    """Passe 1 : icao24 vus au moins une fois dans `bbox`
    (lat_min, lat_max, lon_min, lon_max) ce jour-là."""
    candidates: set[str] = set()
    lat_min, lat_max, lon_min, lon_max = bbox
    for hour in range(24):
        path = download_hour(date, hour)
        for chunk in pd.read_csv(path, usecols=["icao24", "lat", "lon"], dtype={"icao24": str}, chunksize=CHUNKSIZE, compression="gzip"):
            mask = chunk["lat"].between(lat_min, lat_max) & chunk["lon"].between(lon_min, lon_max)
            candidates.update(chunk.loc[mask, "icao24"].dropna().unique())
        print(f"  heure {hour:02d} : {len(candidates)} candidats cumulés")
    return candidates


def extract_candidate_points(date: str, candidates: set[str]) -> dict[str, list[tuple]]:
    """Passe 2 : toutes les positions de la journée pour les icao24 candidats,
    n'importe où dans le monde — format tuple standard du projet
    (timestamp, lat, lon, altitude, cap, vitesse_verticale, au_sol)."""
    points: dict[str, list[tuple]] = {icao: [] for icao in candidates}
    for hour in range(24):
        path = RAW_STATES_DIR / date / f"{hour:02d}.csv.gz"
        for chunk in pd.read_csv(path, usecols=STATE_COLUMNS, dtype={"icao24": str}, chunksize=CHUNKSIZE, compression="gzip"):
            matched = chunk[chunk["icao24"].isin(candidates)]
            for row in matched.itertuples(index=False):
                points[row.icao24].append(
                    (
                        row.time,
                        _nan_to_none(row.lat),
                        _nan_to_none(row.lon),
                        _nan_to_none(row.baroaltitude),
                        _nan_to_none(row.heading),
                        _nan_to_none(row.vertrate),
                        row.onground,
                    )
                )
    return points


def _nan_to_none(value):
    """pandas représente un champ CSV vide par NaN (float), pas None —
    contrairement au reste du projet (ex. preprocess_historical.py, qui
    substitue les `nan` bruts par None avant même le parsing). Sans cette
    conversion, `drop_missing_position` (qui teste `is not None`) laisse
    passer un NaN, `trajectory_distance_km` renvoie NaN, et la comparaison
    `NaN > MAX_PLAUSIBLE_DISTANCE_KM` du modèle d'écart est toujours fausse
    — le garde-fou de distance est alors silencieusement contourné et
    OpenAP reçoit un `range_cr` NaN, dont la boucle ne se termine jamais
    (observé : MemoryError après plusieurs minutes)."""
    return None if pd.isna(value) else value


def segment_flights(points: list[tuple]) -> list[list[tuple]]:
    points = sorted(points, key=lambda p: p[0])
    segments = [[points[0]]]
    for p in points[1:]:
        if p[0] - segments[-1][-1][0] > FLIGHT_GAP_SECONDS:
            segments.append([p])
        else:
            segments[-1].append(p)
    return segments


def flight_touches_bbox(trajectoire: list[dict], bbox: tuple[float, float, float, float]) -> bool:
    """True si au moins un point de la trajectoire tombe dans `bbox` — même
    critère que la candidature d'extraction (scan_candidates), appliqué ici
    a posteriori à une trajectoire déjà construite plutôt qu'à des points
    bruts. Utilisé pour signaler un vol hors de la zone d'entraînement, pas
    pour filtrer/tronquer quoi que ce soit."""
    lat_min, lat_max, lon_min, lon_max = bbox
    return any(lat_min <= p["lat"] <= lat_max and lon_min <= p["lon"] <= lon_max for p in trajectoire)


def build_trajectoire(segment: list[tuple]) -> list[dict] | None:
    pts = trajectory_cleaning.drop_missing_position(segment)
    pts = trajectory_cleaning.dedupe_waypoints(pts)
    pts = trajectory_cleaning.drop_position_teleports(pts)
    pts = trajectory_cleaning.clip_altitude(pts)
    pts = trajectory_cleaning.clip_vertical_rate(pts)
    if len(pts) < MIN_POINTS_PER_FLIGHT:
        return None
    speeds = trajectory_cleaning.compute_speeds(pts)
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
        for i, p in enumerate(pts)
    ]
