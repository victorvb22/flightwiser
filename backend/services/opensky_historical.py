"""Extracting full trajectories from OpenSky's weekly samples (worldwide
states, one .csv.gz file per hour, cached on disk — cf. download_hour) —
generic utilities, with no dependency on a particular geographic area or use
case (batch import into a table, generating a training dataset, etc.),
shared by scripts/extract_historical_for_training.py.

A flight departing from one area can land anywhere in the world (and vice
versa) — a single-pass geographic filter (keeping only points inside a given
bounding box) would truncate the trajectory to the portion flying over that
area. Hence two local passes (the download itself only happens once per
hour, cached on disk):

  1. locate the icao24s that passed through the given bbox that day (scan_candidates);
  2. for those icao24s only, extract ALL of their positions for the day,
     anywhere in the world, for the full trajectory
     (extract_candidate_points).

Then segmentation into individual flights by time gap (segment_flights) and
cleanup/reconstruction into a usable trajectory (build_trajectoire, the same
cleanup as services/trajectory_cleaning.py used by the rest of the project).
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

FLIGHT_GAP_SECONDS = 30 * 60  # time gap beyond which a new flight is considered to have started
MIN_POINTS_PER_FLIGHT = 5
DEFAULT_TYPECODE = "A320"

# Wide bounding box (lat_min, lat_max, lon_min, lon_max) used both for
# extraction candidacy (scripts/extract_historical_for_training.py) and, at
# runtime, to flag a randomly drawn flight as outside the training zone
# (models/anomalie.py) — a single shared definition, not two copies that
# could drift apart. Continental: Iceland/Scandinavia in the north down to
# the Canaries/Cyprus in the south, the Azores/Ireland in the west to the
# Urals in the east.
EUROPE_BBOX = (34.0, 71.0, -25.0, 45.0)

STATE_COLUMNS = ["time", "icao24", "lat", "lon", "baroaltitude", "heading", "vertrate", "onground"]
CHUNKSIZE = 2_000_000


def download_hour(date: str, hour: int) -> Path:
    """Downloads and streams-extracts a given hour's .csv.gz (the .tar itself
    is never kept). Skipped if already cached locally."""
    dest_dir = RAW_STATES_DIR / date
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{hour:02d}.csv.gz"
    if dest.exists():
        return dest

    url = STATES_URL_TEMPLATE.format(date=date, hour=hour)
    print(f"Downloading {url} ...")
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
    """Pass 1: icao24s seen at least once inside `bbox`
    (lat_min, lat_max, lon_min, lon_max) that day."""
    candidates: set[str] = set()
    lat_min, lat_max, lon_min, lon_max = bbox
    for hour in range(24):
        path = download_hour(date, hour)
        for chunk in pd.read_csv(path, usecols=["icao24", "lat", "lon"], dtype={"icao24": str}, chunksize=CHUNKSIZE, compression="gzip"):
            mask = chunk["lat"].between(lat_min, lat_max) & chunk["lon"].between(lon_min, lon_max)
            candidates.update(chunk.loc[mask, "icao24"].dropna().unique())
        print(f"  hour {hour:02d}: {len(candidates)} candidates so far")
    return candidates


def extract_candidate_points(date: str, candidates: set[str]) -> dict[str, list[tuple]]:
    """Pass 2: every position for the day for the candidate icao24s, anywhere
    in the world — the project's standard tuple format (timestamp, lat, lon,
    altitude, cap, vitesse_verticale, au_sol)."""
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
    """pandas represents an empty CSV field as NaN (float), not None —
    unlike the rest of the project (e.g. preprocess_historical.py, which
    substitutes raw `nan` with None before parsing even happens). Without
    this conversion, `drop_missing_position` (which tests `is not None`)
    lets a NaN through, `trajectory_distance_km` returns NaN, and the
    deviation model's `NaN > MAX_PLAUSIBLE_DISTANCE_KM` comparison is always
    false — the distance guard is then silently bypassed and OpenAP receives
    a NaN `range_cr`, whose loop never terminates (observed: MemoryError
    after several minutes)."""
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
    """True if at least one trajectory point falls inside `bbox` — the same
    criterion as extraction candidacy (scan_candidates), applied here after
    the fact to an already-built trajectory rather than to raw points. Used
    to flag a flight as outside the training zone, not to filter/truncate
    anything."""
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
