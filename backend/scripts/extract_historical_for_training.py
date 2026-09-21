"""Reconstructs, from the local cache (raw_states/{date}/*.csv.gz), the raw
trajectories of candidate flights within EUROPE_BBOX for the given day — the
sole data source for training (no more intermediate flights_historical
table: the Aggregate view now reads flights_cache, cf. services/cache.py,
not a separate batch import). Uses services/opensky_historical.py (generic
scan/extraction/segmentation, with no notion of a geographic area); no new
download if the cache already covers the requested date's 24 hours.

Doesn't filter on "airport confirmed within the area": that constraint only
makes sense for a "flights from this area" view, not for training, which
only needs usable trajectories wherever they land — dropping it strictly
gives more usable flights with no quality loss (the "air/ground transition"
filter below already guarantees a complete trajectory).

EUROPE_BBOX rather than FRANCE_BBOX (a requested widening): the bbox only
determines *candidacy* (an aircraft seen at least once in the area that
day) — once a candidate, its full trajectory is extracted anywhere in the
world (the same two-pass mechanism as for France), so widening the bbox
introduces no truncation, only more diversity of trajectories/airports/
aircraft types.

Output: data/processed/flights_historical_features.parquet, with only the
columns train_anomalie.py/train_directness.py need (icao24, callsign,
typecode, still_airborne, waypoints JSON) — deduplicated then concatenated
with flights_clean.parquet at training time (cf. scripts/_training_data.py:
the two sources partially overlap, same day of 2022-06-27).

Usage: python scripts/extract_historical_for_training.py --date 2022-06-27
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import truststore

truststore.inject_into_ssl()

BACKEND_DIR = Path(__file__).resolve().parent.parent
# Needed for "python scripts/extract_historical_for_training.py"
# (sys.path[0] is then scripts/, not backend/); a no-op if already run
# via -m.
sys.path.insert(0, str(BACKEND_DIR))

from services.aircraft_database import get_typecode  # noqa: E402
from services.opensky_historical import (  # noqa: E402
    DEFAULT_TYPECODE,
    EUROPE_BBOX,
    build_trajectoire,
    extract_candidate_points,
    scan_candidates,
    segment_flights,
)

# Written outside the repo (same volume as the raw_states/ cache) rather
# than into data/processed/: this derived file weighs several hundred MB,
# and the repo's disk already ran short of space in practice on a first try.
DATA_DIR = Path(os.environ.get("IMPORT_DATA_DIR", r"D:\ML_data\flightwiser"))
OUTPUT_PARQUET = DATA_DIR / "processed" / "flights_historical_features.parquet"

# Written in batches (pyarrow.parquet.ParquetWriter) rather than one final
# pd.DataFrame(rows): at Europe scale, the "waypoints" column (JSON per
# flight) alone weighs several GB of text once every flight has accumulated
# — converting it to an Arrow table in one shot needs roughly double that
# size in peak memory (the Python strings already in memory + the new
# contiguous Arrow buffer), which failed with an ArrowMemoryError even after
# points_by_icao was freed. Writing in batches of WRITE_CHUNK_SIZE bounds
# that peak to a single batch's size, independent of the total flight count.
WRITE_CHUNK_SIZE = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Day already cached in raw_states/, e.g. 2022-06-27")
    args = parser.parse_args()
    date = args.date

    print(f"Pass 1 — spotting candidates (Europe bbox) for {date} ...")
    candidates = scan_candidates(date, EUROPE_BBOX)
    print(f"{len(candidates)} candidate icao24s.")

    print("\nPass 2 — full extraction of candidates ...")
    points_by_icao = extract_candidate_points(date, candidates)

    print("\nReconstructing and writing trajectories in batches ...")
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_landed = 0
    n_no_transition = 0
    n_too_short = 0
    buffer: list[dict] = []
    writer: pq.ParquetWriter | None = None

    def flush(writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PARQUET, table.schema)
        writer.write_table(table)
        buffer.clear()
        return writer

    # Drained as it's processed (.pop rather than .items()): at Europe
    # scale, points_by_icao weighs several GB — no reason to keep it whole
    # once each icao24 has been handled.
    for icao24 in list(points_by_icao.keys()):
        points = points_by_icao.pop(icao24)
        for segment in segment_flights(points):
            au_sol_values = {p[6] for p in segment}
            if True not in au_sol_values or False not in au_sol_values:
                n_no_transition += 1
                continue
            trajectoire = build_trajectoire(segment)
            if trajectoire is None:
                n_too_short += 1
                continue
            still_airborne = trajectoire[-1]["au_sol"] is False
            buffer.append(
                {
                    "icao24": icao24,
                    "callsign": None,
                    "typecode": get_typecode(icao24) or DEFAULT_TYPECODE,
                    "still_airborne": still_airborne,
                    "waypoints": json.dumps(trajectoire),
                }
            )
            n_rows += 1
            if not still_airborne:
                n_landed += 1
            if len(buffer) >= WRITE_CHUNK_SIZE:
                writer = flush(writer)
    del points_by_icao
    if buffer:
        writer = flush(writer)
    if writer is not None:
        writer.close()

    print(f"{n_rows} flights reconstructed ({n_no_transition} without an air/ground transition, {n_too_short} too short).")
    print(f"Of which {n_landed} landed (usable for training the anomaly model).")
    print(f"\nWrote {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
