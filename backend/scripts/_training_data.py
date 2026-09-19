"""Deduplicated merge of the two training sources (train_anomalie.py,
train_directness.py) — flights_clean.parquet and
flights_historical_features.parquet both cover 2022-06-27, with a real,
non-negligible overlap (empirically verified: ~2200 of flights_clean.parquet's
~3100 flights share an icao24 and a very substantially overlapping time
window — often >80% of the shorter of the two — with a flight from
flights_historical_features.parquet). Merging without deduplicating would
make these flights count twice toward μ/σ (and toward the directness
percentiles), artificially pulling them toward the mean.

The same icao24 can legitimately fly several times the same day (multiple
rotations): detection therefore can't rely on "same icao24" alone, the
[takeoff, landing] time windows also need to overlap significantly — a mere
edge-to-edge contact (an immediately back-to-back rotation) shouldn't count
as a duplicate.

On a duplicate, the flights_historical_features.parquet version is kept
(the source passed last in `input_parquets`) rather than
flights_clean.parquet's: it's the newer, more systematic pipeline (a uniform
30-minute-gap segmentation), and therefore the reference to prefer as future
re-extractions happen (wider geography, new dates...).
"""

import json

import pandas as pd

# Chosen by directly inspecting the edge cases (cf. the module's docstring):
# true duplicate pairs cluster near 100% overlap, distinct back-to-back
# rotations near 0% — 0.5 leaves a wide margin between the two without
# risking merging two genuinely different flights.
MEANINGFUL_OVERLAP_FRACTION = 0.5

_REQUIRED_COLUMNS = ["icao24", "callsign", "typecode", "still_airborne", "waypoints"]


def _time_range(waypoints_json: str) -> tuple[float, float]:
    waypoints = json.loads(waypoints_json)
    timestamps = [w["timestamp"] for w in waypoints]
    return min(timestamps), max(timestamps)


def load_deduplicated_landed_flights(input_parquets: list) -> pd.DataFrame:
    """Loads, filters to landed flights, and merges `input_parquets` (in the
    given order), removing from an earlier source any flight that
    significantly overlaps (cf. MEANINGFUL_OVERLAP_FRACTION) a flight with
    the same icao24 in a later source. Returns the
    icao24/callsign/typecode/still_airborne/waypoints columns, ready for
    feature extraction."""
    frames = []
    for path in input_parquets:
        df = pd.read_parquet(path)
        landed = df[~df["still_airborne"]].reset_index(drop=True)
        print(f"{path.name}: {len(landed)}/{len(df)} landed flights")
        frame = landed[_REQUIRED_COLUMNS].copy()
        ranges = frame["waypoints"].apply(_time_range)
        frame["_takeoff"] = ranges.apply(lambda t: t[0])
        frame["_landing"] = ranges.apply(lambda t: t[1])
        frames.append(frame)

    kept_frames = []
    for i, frame in enumerate(frames):
        later_frames = frames[i + 1 :]
        if not later_frames:
            kept_frames.append(frame)
            continue
        later = pd.concat(later_frames, ignore_index=True)
        later_windows_by_icao = {
            icao: group[["_takeoff", "_landing"]].to_numpy() for icao, group in later.groupby("icao24")
        }

        def is_duplicate_of_later_source(row) -> bool:
            windows = later_windows_by_icao.get(row["icao24"])
            if windows is None:
                return False
            duration = row["_landing"] - row["_takeoff"]
            if duration <= 0:
                return False
            for later_takeoff, later_landing in windows:
                overlap = min(row["_landing"], later_landing) - max(row["_takeoff"], later_takeoff)
                if overlap / duration >= MEANINGFUL_OVERLAP_FRACTION:
                    return True
            return False

        duplicate_mask = frame.apply(is_duplicate_of_later_source, axis=1)
        n_duplicates = int(duplicate_mask.sum())
        if n_duplicates:
            print(f"  -> {n_duplicates} flight(s) already covered by a newer source, dropped (duplicate)")
        kept_frames.append(frame[~duplicate_mask])

    merged = pd.concat(kept_frames, ignore_index=True).drop(columns=["_takeoff", "_landing"])
    print(f"Total landed flights after deduplication (all sources combined): {len(merged)}")
    return merged
