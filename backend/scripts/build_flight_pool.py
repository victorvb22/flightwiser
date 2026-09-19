"""Regenerates data/reference/flight_pool.jsonl (bundled with the
deployment, read by services/flight_pool.py) from the raw file produced by
scripts/collect_opensky_pool.py, run separately on a machine where OpenSky
is reachable (see its docstring) — rerun this here, locally, every time a
new collection session has run and the pool served by the backend needs
refreshing.

Deduplicates by icao24 (a collection session already avoids duplicates in
memory via seen_icao24, but nothing prevents the same aircraft from
reappearing across sessions) and silently skips a corrupted JSON line rather
than failing the whole rebuild — a file written continuously over hours
should normally never contain one, but a rebuild has no reason to depend on
that guarantee to succeed.

Usage: python scripts/build_flight_pool.py
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.flight_pool import POOL_PATH  # noqa: E402

RAW_PATH = Path(r"D:\ML_data\flightwiser\opensky_fetch\pool.jsonl")


def main() -> None:
    if not RAW_PATH.exists():
        print(f"Not found: {RAW_PATH} (has the collection script run yet?)")
        return

    by_icao24: dict[str, dict] = {}
    skipped = 0
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            by_icao24[entry["_icao24"]] = entry

    POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_PATH, "w", encoding="utf-8") as f:
        for entry in by_icao24.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"{POOL_PATH}: {len(by_icao24)} flights ({skipped} line(s) skipped)")


if __name__ == "__main__":
    main()
