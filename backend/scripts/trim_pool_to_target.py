"""One-off trim of the raw OpenSky collection file (scripts/collect_opensky_pool.py's
own output) down to fixed landed/airborne targets, by randomly dropping
entries from each category independently -- a straight random sample within
each category, not a priority order between them. Used whenever the built
flight_pool.jsonl's in-memory footprint (services/flight_pool.py, loaded
whole into the Render backend's process) needs to shrink to stay inside the
instance's memory ceiling; the specific targets are a deliberate per-incident
call, not a fixed policy, so both categories are explicit arguments rather
than one total plus an airborne-only default.

Usage: python scripts/trim_pool_to_target.py <landed_target> <airborne_target>
"""

import json
import random
import sys
from pathlib import Path

RAW_PATH = Path(r"D:\ML_data\flightwiser\opensky_fetch\pool.jsonl")


def main() -> None:
    if not RAW_PATH.exists():
        print(f"Not found: {RAW_PATH} (has the collection script run yet?)")
        return

    if len(sys.argv) != 3:
        print("Usage: python scripts/trim_pool_to_target.py <landed_target> <airborne_target>")
        return
    landed_target = int(sys.argv[1])
    airborne_target = int(sys.argv[2])

    entries: list[dict] = []
    with open(RAW_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    landed = [e for e in entries if e.get("statut") == "atterri"]
    airborne = [e for e in entries if e.get("statut") != "atterri"]
    print(f"Before: {len(entries)} flight(s) - {len(airborne)} airborne, {len(landed)} landed")

    if landed_target > len(landed) or airborne_target > len(airborne):
        print(f"Can't reach {landed_target} landed / {airborne_target} airborne -- only {len(landed)} landed / {len(airborne)} airborne available.")
        return

    random.shuffle(landed)
    random.shuffle(airborne)
    kept = landed[:landed_target] + airborne[:airborne_target]

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Removed {len(landed) - landed_target} random landed and {len(airborne) - airborne_target} random airborne flight(s).")
    print(f"After: {len(kept)} flight(s) - {airborne_target} airborne, {landed_target} landed")


if __name__ == "__main__":
    main()
