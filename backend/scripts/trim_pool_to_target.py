"""One-off trim of the raw OpenSky collection file (scripts/collect_opensky_pool.py's
own output) down to a fixed total size, by randomly dropping airborne
entries only -- landed ones are never touched, cf. that script's own
POOL_CAP/LANDED_TARGET: the whole point of the pool going forward is to grow
the landed share at airborne's expense, at a fixed total, and this is the
one-time correction needed to actually land on that fixed total before the
capped/evicting collection loop takes over.

Usage: python scripts/trim_pool_to_target.py [target_total]
Default target_total: 3000 (collect_opensky_pool.py's own POOL_CAP).
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

    target_total = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

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

    to_remove = len(entries) - target_total
    if to_remove <= 0:
        print(f"Already at or below {target_total} -- nothing to remove.")
        return
    if to_remove > len(airborne):
        print(f"Only {len(airborne)} airborne flight(s) available, can't remove {to_remove} without touching landed ones -- aborting.")
        return

    random.shuffle(airborne)
    kept_airborne = airborne[to_remove:]
    removed = airborne[:to_remove]

    kept = landed + kept_airborne
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Removed {len(removed)} random airborne flight(s).")
    print(f"After: {len(kept)} flight(s) - {len(kept_airborne)} airborne, {len(landed)} landed")


if __name__ == "__main__":
    main()
