"""One-off recovery script: marks every icao24 already in flights_cache as
served in pool_served, so a random draw stops being able to re-pick a
flight someone has already searched or been shown, directly.

Needed once because pool_served (db/schema.sql) didn't exist in Supabase
for a while after it was introduced (services/pool_tracking.py's writes
have always failed silently on a missing table, by design — cf. its own
docstring on why a Supabase error is never allowed to fail a search) —
every flight served during that window went uncounted. Once the table has
actually been created (paste db/schema.sql into the Supabase SQL editor),
run this once to reconcile the two: without it, the pool would still look
"fully available" for every flight already sitting in the visible search
history, and a random draw could immediately re-serve one.

Safe to rerun: mark_served itself is an upsert (on_conflict=icao24), and
this script only ever adds rows, never removes any.

Usage: python scripts/backfill_pool_served.py
"""

import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services import cache, pool_tracking  # noqa: E402


def main() -> None:
    rows = cache.get_all_cached_flights()
    icao24s = {row["id"].partition("_")[0] for row in rows}
    print(f"{len(rows)} cached flight(s), {len(icao24s)} unique icao24(s) to mark served.")

    already_served = pool_tracking.get_served_icao24s()
    to_mark = icao24s - already_served
    print(f"{len(already_served)} already marked served, {len(to_mark)} to add.")

    for icao24 in to_mark:
        pool_tracking.mark_served(icao24)

    print(f"Done: {len(to_mark)} icao24(s) marked served.")


if __name__ == "__main__":
    main()
