"""Local collection of a pool of real flights via the live OpenSky API —
feeds offline the pool the deployed backend now serves instead of OpenSky
calls, which it can no longer make itself (blocked at the network level on
Render/Railway/Cloudflare Workers, cf. the README's Limitations section).
Must run from a machine where OpenSky is actually reachable (verified:
locally, not on a shared cloud host) — not meant to be deployed itself. Once
a collection session is done, see scripts/build_flight_pool.py to refresh
the pool bundled with the backend from the file written here.

This script ONLY replaces the "go fetch the trajectory from OpenSky" step —
neither status nor trajectory are computed any differently than a live call
would have. The scores (anomalie/ecart/directness) are NOT computed here:
they depend on the models' code, which evolves, while this collection can
only be rerun at the cost of a new OpenSky quota. Computing them once and
for all here would have frozen them to whatever the code looked like on
collection day — pipeline.py computes them on demand, at the moment a
flight is drawn from the pool, exactly as it would have for a flight
freshly fetched live. The trajectory is therefore stored here in its full
INTERNAL shape (with vitesse_verticale/cap/au_sol, not just the API
contract's fields) — without that, this deferred computation would be
impossible.

Deliberately WITHOUT going through services/cache.py: writing several
thousand speculative draws into flights_cache would pollute the search
history shown in the Aggregate view with flights nobody actually searched
for — this local file stays a "cold" reserve, kept separate.

Written as JSON Lines (one flight per line), not Parquet: Parquet needs a
footer written cleanly at the end to stay readable, so an interruption
partway through can corrupt the ENTIRE file. In JSON Lines, every line
already written by the time of the interruption stays valid — never more
than the last unflushed batch is lost.

Usage: python scripts/collect_opensky_pool.py
Stop at any time with Ctrl+C — flushes whatever's in memory before exiting,
no loss.
"""

import json
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.opensky_live import _api, _determine_statut, _fetch_and_clean_trajectory  # noqa: E402

OUTPUT_DIR = Path(r"D:\ML_data\flightwiser\opensky_fetch")
OUTPUT_FILE = OUTPUT_DIR / "pool.jsonl"

FLUSH_EVERY = 10
REPORT_EVERY_SECONDS = 5 * 60
# One get_states() call returns EVERY aircraft currently tracked worldwide
# in a single request — no need to repeat it often, we just draw a new
# random sample from it each round.
STATES_REFRESH_SECONDS = 30
NEARBY_MAX_CANDIDATES = 10
# Deliberate spacing between track attempts (get_track_by_aircraft has no
# internal throttle, unlike get_states) — avoids burning through the daily
# OpenSky credit quota too fast. The library doesn't distinguish a 429
# (quota reached) from a simple absence of a track for this aircraft (both
# come back as an empty result) — a cautious default spacing rather than
# detecting it precisely.
DELAY_BETWEEN_ATTEMPTS_SECONDS = 60

buffer: list[dict] = []
seen_icao24: set[str] = set()
total_collected = 0
total_attempts = 0
start_time = time.time()
last_report = start_time
stop_requested = False


def _handle_stop_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\nStop requested - flushing then shutting down cleanly...")


# SIGINT (Ctrl+C) and SIGTERM (the default kill signal, and closing the
# terminal in some cases) — a somewhat abrupt stop without also catching
# SIGTERM could have skipped the final flush and lost the last batch still
# in memory.
signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)


def flush() -> None:
    global buffer
    if not buffer:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        for record in buffer:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  -> {len(buffer)} flight(s) written to {OUTPUT_FILE}")
    buffer = []


def collect_flight(icao24: str, identifiant_affiche: str, on_ground_hint: bool | None) -> dict | None:
    """Fetches trajectory + status, exactly as a live call would have
    (services/opensky_live.py) — no score computed here, cf. the module's
    docstring. Trajectory stored in its full internal shape (not trimmed to
    the API contract): pipeline.py needs it in full to compute the models
    when this flight is served."""
    trajectoire = _fetch_and_clean_trajectory(icao24)
    if trajectoire is None:
        return None

    statut = _determine_statut(icao24, on_ground_hint)

    return {
        "identifiant": identifiant_affiche,
        "statut": statut,
        "trajectoire": trajectoire,
        "_icao24": icao24,
        "_collected_at": datetime.now(timezone.utc).isoformat(),
    }


def report() -> None:
    elapsed_min = (time.time() - start_time) / 60
    rate = total_collected / elapsed_min if elapsed_min > 0 else 0
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"{total_collected} flights collected ({total_attempts} attempts, "
        f"{len(seen_icao24)} distinct aircraft) in {elapsed_min:.0f} min "
        f"- ~{rate:.1f} flights/min"
    )


def main() -> None:
    global total_collected, total_attempts, last_report

    # stdout is block-buffered (not line-buffered) as soon as it isn't
    # attached to an interactive terminal — without this, a redirect to a
    # file (`> log.txt`) or a kill could make reports that were never
    # flushed disappear.
    sys.stdout.reconfigure(line_buffering=True)

    print(f"Collection started. Output: {OUTPUT_FILE}")
    print("Ctrl+C to stop cleanly at any time.\n")

    states_cache: list | None = None
    states_fetched_at = 0.0

    while not stop_requested:
        now = time.time()
        if states_cache is None or now - states_fetched_at > STATES_REFRESH_SECONDS:
            # Not guarded originally: a transient network/SSL error here
            # (observed in practice: HTTPS interception by an antivirus)
            # used to crash the whole script instead of just missing one
            # cycle - given the goal (running unattended for hours), a
            # one-off failure should never be fatal.
            try:
                states = _api.get_states()
            except Exception as exc:
                print(f"  get_states() failed: {exc}")
                time.sleep(30)
                continue
            if states is None or not states.states:
                time.sleep(5)
                continue
            states_cache = list(states.states)
            states_fetched_at = now

        random.shuffle(states_cache)
        for state in states_cache[:NEARBY_MAX_CANDIDATES]:
            if stop_requested:
                break
            if state.icao24 in seen_icao24:
                continue

            total_attempts += 1
            identifiant_affiche = (state.callsign or state.icao24).strip()
            try:
                record = collect_flight(state.icao24, identifiant_affiche, state.on_ground)
            except Exception as exc:
                print(f"  {state.icao24} failed: {exc}")
                time.sleep(DELAY_BETWEEN_ATTEMPTS_SECONDS)
                continue

            seen_icao24.add(state.icao24)
            if record is not None:
                buffer.append(record)
                total_collected += 1
                if len(buffer) >= FLUSH_EVERY:
                    flush()

            time.sleep(DELAY_BETWEEN_ATTEMPTS_SECONDS)

        if time.time() - last_report > REPORT_EVERY_SECONDS:
            report()
            last_report = time.time()

    flush()
    report()
    print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
