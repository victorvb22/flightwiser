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

Fixed-size pool (POOL_CAP), landed-biased: the raw OpenSky snapshot is
overwhelmingly airborne (broadcasting on the ground is brief), so growing
the pool by simply appending every capture would keep diluting the scarce
landed share forever. Airborne flights are no longer added to the pool at
all once running: a landed capture is added, and if that pushes the pool
over POOL_CAP, a random EXISTING airborne entry is evicted to make room --
so the pool stays pinned at POOL_CAP flights total, with the landed share
growing at airborne's expense until LANDED_TARGET is reached (at which
point POOL_CAP - LANDED_TARGET airborne flights necessarily remain).
Written as a full atomic rewrite (temp file + rename) on every landed
capture rather than an in-place edit: a landed capture is rare enough
(single digits per hour) that rewriting the whole file each time costs
nothing, and this is no less crash-safe than pure appending -- the real
file is only ever replaced once a complete write has succeeded, so a kill
mid-write leaves the previous, still-valid file in place, not a corrupt one.

Usage: python scripts/collect_opensky_pool.py
Stop at any time with Ctrl+C — the pool is already durable after every
landed capture (see above), so there is nothing left to flush on exit.
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
DELAY_BETWEEN_ATTEMPTS_SECONDS = 10
# Total pool size, pinned -- cf. the module docstring. Landed flights only
# ever displace airborne ones from here on, never grow the total.
POOL_CAP = 3000
# The landed subset is the scarce one (cf. the on_ground-priority candidate
# selection below) -- stopping on this rather than on a total flight count
# is what actually caps how long an unattended run needs to go. Once
# reached, POOL_CAP - LANDED_TARGET airborne flights remain by construction.
LANDED_TARGET = 2000

# icao24 -> record, the pool's live in-memory state (loaded from OUTPUT_FILE
# at startup, kept in sync with it via _write_pool_atomic after every landed
# capture) -- a dict, not just counts, because eviction needs to pick an
# actual existing airborne entry, not merely track how many there are.
pool: dict[str, dict] = {}
seen_icao24: set[str] = set()
total_collected = 0
total_attempts = 0
start_time = time.time()
last_report = start_time
last_quota_warning = 0.0
stop_requested = False


def _handle_stop_signal(signum, frame):
    global stop_requested
    stop_requested = True
    print("\nStop requested - shutting down cleanly (pool is already durable, nothing to flush)...")


# SIGINT (Ctrl+C) and SIGTERM (the default kill signal, and closing the
# terminal in some cases).
signal.signal(signal.SIGINT, _handle_stop_signal)
signal.signal(signal.SIGTERM, _handle_stop_signal)


def _load_existing_pool() -> dict[str, dict]:
    """Reads whatever's already in OUTPUT_FILE from earlier runs, keyed by
    icao24 -- tolerates a corrupted line the same way build_flight_pool.py's
    own read does, since a rewrite interrupted at exactly the wrong moment
    (before this script's own atomic-rewrite discipline existed) could
    still have left one behind in an older file."""
    if not OUTPUT_FILE.exists():
        return {}
    loaded: dict[str, dict] = {}
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            loaded[entry["_icao24"]] = entry
    return loaded


def _pool_stats(p: dict[str, dict]) -> tuple[int, int, int]:
    """(total, airborne, landed)."""
    landed = sum(1 for v in p.values() if v.get("statut") == "atterri")
    return len(p), len(p) - landed, landed


def _write_pool_atomic(p: dict[str, dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_FILE.with_name(OUTPUT_FILE.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in p.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(OUTPUT_FILE)


def _looks_like_credits_exhausted(exc: Exception) -> bool:
    """OpenSky's own auth server (auth.opensky-network.org) becomes
    unresolvable once the day's request quota is used up (observed directly
    -- a plain DNS/connection failure, not an HTTP error, so it never goes
    through _get_json/last_status_code at all). Every other network hiccup
    this script already tolerates by retrying (get_states() itself timing
    out, a transient DNS blip unrelated to OpenSky) keeps retrying as
    before -- only this specific one means every further attempt today
    would fail the exact same way, so retrying is pointless rather than
    just costly."""
    text = str(exc)
    if "auth.opensky-network.org" not in text:
        return False
    # Several observed wordings for the same underlying DNS-resolution
    # failure, not just urllib3's own NameResolutionError repr: matching
    # only that one exact class name was fragile across
    # requests/urllib3/Python versions and OSes (e.g. "getaddrinfo failed"
    # is the raw socket.gaierror message on Windows). Any of these next to
    # the auth host is the same condition.
    return any(marker in text for marker in ("NameResolutionError", "getaddrinfo failed", "Failed to resolve", "Name or service not known"))


def _warn_if_quota_exhausted() -> None:
    """OpenSky returns 429 once the daily request quota is used up --
    previously indistinguishable from an aircraft simply having no track
    (both came back as `_get_json` returning None, cf. OpenSkyApi's own
    comment). last_status_code (opensky_client.py) is set on every request
    regardless of outcome, so this can now tell the two apart. Throttled to
    once a minute -- once the quota is gone, every single attempt for the
    rest of the run would otherwise print this."""
    global last_quota_warning
    if _api.last_status_code == 429 and time.time() - last_quota_warning > 60:
        print(
            "  [!] OpenSky returned 429 (Too Many Requests) -- the daily "
            "request quota looks exhausted. Collection will keep running "
            "and retrying, but further attempts will likely keep failing "
            "until the quota resets."
        )
        last_quota_warning = time.time()


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
    total, airborne, landed = _pool_stats(pool)
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"this run: {total_collected} flights fetched ({total_attempts} attempts, "
        f"{len(seen_icao24)} distinct aircraft) in {elapsed_min:.0f} min "
        f"- ~{rate:.1f} flights/min"
    )
    print(f"  pool total: {total} flight(s) - {airborne} airborne, {landed} landed ({landed}/{LANDED_TARGET} of target)")


def main() -> None:
    global total_collected, total_attempts, last_report, stop_requested, pool, seen_icao24

    # stdout is block-buffered (not line-buffered) as soon as it isn't
    # attached to an interactive terminal — without this, a redirect to a
    # file (`> log.txt`) or a kill could make reports that were never
    # flushed disappear.
    sys.stdout.reconfigure(line_buffering=True)

    pool = _load_existing_pool()
    seen_icao24 = set(pool.keys())
    total, airborne, landed = _pool_stats(pool)
    print(f"Collection started. Output: {OUTPUT_FILE}")
    print(f"Existing pool on disk: {total} flight(s) - {airborne} airborne, {landed} landed ({landed}/{LANDED_TARGET} of target)")
    if landed >= LANDED_TARGET:
        print("Landed target already reached — nothing to do.")
        return
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
                if _looks_like_credits_exhausted(exc):
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[!] {now_str}: OpenSky's auth server is unreachable (auth.opensky-network.org can't be resolved) -- looks like today's request quota is exhausted.")
                    print(f"    Stopping now, at {now_str}. Restart the script once the quota has reset (daily, roughly 24h after collection started today).")
                    stop_requested = True
                    continue
                print(f"  get_states() failed: {exc}")
                time.sleep(30)
                continue
            if states is None or not states.states:
                _warn_if_quota_exhausted()
                time.sleep(5)
                continue
            states_cache = list(states.states)
            states_fetched_at = now

        # OpenSky's /states/all has no server-side on_ground filter (only
        # time/icao24/bbox) -- landed aircraft are a small, unpredictable
        # slice of each snapshot (broadcasting on the ground is brief), so
        # picking them out here, before the random top-N cut, is the only
        # way to not have them drowned out by the much larger airborne
        # majority.
        landed_states = [s for s in states_cache if s.on_ground]
        airborne_states = [s for s in states_cache if not s.on_ground]
        random.shuffle(landed_states)
        random.shuffle(airborne_states)
        candidates = landed_states[:NEARBY_MAX_CANDIDATES] + airborne_states[: max(0, NEARBY_MAX_CANDIDATES - len(landed_states))]
        for state in candidates:
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
            if record is None:
                _warn_if_quota_exhausted()
            elif record["statut"] != "atterri":
                # Airborne: never added to the pool (cf. module docstring) --
                # fetched only to confirm its status and mark it seen, so it
                # isn't re-attempted again this run.
                total_collected += 1
            else:
                total_collected += 1
                pool[state.icao24] = record
                if len(pool) > POOL_CAP:
                    # .get(), not v["statut"]: _load_existing_pool only guards
                    # against a syntactically-invalid JSON line, not one
                    # that's valid JSON but missing this key (e.g. a stray
                    # line from an older/different tool) -- that shouldn't
                    # crash a multi-hour unattended run over one bad line,
                    # same defensive read _pool_stats already uses.
                    airborne_icao24s = [k for k, v in pool.items() if v.get("statut") != "atterri"]
                    if airborne_icao24s:
                        del pool[random.choice(airborne_icao24s)]
                _write_pool_atomic(pool)
                _, _, landed_count = _pool_stats(pool)
                print(f"  landed: {identifiant_affiche} ({landed_count}/{LANDED_TARGET}) -- pool now {len(pool)} total")
                if landed_count >= LANDED_TARGET:
                    stop_requested = True
                    print(f"\nLanded target reached: {landed_count} landed flight(s) in the pool. Stopping.")
                    break

            time.sleep(DELAY_BETWEEN_ATTEMPTS_SECONDS)

        if time.time() - last_report > REPORT_EVERY_SECONDS:
            report()
            last_report = time.time()

    report()
    print("Clean shutdown complete.")


if __name__ == "__main__":
    main()
