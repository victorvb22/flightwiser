"""services/flight_pool.py -- only the one behaviour that needed a
regression test: a callsign reused by two different aircraft in the pool
(real-world reuse across separate collection sessions, not a collection
bug) used to make the second one permanently unreachable by search --
find_by_identifiant returned whichever entry a plain linear scan hit
first, an accident of file order, not a meaningful choice."""

import services.flight_pool as flight_pool


def test_a_reused_callsign_resolves_to_the_most_recently_collected_aircraft(test_pool, monkeypatch):
    # Same identifiant ("RYR2PY", already in the fixture pool under icao24
    # 4ca56b) reused by a different icao24, collected later.
    pool = flight_pool._get_pool()
    older_entry = next(e for e in pool if e["identifiant"] == "RYR2PY")
    newer_entry = {
        **older_entry,
        "_icao24": "999999",
        "_collected_at": "2099-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(flight_pool, "_pool", [*pool, newer_entry])
    monkeypatch.setattr(flight_pool, "_by_identifiant", None)
    monkeypatch.setattr(flight_pool, "_by_icao24", None)

    result = flight_pool.find_by_identifiant("RYR2PY")

    assert result is not None
    assert result["_icao24"] == "999999"


def test_each_aircraft_is_still_reachable_by_its_own_icao24(test_pool):
    # The older/shadowed entry from the scenario above must still resolve
    # by icao24, even though its callsign now points elsewhere.
    result = flight_pool.find_by_icao24("4ca56b")
    assert result is not None
    assert result["identifiant"] == "RYR2PY"
