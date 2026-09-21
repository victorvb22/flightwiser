"""services/cache.py -- the cache-key format and the graceful-degradation
behaviour every read/write relies on (cf. the module's own docstring: a
network/HTTP error must be a soft miss/no-op, never a failed search)."""

import re
from datetime import datetime, timezone

from services import cache


def test_build_cache_id_is_icao24_plus_todays_utc_date():
    cache_id = cache.build_cache_id("4CA56B")
    today = datetime.now(timezone.utc).date().isoformat()
    assert cache_id == f"4CA56B_{today}"
    assert re.fullmatch(r".+_\d{4}-\d{2}-\d{2}", cache_id)


def test_get_cached_flight_is_a_silent_miss_on_a_network_error():
    # block_network (conftest.py, autouse) makes the underlying
    # requests.get raise -- this must come back as None, not propagate.
    assert cache.get_cached_flight("does-not-matter") is None


def test_get_all_cached_flights_is_a_silent_empty_list_on_a_network_error():
    assert cache.get_all_cached_flights() == []


def test_touch_flight_does_not_raise_on_a_network_error():
    # Same best-effort policy as store_flight.
    cache.touch_flight("does-not-matter")


def test_store_flight_does_not_raise_on_a_network_error():
    # Must never surface a write failure to the caller (the user's response
    # is already computed) -- this call simply must not raise.
    cache.store_flight(
        "does-not-matter",
        {
            "statut": "atterri",
            "trajectoire": [],
            "ecart_trajectoire": None,
            "anomalie": None,
            "directness": None,
            "kpi_bonus": None,
            "source": "vedette",
        },
    )
