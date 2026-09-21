"""pipeline.py -- the orchestrator tying the pool, cache, and models
together. Only the one behaviour that needed a regression test for a real
bug (see below); the rest of this module is already covered end to end by
tests/test_api.py."""

import pipeline
import services.cache as cache


def test_a_cache_hit_still_touches_calcule_le(test_pool, monkeypatch):
    # Real bug, reported and reproduced: a flight searched again hours
    # after its first search didn't show up near the top of the Aggregate
    # view's "Searched" column, because _serve_pool_entry's cache-HIT path
    # (this test) never called store_flight again -- calcule_le stayed
    # frozen at the first computation, even though the flight had genuinely
    # just been searched a second time. touch_flight() is the fix; this
    # confirms the cache-hit path actually calls it, without needing a real
    # Supabase round trip (block_network already forces every real request
    # to fail, so get_cached_flight is monkeypatched directly to simulate
    # the hit).
    touched_ids = []
    monkeypatch.setattr(cache, "touch_flight", lambda cache_id: touched_ids.append(cache_id))
    monkeypatch.setattr(
        cache,
        "get_cached_flight",
        lambda cache_id: {
            "statut": "atterri",
            # A single point, not empty -- _flight_metadata (called on
            # every response, cache hit or not) indexes the trajectory's
            # last point for altitude/speed/destination.
            "trajectoire": [{"lat": 48.0, "lon": 2.0, "altitude": 0.0, "vitesse": 0.0, "timestamp": 0}],
            "ecart_trajectoire": None,
            "anomalie": None,
            "directness": None,
            "kpi_bonus": None,
        },
    )

    result = pipeline.get_or_compute_flight("RYR2PY")

    assert result is not None
    assert result["source"] == "cache"
    assert touched_ids == [cache.build_cache_id("4ca56b")]
