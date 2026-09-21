"""Shared test setup.

Two things every test in this suite relies on:

1. Required env vars are set here, before any app module is imported --
   config.py reads them at class-definition time (module import time), so
   setting them inside a fixture would be too late for anything imported at
   collection time (e.g. a `from main import app` at the top of a test
   file). conftest.py is always imported first by pytest, which is what
   makes this reliable.
2. `block_network` (autouse) makes every outbound `requests.get`/`.post`
   call raise immediately. services/cache.py and services/pool_tracking.py
   already treat a network/HTTP error as a soft failure (cache miss on
   read, no-op on write, cf. their own docstrings) -- so this doesn't need
   a mock Supabase, it just exercises the exact same "Supabase is
   unreachable" path production already has to tolerate. It also means the
   test suite can never accidentally write to (or read stale data from) a
   real Supabase project, whatever SUPABASE_URL happens to be set to.
"""

import os

os.environ.setdefault("SUPABASE_URL", "http://supabase.invalid")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:5173")

from pathlib import Path

import pytest
import requests

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise requests.RequestException("real network calls are blocked in tests")

    monkeypatch.setattr(requests, "get", _blocked)
    monkeypatch.setattr(requests, "post", _blocked)


@pytest.fixture
def test_pool(monkeypatch):
    """Points services.flight_pool at a small, frozen 3-flight fixture
    (fixtures/flight_pool_sample.jsonl -- two real flights copied verbatim
    out of data/reference/flight_pool.jsonl, plus one synthetic
    unidentified aircraft) instead of the real pool, so API tests stay
    correct and fast regardless of how many flights the user has collected
    since. Resets the module's load-once cache before and after (monkeypatch
    auto-reverts the attribute itself, but the cached list needs its own
    reset since nothing else would reload it)."""
    import services.flight_pool as flight_pool

    monkeypatch.setattr(flight_pool, "POOL_PATH", FIXTURES_DIR / "flight_pool_sample.jsonl")
    monkeypatch.setattr(flight_pool, "_pool", None)
    yield
    flight_pool._pool = None
