"""services/aircraft_database.py -- get_typecode/get_registration used to
scan the whole (~94 MB) DataFrame on every call; now backed by the
DataFrame's own icao24 index (set_index, built once) -- a first fix used a
plain Python dict instead, which measurably added ~180 MB for this
database's ~520k rows and contributed to a real Render OOM crash (512 MB
instance) shortly after deploy. Only the behaviour that needed locking
down: same answers as before, for a known icao24 (real bundled data, cf.
tests/test_api.py's own use of the same aircraft) and an unknown one."""

from services.aircraft_database import get_registration, get_typecode


def test_known_icao24_resolves_typecode_and_registration():
    # RYR2PY / 4ca56b / EI-DWF -- verified by hand against the real bundled
    # aircraft database, same flight used throughout tests/test_api.py.
    assert get_typecode("4ca56b") == "B738"
    assert get_registration("4ca56b") == "EI-DWF"
    # Case/whitespace-insensitive, same as before the dict-index change.
    assert get_typecode(" 4CA56B ") == "B738"


def test_unknown_icao24_returns_none():
    assert get_typecode("ffffff") is None
    assert get_registration("ffffff") is None
