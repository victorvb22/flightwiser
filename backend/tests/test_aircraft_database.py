"""services/aircraft_database.py -- get_typecode/get_registration used to
scan the whole (~94 MB) DataFrame on every call; now backed by a dict index
built once (_get_by_icao24). Only the behaviour that needed locking down:
same answers as before, for a known icao24 (real bundled data, cf.
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
