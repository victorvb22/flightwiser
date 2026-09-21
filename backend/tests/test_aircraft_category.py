"""services/aircraft_category.py -- the typecode -> category lookup every
scored diagnostic goes through first."""

from services.aircraft_category import CATEGORIES, categorize


def test_known_typecodes_resolve_to_the_expected_category():
    # One real typecode per category, cf. data/reference/aircraft_categories.csv.
    assert categorize("A320") == "avion_ligne"
    assert categorize("B350") == "jet_affaire"
    assert categorize("AA5") == "petit_avion"
    assert categorize("A109") == "helicoptere"


def test_unknown_typecode_returns_none_not_a_guessed_default():
    assert categorize("ZZZZ999") is None


def test_lookup_is_case_and_whitespace_insensitive():
    assert categorize("a320") == categorize("A320")
    assert categorize("  A320  ") == categorize("A320")


def test_categories_constant_has_exactly_the_four_documented_keys():
    assert set(CATEGORIES) == {"avion_ligne", "jet_affaire", "petit_avion", "helicoptere"}
