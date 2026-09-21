"""models/_ecart_features.py -- OpenAP typecode resolution and the
great-circle distance feeding the trajectory-deviation model. Deliberately
doesn't exercise FlightGenerator's own simulation (that's OpenAP's own
concern, not this project's) -- only the parts flightwiser's code owns."""

from models._ecart_features import resolve_typecode, trajectory_distance_km


def test_resolve_typecode_recognizes_a_mainline_airliner():
    # A320: the project's own default/reference typecode, definitely on
    # OpenAP's list.
    assert resolve_typecode("A320") is not None


def test_resolve_typecode_recognizes_a_business_jet_openap_also_models():
    # cf. data/reference/aircraft_categories.csv and
    # scripts/build_aircraft_categories.py's _OPENAP_BUSINESS_JETS: c550 is
    # one of the two business-jet families OpenAP happens to include, even
    # though it's categorized jet_affaire, not avion_ligne.
    assert resolve_typecode("C550") is not None


def test_resolve_typecode_returns_none_for_a_typecode_openap_does_not_model():
    # B350 (jet_affaire via MANUAL_OVERRIDES, cf.
    # scripts/build_aircraft_categories.py) is not on OpenAP's ~40-typecode
    # list -- this is the case that makes most of jet_affaire, and all of
    # petit_avion/helicoptere, score null for trajectory deviation.
    assert resolve_typecode("B350") is None


def test_resolve_typecode_returns_none_for_a_helicopter():
    # OpenAP has no rotary-wing performance models at all.
    assert resolve_typecode("A109") is None


def test_trajectory_distance_km_matches_known_reference():
    trajectoire = [
        {"lat": 49.0097, "lon": 2.5479},  # Paris CDG
        {"lat": 51.4700, "lon": -0.4543},  # London LHR
    ]
    km = trajectory_distance_km(trajectoire)
    assert 335 < km < 355
