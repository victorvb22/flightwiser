"""End-to-end HTTP contract tests, through FastAPI's TestClient (in-process,
no real socket, no dependency on requests -- block_network in conftest.py
doesn't interfere with it).

Run against the `test_pool` fixture (fixtures/flight_pool_sample.jsonl), a
small, frozen copy of three real/synthetic flights -- not the real, growing
data/reference/flight_pool.jsonl -- so these stay correct and fast no
matter how many flights the user has collected since. The exact numbers
asserted below were verified by hand against this fixture (cf. the fixture
file's own three entries) before being written down as tests.
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_landed_identified_airliner_gets_all_three_scores(test_pool):
    response = client.get("/api/v1/flights/RYR2PY")
    assert response.status_code == 200
    body = response.json()
    assert body["identifiant"] == "RYR2PY"
    assert body["statut"] == "atterri"
    assert body["typecode"] == "B738"
    assert body["anomalie"] is not None
    assert 0.0 <= body["anomalie"]["score"] <= 1.0
    assert body["directness"] is not None
    assert 0.0 <= body["directness"]["score"] <= 1.0
    assert body["ecart_trajectoire"] is not None


def test_unknown_identifier_returns_404(test_pool):
    response = client.get("/api/v1/flights/DOES-NOT-EXIST")
    assert response.status_code == 404


def test_airborne_flight_has_no_scores(test_pool):
    # Destination/full trajectory unknown while still airborne -- brief
    # section 12's MVP constraint, cf. pipeline.py's own docstring.
    response = client.get("/api/v1/flights/LTA660")
    assert response.status_code == 200
    body = response.json()
    assert body["statut"] == "en_vol"
    assert body["anomalie"] is None
    assert body["ecart_trajectoire"] is None
    assert body["directness"] is None


def test_unidentified_aircraft_gets_no_scores_not_a_guessed_category(test_pool):
    response = client.get("/api/v1/flights/TEST999")
    assert response.status_code == 200
    body = response.json()
    assert body["typecode"] is None
    assert body["anomalie"] is None
    assert body["ecart_trajectoire"] is None
    assert body["directness"] is None


def test_random_flight_respects_the_statut_filter(test_pool):
    response = client.get("/api/v1/flights/random/search?statut=atterri")
    assert response.status_code == 200
    assert response.json()["statut"] == "atterri"

    response = client.get("/api/v1/flights/random/search?statut=en_vol")
    assert response.status_code == 200
    assert response.json()["statut"] == "en_vol"


def test_example_identifiant_is_one_of_the_pool_flights(test_pool):
    # Bare identifier, not a full flight: no scoring, no cache write, no
    # pool_served write -- cf. the route's own docstring.
    response = client.get("/api/v1/flights/example")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"identifiant"}
    assert body["identifiant"] in {"RYR2PY", "LTA660", "TEST999"}


def test_search_history_endpoint_returns_a_list():
    # block_network makes cache.get_all_cached_flights() a silent empty
    # list -- this only checks the route wires that up correctly, not real
    # history content (which needs a live Supabase).
    response = client.get("/api/v1/flights/history")
    assert response.status_code == 200
    assert response.json() == []


def test_anomalie_model_params_cover_all_four_categories():
    response = client.get("/api/v1/models/anomalie")
    assert response.status_code == 200
    assert set(response.json().keys()) == {"avion_ligne", "jet_affaire", "petit_avion", "helicoptere"}
