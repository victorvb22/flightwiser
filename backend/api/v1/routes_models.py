"""Read-only introspection of trained model parameters — for the
Documentation page's charts (real trained numbers, not illustrative ones),
not part of the flight-search contract (brief section 8), hence its own
router rather than living under routes_flights.py.
"""

from fastapi import APIRouter

from models import anomalie

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/anomalie")
def get_anomalie_params():
    return anomalie.get_category_summaries()
