from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.v1.routes_flights import router as flights_router
from api.v1.routes_models import router as models_router
from config import settings

app = FastAPI(title="Flightwiser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(flights_router)
app.include_router(models_router)


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}
