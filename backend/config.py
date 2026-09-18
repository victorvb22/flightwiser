"""Configuration centralisée — toutes les clés/URLs viennent des variables d'environnement.

Aucun autre module ne doit lire os.environ directement ; tout passe par les
attributs de `settings` ci-dessous.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    opensky_client_id: str | None = os.getenv("OPENSKY_CLIENT_ID")
    opensky_client_secret: str | None = os.getenv("OPENSKY_CLIENT_SECRET")
    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_key: str | None = os.getenv("SUPABASE_KEY")
    cors_allowed_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]


settings = Settings()
