"""Centralized configuration — every key/URL comes from environment variables.

No other module should read os.environ directly; everything goes through the
`settings` attributes below.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    opensky_client_id: str | None = os.getenv("OPENSKY_CLIENT_ID")
    opensky_client_secret: str | None = os.getenv("OPENSKY_CLIENT_SECRET")
    # Optional Cloudflare Worker relay (services/opensky_client.py) — some
    # hosting providers' outbound network can't reach opensky-network.org at
    # all (confirmed on Render: full connection timeout, not a slow response
    # or a rejection — traced to a Render-specific IP-range block, since the
    # exact same hosts respond normally from elsewhere). Unset for a local
    # dev machine that can already reach OpenSky directly.
    opensky_relay_url: str | None = os.getenv("OPENSKY_RELAY_URL")
    opensky_relay_secret: str | None = os.getenv("OPENSKY_RELAY_SECRET")
    supabase_url: str | None = os.getenv("SUPABASE_URL")
    supabase_key: str | None = os.getenv("SUPABASE_KEY")
    cors_allowed_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    ]


settings = Settings()
