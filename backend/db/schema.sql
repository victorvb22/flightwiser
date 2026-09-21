-- Flightwiser — Supabase schema (brief section 9)
-- Paste this once into the Supabase dashboard's SQL editor (Project > SQL Editor).
-- Accessed exclusively via services/cache.py, using the secret key (bypasses RLS) —
-- no other client (frontend included) should touch these tables directly,
-- so no RLS policy needs to be defined here.

create table if not exists flights_cache (
    id text primary key,               -- icao24 + UTC date, e.g. "39de4e_2026-09-16"
    statut text not null,              -- en_vol / atterri (airborne / landed)
    trajectoire jsonb not null,
    ecart_trajectoire jsonb,           -- nullable: airborne flight, or too few points
    anomalie jsonb,                    -- nullable: same
    directness jsonb,                  -- nullable: same (great-circle distance too short, or airborne)
    kpi_bonus jsonb,                   -- nullable, always null: a bonus KPI isn't part of the product's scope, field kept for API-contract compatibility (brief section 8)
    source text not null,              -- "vedette" (served from the pool) at write time (never "cache" in the DB, "direct" only in older rows)
    calcule_le timestamptz not null default now()
);

-- Added after the table's initial creation (the route-directness score,
-- cf. models/directness.py) — the CREATE TABLE IF NOT EXISTS above doesn't
-- touch an already-existing table, so this line needs to be pasted (once)
-- into the Supabase SQL editor for a database already in place.
alter table flights_cache add column if not exists directness jsonb;

-- flights_historical (a separate batch import, once used by the Aggregate
-- view by date) has been removed: the Aggregate view now reads
-- flights_cache (GET /api/v1/flights/history) — a flight searched through
-- the app is already there, no need for a separately fed table. On a
-- database created before this change: `drop table if exists
-- flights_historical;` in the Supabase SQL editor to remove it.

-- Tracks which pool flights (services/flight_pool.py) have already been
-- served — persistent here, not just in the backend's memory, because
-- Render's disk is ephemeral between redeploys: without this table, a
-- redeploy would make every pool flight look "never served" again, and a
-- random draw could serve an already-seen flight a second time.
create table if not exists pool_served (
    icao24 text primary key,
    servi_le timestamptz not null default now()
);
