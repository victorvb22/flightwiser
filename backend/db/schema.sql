-- Flightwiser — schéma Supabase (brief section 9)
-- À coller une fois dans l'éditeur SQL du dashboard Supabase (Project > SQL Editor).
-- Accès exclusivement via services/cache.py, avec la clé secrète (bypasse RLS) —
-- aucun autre client (frontend inclus) ne doit toucher ces tables directement,
-- donc pas de policy RLS à définir ici.

create table if not exists flights_cache (
    id text primary key,               -- icao24 + date UTC, ex. "39de4e_2026-09-16"
    statut text not null,              -- en_vol / atterri
    trajectoire jsonb not null,
    ecart_trajectoire jsonb,           -- nullable : vol en_vol, ou trop peu de points
    anomalie jsonb,                    -- nullable : idem
    directness jsonb,                  -- nullable : idem (distance grand-cercle trop courte, ou vol en_vol)
    kpi_bonus jsonb,                   -- nullable : aucun KPI bonus implémenté pour l'instant
    source text not null,              -- direct au moment de l'écriture (jamais "cache" en base)
    calcule_le timestamptz not null default now()
);

-- Ajouté après la création initiale de la table (le score de trajet direct,
-- cf. models/directness.py) — CREATE TABLE IF NOT EXISTS ci-dessus ne
-- retouche pas une table déjà existante, donc cette ligne doit être collée
-- (une fois) dans l'éditeur SQL Supabase pour une base déjà en place.
alter table flights_cache add column if not exists directness jsonb;

-- flights_historical (import batch séparé, un temps utilisé par la Vue
-- agrégée par date) a été retirée : la Vue agrégée lit désormais
-- flights_cache (GET /api/v1/flights/history) — un vol recherché via l'app
-- y est déjà, pas besoin d'une table alimentée séparément. Sur une base
-- existante créée avant ce changement : `drop table if exists
-- flights_historical;` dans l'éditeur SQL Supabase pour la retirer.
