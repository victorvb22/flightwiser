# Flightwiser

Outil de recherche et de diagnostic ML sur des vols individuels, à partir de données ADS-B réelles (OpenSky Network) : trajectoire réelle, score d'anomalie de trajectoire, écart vs profil de vol optimal simulé (OpenAP), et directivité de la route (distance grand-cercle vs distance parcourue). v1 : pas de catégorisation d'anomalie par carburant/mission, pas de KPI bonus.

## Structure

```
backend/    API REST (FastAPI), clients de données, modèles ML, scripts de prétraitement
frontend/   SPA (React + TypeScript, Vite)
data/       data/raw/ (dataset historique brut, non versionné) — data/reference/ et data/processed/ regénérés par les scripts
```

## Backend — setup local

```
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # puis renseigner les identifiants OpenSky
uvicorn main:app --reload
```

Vérifier que ça tourne : `GET http://127.0.0.1:8000/api/v1/health`.

## Prétraitement du dataset historique

```
cd backend
python scripts/preprocess_historical.py
```

Lit `data/raw/flights_raw.csv`, produit `data/processed/flights_clean.parquet` (dédoublonnage des points, filtrage des artefacts ADS-B — position/altitude/vitesse verticale aberrantes —, recalcul de vitesse, déduction de l'aéroport de destination, enrichissement du type d'appareil).

## Catégorisation des appareils

Avant tout calcul de score, chaque vol est classé par typecode (et icao24 pour les hélicoptères) :

- `avion_ligne` / `petit_avion` (`services/aircraft_category.py`) : frontière = "OpenAP reconnaît ce typecode" — une gaussienne unique sur tout le parc pénaliserait systématiquement le groupe minoritaire. Utilisée par le modèle d'écart et le modèle de directivité.
- `helicoptere` (`models/_anomalie_features.categorize`, propre au modèle d'anomalie) : détecté via le champ `icaoaircrafttype` de la base OpenSky (`services/aircraft_database.is_helicopter`), isolé de `petit_avion` — le profil d'un hélicoptère (vol stationnaire, pas de phases montée/croisière/descente classiques) est structurellement hors de la distribution d'un appareil à voilure fixe, même pour un vol normal.

L'entraînement (voir ci-dessous) est restreint aux appareils passés par l'Europe le 27/06/2022 ; un tirage aléatoire de vol est en revanche mondial. Un vol hélicoptère dont la trajectoire ne touche jamais l'Europe reçoit un flag `out_of_training_scope` sur son score d'anomalie (mission trop variable par région pour faire confiance à la baseline hors zone d'entraînement) — vérifié pour `avion_ligne`/`petit_avion` que ce n'était pas nécessaire (`scripts/check_geographic_consistency.py`).

## Modèle d'anomalie de trajectoire

```
cd backend
python scripts/extract_historical_for_training.py --date 2022-06-27   # une fois, ~20-45 min selon la bbox
python scripts/train_anomalie.py
```

Ajuste une gaussienne diagonale par catégorie (avion_ligne / petit_avion / helicoptere) sur 7 features résumées de vol (brief section 11), sur les vols atterris de deux sources fusionnées et dédoublonnées (`scripts/_training_data.py`) — `data/processed/flights_clean.parquet` et `data/processed/flights_historical_features.parquet` (extraction Europe ci-dessous) — et écrit les paramètres dans `models/artifacts/anomalie_params.json`, utilisés par `models/anomalie.py` au moment du calcul.

## Modèle d'écart trajectoire

`models/ecart_trajectoire.py` compare la trajectoire réelle d'un vol à un profil simulé par [OpenAP](https://openap.dev) pour le même type d'appareil, phase par phase (montée/croisière/descente). Pas d'entraînement ni d'artefact — OpenAP est appelé à la demande, à partir de `compute(trajectoire, typecode)`. Un typecode qu'OpenAP ne reconnaît pas (catégorie `petit_avion`, hélicoptères inclus) renvoie `None` plutôt qu'un score calculé contre un modèle de performance A320 substitué silencieusement — pas de simulation de repli.

## Modèle de directivité de route

`models/directness.py` calcule le ratio distance grand-cercle / distance réellement parcourue entre le premier et le dernier point de la trajectoire — un ratio proche de 1 signale une route directe, plus bas suggère un circuit d'attente, un détournement, ou un vectoring ATC prolongé. Seul des trois à regarder la forme de la trajectoire plutôt que ses valeurs de vitesse/altitude. Entraînement (percentiles empiriques, pas de gaussienne) :

```
cd backend
python scripts/train_directness.py
```

## Recherche libre (`GET /api/v1/flights/{identifiant}`)

Fonctionnelle de bout en bout, avec cache : `pipeline.py` résout l'identifiant (immatriculation ou callsign) en icao24, vérifie `flights_cache` (Supabase) sous la clé `{icao24}_{date UTC}`, et si absente récupère la trajectoire live/récente via OpenSky (même nettoyage que le prétraitement batch, `services/trajectory_cleaning.py`, partagé), calcule écart et anomalie pour un vol atterri, puis écrit le résultat en cache. Une recherche déjà vue le même jour ne redéclenche donc aucun appel OpenSky superflu (`source: "cache"` vs `"direct"`).

### Setup Supabase

Coller une fois `backend/db/schema.sql` dans le SQL Editor du dashboard Supabase (crée `flights_cache`), puis renseigner `SUPABASE_URL`/`SUPABASE_KEY` (clé secrète, Project Settings > API) dans `backend/.env`. Accès en REST direct (PostgREST) via `services/cache.py`, pas de SDK — le cache est un best-effort : toute erreur réseau/HTTP est journalisée et traitée comme un miss/no-op plutôt que de faire échouer une recherche.

## Recherche géolocalisée (`GET /api/v1/flights/nearest/search?lat=&lon=`)

Bounding box de 100 km autour de la position (`services.opensky_client.get_states(bbox=...)`), tri par distance haversine, essaie jusqu'à 10 avions les plus proches (`tracks/all` est documenté "purement expérimental" par OpenSky — le plus proche n'a pas toujours de trajectoire disponible) jusqu'à un résultat exploitable, puis délègue au même pipeline que la recherche par identifiant (même cache, mêmes modèles). 400 si les coordonnées sont hors plage, 404 si rien d'exploitable à proximité.

## Données d'entraînement historiques (`scripts/extract_historical_for_training.py`)

```
cd backend
python scripts/extract_historical_for_training.py --date 2022-06-27
```

OpenSky ne publie pas de Trino public, mais un bucket public (`s3.opensky-network.org/data-samples`) contient une extraction mondiale complète des vecteurs d'état chaque mardi soir (journée précédente), du 05/06/2017 au 27/06/2022. Le script télécharge les 24 fichiers horaires (mis en cache sur `D:/ML_data/flightwiser`, ou `IMPORT_DATA_DIR` si défini — jamais dans le repo), repère en deux passes locales (`services/opensky_historical.py`) les avions passés par une large bbox Europe ce jour-là (un vol qui y décolle peut atterrir n'importe où dans le monde, donc un filtrage géographique en une seule passe tronquerait la trajectoire), reconstruit et nettoie chaque vol (`services/trajectory_cleaning.py`, partagé avec le reste du projet), et écrit `data/processed/flights_historical_features.parquet` — utilisé, dédoublonné avec `flights_clean.parquet` (`scripts/_training_data.py`), par `scripts/train_anomalie.py` et `scripts/train_directness.py`.

La bbox ne fait que décider quels avions deviennent candidats ; une fois candidat, la trajectoire complète est extraite où qu'elle aille dans le monde — élargir la bbox n'introduit donc aucune troncature, seulement plus de diversité.

## Frontend — setup local

```
cd frontend
npm install
copy .env.example .env   # optionnel en local, la valeur par défaut pointe déjà vers le backend local
npm run dev
```

Ouvrir `http://localhost:5173` (backend démarré en parallèle sur le port 8000, CORS déjà configuré). Trois pages, toutes branchées au backend réel : `pages/RechercheVol.tsx` (recherche libre + vol aléatoire), `pages/VueAgregee.tsx` (historique des recherches, `GET /api/v1/flights/history`), `pages/Documentation.tsx` (paramètres réels des modèles entraînés, `GET /api/v1/models/anomalie`).

### Visualisations

`components/visualisations/` a son rendu graphique réel (brief section 7), construit avec la méthode de la skill dataviz — palette validée CVD-safe, mode clair/sombre sélectionné (pas un simple flip), rôles couleur cohérents avec leur job (séquentiel pour l'altitude, statut pour la sévérité, une seule couleur pour un histogramme à série unique — jamais l'un pour l'autre) :

- **VueTrajectoire** : tracé SVG de la trajectoire (projection équirectangulaire), coloré par altitude (rampe séquentielle bleue), légende de la rampe, curseur croisé + infobulle au survol ou à la navigation clavier (←/→), table de données repliable (`<details>`) pour que chaque valeur reste accessible sans survol.
- **JaugeEcart** : un meter par phase (montée/croisière/descente) + écart global, couleur de sévérité par palette de statut (jamais une couleur de série), détail textuel toujours visible sous la barre.
- **ScoreAnomalie** : stat tile (rang percentile + badge de sévérité), features contributives en tags, avertissement `out_of_training_scope` pour un hélicoptère hors zone d'entraînement.
- **DirectnessGauge** : même langage que ScoreAnomalie pour le score de directivité.
- **HistogrammeScores** : bar chart à 10 bins, utilisé pour les distributions de score de la Vue agrégée.

## Vue agrégée (`GET /api/v1/flights/history`)

Lit `flights_cache` (Supabase, `services/cache.get_all_cached_flights`) — tout vol déjà recherché via l'app, pas un import batch séparé. Résout chaque icao24 en immatriculation quand connue (`services/aircraft_database.get_registration`, sinon icao24 brut), en typecode, et en catégorie (même catégorisation que le modèle d'anomalie) ; tableau triable sur chaque colonne, histogrammes de score et stats calculés côté frontend (`pages/VueAgregee.tsx`) à partir de cette même liste.

## Déploiement

Frontend (Vercel) et backend (Render) hébergés séparément — Vercel n'est pas adapté au backend : dépendances lourdes (pandas/numpy/pyarrow/openap/scipy) et une base aéronefs de ~94 Mo gardée en mémoire, qu'un serverless à froid rechargerait à chaque requête.

**Backend (Render)** : `render.yaml` à la racine du repo décrit le service (`New` > `Blueprint` sur le dashboard Render plutôt qu'une config manuelle). Secrets à renseigner dans le dashboard (jamais dans le repo) : `OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET`, `SUPABASE_URL`, `SUPABASE_KEY`, `CORS_ALLOWED_ORIGINS` (domaine Vercel une fois connu), `OPENSKY_RELAY_URL`/`OPENSKY_RELAY_SECRET` (voir ci-dessous).

**Relais OpenSky (`cloudflare-relay/worker.js`)** : le réseau sortant de Render ne peut pas joindre `opensky-network.org` du tout (timeout de connexion complet, confirmé sur deux hôtes différents, alors que les mêmes hôtes répondent normalement ailleurs — probablement un blocage des plages IP des hébergeurs cloud, qu'OpenSky documente pratiquer contre les abus). Un Worker Cloudflare gratuit relaie les seuls appels vers `opensky-network.org`/`auth.opensky-network.org` (jamais un proxy ouvert — deux hôtes fixes, secret partagé requis). Déploiement : coller `cloudflare-relay/worker.js` dans le dashboard Cloudflare (Workers & Pages > Create), ajouter la variable `RELAY_SECRET`, puis renseigner `OPENSKY_RELAY_URL` (l'URL du Worker) et `OPENSKY_RELAY_SECRET` (même valeur) côté Render. Vide en local — `services/opensky_client.py` retombe sur un accès direct à OpenSky si ces variables ne sont pas définies.

**Frontend (Vercel)** : Root Directory = `frontend` (monorepo). `frontend/vercel.json` ajoute la réécriture SPA nécessaire (`BrowserRouter` : sans elle, un lien direct ou un F5 sur `/agregee` ou `/documentation` renvoie une 404 sur un hébergement statique). Variable d'environnement `VITE_API_BASE_URL` = URL du backend Render.

Ordre de bootstrap (dépendance croisée) : déployer le backend d'abord, récupérer son URL, la mettre dans `VITE_API_BASE_URL` côté Vercel, déployer le frontend, récupérer son URL, puis mettre à jour `CORS_ALLOWED_ORIGINS` côté Render et redéployer.
