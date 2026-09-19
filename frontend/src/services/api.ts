/**
 * Unique point d'appel vers le backend (brief section 7) — aucun composant
 * ne doit appeler fetch()/axios directement, tout passe par ici.
 */

export interface TrajectoirePoint {
  lat: number;
  lon: number;
  altitude: number | null;
  vitesse: number | null;
  timestamp: number;
}

export interface EcartPhase {
  ecart: number | null;
  detail: string;
}

export interface EcartTrajectoire {
  score_global: number;
  par_phase: {
    montee: EcartPhase;
    croisiere: EcartPhase;
    descente: EcartPhase;
  };
}

export interface AnomalieFeatureDetail {
  feature: string;
  valeur: number;
  reference: number;
}

/** One of the three patterns a separate, classical ML project (Random
 * Forest, trained on synthetic anomaly injections — never a real one, cf.
 * backend/models/anomaly_type.py) was built to recognise, or "normal" when
 * none of the three match. Airliner-only (backend restricts the call to
 * avion_ligne) and always computed when applicable, regardless of whether
 * this flight's own score actually flags an anomaly — components decide
 * when to surface it (cf. ScoreAnomalie.tsx: only shown for a flagged score,
 * and only when it isn't "normal" — that combination isn't a contradiction,
 * it means "flagged by the score model, but not one of these three known
 * patterns"). Absent on a flights_cache row written before this field
 * existed, or null for a non-airliner/insufficient-trajectory flight. */
export type TypeAnomalie = "go_around" | "holding" | "emergency_descent" | "normal";

export interface Anomalie {
  score: number;
  features_contributives: string[];
  /** Absent on a flights_cache row written before this field existed. */
  features_detail?: AnomalieFeatureDetail[];
  /** True for a helicopter flight whose trajectory never touches the
   * training region (Europe) — random draws are worldwide now, so this can
   * happen. Absent on a flights_cache row written before this field
   * existed; treat as false in that case, same as features_detail. */
  out_of_training_scope?: boolean;
  type_anomalie?: TypeAnomalie | null;
}

export interface Directness {
  score: number;
  ratio: number;
}

export type StatutVol = "en_vol" | "atterri";
export type SourceVol = "vedette" | "cache" | "direct";

export interface FlightResponse {
  identifiant: string;
  statut: StatutVol;
  trajectoire: TrajectoirePoint[];
  ecart_trajectoire: EcartTrajectoire | null;
  anomalie: Anomalie | null;
  /** Absent on a flights_cache row written before this field existed. */
  directness?: Directness | null;
  kpi_bonus: unknown | null;
  source: SourceVol;
  /** Métadonnées façon FlightRadar24 — chacune peut être absente selon ce
   * que la base aéronefs / la géométrie de la trajectoire permettent de
   * résoudre (brief : "si elles sont dispo"), jamais fabriquée. */
  typecode: string | null;
  immatriculation: string | null;
  origine: string | null;
  origine_ville: string | null;
  destination: string | null;
  destination_ville: string | null;
  altitude_actuelle: number | null;
  vitesse_actuelle: number | null;
}

export interface HealthResponse {
  status: string;
}

/** One row of every flight ever searched (and therefore cached) — backs the
 * Aggregate view's chart/history table. Trimmed vs. FlightResponse: no
 * immatriculation/origine/... metadata, since flights_cache never stored
 * those (only what the diagnostic models themselves need) — typecode is the
 * one exception, resolved from icao24 on the backend same as identifiant. */
/** Internal category keys, shared with the backend — cf.
 * models/_anomalie_features.categorize. Same three keys as
 * Documentation.tsx's CATEGORY_LABELS. */
export type Categorie = "avion_ligne" | "petit_avion" | "helicoptere";

export interface HistoryFlightEntry {
  identifiant: string;
  typecode: string | null;
  /** The exact category the anomaly model scored this flight against — null
   * only when typecode itself is unknown (nothing to categorize; the flight
   * also has no anomalie/ecart_trajectoire/directness score in that case). */
  categorie: Categorie | null;
  statut: StatutVol;
  trajectoire: TrajectoirePoint[];
  ecart_trajectoire: EcartTrajectoire | null;
  anomalie: Anomalie | null;
  directness: Directness | null;
  calcule_le: string;
}

export interface AnomalieCategoryParams {
  n_train: number;
  n_validation: number;
  epsilon_log_vraisemblance: number;
  percentile_breakpoints: number[];
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

/** Vol introuvable (404) — distinct d'une erreur réseau/serveur pour que
 * l'UI puisse afficher un message adapté plutôt qu'une erreur générique. */
export class FlightNotFoundError extends Error {}

/** Toute autre erreur (réseau, backend indisponible, statut inattendu). */
export class ApiError extends Error {}

export async function getHealth(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/health`);
  } catch {
    throw new ApiError("Backend unreachable");
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded ${response.status}`);
  }
  return response.json();
}

export async function getFlight(identifiant: string): Promise<FlightResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/flights/${encodeURIComponent(identifiant)}`);
  } catch {
    throw new ApiError("Backend unreachable");
  }
  if (response.status === 404) {
    throw new FlightNotFoundError(`Flight "${identifiant}" not found`);
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded ${response.status}`);
  }
  return response.json();
}

export async function getRandomFlight(statut?: StatutVol): Promise<FlightResponse> {
  const params = statut ? `?statut=${statut}` : "";
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/flights/random/search${params}`);
  } catch {
    throw new ApiError("Backend unreachable");
  }
  if (response.status === 404) {
    throw new FlightNotFoundError("No flight available right now");
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded ${response.status}`);
  }
  return response.json();
}

export async function getSearchHistory(): Promise<HistoryFlightEntry[]> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/flights/history`);
  } catch {
    throw new ApiError("Backend unreachable");
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded ${response.status}`);
  }
  return response.json();
}

export async function getAnomalieModelParams(): Promise<Record<string, AnomalieCategoryParams>> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/models/anomalie`);
  } catch {
    throw new ApiError("Backend unreachable");
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded ${response.status}`);
  }
  return response.json();
}
