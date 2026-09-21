/**
 * The single point of contact with the backend (brief section 7) — no
 * component should call fetch()/axios directly, everything goes through here.
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
  /** FlightRadar24-style metadata — each field can be absent depending on
   * what the aircraft database / the trajectory's geometry actually let us
   * resolve (brief: "if available"), never fabricated. */
  typecode: string | null;
  /** The exact category the anomaly/directness models scored this flight
   * against (services/aircraft_category.categorize) — null when the
   * typecode itself is unresolved. Same field HistoryFlightEntry already
   * carries; lets the frontend explain *why* a diagnostic is unavailable
   * instead of guessing (e.g. JaugeEcart.tsx: a landed, fully-categorized
   * flight can still have no ecart_trajectoire score if OpenAP doesn't
   * model that specific typecode — a different reason than an
   * unidentified aircraft). */
  categorie: Categorie | null;
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
 * services/aircraft_category.py's CATEGORIES. Same four keys as
 * Documentation.tsx's CATEGORY_LABELS. */
export type Categorie = "avion_ligne" | "jet_affaire" | "petit_avion" | "helicoptere";

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

/** Flight not found (404) — distinct from a network/server error so the UI
 * can show a tailored message instead of a generic one. */
export class FlightNotFoundError extends Error {}

/** Any other error (network, backend unavailable, unexpected status). */
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

/** A real, not-yet-served pool identifier — backs the Search page's
 * placeholder example, so "e.g. ..." always names a flight that actually
 * exists in the pool. */
export async function getExampleIdentifiant(): Promise<string> {
  let response: Response;
  try {
    // no-store: this same URL (no params to vary it) is deliberately
    // fetched again after every search/random draw so the example stays
    // fresh — the browser's own HTTP cache otherwise has no reason to know
    // the response should differ each time and can silently reuse the
    // first one, observed directly (repeated draws left the placeholder
    // showing the exact same identifier). A bounded attempt timeout (cf.
    // useBackendWakeup.ts's own ATTEMPT_TIMEOUT_MS) so a single attempt
    // hanging while Render's free-tier container is still booting can't
    // stall AppDataContext's retry loop for the whole call.
    response = await fetch(`${API_BASE_URL}/api/v1/flights/example`, { cache: "no-store", signal: AbortSignal.timeout(15_000) });
  } catch {
    throw new ApiError("Backend unreachable");
  }
  if (!response.ok) {
    throw new ApiError(`Backend responded ${response.status}`);
  }
  const body: { identifiant: string } = await response.json();
  return body.identifiant;
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
