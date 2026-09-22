import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import {
  ApiError,
  getAnomalieModelParams,
  getExampleIdentifiant,
  getSearchHistory,
  type AnomalieCategoryParams,
  type FlightResponse,
  type HistoryFlightEntry,
} from "../services/api";

/**
 * Cross-page cache for the two datasets both the Aggregate view and the
 * Documentation page fetch from the backend — held here, above the router
 * (provided in App.tsx, outside <Routes>), so navigating away and back
 * doesn't refetch, and doesn't blank the page while it does. The page
 * component itself still fully unmounts and remounts on every navigation
 * (React Router's normal behaviour, left as is) — only the data survives,
 * in memory, for the lifetime of this browser tab; a hard reload or a new
 * tab always starts empty, deliberately, this was never meant to be a
 * persistent cache.
 *
 * Both datasets follow the same pattern: `entries`/`params` stay whatever
 * was last fetched while a refresh runs — `loading` only ever turns true
 * when there is nothing cached yet (first-ever visit this session), so a
 * revisit shows the previous data immediately and swaps it out silently
 * once the background refresh resolves, instead of the page going blank
 * and reloading every time.
 */

interface HistoryState {
  entries: HistoryFlightEntry[] | null;
  loading: boolean;
  error: string | null;
}

interface AnomalieParamsState {
  params: Record<string, AnomalieCategoryParams> | null;
  loading: boolean;
  error: string | null;
}

interface AppDataContextValue {
  history: HistoryState;
  /** Re-fetches the search history in the background — the eventually-
   * consistent half of showing a just-searched flight (RechercheVol.tsx
   * calls both this and addToHistory below, cf. its own comment). Called
   * on every visit to the Aggregate view too. */
  refreshHistory: () => void;
  /** Splices one flight into `history.entries` immediately, client-side —
   * no network round trip, so the flight a search just found shows up in
   * the Aggregate view the instant the search itself resolves, rather than
   * waiting on a second request. Safe now in a way it wasn't originally:
   * every field HistoryFlightEntry needs (`categorie` included) is already
   * on FlightResponse (pipeline.py's `_flight_metadata` resolves it
   * server-side either way, cf. api.ts), so this never has to guess at
   * anything the backend itself computed — only `calcule_le` is a local
   * approximation (the moment this function runs, not the server's own
   * write time, which refreshHistory's follow-up call corrects to the
   * real value once it resolves). Replaces any existing entry with the
   * same identifiant rather than duplicating it, so the two rows don't
   * both show up for the few hundred ms until that reconciliation lands. */
  addToHistory: (vol: FlightResponse) => void;
  anomalieParams: AnomalieParamsState;
  refreshAnomalieParams: () => void;
  /** A real, not-yet-served pool identifier for the Search page's
   * placeholder — pinned (localStorage, cf. EXAMPLE_STORAGE_KEY below) once
   * fetched, so it survives a hard reload too, not just cross-page
   * navigation within the tab: null only on a device that has genuinely
   * never fetched one yet (retrying in the background if the backend is
   * still asleep, cf. refreshExample's own comment). Refreshed only after
   * every successful search/random draw (RechercheVol.tsx), i.e. only once
   * this exact flight has actually been used — never on a timer, and never
   * just because the page remounted: the whole point is that this stays
   * whatever it last was, correct or not, across however long the backend
   * then sits idle (cf. Render's free-tier sleep), rather than something
   * that has to be kept fresh in the background. */
  exampleIdentifiant: string | null;
  refreshExample: () => void;
}

const AppDataContext = createContext<AppDataContextValue | null>(null);

// Persisted (not just held in this context's in-memory state, unlike
// history/anomalieParams above) so the placeholder stays pinned to the
// same real flight across a hard reload too, not just cross-page
// navigation within one tab — cleaner to look at (doesn't visibly swap to
// a different example on every refresh) and doesn't need a fetch every
// time the Search page happens to remount. Wrapped in try/catch: a private
// window or blocked site data can make localStorage throw on access, and
// this is only ever a per-viewer convenience, never something that has to
// succeed.
const EXAMPLE_STORAGE_KEY = "flightwiser.exampleIdentifiant";

function readStoredExample(): string | null {
  try {
    return localStorage.getItem(EXAMPLE_STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredExample(identifiant: string): void {
  try {
    localStorage.setItem(EXAMPLE_STORAGE_KEY, identifiant);
  } catch {
    // Ignored — the in-memory state (below) still gets the fresh value for
    // the rest of this session either way.
  }
}

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<HistoryState>({ entries: null, loading: false, error: null });
  const [anomalieParams, setAnomalieParams] = useState<AnomalieParamsState>({ params: null, loading: false, error: null });
  const [exampleIdentifiant, setExampleIdentifiant] = useState<string | null>(readStoredExample);

  // Guards against the same class of race already found and fixed for
  // exampleIdentifiant below: VueAgregee's own mount effect calls
  // refreshHistory() every visit, StrictMode double-invokes that in dev,
  // and a search on the Search page calls it too (right after the flight
  // that search just found was written to flights_cache) — nothing
  // enforced that the LAST of several overlapping calls to actually
  // resolve was also the last one issued, so a slower, earlier-issued
  // fetch (e.g. one kicked off before that write happened) could still
  // land after a newer one and silently show the just-searched flight as
  // missing. Only the response to the most recently issued call is ever
  // allowed to apply.
  const historyRequestId = useRef(0);
  const refreshHistory = useCallback(() => {
    const requestId = ++historyRequestId.current;
    setHistory((s) => ({ ...s, loading: s.entries === null, error: null }));
    (async () => {
      // Retries rather than a single attempt: this is the very first
      // history fetch of the session when it's called right after a
      // search on the Search page (RechercheVol.tsx), which can land
      // while the backend is still waking up (Render's free-tier sleep,
      // same as useBackendWakeup.ts/refreshExample below) — a single
      // failed attempt here used to leave the Aggregate view stuck
      // showing only the one flight addToHistory had already spliced in
      // client-side (plus "No values yet" on every chart), with nothing
      // left to retry it short of a full page reload: VueAgregee's own
      // mount effect calls this exact function, so revisiting the page
      // without reloading never issues a fresh attempt on its own. Same
      // retry cadence/budget as refreshExample, for consistency.
      const deadline = Date.now() + 75_000;
      while (requestId === historyRequestId.current && Date.now() < deadline) {
        try {
          const entries = await getSearchHistory();
          if (requestId === historyRequestId.current) setHistory({ entries, loading: false, error: null });
          return;
        } catch (err) {
          if (requestId !== historyRequestId.current) return;
          if (Date.now() + 3_000 >= deadline) {
            setHistory((s) => ({ ...s, loading: false, error: err instanceof ApiError ? err.message : "Unexpected error" }));
            return;
          }
          await new Promise((resolve) => setTimeout(resolve, 3_000));
        }
      }
    })();
  }, []);

  const addToHistory = useCallback((vol: FlightResponse) => {
    const entry: HistoryFlightEntry = {
      identifiant: vol.identifiant,
      typecode: vol.typecode,
      categorie: vol.categorie,
      statut: vol.statut,
      trajectoire: vol.trajectoire,
      ecart_trajectoire: vol.ecart_trajectoire,
      anomalie: vol.anomalie,
      directness: vol.directness ?? null,
      calcule_le: new Date().toISOString(),
    };
    setHistory((s) => ({
      ...s,
      // Same identifiant, not the same flight necessarily (cf. rowKey's
      // own comment in VueAgregee.tsx) -- close enough for a display that
      // self-corrects within one round trip either way, and simpler than
      // threading icao24 through just for this.
      entries: [entry, ...(s.entries ?? []).filter((e) => e.identifiant !== entry.identifiant)],
    }));
  }, []);

  // Same reasoning as refreshHistory above — Documentation.tsx's own mount
  // effect has the identical shape (calls this every visit, StrictMode
  // double-invokes it too).
  const anomalieParamsRequestId = useRef(0);
  const refreshAnomalieParams = useCallback(() => {
    const requestId = ++anomalieParamsRequestId.current;
    setAnomalieParams((s) => ({ ...s, loading: s.params === null, error: null }));
    getAnomalieModelParams()
      .then((params) => {
        if (requestId === anomalieParamsRequestId.current) setAnomalieParams({ params, loading: false, error: null });
      })
      .catch((err) => {
        if (requestId === anomalieParamsRequestId.current) {
          setAnomalieParams((s) => ({ ...s, loading: false, error: err instanceof ApiError ? err.message : "Unexpected error" }));
        }
      });
  }, []);

  // Guards against a real race, not a hypothetical one: RechercheVol.tsx's
  // own mount effect calls refreshExample() once, but StrictMode's
  // dev-only double-invoke fires it twice, and a fast user can trigger it
  // again (another search/random draw) before an earlier call has
  // resolved — reproduced directly, two in-flight requests do not
  // necessarily resolve in the order they were sent, so whichever happened
  // to answer last could silently overwrite a newer, already-correct
  // example with a stale one. Only the response to the most recently
  // issued call is ever allowed to apply; it also doubles as the retry
  // loop's own cancellation check below (a newer call supersedes an older
  // one's retries, not just its eventual success).
  const exampleRequestId = useRef(0);
  const refreshExample = useCallback(() => {
    const requestId = ++exampleRequestId.current;
    (async () => {
      // Retries rather than a single attempt: the very first call happens
      // on this page's first-ever mount, which can land while the backend
      // is still asleep (Render's free-tier sleep, same as
      // useBackendWakeup.ts) — a single failed attempt there used to leave
      // the placeholder on its plain fallback for the rest of the session,
      // since nothing else was going to retry it. Same retry cadence/budget
      // as useBackendWakeup.ts, for consistency, though this has no banner
      // of its own to show while it waits.
      const deadline = Date.now() + 75_000;
      while (requestId === exampleRequestId.current && Date.now() < deadline) {
        try {
          const identifiant = await getExampleIdentifiant();
          if (requestId === exampleRequestId.current) {
            setExampleIdentifiant(identifiant);
            writeStoredExample(identifiant);
          }
          return;
        } catch {
          await new Promise((resolve) => setTimeout(resolve, 3_000));
        }
      }
    })();
  }, []);

  return (
    <AppDataContext.Provider
      value={{ history, refreshHistory, addToHistory, anomalieParams, refreshAnomalieParams, exampleIdentifiant, refreshExample }}
    >
      {children}
    </AppDataContext.Provider>
  );
}

export function useAppData(): AppDataContextValue {
  const ctx = useContext(AppDataContext);
  if (!ctx) throw new Error("useAppData must be used within AppDataProvider");
  return ctx;
}
