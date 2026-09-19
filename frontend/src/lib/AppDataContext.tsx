import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import {
  ApiError,
  getAnomalieModelParams,
  getSearchHistory,
  type AnomalieCategoryParams,
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
  /** Re-fetches the search history in the background. Called on every visit
   * to the Aggregate view, and right after a search succeeds on the Search
   * page — the second call is what lets a just-searched flight show up
   * without the user having to leave and come back. A plain re-fetch rather
   * than splicing the new flight in locally: `categorie` is resolved
   * server-side (services/aircraft_category.py) from data the frontend
   * doesn't have, so a locally-built entry could never be trusted to match
   * what the backend would actually compute. */
  refreshHistory: () => void;
  anomalieParams: AnomalieParamsState;
  refreshAnomalieParams: () => void;
}

const AppDataContext = createContext<AppDataContextValue | null>(null);

export function AppDataProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<HistoryState>({ entries: null, loading: false, error: null });
  const [anomalieParams, setAnomalieParams] = useState<AnomalieParamsState>({ params: null, loading: false, error: null });

  const refreshHistory = useCallback(() => {
    setHistory((s) => ({ ...s, loading: s.entries === null, error: null }));
    getSearchHistory()
      .then((entries) => setHistory({ entries, loading: false, error: null }))
      .catch((err) => setHistory((s) => ({ ...s, loading: false, error: err instanceof ApiError ? err.message : "Unexpected error" })));
  }, []);

  const refreshAnomalieParams = useCallback(() => {
    setAnomalieParams((s) => ({ ...s, loading: s.params === null, error: null }));
    getAnomalieModelParams()
      .then((params) => setAnomalieParams({ params, loading: false, error: null }))
      .catch((err) => setAnomalieParams((s) => ({ ...s, loading: false, error: err instanceof ApiError ? err.message : "Unexpected error" })));
  }, []);

  return (
    <AppDataContext.Provider value={{ history, refreshHistory, anomalieParams, refreshAnomalieParams }}>{children}</AppDataContext.Provider>
  );
}

export function useAppData(): AppDataContextValue {
  const ctx = useContext(AppDataContext);
  if (!ctx) throw new Error("useAppData must be used within AppDataProvider");
  return ctx;
}
