import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Search as SearchIcon, X } from "lucide-react";
import { SplitFlapText } from "../components/SplitFlapText";
import { PulseRing } from "../components/PulseRing";
import { WakeupBanner } from "../components/WakeupBanner";
import { VueTrajectoire } from "../components/visualisations/VueTrajectoire";
import { HistogrammeScores } from "../components/visualisations/HistogrammeScores";
import { ScoreAnomalie } from "../components/visualisations/ScoreAnomalie";
import { JaugeEcart } from "../components/visualisations/JaugeEcart";
import { DirectnessGauge } from "../components/visualisations/DirectnessGauge";
import type { Categorie, HistoryFlightEntry } from "../services/api";
import { severityColor, worstSeverity } from "../lib/severity";
import { useAppData } from "../lib/AppDataContext";
import { useIsMobile } from "../lib/useIsMobile";
import { useBackendWakeup } from "../lib/useBackendWakeup";

type FeatureKey = "anomalie" | "ecart" | "directness";
type SortKey = "identifiant" | "typecode" | "categorie" | "statut" | FeatureKey | "calcule_le";

// A flight can legitimately appear more than once in the history (the same
// aircraft searched again on a different day, cf. the cache key in
// services/cache.py) -- entry.identifiant alone is therefore NOT unique
// across the list. calcule_le (when the cache row was actually computed)
// is, in practice, always distinct even for the same identifiant. Using
// bare identifiant as a React key/selection id here was exactly what broke
// sorting: with duplicate keys, React can't tell which element goes with
// which array entry any more once the order changes, so it reused stale
// DOM rows instead of reordering them (reproduced directly: sorting left
// leftover, wrongly-ordered rows above the correctly sorted list).
function rowKey(entry: HistoryFlightEntry): string {
  return `${entry.identifiant}::${entry.calcule_le}`;
}

const FEATURE_OPTIONS: { key: FeatureKey; label: string; good: 0 | 1 }[] = [
  { key: "anomalie", label: "Anomaly score", good: 1 },
  { key: "ecart", label: "Trajectory deviation", good: 0 },
  { key: "directness", label: "Route directness", good: 1 },
];

// Same keys/labels as Documentation.tsx's CATEGORY_LABELS — shown here so a
// surprising score can be explained by which category the flight actually
// landed in (cf. services/aircraft_category.categorize), not just guessed at.
const CATEGORY_LABELS: Record<Categorie, string> = {
  avion_ligne: "Airliner",
  jet_affaire: "Business jet",
  petit_avion: "Small aircraft",
  helicoptere: "Helicopter",
};

function extractScore(entry: HistoryFlightEntry, key: FeatureKey): number | null {
  if (key === "anomalie") return entry.anomalie?.score ?? null;
  if (key === "ecart") return entry.ecart_trajectoire?.score_global ?? null;
  return entry.directness?.score ?? null;
}

function binIndexFor(score: number | null): number | null {
  return score === null ? null : Math.min(9, Math.max(0, Math.floor(score * 10)));
}

// Same 10-bin shape (0-0.1, ..., 0.9-1.0) as the backend's own date-based
// aggregate route already used — computed client-side here since this
// chart reads the full search-history endpoint, not a single date query.
function toBins(values: number[]): number[] {
  const bins = new Array(10).fill(0) as number[];
  for (const v of values) {
    const idx = Math.min(9, Math.max(0, Math.floor(v * 10)));
    bins[idx] += 1;
  }
  return bins;
}

const cardStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  background: "var(--surface)",
  borderRadius: 14,
  padding: 20,
};

const cardTitleStyle: React.CSSProperties = {
  marginTop: 0,
  marginBottom: 0,
  fontSize: 13,
  color: "var(--text-faint)",
  fontWeight: 700,
  textTransform: "uppercase",
  letterSpacing: 1,
};

const selectStyle: React.CSSProperties = {
  background: "var(--surface-2)",
  color: "var(--text)",
  border: "1px solid var(--border-strong)",
  borderRadius: 8,
  padding: "8px 12px",
  fontSize: 13.5,
  font: "inherit",
};

/** Animates a freshly-mounted block from 0 to its natural height — remount
 * (via a `key` on the caller's side, one per row) restarts the animation,
 * which is exactly what's wanted here: a fresh "opening" transition each
 * time a different flight's detail is expanded, rather than an instant
 * snap. Closing isn't animated the same way (the row just unmounts
 * immediately) — the ask was specifically for the *open* to transition. */
function ExpandingDetail({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [maxHeight, setMaxHeight] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // A ResizeObserver, not a one-shot rAF measurement: VueTrajectoire loads
    // its world-map data asynchronously and grows once it arrives, and the
    // score panels' own content can reflow too — a single measurement taken
    // right after mount locks in whatever (smaller) height existed at that
    // instant, silently clipping everything that grows in afterwards. This
    // keeps re-measuring for as long as the block stays expanded.
    const observer = new ResizeObserver(() => setMaxHeight(el.scrollHeight));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <div ref={ref} style={{ maxHeight, overflow: "hidden", transition: "max-height 450ms cubic-bezier(0.22, 1, 0.36, 1)" }}>
      {children}
    </div>
  );
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  dir,
  onClick,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: "asc" | "desc";
  onClick: (k: SortKey) => void;
}) {
  const active = activeKey === sortKey;
  return (
    <th style={{ padding: "6px 8px", cursor: "pointer", userSelect: "none" }} onClick={() => onClick(sortKey)}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: active ? "var(--text)" : "var(--text-faint)" }}>
        {label}
        {active && (dir === "asc" ? <ChevronUp size={12} /> : <ChevronDown size={12} />)}
      </span>
    </th>
  );
}

export function VueAgregee() {
  // Cached above the router (lib/AppDataContext.tsx) — entries/loading/error
  // survive leaving and coming back to this page within the same session,
  // so a revisit shows the last-known list immediately instead of the page
  // going blank and refetching every time. This component itself still
  // fully unmounts/remounts on navigation as before, so the title's own
  // split-flap animation below still replays on every visit regardless.
  const isMobile = useIsMobile();
  const wakingUp = useBackendWakeup();
  // The table is wider than a phone screen (8 columns) and scrolls
  // horizontally inside its own box, so an open flight's panel can't just
  // fill the table's width — it would run off-screen. On mobile it's sized
  // from the scroll box's own visible width instead, and stays put
  // (sticky) while the table scrolls sideways underneath.
  const scrollBoxRef = useRef<HTMLDivElement>(null);
  const [scrollBoxWidth, setScrollBoxWidth] = useState(0);
  useEffect(() => {
    const el = scrollBoxRef.current;
    if (!el) return;
    const measure = () => setScrollBoxWidth(el.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  });
  const { history: historyState, refreshHistory } = useAppData();
  const history = historyState.entries ?? [];
  const loadError = historyState.error;
  const historyLoading = historyState.loading;
  const [filterText, setFilterText] = useState("");
  const [statutFilter, setStatutFilter] = useState<"all" | "en_vol" | "atterri">("all");
  const [sortKey, setSortKey] = useState<SortKey>("calcule_le");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Refreshed on every visit — silently, in the background, if a previous
  // visit already populated the cache (refreshHistory only flips `loading`
  // back on when there's nothing cached yet, cf. AppDataContext), so this
  // never blanks a list that's already on screen just to show the same
  // data again a moment later.
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  // Same pulse indicator as the Search page's title, timed to the title's
  // own split-flap flicker rather than appearing immediately — starting it
  // before "AGGREGATE VIEW" has finished settling would draw the eye away
  // from a title that isn't done animating yet. Only ever shows while the
  // history fetch is still in flight; if it resolves before the title
  // settles, indicatorKind below just never turns "loading" at all.
  const [titleSettled, setTitleSettled] = useState(false);
  const FADE_MS = 320;
  const indicatorKind: "loading" | null = titleSettled && historyLoading ? "loading" : null;
  const [renderedKind, setRenderedKind] = useState<"loading" | null>(null);
  const [indicatorVisible, setIndicatorVisible] = useState(false);
  const indicatorAnimRef = useRef(0);
  useEffect(() => {
    if (indicatorKind) {
      setRenderedKind(indicatorKind);
      // Double rAF — see RechercheVol.tsx's identical pulse for why a
      // single rAF isn't enough to guarantee the opacity:0 state actually
      // paints before switching to opacity:1.
      const id1 = requestAnimationFrame(() => {
        const id2 = requestAnimationFrame(() => setIndicatorVisible(true));
        indicatorAnimRef.current = id2;
      });
      indicatorAnimRef.current = id1;
      return () => cancelAnimationFrame(indicatorAnimRef.current);
    }
    setIndicatorVisible(false);
    const timer = setTimeout(() => setRenderedKind(null), FADE_MS);
    return () => clearTimeout(timer);
  }, [indicatorKind]);
  const PULSE_SLOT = 34; // matches PulseRing's own SVG size exactly

  const scoresByFeature = useMemo(() => {
    const result: Record<FeatureKey, number[]> = { anomalie: [], ecart: [], directness: [] };
    for (const entry of history) {
      for (const { key } of FEATURE_OPTIONS) {
        const v = extractScore(entry, key);
        if (v !== null) result[key].push(v);
      }
    }
    return result;
  }, [history]);

  const filtered = useMemo(() => {
    let rows = history;
    if (filterText.trim()) {
      const q = filterText.trim().toLowerCase();
      rows = rows.filter((h) => h.identifiant.toLowerCase().includes(q));
    }
    if (statutFilter !== "all") rows = rows.filter((h) => h.statut === statutFilter);

    const dir = sortDir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      if (sortKey === "identifiant") return dir * a.identifiant.localeCompare(b.identifiant);
      if (sortKey === "statut") return dir * a.statut.localeCompare(b.statut);
      if (sortKey === "calcule_le") return dir * (new Date(a.calcule_le).getTime() - new Date(b.calcule_le).getTime());
      // typecode/categorie: null (unresolved aircraft, cf. pipeline.py) goes
      // last regardless of direction — same "missing isn't the worst real
      // value" convention as the score columns below.
      if (sortKey === "typecode") {
        if (a.typecode === null && b.typecode === null) return 0;
        if (a.typecode === null) return 1;
        if (b.typecode === null) return -1;
        return dir * a.typecode.localeCompare(b.typecode);
      }
      if (sortKey === "categorie") {
        if (a.categorie === null && b.categorie === null) return 0;
        if (a.categorie === null) return 1;
        if (b.categorie === null) return -1;
        return dir * CATEGORY_LABELS[a.categorie].localeCompare(CATEGORY_LABELS[b.categorie]);
      }
      // Numeric score columns: airborne flights have no score at all (null,
      // not just a low one) — they belong after every scored flight
      // regardless of which direction is currently showing (max-first or
      // min-first), not sorted as if a missing score were the worst
      // possible real one.
      const av = extractScore(a, sortKey);
      const bv = extractScore(b, sortKey);
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return dir * (av - bv);
    });
  }, [history, filterText, statutFilter, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("desc");
    }
  }

  // The currently-expanded row's own scores drive which bar lights up in
  // each of the three charts above — null (no selection, or that feature
  // has no score for this flight) means "highlight nothing", not "highlight
  // bin 0".
  const selectedEntry = expandedId ? (history.find((h) => rowKey(h) === expandedId) ?? null) : null;

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ margin: 0, marginBottom: 8, height: 40, display: "flex", alignItems: "center" }}>
          <SplitFlapText
            text="AGGREGATE VIEW"
            style={{ fontFamily: "var(--font-mono)", fontSize: 31, fontWeight: 700, letterSpacing: 2 }}
            onSettled={() => setTitleSettled(true)}
          />
          <span style={{ marginLeft: 12, width: PULSE_SLOT, height: PULSE_SLOT, flexShrink: 0, position: "relative" }} aria-hidden="true">
            {renderedKind && (
              <span
                style={{ position: "absolute", inset: 0, opacity: indicatorVisible ? 1 : 0, transition: `opacity ${FADE_MS}ms ease` }}
                aria-label="Loading"
                role="status"
              >
                <PulseRing color="var(--green)" />
              </span>
            )}
          </span>
        </h1>
        <p style={{ color: "var(--text-faint)", fontSize: 14.5, margin: 0 }}>
          Score distributions and full history across every flight ever searched — not just this session.
        </p>
        <WakeupBanner visible={wakingUp} />
      </div>

      {loadError ? (
        <div style={{ ...cardStyle, marginBottom: 20 }}>
          <p role="alert" style={{ color: "var(--red)", fontSize: 14.5, margin: 0 }}>
            {loadError}
          </p>
        </div>
      ) : (
        <div style={{ display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 20 }}>
          {FEATURE_OPTIONS.map((f) => {
            const values = scoresByFeature[f.key];
            const highlightBin = selectedEntry ? binIndexFor(extractScore(selectedEntry, f.key)) : null;
            return (
              <div key={f.key} style={{ ...cardStyle, flex: "1 1 300px", minWidth: 0 }}>
                {values.length > 0 ? (
                  <HistogrammeScores title={f.label} counts={toBins(values)} good={f.good} highlightBin={highlightBin} />
                ) : (
                  <>
                    <p style={{ fontSize: 13, color: "#fff", opacity: 0.8, margin: "0 0 8px", textAlign: "center", textTransform: "uppercase", letterSpacing: 0.6 }}>{f.label}</p>
                    <p style={{ color: "var(--text-faint)", fontSize: 13.5, margin: 0, textAlign: "center" }}>
                      {historyLoading ? "Values loading" : "No values yet."}
                    </p>
                  </>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
          <h3 style={{ ...cardTitleStyle, color: "#fff", opacity: 0.8 }}>Flight search history ({history.length})</h3>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <div style={{ position: "relative" }}>
              <SearchIcon size={14} style={{ position: "absolute", left: 10, top: "50%", transform: "translateY(-50%)", color: "var(--text-faint)" }} />
              <input
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                placeholder="Filter by identifier"
                style={{ ...selectStyle, padding: "8px 12px 8px 30px" }}
              />
            </div>
            <select value={statutFilter} onChange={(e) => setStatutFilter(e.target.value as typeof statutFilter)} style={selectStyle}>
              <option value="all">All statuses</option>
              <option value="en_vol">Airborne</option>
              <option value="atterri">Landed</option>
            </select>
          </div>
        </div>

        {history.length === 0 ? (
          <p style={{ color: "var(--text-faint)", fontSize: 14.5, margin: 0 }}>
            {loadError
              ? loadError
              : historyLoading
                ? "Flight history loading"
                : "Nothing searched yet — flights looked up on the Search page will show up here."}
          </p>
        ) : filtered.length === 0 ? (
          <p style={{ color: "var(--text-faint)", fontSize: 14.5, margin: 0 }}>No flight matches this filter.</p>
        ) : (
          <div ref={scrollBoxRef} style={{ maxHeight: 560, overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead>
                <tr style={{ textAlign: "left", fontSize: 11.5, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  <SortHeader label="Flight" sortKey="identifiant" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Aircraft type" sortKey="typecode" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Category" sortKey="categorie" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Status" sortKey="statut" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Anomaly" sortKey="anomalie" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Deviation" sortKey="ecart" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Directness" sortKey="directness" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                  <SortHeader label="Searched" sortKey="calcule_le" activeKey={sortKey} dir={sortDir} onClick={toggleSort} />
                </tr>
              </thead>
              <tbody style={{ fontFamily: "var(--font-mono)" }}>
                {filtered.map((entry) => {
                  const key = rowKey(entry);
                  const isOpen = expandedId === key;
                  const anomScore = entry.anomalie?.score ?? null;
                  const ecartScore = entry.ecart_trajectoire?.score_global ?? null;
                  const dirScore = entry.directness?.score ?? null;
                  const worst = worstSeverity([
                    { score: anomScore, good: 1 },
                    { score: ecartScore, good: 0 },
                    { score: dirScore, good: 1 },
                  ]);
                  const panelAccentColor = worst !== null ? severityColor(worst.score, worst.good) : "var(--border)";
                  return (
                    <Fragment key={key}>
                      <tr onClick={() => setExpandedId(isOpen ? null : key)} style={{ borderTop: "1px solid var(--border)", cursor: "pointer" }}>
                        <td style={{ padding: "8px", fontWeight: 600 }}>{entry.identifiant}</td>
                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>{entry.typecode ?? "—"}</td>
                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>{entry.categorie ? CATEGORY_LABELS[entry.categorie] : "—"}</td>
                        <td style={{ padding: "8px", color: "var(--text-muted)" }}>{entry.statut === "en_vol" ? "Airborne" : "Landed"}</td>
                        <td style={{ padding: "8px", color: anomScore !== null ? severityColor(anomScore, 1) : "var(--text-faint)" }}>
                          {anomScore !== null ? anomScore.toFixed(3) : "—"}
                        </td>
                        <td style={{ padding: "8px", color: ecartScore !== null ? severityColor(ecartScore, 0) : "var(--text-faint)" }}>
                          {ecartScore !== null ? ecartScore.toFixed(3) : "—"}
                        </td>
                        <td style={{ padding: "8px", color: dirScore !== null ? severityColor(dirScore, 1) : "var(--text-faint)" }}>
                          {dirScore !== null ? dirScore.toFixed(3) : "—"}
                        </td>
                        <td style={{ padding: "8px", color: "var(--text-faint)", fontSize: 12.5 }}>{new Date(entry.calcule_le).toLocaleString()}</td>
                      </tr>
                      {isOpen && (
                        <tr>
                          <td colSpan={8} style={{ padding: "10px 8px" }}>
                            <ExpandingDetail key={key}>
                              {/* Half the table's width (was nearly full) —
                                  a 3-column side-by-side layout doesn't fit
                                  cleanly at that width any more, so this is a
                                  single stacked column instead: a compact
                                  trajectory strip up top, then each score
                                  detail gets the panel's full (now narrower)
                                  width in turn rather than a cramped sliver
                                  of it. */}
                              <div
                                style={{
                                  ...(isMobile
                                    ? // Fills the visible box, with the cell's own 8px
                                      // on the left and a wider 20px on the right.
                                      { width: Math.max(scrollBoxWidth - 28, 0), boxSizing: "border-box", position: "sticky", left: 8 }
                                    : { maxWidth: "50%", minWidth: 340 }),
                                  // Worst of the flight's three scores frames
                                  // the whole panel — same "which diagnostic
                                  // is driving concern" signal as the Search
                                  // page's status pill, just spelled as a
                                  // border here since this panel has no
                                  // single badge to colour.
                                  border: `2px solid ${panelAccentColor}`,
                                  borderRadius: 10,
                                  background: "var(--surface-2)",
                                  padding: 14,
                                }}
                              >
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                                  <span style={{ fontSize: 13, color: "var(--text-muted)" }}>{entry.identifiant}</span>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      setExpandedId(null);
                                    }}
                                    style={{ background: "none", border: "none", color: "var(--text-faint)", cursor: "pointer", display: "flex", alignItems: "center", gap: 4, fontSize: 13 }}
                                  >
                                    <X size={14} /> Close
                                  </button>
                                </div>
                                <div style={{ marginBottom: 16 }}>
                                  <VueTrajectoire trajectoire={entry.trajectoire} severity={anomScore} enVol={entry.statut === "en_vol"} />
                                </div>
                                <div style={{ paddingTop: 16, borderTop: "1px solid var(--border)", marginBottom: 16 }}>
                                  <p style={{ fontSize: 11.5, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: 0.5, margin: "0 0 10px" }}>
                                    Trajectory anomaly
                                  </p>
                                  <ScoreAnomalie anomalie={entry.anomalie} enVol={entry.statut === "en_vol"} />
                                </div>
                                <div style={{ paddingTop: 16, borderTop: "1px solid var(--border)", marginBottom: 16 }}>
                                  <p style={{ fontSize: 11.5, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: 0.5, margin: "0 0 10px" }}>
                                    Deviation vs. optimal profile
                                  </p>
                                  <JaugeEcart ecart={entry.ecart_trajectoire} enVol={entry.statut === "en_vol"} categorie={entry.categorie} />
                                </div>
                                <div style={{ paddingTop: 16, borderTop: "1px solid var(--border)" }}>
                                  <p style={{ fontSize: 11.5, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: 0.5, margin: "0 0 10px" }}>
                                    Route directness
                                  </p>
                                  <DirectnessGauge directness={entry.directness} enVol={entry.statut === "en_vol"} />
                                </div>
                              </div>
                            </ExpandingDetail>
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
