import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";
import { Fingerprint, Gauge, Plane, PlaneLanding, PlaneTakeoff, Search, Shuffle, Wind } from "lucide-react";
import { ApiError, FlightNotFoundError, getFlight, getRandomFlight, type FlightResponse, type StatutVol } from "../services/api";
import { SplitFlapText } from "../components/SplitFlapText";
import { PulseRing } from "../components/PulseRing";
import { WakeupBanner } from "../components/WakeupBanner";
import { VueTrajectoire } from "../components/visualisations/VueTrajectoire";
import { JaugeEcart } from "../components/visualisations/JaugeEcart";
import { ScoreAnomalie } from "../components/visualisations/ScoreAnomalie";
import { DirectnessGauge } from "../components/visualisations/DirectnessGauge";
import { severityColor, worstSeverity } from "../lib/severity";
import { useIsMobile } from "../lib/useIsMobile";
import { useAppData } from "../lib/AppDataContext";
import { useBackendWakeup } from "../lib/useBackendWakeup";

type Etat =
  | { statut: "repos" }
  | { statut: "chargement" }
  | { statut: "succes"; vol: FlightResponse }
  | { statut: "erreur"; message: string };

const inputStyle: React.CSSProperties = {
  padding: "13px 18px",
  // Border colour lives in the .search-input CSS rule instead of here —
  // an inline style always wins over a stylesheet rule regardless of
  // selector, so a border set here would make the .search-input:focus
  // green border unreachable.
  borderRadius: 8,
  // Subtle green tint, fading left-to-right, layered over the usual dark
  // surface colour (kept behind so the tint stays faint and text stays
  // legible either way).
  background: "linear-gradient(to right, rgba(47, 230, 164, 0.1), rgba(47, 230, 164, 0.02)), var(--surface)",
  color: "var(--text)",
  font: "inherit",
  fontSize: 16.5,
  flex: "1 1 320px",
  minWidth: 0,
};

const buttonStyle: React.CSSProperties = {
  padding: "17px 24px",
  // Same 1px width as ghostButtonStyle's border, just the same colour as
  // the background (i.e. invisible) — the point is that only the *colour*
  // ever needs to transition between the two variants, not the width too
  // (a border-none -> 1px-solid swap would still jump instead of animate).
  border: "1px solid var(--green)",
  borderRadius: 8,
  background: "var(--green)",
  color: "#04110c",
  font: "inherit",
  fontSize: 15,
  fontWeight: 700,
  cursor: "pointer",
  display: "inline-flex",
  alignItems: "center",
  gap: 8,
  whiteSpace: "nowrap",
  // Longer than most transitions on this page (was 300ms) — deliberately,
  // to survive the frame-timing collision described where this style is
  // swapped (RechercheVol's own rechercherAleatoire): even with that fixed,
  // a bit more runway makes the fade read as smooth rather than snappy on
  // a slower device. willChange hints the browser to give this button its
  // own compositing layer, so its paint doesn't get bundled in with the
  // surrounding cards/board repainting at the same time.
  transition: "background-color 400ms ease, color 400ms ease, border-color 400ms ease",
  willChange: "background-color, color, border-color",
};

const ghostButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  background: "var(--surface-3)",
  color: "var(--text)",
  border: "1px solid rgba(47, 230, 164, 0.45)",
};

const randomMenuItemStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  width: "100%",
  padding: "10px 16px",
  border: "none",
  background: "transparent",
  color: "var(--text)",
  font: "inherit",
  fontSize: 12,
  fontWeight: 300,
  textTransform: "uppercase",
  letterSpacing: 0.8,
  cursor: "pointer",
  whiteSpace: "nowrap",
  textAlign: "left",
};

const randomMenuItemActiveStyleAirborne: React.CSSProperties = {
  background: "var(--blue-dim)",
  color: "var(--blue)",
  fontWeight: 400,
};

const randomMenuItemActiveStyleLanded: React.CSSProperties = {
  background: "var(--green-dim)",
  color: "var(--green)",
  fontWeight: 400,
};

// Placeholder for the flight identifier — chosen once, roughly matching a
// typical callsign/registration's length, and never recomputed, so the
// board doesn't shift width once the real identifier replaces it.
const PLACEHOLDER_IDENTIFIANT = "-------";

// Same border as the characteristics tiles — "more white" was about the
// card *titles*, not this outline.
const cardStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  background: "var(--surface-3)",
  borderRadius: 14,
  padding: 20,
};

const cardTitleStyle: React.CSSProperties = {
  marginTop: 0,
  marginBottom: 16,
  fontSize: 13,
  color: "var(--text-faint)",
  fontWeight: 700,
  textTransform: "uppercase",
  letterSpacing: 1,
};

// The "Trajectory" / "Flight diagnostic" card titles specifically — bright
// white and a size of their own, unlike the muted, smaller cardTitleStyle
// used for their sub-headings ("Trajectory anomaly", etc.).
const cardTitleWhiteStyle: React.CSSProperties = {
  ...cardTitleStyle,
  fontSize: 14,
  color: "rgba(255, 255, 255, 0.9)",
};

const placeholderTextStyle: React.CSSProperties = {
  color: "var(--text-faint)",
  fontSize: 13.5,
  margin: 0,
};

// Collapsed to a sliver (just enough for the placeholder line, or a custom
// height to match a neighbouring element) until a flight is loaded, then
// opens up to reveal the map / gauges — a max-height transition rather than
// the content just appearing at full size. min-height is pinned to the same
// collapsed value: max-height alone is only a ceiling, so when the
// placeholder text is shorter than the target collapsed height the box
// would just shrink to fit its content instead of actually matching it —
// min-height forces it open to that height even though there's nothing
// tall enough inside to reach it on its own.
const EXPAND_EASE = "950ms cubic-bezier(0.22, 1, 0.36, 1)";
const EXPAND_TRANSITION = `max-height ${EXPAND_EASE}, min-height ${EXPAND_EASE}`;
// A gentler curve for the Trajectory card specifically: the shared
// EXPAND_EASE above is heavily front-loaded (reaches ~90% of its target in
// the first fifth of its own duration, then eases a long, barely-visible
// tail) — snappy for the panels it was designed for, but felt like a jump
// cut for this one now that its *content* also crossfades (below), since
// the box was already basically at full height before the fade had gotten
// anywhere. A more even ease-out and a touch more time gives the two
// motions room to actually read as one coordinated reveal.
const TRAJECTORY_EXPAND_EASE = "1100ms cubic-bezier(0.33, 1, 0.68, 1)";
const TRAJECTORY_EXPAND_TRANSITION = `max-height ${TRAJECTORY_EXPAND_EASE}, min-height ${TRAJECTORY_EXPAND_EASE}`;

function expandStyle(expanded: boolean, maxHeightOpen: number, collapsedHeight = 28): React.CSSProperties {
  return {
    maxHeight: expanded ? maxHeightOpen : collapsedHeight,
    minHeight: expanded ? 0 : collapsedHeight,
    overflow: "hidden",
    transition: EXPAND_TRANSITION,
  };
}

// Measures the *natural* height of whatever is currently rendered inside —
// a hardcoded "big enough" ceiling for the open state has twice now turned
// out not to be (a flight with a longer contributing-features list, or the
// trajectory's own raw-data table once expanded, both overflow a fixed
// guess and get clipped by expandStyle's overflow:hidden). A ResizeObserver
// keeps this correct no matter what's inside, including content that
// changes size after the fact, like that same raw-data table.
function useAutoHeight(): [React.RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setHeight(el.scrollHeight);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, height];
}

// The data source ("vedette"/"cache") used to show here as a small tally-
// light-style indicator next to the pill — removed once "vedette" became
// the normal, expected case for every fresh pool draw rather than a
// noteworthy state worth calling out.
function StatutBadge({
  statut,
  anomalieScore,
  ecartScore,
  directnessScore,
}: {
  statut: string;
  anomalieScore: number | null;
  ecartScore: number | null;
  directnessScore: number | null;
}) {
  const enVol = statut === "en_vol";
  // Worst of the three landed-flight diagnostics drives the pill colour.
  const worst = worstSeverity([
    { score: anomalieScore, good: 1 },
    { score: ecartScore, good: 0 },
    { score: directnessScore, good: 1 },
  ]);
  // Landed but no score at all (unidentified aircraft, cf. pipeline.py --
  // the Flight diagnostic card below stays on its "not available" state):
  // the pill falls back to the neutral --surface-2 background, against
  // which the near-black text used everywhere else would be almost
  // unreadable -- white at 80% opacity instead, same as the rest of the
  // app's muted-on-dark text.
  const noScore = !enVol && worst === null;
  const background = enVol ? "var(--blue)" : worst !== null ? severityColor(worst.score, worst.good) : "var(--surface-2)";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "7px 14px",
        borderRadius: 999,
        border: "1px solid var(--border-strong)",
        background,
        color: noScore ? "rgba(255, 255, 255, 0.8)" : "#04110c",
        fontWeight: 700,
        fontSize: 14,
      }}
    >
      {enVol ? "Airborne" : "Landed"}
    </span>
  );
}

function MetaTile({
  label,
  mobileLabel,
  value,
  unit,
  connu,
  icon,
}: {
  label: string;
  /** Shorter stand-in for phone-width screens — the tile grid stays a
   * single row of 6 there too (unlike everything else on this page, that's
   * deliberately not changing on mobile), so a shrunk font alone isn't
   * enough for the longest labels ("Current altitude"/"Current speed") to
   * stay on one line without clipping against the tile's own overflow:hidden. */
  mobileLabel?: string;
  value: string;
  unit?: string;
  connu: boolean;
  icon: ReactNode;
}) {
  const isMobile = useIsMobile();
  const displayLabel = isMobile && mobileLabel ? mobileLabel : label;
  return (
    <div
      style={{
        position: "relative",
        overflow: "hidden",
        // Tighter on a phone-width screen: six tiles share one row there,
        // so every horizontal pixel of padding costs value/unit room.
        padding: isMobile ? "10px 3px" : "12px 12px",
        border: "1px solid var(--border)",
        // Same background as the trajectory/diagnostic cards — no separate
        // tint of its own.
        background: "var(--surface-3)",
        borderRadius: 12,
        minWidth: 0,
      }}
    >
      {/* Oversized watermark, deliberately cropped by the tile — decoration
          only, the label/value pair below still carries all the meaning. */}
      <span
        aria-hidden="true"
        style={{
          position: "absolute",
          right: -10,
          bottom: -14,
          display: "flex",
          color: "#fff",
          opacity: 0.25,
          pointerEvents: "none",
        }}
      >
        {icon}
      </span>
      <div style={{ position: "relative", minWidth: 0 }}>
        <div
          style={{
            fontSize: isMobile ? 7 : 10,
            color: "#fff",
            opacity: 0.8,
            textTransform: "uppercase",
            letterSpacing: isMobile ? 0.2 : 0.4,
            // Phone widths only: the name is centred across the tile.
            textAlign: isMobile ? "center" : undefined,
            whiteSpace: "nowrap",
          }}
        >
          {displayLabel}
        </div>
        {/* Unit rendered as its own static span, never fed to SplitFlapText —
            it sits at a fixed spot right after the value and never flickers,
            so the tile's internal layout doesn't shift with the value's
            length the way it would if the unit were part of the animated
            string. */}
        {/* Phone widths: the reading (value plus its unit) is centred
            across the tile, like the name above it. */}
        <div style={{ whiteSpace: "nowrap", textAlign: isMobile ? "center" : undefined }}>
          <SplitFlapText
            text={value}
            tickMs={90}
            reverse={!connu}
            sequential
            style={{
              display: "inline-block",
              fontFamily: "var(--font-mono)",
              // Same value size for all six tiles on a phone (not sized per
              // tile), tracking the viewport: the formula is the largest
              // size at which a 6-character reading (a registration) still
              // fits a tile's inner width at that screen width, so it
              // grows on wider phones instead of staying pinned to the
              // narrowest one. Floor/ceiling keep it sane at the extremes.
              fontSize: isMobile ? "clamp(11px, calc(4.63vw - 4.63px), 15px)" : 18,
              fontWeight: 700,
              color: "#fff",
              whiteSpace: "nowrap",
            }}
          />
          {/* Shown regardless of whether the value is known yet, so the
              unit's position never shifts once the real reading arrives. */}
          {unit && (
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: isMobile ? 7 : 13.5,
                fontWeight: 400,
                color: "var(--text)",
                opacity: 0.45,
                marginLeft: isMobile ? 1 : 8,
              }}
            >
              {unit}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

/** Crossfades the "run a search" placeholder out and the map in, instead of
 * the instant content swap a plain `vol ? A : B` ternary gives — the
 * placeholder stays mounted (opacity 0, absolutely positioned so it doesn't
 * hold the layout open) rather than unmounting immediately, and the map
 * mounts at opacity 0 and is nudged to 1 a frame later (double rAF, same
 * "guarantee a real paint at the starting state first" reasoning as the
 * search-status ring's own fade) so the transition actually plays instead
 * of the two style changes landing in the same paint. */
function TrajectoryReveal({ vol }: { vol: FlightResponse | null }) {
  const [mapVisible, setMapVisible] = useState(false);
  useEffect(() => {
    if (!vol) {
      setMapVisible(false);
      return;
    }
    let id2 = 0;
    const id1 = requestAnimationFrame(() => {
      id2 = requestAnimationFrame(() => setMapVisible(true));
    });
    return () => {
      cancelAnimationFrame(id1);
      cancelAnimationFrame(id2);
    };
  }, [vol]);

  return (
    <div style={{ position: "relative" }}>
      <p
        style={{
          ...placeholderTextStyle,
          opacity: vol ? 0 : 1,
          transition: "opacity 350ms ease",
          ...(vol ? { position: "absolute", top: 0, left: 0, pointerEvents: "none" } : null),
        }}
      >
        Run a search to plot the trajectory.
      </p>
      {vol && (
        <div style={{ opacity: mapVisible ? 1 : 0, transition: "opacity 500ms ease" }}>
          <VueTrajectoire
            trajectoire={vol.trajectoire}
            severity={vol.anomalie?.score ?? null}
            enVol={vol.statut === "en_vol"}
            origineVille={vol.origine_ville}
            destinationVille={vol.destination_ville}
          />
        </div>
      )}
    </div>
  );
}

function PanneauAppareil({ vol }: { vol: FlightResponse | null }) {
  const isMobile = useIsMobile();
  const altitude = vol?.altitude_actuelle != null ? `${Math.round(vol.altitude_actuelle)}` : "-----";
  const vitesse = vol?.vitesse_actuelle != null ? `${Math.round(vol.vitesse_actuelle * 3.6)}` : "---";
  return (
    // Fixed 6-column grid so all six tiles sit on a single row — the row's
    // total width always matches the trajectory card below it (same parent
    // column), so "Current speed"'s right edge lines up with the map's.
    // Each placeholder's dash count is chosen once to roughly match that
    // field's typical real length (e.g. a 4-letter ICAO code, a 3-digit
    // speed), so the board doesn't visibly resize once real data lands.
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: isMobile ? 4 : 10 }}>
      <MetaTile
        label="Aircraft type"
        mobileLabel="Aircraft"
        value={vol?.typecode ?? "----"}
        connu={!!vol?.typecode}
        icon={<Plane size={70} strokeWidth={0.75} />}
      />
      <MetaTile
        label="Registration"
        value={vol?.immatriculation ?? "------"}
        connu={!!vol?.immatriculation}
        icon={<Fingerprint size={70} strokeWidth={0.75} />}
      />
      <MetaTile label="Origin" value={vol?.origine ?? "----"} connu={!!vol?.origine} icon={<PlaneTakeoff size={70} strokeWidth={0.75} />} />
      <MetaTile
        label="Destination"
        value={vol?.destination ?? "----"}
        connu={!!vol?.destination}
        icon={<PlaneLanding size={70} strokeWidth={0.75} />}
      />
      <MetaTile
        label="Current altitude"
        mobileLabel="Cur altitude"
        value={altitude}
        unit="m"
        connu={vol?.altitude_actuelle != null}
        icon={<Gauge size={70} strokeWidth={0.75} />}
      />
      <MetaTile
        label="Current speed"
        mobileLabel="Cur speed"
        value={vitesse}
        unit="km/h"
        connu={vol?.vitesse_actuelle != null}
        icon={<Wind size={70} strokeWidth={0.75} />}
      />
    </div>
  );
}

export function RechercheVol() {
  const isMobile = useIsMobile();
  const wakingUp = useBackendWakeup();
  const { history, refreshHistory, addToHistory, exampleIdentifiant, refreshExample } = useAppData();
  // Only on the very first visit this session (exampleIdentifiant is cached
  // above the router, cf. AppDataContext.tsx) — a revisit already has one,
  // and every successful search/random draw below refreshes it anyway.
  useEffect(() => {
    if (exampleIdentifiant === null) refreshExample();
  }, [exampleIdentifiant, refreshExample]);
  // Same idea, for the Aggregate view's history: pre-fetches it in the
  // background the moment this page (almost always the first one visited)
  // mounts, rather than only starting that fetch once the user actually
  // navigates to Aggregate view — so it's often already loaded, or at least
  // well under way, by the time they get there. Only when nothing's loaded
  // yet this session (entries === null): a revisit doesn't need to refetch
  // just from passing through Search again, and a search run before this
  // resolves is harmless either way (AppDataContext.tsx's own requestId
  // guard means whichever refreshHistory() call was issued last always
  // wins — the one below, after a search, or this one, whichever is later).
  useEffect(() => {
    if (history.entries === null) refreshHistory();
  }, [history.entries, refreshHistory]);
  const [identifiant, setIdentifiant] = useState("");
  const [etat, setEtat] = useState<Etat>({ statut: "repos" });
  const [showRandomMenu, setShowRandomMenu] = useState(false);
  // Chosen by clicking a menu option while hovering — persists as the
  // current filter until changed. The search itself only ever runs from
  // the main button's own click, never from picking a filter.
  const [randomFilter, setRandomFilter] = useState<StatutVol | null>(null);
  // True only while a random-flight fetch is actually in flight — lets the
  // button swap to its active look even when no filter is armed (a bare
  // click with nothing selected in the dropdown), which randomFilter alone
  // can't express.
  const [randomLoading, setRandomLoading] = useState(false);
  const vol = etat.statut === "succes" ? etat.vol : null;

  // One last red blink of the search dot on error, then it's replaced by
  // the message text — rather than the message appearing immediately
  // alongside a dot that's still mid-animation.
  const [flashError, setFlashError] = useState(false);
  useEffect(() => {
    if (etat.statut !== "erreur") return;
    setFlashError(true);
    const timer = setTimeout(() => setFlashError(false), 550);
    return () => clearTimeout(timer);
  }, [etat.statut]);

  // One persistent slot for the pulse indicator, faded in/out via opacity
  // rather than mounted/unmounted outright — a plain `&&` conditional gives
  // the browser no time to animate before the element is simply gone.
  // Loading -> error swaps colour/animation in place (still visible, no
  // fade needed there, and no risk of both indicators's margins briefly
  // coexisting) while both loading -> idle and error -> idle fade out over
  // FADE_MS before actually unmounting.
  const FADE_MS = 320;
  const indicatorKind: "loading" | "error" | null = etat.statut === "chargement" ? "loading" : flashError ? "error" : null;
  const [renderedKind, setRenderedKind] = useState<"loading" | "error" | null>(null);
  const [indicatorVisible, setIndicatorVisible] = useState(false);
  const idRef = useRef(0);
  useEffect(() => {
    if (indicatorKind) {
      setRenderedKind(indicatorKind);
      // Double rAF, not one: a single rAF callback still runs before the
      // browser has necessarily *painted* the just-mounted opacity:0 state,
      // so the very next style change (to opacity:1) can land in the same
      // paint and skip the transition entirely — the classic "no reflow
      // between the two states" gotcha. Waiting a full extra frame
      // guarantees a real paint at 0 happens first.
      const id1 = requestAnimationFrame(() => {
        const id2 = requestAnimationFrame(() => setIndicatorVisible(true));
        idRef.current = id2;
      });
      idRef.current = id1;
      return () => cancelAnimationFrame(idRef.current);
    }
    setIndicatorVisible(false);
    const timer = setTimeout(() => setRenderedKind(null), FADE_MS);
    return () => clearTimeout(timer);
  }, [indicatorKind]);

  const PULSE_SLOT = 34; // matches PulseRing's own SVG size exactly

  // The Flight Diagnostic card's collapsed (pre-search) height is pinned to
  // the characteristics row's own rendered height, so the two line up
  // side by side instead of the diagnostic card being a token sliver next
  // to a much taller row. Measured once — the row's height doesn't change
  // with its content (always a single line of tiles).
  const panneauRef = useRef<HTMLDivElement>(null);
  const [panneauHeight, setPanneauHeight] = useState(28);
  useEffect(() => {
    if (panneauRef.current) setPanneauHeight(panneauRef.current.offsetHeight);
  }, []);

  const [trajContentRef, trajContentHeight] = useAutoHeight();
  const [diagContentRef, diagContentHeight] = useAutoHeight();

  // On a phone the cards are stacked, so there is no neighbouring row to
  // line up with — and the tile row's height is too short for the title
  // plus the placeholder line, which wraps onto two lines at that width.
  // The collapsed card is sized to its own content instead (measured
  // placeholder + title, padding and border), never below the tile row.
  const diagCollapsedHeight = isMobile ? Math.max(panneauHeight, diagContentHeight + 48) : panneauHeight;

  // Random flight is "active" off randomFilter (a filter is armed) or
  // randomLoading (a random search is running right now even with no
  // filter picked) — either way the buttons should swap to show Random
  // flight as the action currently in charge, then fall back to normal
  // Search-mode styling once the fetch settles (randomLoading clears and,
  // absent a filter, randomFilter was never set in the first place).
  const randomActive = randomFilter !== null || randomLoading;

  // Typing means "I want Search now": the random-flight filter is actually
  // cleared, not just visually hidden, so clicking Random flight afterwards
  // doesn't silently still apply a filter the UI no longer shows.
  function handleIdentifiantChange(event: React.ChangeEvent<HTMLInputElement>) {
    setIdentifiant(event.target.value);
    if (event.target.value !== "" && randomFilter !== null) setRandomFilter(null);
  }

  // Closing the dropdown is debounced: a real mouse path from the button
  // down into the menu can (browser-dependent) report a fleeting
  // mouseleave on the wrapper mid-transit, which — without this — closes
  // the menu right as the pointer arrives, so it only reopens on a second
  // hover. Cancelling a pending close on the next mouseenter absorbs that.
  const closeMenuTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  function openRandomMenu() {
    if (closeMenuTimer.current) {
      clearTimeout(closeMenuTimer.current);
      closeMenuTimer.current = null;
    }
    setShowRandomMenu(true);
  }
  function closeRandomMenuSoon() {
    closeMenuTimer.current = setTimeout(() => setShowRandomMenu(false), 200);
  }

  async function rechercher(event: FormEvent) {
    event.preventDefault();
    const valeur = identifiant.trim();
    if (!valeur) return;
    setEtat({ statut: "chargement" });
    try {
      const vol = await getFlight(valeur);
      setEtat({ statut: "succes", vol });
      // addToHistory shows this flight in the Aggregate view immediately,
      // client-side, no extra request; refreshHistory's own (slower)
      // background fetch then reconciles it against the real server-side
      // row (mainly calcule_le, an exact timestamp addToHistory can only
      // approximate) once that resolves — see both their own docstrings.
      addToHistory(vol);
      refreshHistory();
      // Keeps the placeholder's example naming a flight that's still
      // "undiscovered" rather than the one just searched — cheap to do
      // here since the backend is already known to be awake at this exact
      // moment (cf. AppDataContext.tsx's own docstring on exampleIdentifiant).
      refreshExample();
      setIdentifiant("");
    } catch (err) {
      setEtat({ statut: "erreur", message: messageErreur(err) });
      setIdentifiant("");
    }
  }

  async function rechercherAleatoire() {
    // Only actually clears anything when no filter was picked: choosing a
    // filter already clears the search text itself (see the menu item
    // handlers below). A bare click with nothing selected is the one path
    // that could otherwise leave stale search text sitting next to a
    // random result that has nothing to do with it.
    setIdentifiant("");
    setRandomLoading(true);
    setEtat({ statut: "chargement" });
    try {
      const vol = await getRandomFlight(randomFilter ?? undefined);
      setEtat({ statut: "succes", vol });
      // Deferred a frame (both here and in `finally` below) rather than set
      // in the same commit as the "succes" state above: that commit already
      // kicks off the heaviest work on the page at once — the trajectory/
      // diagnostic cards unfolding and the identifier board's split-flap
      // animation starting — and measured directly (instrumented, not
      // guessed), the button's own background/text transition was landing
      // in that same busy frame and visibly stalling for ~150-270ms before
      // it could even start, instead of fading immediately. Letting that
      // first heavy frame commit and paint on its own, then flipping the
      // buttons back a frame later, gives the color transition a clear run
      // — most noticeable on mobile, where there's less headroom to absorb
      // the collision. Once a flight has actually been found, the filter's
      // job is done — clearing it drops the label and swaps the buttons
      // back to normal Search-mode styling, rather than leaving Random
      // flight looking "still armed" for a search that already ran.
      requestAnimationFrame(() => setRandomFilter(null));
      // Same reasoning as rechercher() above.
      addToHistory(vol);
      refreshHistory();
      refreshExample();
    } catch (err) {
      setEtat({ statut: "erreur", message: messageErreur(err) });
    } finally {
      requestAnimationFrame(() => setRandomLoading(false));
    }
  }

  // Desktop reveals the filter dropdown on hover, so a click on the button
  // itself has always meant "run it" — there's no separate reveal step to
  // skip. Touch has no hover at all, so without this, a phone's very first
  // tap on the button would fire a search immediately and Airborne/Landed
  // would never be reachable. Only the first tap (menu not open yet, no
  // filter armed) is diverted to opening the menu instead; every other tap
  // — a filter already picked, or a second tap with the menu still open and
  // nothing picked (i.e. "I looked, I want any flight") — runs the search
  // exactly like desktop's click always has. Never triggered on desktop
  // (isMobile is false there), so this changes nothing about its behaviour.
  function handleRandomButtonClick() {
    // A leftover "not found" from a previous free-text search no longer
    // describes anything relevant once Random flight is engaged, even in
    // the mobile branch below where opening the menu doesn't run a search
    // by itself yet — rechercherAleatoire() would clear it once it
    // actually runs, but that's not guaranteed to happen from this one tap.
    if (etat.statut === "erreur") setEtat({ statut: "repos" });
    if (isMobile && randomFilter === null && !randomLoading && !showRandomMenu) {
      setShowRandomMenu(true);
      return;
    }
    setShowRandomMenu(false);
    rechercherAleatoire();
  }

  function messageErreur(err: unknown): string {
    if (err instanceof FlightNotFoundError) return err.message;
    if (err instanceof ApiError) return err.message;
    return "Unexpected error";
  }

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        {/* Fixed height, not just min-height: a flex row otherwise grows to
            fit its tallest child, and the pulse ring (below) is taller than
            the text's own line box — without this, the ring appearing/
            disappearing shifted the title and everything under it up and
            down. Pinning the row's height means the ring can never affect
            layout no matter how it's nudged into vertical alignment. */}
        <h1 style={{ margin: 0, marginBottom: 8, height: 40, display: "flex", alignItems: "center" }}>
          <SplitFlapText
            text="FLIGHT SEARCH"
            style={{ fontFamily: "var(--font-mono)", fontSize: 31, fontWeight: 700, letterSpacing: 2 }}
          />
          {/* A fixed-size slot (34x34, PulseRing's own SVG size exactly),
              always reserved via flexShrink:0 whether or not a ring is
              currently rendered inside it — simpler and more robust than
              measuring the title's width: this way "where does the ring go"
              never depends on the title's rendered size at all, just normal
              flex flow right after it, same as it always was. Its size
              never changes, so the error message placed right after it
              (below) never moves either, whether the ring is showing,
              fading out, or gone. */}
          <span style={{ marginLeft: 12, width: PULSE_SLOT, height: PULSE_SLOT, flexShrink: 0, position: "relative" }} aria-hidden="true">
            {renderedKind && (
              // Opacity is the only thing that transitions — dims up on
              // appearing, dims down on disappearing, instead of popping
              // instantly. No vertical nudge: the slot is already centered
              // in the 40px title row via align-items:center, same as the
              // text next to it — an earlier translateY(6px) here pushed the
              // ring visibly below the text's own baseline.
              <span
                style={{ position: "absolute", inset: 0, opacity: indicatorVisible ? 1 : 0, transition: `opacity ${FADE_MS}ms ease` }}
                aria-label={renderedKind === "loading" ? "Searching" : undefined}
                role={renderedKind === "loading" ? "status" : undefined}
              >
                <PulseRing key={renderedKind} color={renderedKind === "error" ? "var(--red)" : "var(--green)"} singlePulse={renderedKind === "error"} />
              </span>
            )}
          </span>
          {/* Desktop only — to the right of the title, where there's room
              for it next to "FLIGHT SEARCH". On a phone-width screen this
              same span would sit hard against (or wrap under) the title
              itself; the mobile version below the search bar instead has
              its own full-width row. Not gated on !flashError any more
              (same fix as the mobile span below) — withholding the text
              until the pulse dot's own 550ms flash finished, then popping
              it in, read as a blink rather than a clean, immediate
              appearance. */}
          {!isMobile && etat.statut === "erreur" && (
            <span role="alert" style={{ marginLeft: 12, color: "var(--red)", fontSize: 14.5, fontWeight: 600, whiteSpace: "nowrap" }}>
              {etat.message}
            </span>
          )}
        </h1>
        <p style={{ color: "var(--text-faint)", fontSize: 14.5, margin: 0 }}>
          Real trajectory, deviation vs. optimal profile, anomaly score, route directness — from pre-recorded ADS-B data.
        </p>
        <WakeupBanner visible={wakingUp} />
      </div>

      {/* Wrapped in the same grid as the panel below (reused, not
          duplicated) purely so its column track computes to the same width
          — the *search form itself* (input + Search button) then has its
          right edge land exactly on the trajectory card's right edge.
          Random flight sits outside that constraint, positioned right after
          it, so it's the one allowed to extend past that boundary rather
          than the other way around. */}
      <div className="dashboard-grid">
        <div style={{ position: "relative" }}>
          <form onSubmit={rechercher} style={{ display: "flex", gap: 10 }}>
            <input
              type="text"
              className="search-input"
              value={identifiant}
              onChange={handleIdentifiantChange}
              placeholder={exampleIdentifiant ? `Registration or callsign (e.g. ${exampleIdentifiant})` : "Registration or callsign"}
              style={inputStyle}
            />
            {/* Once a filter is picked, Random flight becomes the "loaded and
                ready" action — so the two buttons swap looks (filled vs.
                outline) to signal that, rather than Search staying visually
                primary while a different action is actually queued up.
                Typing overrides this back to Search, since that's now the
                action that's actually about to run. */}
            <button type="submit" disabled={etat.statut === "chargement"} style={randomActive ? ghostButtonStyle : buttonStyle}>
              <Search size={16} color={randomActive ? "var(--green)" : "#04110c"} style={{ transition: "stroke 400ms ease" }} />
              Search
            </button>
          </form>
          {/* className, not inline position — index.css flips this whole
              layout (below the form, right-aligned) under 640px, and the
              filter label with it (RechercheVol section above explains why
              the dropdown itself lives in its own always-position:relative
              div just below rather than moving with this one: its own
              anchor point needs to stay put across that flip). */}
          <div className="random-flight-wrapper" onMouseEnter={openRandomMenu} onMouseLeave={closeRandomMenuSoon}>
            {/* Mobile only — desktop keeps the error inline next to
                "FLIGHT SEARCH" (the title row above), where there's room
                for it. On a phone, this row (already flex, already
                flex-end, cf. index.css's 640px block) is the only spare
                horizontal space below the search bar that doesn't cost an
                extra line of its own: flex-grow claims whatever the
                filter label + button don't need, so the row's height never
                changes — a long message is clipped with an ellipsis rather
                than wrapping, which would grow it. Not gated on
                !flashError (unlike the desktop span above) — that gate
                exists to let the title row's own red pulse-dot finish its
                550ms flash before the text sits down next to it; this
                message lives nowhere near that dot, so gating it the same
                way just withheld it for 550ms and then popped it in,
                reading as a blink rather than a clean, immediate
                appearance. */}
            {isMobile && etat.statut === "erreur" && (
              <span
                role="alert"
                style={{
                  flex: "1 1 auto",
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: "var(--red)",
                  fontSize: 13,
                  fontWeight: 600,
                  marginRight: 10,
                }}
              >
                {etat.message}
              </span>
            )}
            {/* Stays visible once picked, hover or not — it's informational,
                not part of the hover-to-open dropdown mechanism. Hidden
                while typing a search (same condition as the button swap
                above), since it no longer reflects what's about to run.
                Keyed off randomFilter directly rather than randomActive,
                since a bare click with nothing picked also flips
                randomActive on (via randomLoading) but has no filter to
                actually label. Rendered before the button in the markup
                (not just visually via absolute positioning) so it lands on
                the correct side once the mobile layout turns it into a
                normal flex item, left-to-right. */}
            {randomFilter !== null && (
              <span className="random-filter-label" style={{ color: randomFilter === "en_vol" ? "var(--blue)" : "var(--green)" }}>
                {randomFilter === "en_vol" ? "Airborne" : "Landed"}
              </span>
            )}
            {/* Always position:relative, on both breakpoints — the dropdown
                below anchors to this specific box regardless of whether the
                wrapper around it is itself absolutely positioned (desktop)
                or a static flex item (mobile), so it doesn't need its own
                mobile-specific positioning at all. */}
            <div style={{ position: "relative" }}>
              <button
                type="button"
                onClick={handleRandomButtonClick}
                disabled={etat.statut === "chargement"}
                style={randomActive ? buttonStyle : ghostButtonStyle}
              >
                <Shuffle size={16} color={randomActive ? "#04110c" : "var(--green)"} style={{ transition: "stroke 400ms ease" }} />
                Random flight
              </button>
              {showRandomMenu && (
                <div
                  style={{
                    position: "absolute",
                    // Flush against the button (no gap): a marginTop here would
                    // leave a sliver that belongs to neither element, and moving
                    // the mouse through it fires mouseLeave on the wrapper
                    // before the pointer ever reaches the dropdown — exactly
                    // the "menu doesn't stay open" bug.
                    top: "100%",
                    left: 0,
                    minWidth: "100%",
                    background: "var(--surface-2)",
                    border: "1px solid var(--border-strong)",
                    borderRadius: 8,
                    overflow: "hidden",
                    zIndex: 10,
                    boxShadow: "0 8px 24px rgba(0,0,0,0.4)",
                  }}
                >
                  <button
                    type="button"
                    className="random-menu-item"
                    onClick={() => {
                      setRandomFilter(randomFilter === "en_vol" ? null : "en_vol");
                      setIdentifiant("");
                      if (etat.statut === "erreur") setEtat({ statut: "repos" });
                      // Touch has no hover to close this with afterward (the
                      // desktop mouseleave-based auto-close never fires) —
                      // closing here only on mobile leaves desktop's own
                      // hover-driven behaviour untouched.
                      if (isMobile) setShowRandomMenu(false);
                    }}
                    style={{ ...randomMenuItemStyle, ...(randomFilter === "en_vol" ? randomMenuItemActiveStyleAirborne : null) }}
                  >
                    Airborne
                  </button>
                  <button
                    type="button"
                    className="random-menu-item"
                    onClick={() => {
                      setRandomFilter(randomFilter === "atterri" ? null : "atterri");
                      setIdentifiant("");
                      if (etat.statut === "erreur") setEtat({ statut: "repos" });
                      if (isMobile) setShowRandomMenu(false);
                    }}
                    style={{ ...randomMenuItemStyle, ...(randomFilter === "atterri" ? randomMenuItemActiveStyleLanded : null) }}
                  >
                    Landed
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Always on screen, like a permanent airport display board — filled
          with dashes until a search resolves, then the split-flap effect on
          SplitFlapText fires on its own the moment the real text replaces
          the placeholder, and these two panels expand open to reveal the
          map/gauges rather than popping in already full-size. */}
      <div style={{ marginTop: 28 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
          <h2 style={{ margin: 0 }}>
            <SplitFlapText
              text={vol ? vol.identifiant : PLACEHOLDER_IDENTIFIANT}
              tickMs={60}
              reverse={!vol}
              sequential
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 27,
                fontWeight: 700,
                letterSpacing: 0.5,
                color: "var(--green)",
              }}
            />
          </h2>
          {vol && (
            <StatutBadge
              key={vol.identifiant}
              statut={vol.statut}
              anomalieScore={vol.anomalie?.score ?? null}
              ecartScore={vol.ecart_trajectoire?.score_global ?? null}
              directnessScore={vol.directness?.score ?? null}
            />
          )}
        </div>

        <div className="dashboard-grid">
          {/* Aircraft-characteristics tiles live in this same grid column
              so their total width always matches the trajectory card
              directly below them, instead of spanning the full page. */}
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div ref={panneauRef}>
              <PanneauAppareil vol={vol} />
            </div>
            <div style={cardStyle}>
              <h3 style={cardTitleWhiteStyle}>Trajectory</h3>
              <div style={{ ...expandStyle(vol !== null, Math.max(trajContentHeight, 28)), transition: TRAJECTORY_EXPAND_TRANSITION }}>
                <div ref={trajContentRef}>
                  <TrajectoryReveal vol={vol} />
                </div>
              </div>
            </div>
          </div>

          {/* One large "Flight diagnostic" frame instead of two separate
              stacked cards — anomaly score and trajectory deviation read
              as a single diagnostic reading, not two unrelated widgets.
              expandStyle is applied to this *outer* box directly (not an
              inner wrapper) so its total height — chrome included — is what
              gets pinned to the characteristics row's height before a
              search; padding/title spacing shrink to fit in that state
              (the chrome alone would otherwise be taller than the target),
              and those two properties are themselves transitioned in step
              with max-height so the whole card animates as one coordinated
              move instead of the height easing smoothly while the padding
              snaps instantly. */}
          <div
            style={{
              ...cardStyle,
              padding: vol ? cardStyle.padding : "10px 20px",
              ...expandStyle(vol !== null, Math.max(diagContentHeight + 78, 720), diagCollapsedHeight),
              // expandStyle's own transition covers max-height/min-height —
              // padding is appended to that same list rather than replacing
              // it, so all three animate together instead of padding
              // snapping instantly while height eases.
              transition: `${EXPAND_TRANSITION}, padding ${EXPAND_EASE}`,
            }}
          >
            <h3
              style={{
                ...cardTitleWhiteStyle,
                marginBottom: vol ? 16 : 4,
                transition: "margin-bottom 950ms cubic-bezier(0.22, 1, 0.36, 1)",
              }}
            >
              Flight diagnostic
            </h3>
            <div style={{ overflow: "hidden" }}>
              <div ref={diagContentRef}>
                {vol ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
                    <div>
                      <h4 style={{ ...cardTitleStyle, fontSize: 12, marginBottom: 12 }}>Trajectory anomaly</h4>
                      <ScoreAnomalie anomalie={vol.anomalie} enVol={vol.statut === "en_vol"} />
                    </div>
                    <div>
                      <h4 style={{ ...cardTitleStyle, fontSize: 12, marginBottom: 12 }}>Deviation vs. optimal profile</h4>
                      <JaugeEcart ecart={vol.ecart_trajectoire} enVol={vol.statut === "en_vol"} categorie={vol.categorie} />
                    </div>
                    <div>
                      <h4 style={{ ...cardTitleStyle, fontSize: 12, marginBottom: 12 }}>Route directness</h4>
                      <DirectnessGauge directness={vol.directness} enVol={vol.statut === "en_vol"} />
                    </div>
                  </div>
                ) : (
                  <p style={placeholderTextStyle}>Run a search to compute the flight diagnostic.</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
