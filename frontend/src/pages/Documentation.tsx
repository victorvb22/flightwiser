import { useEffect } from "react";
import { ExternalLink } from "lucide-react";
import { SplitFlapText } from "../components/SplitFlapText";
import { useIsMobile } from "../lib/useIsMobile";
import { useAppData } from "../lib/AppDataContext";
import type { AnomalieCategoryParams } from "../services/api";

const cardStyle: React.CSSProperties = {
  border: "1px solid var(--border)",
  background: "var(--surface)",
  borderRadius: 14,
  padding: 24,
  marginBottom: 20,
};

const cardTitleStyle: React.CSSProperties = {
  marginTop: 0,
  marginBottom: 14,
  fontSize: 13,
  color: "var(--green)",
  fontWeight: 700,
  textTransform: "uppercase",
  letterSpacing: 1,
};

const pStyle: React.CSSProperties = {
  color: "var(--text-muted)",
  fontSize: 14.5,
  lineHeight: 1.7,
  margin: "0 0 12px",
  textAlign: "justify",
};

const codeStyle: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  background: "var(--surface-2)",
  padding: "1px 6px",
  borderRadius: 4,
  fontSize: 13.5,
  color: "var(--text)",
};

// Internal category keys mapped to the English labels shown here. All four
// come from services/aircraft_category.py's table (typecode -> category) —
// the anomaly and route-directness models both train on all four;
// trajectory-deviation doesn't use categories at all (a live per-typecode
// OpenAP simulation, see its own section below).
const CATEGORY_LABELS: Record<string, string> = {
  avion_ligne: "Airliner",
  jet_affaire: "Business jet",
  petit_avion: "Small aircraft",
  helicoptere: "Helicopter",
};

// Horizontal padding of a card on a phone-width screen (cardStyle's 24
// everywhere else) — a Table there bleeds out by exactly this much on both
// sides (negative margin), so it spans the card edge to edge instead of
// staying inset by its padding like the paragraphs do.
const CARD_PADDING_MOBILE = 12;

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  const isMobile = useIsMobile();
  const cellPaddingRest = isMobile ? "6px" : "10px";
  // The first column's own left padding, separate from every other cell's:
  // on desktop the table sits directly inside the card at its own 24px
  // padding, same as a paragraph (pStyle) — a plain shared cellPadding would
  // add its horizontal padding on top of that, pushing the first column's
  // text 10px further right than a paragraph's. On mobile the table instead
  // bleeds out of the card by CARD_PADDING_MOBILE (the comment on that
  // constant), landing it flush with the card's own edge — so it needs that
  // same amount back as left padding to land text at the paragraph's inset,
  // not the smaller 6px every other cell uses.
  const firstCellPaddingLeft = isMobile ? CARD_PADDING_MOBILE : 0;
  const firstCellPadding = `6px ${cellPaddingRest} 6px ${firstCellPaddingLeft}px`;
  const cellPadding = `6px ${cellPaddingRest}`;
  return (
    <div style={isMobile ? { margin: `0 -${CARD_PADDING_MOBILE}px`, overflowX: "auto" } : undefined}>
      {/* Phone-width only: equal-width columns and justified text spread the
          table evenly across the full width instead of letting the columns
          bunch up to the left at their natural content width. */}
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: isMobile ? 12.5 : 13.5,
          marginTop: 8,
          marginBottom: 12,
          ...(isMobile ? { tableLayout: "fixed", textAlign: "justify" } : null),
        }}
      >
        <thead>
          <tr style={{ textAlign: "left", color: "var(--text-faint)", fontSize: isMobile ? 10.5 : 11.5, textTransform: "uppercase", letterSpacing: 0.5 }}>
            {head.map((h, j) => (
              <th key={h} style={{ padding: j === 0 ? firstCellPadding : cellPadding, borderBottom: "1px solid var(--border-strong)" }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody style={{ fontFamily: "var(--font-mono)" }}>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderTop: "1px solid var(--border)" }}>
              {row.map((cell, j) => (
                <td key={j} style={{ padding: j === 0 ? firstCellPadding : cellPadding }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Smooths a series by averaging each point with its ±`radius` neighbours
 * (clamped at the edges) — the raw per-point values are noisy (each is
 * derived from a single 1-percentile slice of a few thousand training
 * flights at most), and this is a visualisation of the overall shape, not
 * a claim about any one point's exact density. */
function smooth(values: number[], radius = 2): number[] {
  return values.map((_, i) => {
    const lo = Math.max(0, i - radius);
    const hi = Math.min(values.length - 1, i + radius);
    let sum = 0;
    for (let j = lo; j <= hi; j++) sum += values[j];
    return sum / (hi - lo + 1);
  });
}

/**
 * The real, trained empirical distribution — not an illustrative bell curve.
 * `percentile_breakpoints` (101 points, one per percentile 0-100 of
 * validation log-likelihood) is already a genuine 1-D distribution by
 * construction: the width of each 1-percentile slice is inversely
 * proportional to density there, so differencing consecutive breakpoints
 * and inverting recovers an empirical density curve directly from the
 * artifact actually deployed, with no synthetic data involved.
 */
function DensityCurve({ label, params }: { label: string; params: AnomalieCategoryParams }) {
  const W = 620;
  const H = 360;
  const PAD_L = 20;
  const PAD_R = 20;
  const PAD_TOP = 40;
  const PAD_BOTTOM = 48;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_TOP - PAD_BOTTOM;

  const bp = params.percentile_breakpoints;
  // epsilon is params.epsilon_log_vraisemblance itself — the *exact* number
  // printed in the table right below this chart, not a stand-in from this
  // curve's own percentile breakpoints (an earlier version used bp[1], this
  // category's own train-set 1st percentile, reasoning it was "the same
  // population as the curve" — but that meant the line on the chart and the
  // ε value in the table were two different numbers).
  //
  // The curve itself now covers the *entire* range from bp[0] (the train
  // set's true minimum) to the max, including the bin an earlier version
  // skipped specifically to keep that one extreme outlier from stretching
  // the axis. Skipping it had a real cost that wasn't obvious until someone
  // asked directly: with the domain starting after bp[0], ε (which sits
  // between bp[0] and bp[1] for both categories — verified directly against
  // the live API) landed in a stretch with *no curve data plotted at all*,
  // which reads as "this line marks nothing," the opposite of the point.
  // There genuinely is data there — it's just the sparse bottom ~1% of
  // flights spread across a wide range, hence the long, low, mostly-flat
  // segment on the left now. That's the real shape, not a rendering gap.
  const epsilon = params.epsilon_log_vraisemblance;
  const xMin = bp[0];
  const xMax = bp[bp.length - 1];

  const xs: number[] = [];
  const rawDensities: number[] = [];
  for (let i = 0; i < bp.length - 1; i++) {
    const width = bp[i + 1] - bp[i];
    xs.push((bp[i] + bp[i + 1]) / 2);
    rawDensities.push(width > 0 ? 1 / width : 0);
  }
  const densities = smooth(rawDensities, 2);
  const maxD = Math.max(...densities, 1e-9);

  const toPx = (x: number) => PAD_L + ((x - xMin) / (xMax - xMin)) * plotW;
  // Square-root, not linear: the sparse bin at bp[0]-bp[1] is genuinely
  // ~500-1000x less dense than the peak (verified against the live API —
  // e.g. avion_ligne's bin containing ε is 0.06% of peak density), so a
  // linear y-axis rendered it as an exact flat line at the baseline —
  // indistinguishable from zero, which was the actual bug once someone
  // pointed at ε and asked where its data was. sqrt keeps the ordering and
  // still shows the peak as tallest, but compresses the axis enough that a
  // real, non-zero bin reads as a real, non-zero bin instead of rounding
  // away. Labelled on the axis below so the compression is disclosed, not
  // hidden — same "no silent misleading encoding" rule as HistogrammeScores.
  const toPy = (d: number) => PAD_TOP + plotH - Math.sqrt(Math.max(d, 0) / maxD) * plotH;
  const baseY = PAD_TOP + plotH;

  const pts = xs.map((x, i) => [toPx(x), toPy(densities[i])] as const);
  const polylinePts = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const cutX = toPx(epsilon);
  // The curve now genuinely starts at PAD_L (xs[0] is the midpoint of the
  // bp[0]-bp[1] bin, essentially right at xMin), so the shaded "below ε"
  // region can just close a normal polygon along the real curve down to the
  // baseline — no more manually stitching a straight line to the left edge
  // to paper over a gap that no longer exists.
  const shadedPts = pts.filter(([x]) => x <= cutX);
  const shadedPath = shadedPts.length ? `${PAD_L.toFixed(1)},${baseY} ${shadedPts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")} ${cutX.toFixed(1)},${baseY}` : "";

  // Gaussian fit for shape comparison, in a distinct colour — estimated
  // from the same percentile breakpoints (median for the mean, the 16th/
  // 84th-percentile span for the std, the standard percentile estimator for
  // a normal's ±1σ) rather than fit separately, so it's directly comparable
  // to the empirical curve above. Peak-matched to the empirical curve's own
  // peak height (there's no shared absolute density scale between a smoothed
  // empirical histogram and a continuous PDF) — the point is the shape
  // difference, not the two curves' raw magnitudes.
  const median = bp[50];
  const std = Math.max((bp[84] - bp[16]) / 2, 1e-6);
  const gaussianRaw = xs.map((x) => Math.exp(-0.5 * ((x - median) / std) ** 2));
  const maxGaussianRaw = Math.max(...gaussianRaw, 1e-9);
  const gaussianPts = xs.map((x, i) => [toPx(x), toPy((gaussianRaw[i] / maxGaussianRaw) * maxD)] as const);
  const gaussianPolylinePts = gaussianPts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

  // Epsilon label placed above the plot, near the marker but clamped so it
  // never runs past either edge of the chart — at the true (validation) ε,
  // that line can land very close to the left wall for some categories.
  const epsilonLabelX = Math.min(Math.max(cutX, PAD_L + 34), W - PAD_R - 34);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
      <p style={{ fontSize: 13, color: "#fff", opacity: 0.85, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: 0.6 }}>{label}</p>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", maxWidth: W }}>
        <line x1={PAD_L} y1={baseY} x2={W - PAD_R} y2={baseY} stroke="var(--border-strong)" strokeWidth={1} />
        {shadedPath && <polygon points={shadedPath} fill="var(--red)" opacity={0.22} />}
        <polyline points={gaussianPolylinePts} fill="none" stroke="var(--blue)" strokeWidth={1} strokeLinejoin="round" />
        <polyline points={polylinePts} fill="none" stroke="var(--green)" strokeWidth={1.6} strokeLinejoin="round" />
        <line x1={cutX} y1={PAD_TOP - 14} x2={cutX} y2={baseY} stroke="var(--red)" strokeWidth={1.2} strokeDasharray="3,3" />
        <text x={epsilonLabelX} y={PAD_TOP - 20} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--red)">
          ε = {epsilon.toFixed(1)}
        </text>
        {/* x-axis min/max, in the same log-likelihood units as ε — without
            these the axis had no numbers on it at all, just a shape, which
            made it impossible to check anything about the chart against the
            table beside it. */}
        <text x={PAD_L} y={baseY + 16} textAnchor="start" fontSize={8.5} fontFamily="var(--font-mono)" fill="var(--text-faint)">
          {xMin.toFixed(0)}
        </text>
        <text x={W - PAD_R} y={baseY + 16} textAnchor="end" fontSize={8.5} fontFamily="var(--font-mono)" fill="var(--text-faint)">
          {xMax.toFixed(0)}
        </text>
        <text x={(PAD_L + W - PAD_R) / 2} y={baseY + 16} textAnchor="middle" fontSize={8.5} fontFamily="var(--font-mono)" fill="var(--text-faint)">
          log-likelihood
        </text>
        <text
          x={PAD_L}
          y={PAD_TOP - 20}
          textAnchor="start"
          fontSize={8.5}
          fontFamily="var(--font-mono)"
          fill="var(--text-faint)"
          opacity={0.7}
        >
          density (√ scale)
        </text>
      </svg>
      <div style={{ display: "flex", justifyContent: "center", gap: 14, marginTop: 2 }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 10.5, color: "var(--text-faint)" }}>
          <span style={{ width: 10, height: 2, background: "var(--green)", display: "inline-block" }} /> empirical
        </span>
        <span style={{ display: "inline-flex", alignItems: "center", gap: 5, fontSize: 10.5, color: "var(--text-faint)" }}>
          <span style={{ width: 10, height: 2, background: "var(--blue)", display: "inline-block" }} /> Gaussian fit
        </span>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  const isMobile = useIsMobile();
  return (
    <div style={isMobile ? { ...cardStyle, padding: `${cardStyle.padding}px ${CARD_PADDING_MOBILE}px` } : cardStyle}>
      <h3 style={cardTitleStyle}>{title}</h3>
      {children}
    </div>
  );
}

export function Documentation() {
  const isMobile = useIsMobile();
  // Cached above the router (lib/AppDataContext.tsx) — same reasoning as
  // VueAgregee.tsx: a revisit shows the last-known values immediately
  // instead of the page going blank and refetching every time, while this
  // component itself still fully unmounts/remounts on navigation as
  // before, so nothing here needed to change for that.
  const { anomalieParams: anomalieParamsState, refreshAnomalieParams } = useAppData();
  const anomalieParams = anomalieParamsState.params;
  const loadError = anomalieParamsState.error;

  useEffect(() => {
    refreshAnomalieParams();
  }, [refreshAnomalieParams]);

  // Anomaly-model category order — directness (further down) trains on the
  // same four categories and reuses this exact order/labels; trajectory-
  // deviation doesn't use categories at all (cf. CATEGORY_LABELS comment
  // above).
  const categoryOrder = ["avion_ligne", "jet_affaire", "petit_avion", "helicoptere"];

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ margin: 0, marginBottom: 8 }}>
          <SplitFlapText text="DOCUMENTATION" style={{ fontFamily: "var(--font-mono)", fontSize: 31, fontWeight: 700, letterSpacing: 2 }} />
        </h1>
        <p style={{ color: "var(--text-faint)", fontSize: 14.5, margin: 0 }}>How the three flight diagnostics are actually computed — models, data, and the thresholds behind them.</p>
      </div>

      <Section title="Architecture">
        <p style={pStyle}>
          FastAPI backend on Render, React/TypeScript frontend on Vercel, Supabase (PostgREST) for caching and the search history's source data. A
          search resolves an identifier to a pre-recorded real flight, runs it through three independent diagnostic models, then caches the result
          per aircraft/day so repeat lookups don't re-score. See <strong style={{ color: "var(--text)" }}>Limitations</strong> below for why flights
          come from a pre-recorded pool rather than a live OpenSky call.
        </p>
      </Section>

      <Section title="Limitations">
        <p style={pStyle}>
          The original design resolved an identifier straight to an ICAO24 address and fetched its trajectory live from the OpenSky Network on every
          search — no pool, no pre-recording. That worked from a local machine, but OpenSky blocks outbound requests from the shared, dynamic IP
          ranges of small cloud platforms as an anti-abuse measure — confirmed directly against Render, Railway, and a Cloudflare Worker relay, all
          blocked at the network level, while the same calls succeeded cleanly from GitHub Actions (a large, well-managed host OpenSky doesn't flag).
        </p>
        <p style={pStyle}>
          Rather than keep fighting a block outside this project's control, live search was replaced with a pool of real flights collected in
          advance from a reachable machine, using the exact same fetch-and-score code path a live search would have used — so each pooled flight is
          identical in shape and content to what a live call would have returned, just recorded ahead of time instead of on demand. A search now
          draws from this pool instead of OpenSky directly; once a flight has been served, it's marked as such (persisted in Supabase) so a random
          draw never repeats it.
        </p>
      </Section>

      <Section title="Dataset">
        <p style={pStyle}>
          Two sources, merged for training: <code style={codeStyle}>flights_clean.parquet</code> (3,132 flights from an earlier historical batch)
          and a reconstruction of the same day (2022-06-27) from OpenSky's weekly global state-vector sample, restricted to aircraft seen at least
          once over Europe that day (21,369 reconstructed flights, 12,966 landed) — a candidacy filter only, not a spatial clip: once an aircraft
          qualifies, its full trajectory is extracted wherever it actually went. The two sources overlap (same day, real duplicate flights) — 1,091
          flights dropped from the older source by (icao24, overlapping time window) before merging, keeping the newer extraction's version, so no
          flight counts twice toward a category's mean/std. <strong style={{ color: "var(--text)" }}>14,330 landed flights</strong> after
          deduplication, feeding both models below (each then applies its own further filtering — a usable trajectory for the anomaly model, a long
          enough great-circle distance for route directness).
        </p>
        <p style={pStyle}>
          Every flight is split into four categories before any model sees it — internally <code style={codeStyle}>avion_ligne</code>,{" "}
          <code style={codeStyle}>jet_affaire</code>, <code style={codeStyle}>petit_avion</code>, and <code style={codeStyle}>helicoptere</code>,
          shown here as <strong style={{ color: "var(--text)" }}>Airliner</strong>,{" "}
          <strong style={{ color: "var(--text)" }}>Business jet</strong>,{" "}
          <strong style={{ color: "var(--text)" }}>Small aircraft</strong> (flight school, touring, ULM), and{" "}
          <strong style={{ color: "var(--text)" }}>Helicopter</strong> — because a single shared distribution systematically penalises whichever
          group is a minority in the data. The boundary is a lookup table (typecode → category), not OpenAP's own supported-aircraft list as an
          earlier version used: OpenAP's coverage is built for flight-performance simulation, not real-world classification, and reusing it as a
          category boundary let genuine airliners it simply hasn't modelled yet (A330-900, A220, CRJ-1000, 787-10, among others) fall into Small
          aircraft by default — found directly on this dataset, not a hypothetical edge case. Helicopters are identified automatically from the
          aircraft database's own type designator (rotorcraft class) rather than listed by hand.
        </p>
        <p style={pStyle}>
          The anomaly and route-directness models below both use all four categories: a helicopter's baseline profile (hover, no fixed-wing
          climb/cruise/descent phases) and a business jet's (cruising far faster and higher than typical general aviation, and more directly point-
          to-point than typical training/touring circuits) each sit meaningfully outside the other distributions even for a perfectly ordinary
          flight, so pooling either into Small aircraft risked flagging most of that population as anomalous regardless of the actual flight. Only
          the trajectory-deviation model doesn't make this split — it isn't trained on a per-category distribution at all, just a live OpenAP
          simulation keyed to the exact typecode (see below): in practice this scores almost every Airliner flight, plus the couple of Business jet
          typecodes OpenAP happens to model too, and nothing else.
        </p>
        <Table
          head={["Category", "Flights (anomaly model)", "Flights (directness model)"]}
          rows={[
            ["Airliner", "12,059", "11,376"],
            ["Business jet", "1,055", "995"],
            ["Small aircraft", "229", "144"],
            ["Helicopter", "191", "122"],
          ]}
        />
      </Section>

      <Section title="1. Anomaly score">
        <p style={pStyle}>
          A diagonal Gaussian — one independent normal distribution per feature, no cross-feature correlation — fit per category on 7 summary
          statistics of the flight: average/max/std speed, max altitude, max climb rate, max descent rate, and duration (log-transformed; it's the
          only one of the seven with a strongly skewed raw distribution). The score shown is the flight's log-likelihood converted to a percentile
          rank against its category's own training distribution — 0-1, where a lower number means the flight sits further into the tail of what's
          normal for aircraft like it.
        </p>
        <p style={pStyle}>
          There are no real anomaly labels in this dataset, so the threshold below isn't a precision/recall-tuned cutoff — it's the 1st percentile
          of held-out validation log-likelihood, used to sanity-check the lowest-scoring flights by hand during training, not to hard-flag anomalies
          at serving time (the percentile score itself is what's shown, unthresholded). The curves below are the real, trained distributions —
          derived directly from each category's own 101 percentile breakpoints, not an illustrative sketch.
        </p>
        {loadError ? (
          <p role="alert" style={{ color: "var(--red)", fontSize: 14, margin: "0 0 12px" }}>
            Couldn't load live model parameters ({loadError}) — showing the reference values from the last training run instead.
          </p>
        ) : null}
        {anomalieParams ? (
          <>
            {/* One column per category, each curve centred *within its own
                column* rather than the group being centred together as one
                block — at this size that would otherwise crowd the middle
                with empty margins instead of using the section's full
                width. Three now (helicoptere added), not the two this was
                originally built for — a fixed-count grid still fits since
                categoryOrder itself is a fixed, known list. */}
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${isMobile ? 2 : categoryOrder.length}, 1fr)`, gap: 16, marginBottom: 8 }}>
              {categoryOrder.map((cat) => (
                <div key={cat} style={{ display: "flex", justifyContent: "center", minWidth: 0 }}>
                  <DensityCurve label={CATEGORY_LABELS[cat]} params={anomalieParams[cat]} />
                </div>
              ))}
            </div>
            <Table
              head={["Category", "Train", "Validation", "ε (1st pct. log-likelihood)"]}
              rows={categoryOrder.map((cat) => [
                CATEGORY_LABELS[cat],
                anomalieParams[cat].n_train.toLocaleString("en-US"),
                anomalieParams[cat].n_validation.toLocaleString("en-US"),
                anomalieParams[cat].epsilon_log_vraisemblance.toFixed(2),
              ])}
            />
          </>
        ) : !loadError ? (
          <p style={{ color: "var(--text-faint)", fontSize: 14, margin: 0 }}>Loading trained parameters…</p>
        ) : (
          <Table
            head={["Category", "Train", "Validation", "ε (1st pct. log-likelihood)"]}
            rows={[
              ["Airliner", "10,854", "1,205", "-47.03"],
              ["Business jet", "950", "105", "-61.23"],
              ["Small aircraft", "207", "22", "-40.43"],
              ["Helicopter", "172", "19", "-44.41"],
            ]}
          />
        )}
        <p style={{ ...pStyle, marginTop: 12 }}>
          Training is Europe-scoped, but a random flight draw is worldwide now (no more geographic-proximity search) — checked directly rather than
          assumed: a modest out-of-Europe sample (60 usable flights, same day) scored comparably to the Europe baseline for both Airliner and Small
          aircraft (every feature's out-of-Europe mean within ~0.6 standard deviations of the Europe-trained one — see{" "}
          <code style={codeStyle}>scripts/check_geographic_consistency.py</code>), so no extra flag for those two. Helicopters are different: mission
          profile varies far more by region (offshore, medical, touristic) than a fixed-wing flight's does, so a random helicopter draw whose
          trajectory never touches Europe gets an <strong style={{ color: "#fff" }}>out of the training scope</strong> warning on its score
          regardless — not contingent on this check, a standing caution for that category.
        </p>
        <p style={{ ...pStyle, marginTop: 12 }}>
          For an Airliner flight whose score flags an anomaly, a second, entirely separate model attempts to name which of three known patterns it
          matches: <strong style={{ color: "var(--text)" }}>go-around</strong>, <strong style={{ color: "var(--text)" }}>holding pattern</strong>,
          or <strong style={{ color: "var(--text)" }}>emergency descent</strong> — shown as a small tag next to the badge above. It's a Random Forest
          from a separate project, trained on the same OpenSky day but on synthetically injected anomalies (parametrised go-arounds, holding
          circuits, and rapid descents — real labelled examples of any of these are scarce by nature) rather than confirmed real ones, so treat the
          tag as an indicative best guess, not a diagnosis. A "normal" result from that model isn't a contradiction of the flagged score above — it
          means the anomaly doesn't match any of these three specific patterns, which is itself useful information. Full write-up, dataset
          construction, and evaluation:{" "}
          <a href="https://github.com/victorvb22/flight-trajectory-detection" target="_blank" rel="noreferrer" style={{ color: "#fff" }}>
            github.com/victorvb22/flight-trajectory-detection
          </a>
          .
        </p>
      </Section>

      <Section title="2. Trajectory deviation">
        <p style={pStyle}>
          Independent of the anomaly score. The real climb/cruise/descent phases (segmented from vertical speed and altitude) are compared against
          an OpenAP-simulated optimal profile for the same aircraft type — same route distance, same typecode's performance model. Each phase gets
          its own deviation percentage (actual vs. optimal vertical speed and phase duration); the overall score is their combination. A typecode
          OpenAP doesn't recognize gets no score at all rather than a number computed by comparing it against a substituted A320 simulation — this
          excludes Small aircraft and Helicopter outright (OpenAP models neither) and most of Business jet too: OpenAP's own list covers only about
          forty typecodes, almost all mainline/regional airliners, plus two business-jet families (Cessna Citation and Gulfstream, and their
          synonyms) that happen to be on it as well — those specific typecodes do get a real score. A comparison against a performance profile for
          a different kind of aircraft entirely isn't a deviation measurement, just a misleading number that happens to look like one.
        </p>
      </Section>

      <Section title="3. Route directness">
        <p style={pStyle}>
          The newest of the three, and the only one that looks at the trajectory's <em>shape</em> rather than its speed/altitude values: great-circle
          distance between the flight's first and last recorded position, divided by the distance actually flown along the path. A ratio near 1
          means a direct route; a lower one suggests holding patterns, a diversion, or extended ATC vectoring — something neither of the other two
          models can see, since they never look at lateral routing at all.
        </p>
        <p style={pStyle}>
          Scored the same way as the anomaly model (percentile rank against its category's training distribution) but without a Gaussian — a single
          bounded ratio doesn't need the multi-feature machinery built for combining seven dimensions, so its percentiles are computed directly
          (empirically) from the training data instead. Uses all four categories, same as the anomaly model — checked directly on real data rather
          than assumed: Business jet's median/mean ratio (0.895/0.839) is meaningfully higher than Small aircraft's (0.851/0.761), so pooling them
          would have diluted Small aircraft's own baseline with flights that don't share its profile.
        </p>
        <Table
          head={["Category", "Median ratio", "Mean ratio"]}
          rows={[
            ["Airliner", "0.928", "0.894"],
            ["Business jet", "0.895", "0.839"],
            ["Small aircraft", "0.851", "0.761"],
            ["Helicopter", "0.921", "0.847"],
          ]}
        />
      </Section>

      <div style={{ display: "flex", justifyContent: "center", padding: "24px 0 8px" }}>
        <a
          href="https://github.com/victorvb22/flightwiser"
          target="_blank"
          rel="noreferrer"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            color: "var(--text-muted)",
            fontSize: 14,
            fontFamily: "var(--font-mono)",
            textDecoration: "none",
            border: "1px solid var(--border-strong)",
            borderRadius: 8,
            padding: "10px 18px",
          }}
        >
          <ExternalLink size={16} />
          View source on GitHub
        </a>
      </div>
    </div>
  );
}
