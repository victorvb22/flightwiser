import type { Directness } from "../../services/api";
import { severityColor, severityLabel } from "../../lib/severity";
import { useCountUp } from "../../lib/useCountUp";

/** Flown-path length vs. the great-circle distance between the trajectory's
 * own endpoints — a signal neither the anomaly score (blind to route shape)
 * nor the deviation gauge (compares altitude/speed profile, not routing)
 * can see: holding patterns, diversions, or extended ATC vectoring. */
export function DirectnessGauge({
  directness,
  enVol = false,
}: {
  directness: Directness | null | undefined;
  enVol?: boolean;
}) {
  // Called unconditionally (rules of hooks) — target is 0 while there's
  // nothing to show yet, same pattern as JaugeEcart.
  const pct = useCountUp(directness ? directness.ratio * 100 : 0);

  if (!directness) {
    // Same distinction as ScoreAnomalie/JaugeEcart: a landed flight can
    // also legitimately have no score (unrecognized aircraft type), so
    // "flight in progress" would be a fabricated reason in that case.
    return (
      <p style={{ color: "var(--text-faint)", fontSize: 14.5 }}>
        {enVol ? "Route directness not available (flight in progress)." : "Route directness not available (aircraft type not recognized)."}
      </p>
    );
  }

  const color = severityColor(directness.score, 1);
  const label = severityLabel(directness.score, 1);

  return (
    <section>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 38, fontWeight: 700, color }}>{Math.round(pct)}%</span>
        <span style={{ fontSize: 13, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>
          of straight-line distance · {label}
        </span>
      </div>
      <div style={{ height: 7, borderRadius: 4, background: "var(--surface-3)", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4, boxShadow: `0 0 10px ${color}` }} />
      </div>
      <p style={{ fontSize: 13, color: "var(--text-faint)", margin: "10px 0 0", lineHeight: 1.5 }}>
        Flown distance vs. the great-circle distance between the trajectory's endpoints — a lower ratio suggests holding patterns, a
        diversion, or extended vectoring.
      </p>
    </section>
  );
}
