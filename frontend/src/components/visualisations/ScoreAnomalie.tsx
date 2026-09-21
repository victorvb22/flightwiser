import { AlertTriangle, CheckCircle2 } from "lucide-react";
import type { Anomalie } from "../../services/api";
import { severityColor, severityLabel } from "../../lib/severity";
import { useCountUp } from "../../lib/useCountUp";

// Display labels for type_anomalie (backend/models/anomaly_type.py) — "normal"
// is never shown as a tag (cf. ScoreAnomalie below): it means the flagged
// score doesn't match any of these three known patterns, not that nothing
// was computed.
const ANOMALY_TYPE_LABELS: Record<string, string> = {
  go_around: "Go-around",
  holding: "Holding pattern",
  emergency_descent: "Emergency descent",
};

const FEATURE_LABELS: Record<string, string> = {
  vitesse_moyenne: "average speed",
  vitesse_max: "max speed",
  vitesse_std: "speed variability",
  altitude_max: "max altitude",
  taux_montee_max: "climb rate",
  taux_descente_max: "descent rate",
  duration_min: "duration",
};

// Raw feature values from the backend are all in their model-native SI unit
// (m/s, m, minutes) — converted here only where the app already has an
// established display convention elsewhere (speed -> km/h, like
// PanneauAppareil's "Current speed"); altitude/rate/duration are shown as-is
// since there's no such precedent for those.
function formatFeatureValue(feature: string, value: number): string {
  if (feature === "vitesse_moyenne" || feature === "vitesse_max" || feature === "vitesse_std") {
    return `${Math.round(value * 3.6)} km/h`;
  }
  if (feature === "altitude_max") return `${Math.round(value)} m`;
  if (feature === "taux_montee_max" || feature === "taux_descente_max") return `${value.toFixed(1)} m/s`;
  if (feature === "duration_min") return `${Math.round(value)} min`;
  return `${value}`;
}

const R = 52;
const CIRC = 2 * Math.PI * R;

/** Circular gauge: the filled arc follows the score (likelihood percentile
 * rank), the color follows the shared green -> red gradient. */
function RingGauge({ score, color }: { score: number; color: string }) {
  // Fills from 0 up to the real score on arrival rather than snapping
  // straight to it — driven every frame, so no CSS transition on the arc
  // itself (that would fight with these per-frame attribute updates).
  const display = useCountUp(score);
  const filled = CIRC * Math.max(0.02, Math.min(display, 1));
  return (
    <svg viewBox="0 0 128 128" width={128} height={128} role="img" aria-label={`Anomaly score ${Math.round(score * 100)}%`}>
      <circle cx={64} cy={64} r={R} fill="none" stroke="var(--surface-3)" strokeWidth={9} />
      <circle
        cx={64}
        cy={64}
        r={R}
        fill="none"
        stroke={color}
        strokeWidth={9}
        strokeLinecap="round"
        strokeDasharray={`${filled} ${CIRC - filled}`}
        transform="rotate(-90 64 64)"
        style={{ filter: `drop-shadow(0 0 6px ${color})` }}
      />
      <text x={64} y={60} textAnchor="middle" fontFamily="var(--font-mono)" fontSize={27} fontWeight={700} fill={color}>
        {Math.round(display * 100)}
      </text>
      <text x={64} y={78} textAnchor="middle" fontSize={11} fill="var(--text-faint)" letterSpacing={0.8}>
        PERCENTILE
      </text>
    </svg>
  );
}

export function ScoreAnomalie({ anomalie, enVol = false }: { anomalie: Anomalie | null; enVol?: boolean }) {
  if (anomalie === null) {
    // A landed flight can also legitimately have no score — pipeline.py
    // only computes it when the aircraft's typecode resolves to one of the
    // four known categories, cf. services/aircraft_category.categorize.
    // "Flight in progress" would be a fabricated reason in that case
    // (same distinction JaugeEcart.tsx already makes), so only claim it
    // when it's actually true.
    return (
      <p style={{ color: "var(--text-faint)", fontSize: 14.5 }}>
        {enVol ? "Anomaly score not available (flight in progress)." : "Anomaly score not available (aircraft type not recognized)."}
      </p>
    );
  }

  const color = severityColor(anomalie.score, 1);
  const label = severityLabel(anomalie.score, 1);
  const isAnomaly = label === "anomaly";
  const Icon = isAnomaly ? AlertTriangle : CheckCircle2;
  // Shown only once the score itself is flagged, and only when it isn't
  // "normal" — a flagged score whose type comes back "normal" means none of
  // these three known patterns match, which isn't the same as nothing to
  // show (cf. ANOMALY_TYPE_LABELS comment).
  const anomalyTypeLabel = isAnomaly && anomalie.type_anomalie ? ANOMALY_TYPE_LABELS[anomalie.type_anomalie] : undefined;

  return (
    <section style={{ display: "flex", gap: 20, alignItems: "center" }}>
      <RingGauge score={anomalie.score} color={color} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6, flexWrap: "wrap" }}>
          <Icon size={19} color={color} />
          <span style={{ fontWeight: 700, fontSize: 17.5, color, textTransform: "capitalize" }}>{label}</span>
          {anomalyTypeLabel && (
            <span
              title="Best-effort guess from a separate model, trained only on synthetic examples — never a confirmed real one. Treat as indicative, not a diagnosis."
              style={{
                fontSize: 11.5,
                fontFamily: "var(--font-mono)",
                color: "var(--text-muted)",
                background: "var(--surface-2)",
                border: "1px solid var(--border-strong)",
                borderRadius: 999,
                padding: "3px 10px",
                cursor: "help",
              }}
            >
              {anomalyTypeLabel}
            </span>
          )}
        </div>
        <p style={{ fontSize: 13, color: "var(--text-faint)", margin: "0 0 10px", lineHeight: 1.5 }}>
          Likelihood rank against normal flights — the lower it is, the more this flight departs from typical behaviour.
        </p>
        {anomalie.out_of_training_scope && (
          <p
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12.5,
              color: "var(--amber)",
              background: "rgba(245, 185, 66, 0.12)",
              border: "1px solid rgba(245, 185, 66, 0.35)",
              borderRadius: 6,
              padding: "5px 9px",
              margin: "0 0 10px",
            }}
          >
            <AlertTriangle size={13} />
            Out of the training scope — this helicopter flight is far outside Europe, where the baseline was trained; treat the score with caution.
          </p>
        )}
        {anomalie.features_contributives.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {anomalie.features_contributives.map((feature) => {
              // features_detail may be absent on a cache row written before
              // this field existed — fall back to no tooltip rather than
              // crashing on older cached flights.
              const detail = anomalie.features_detail?.find((d) => d.feature === feature);
              return (
                <span key={feature} className="anomaly-feature-chip">
                  <span
                    style={{
                      fontSize: 12.5,
                      fontFamily: "var(--font-mono)",
                      padding: "3px 9px",
                      borderRadius: 999,
                      border: `1px solid ${color}55`,
                      color,
                      background: `${color}14`,
                    }}
                  >
                    {FEATURE_LABELS[feature] ?? feature}
                  </span>
                  {detail && (
                    <span className="anomaly-feature-tooltip">
                      {formatFeatureValue(feature, detail.valeur)}
                      <span style={{ color: "var(--text-faint)" }}> / ref. {formatFeatureValue(feature, detail.reference)}</span>
                    </span>
                  )}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
