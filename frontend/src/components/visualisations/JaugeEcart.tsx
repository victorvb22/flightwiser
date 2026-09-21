import { PlaneLanding, PlaneTakeoff, Gauge as GaugeIcon } from "lucide-react";
import type { Categorie, EcartTrajectoire } from "../../services/api";
import { severityColor, severityLabel } from "../../lib/severity";
import { useCountUp } from "../../lib/useCountUp";

const PHASE_ICONS = {
  montee: PlaneTakeoff,
  croisiere: GaugeIcon,
  descente: PlaneLanding,
} as const;

const PHASE_LABELS = {
  montee: "Climb",
  croisiere: "Cruise",
  descente: "Descent",
} as const;

function Meter({ label, ecart, detail }: { label: string; ecart: number | null; detail: string }) {
  const Icon = PHASE_ICONS[label as keyof typeof PHASE_ICONS] ?? GaugeIcon;
  const displayLabel = PHASE_LABELS[label as keyof typeof PHASE_LABELS] ?? label;
  const targetPct = ecart !== null ? Math.min(ecart, 1) * 100 : 0;
  const pct = useCountUp(targetPct);
  const color = ecart !== null ? severityColor(ecart, 0) : "var(--text-faint)";

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14.5, fontWeight: 600 }}>
          <Icon size={16} color={color} />
          {displayLabel}
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 14.5, color }}>
          {ecart !== null ? `${Math.round(pct)}%` : "—"}
        </span>
      </div>
      <div style={{ height: 7, borderRadius: 4, background: "var(--surface-3)", overflow: "hidden" }}>
        {ecart !== null && (
          <div
            style={{
              width: `${pct}%`,
              height: "100%",
              background: color,
              borderRadius: 4,
              boxShadow: `0 0 10px ${color}`,
            }}
          />
        )}
      </div>
      <p style={{ fontSize: 13, color: "var(--text-faint)", margin: "6px 0 0", lineHeight: 1.5 }}>
        {ecart !== null ? detail : "phase not detected in the real trajectory"}
      </p>
    </div>
  );
}

export function JaugeEcart({
  ecart,
  enVol = false,
  categorie = null,
}: {
  ecart: EcartTrajectoire | null;
  enVol?: boolean;
  /** Drives the succinct reason shown when landed and unavailable — cf.
   * pipeline.py: null itself (unidentified aircraft), "helicoptere" (no
   * rotary-wing performance model exists at all), or a resolved fixed-wing
   * category whose specific typecode OpenAP just doesn't have a
   * performance sheet for (the common real case, e.g. A330-800/A337). */
  categorie?: Categorie | null;
}) {
  // Called unconditionally (rules of hooks) — target is 0 while there's
  // nothing to show yet, so it's a no-op until a real score arrives.
  const globalPct = useCountUp(ecart !== null ? Math.min(ecart.score_global, 1) * 100 : 0);

  if (ecart === null) {
    // A landed flight can also legitimately have no score now — a typecode
    // OpenAP doesn't recognize (the whole petit_avion category, not just
    // helicopters) gets no simulated performance model to compare against,
    // same as an in-progress flight gets no destination. "Flight in
    // progress" would be a wrong, fabricated reason in that landed case, so
    // only claim it when it's actually true; otherwise the succinct reason
    // is picked from categorie, the same signal ScoreAnomalie/
    // DirectnessGauge use for their own version of this message. Not fully
    // precise in one narrow edge case (a resolved, OpenAP-recognized
    // category whose trajectory was itself too short/undetectable) — the
    // API doesn't expose that distinction, and it's rare enough not to be
    // worth a reason code of its own.
    let reason = "aircraft type not modeled by OpenAP";
    if (categorie === null) reason = "aircraft type not recognized";
    else if (categorie === "helicoptere") reason = "no performance model for rotorcraft";
    return (
      <p style={{ color: "var(--text-faint)", fontSize: 14.5 }}>
        {enVol ? "Trajectory deviation not available (flight in progress)." : `Trajectory deviation not available (${reason}).`}
      </p>
    );
  }

  const globalColor = severityColor(ecart.score_global, 0);

  return (
    <section>
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 16 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 38, fontWeight: 700, color: globalColor }}>
          {Math.round(globalPct)}%
        </span>
        <span style={{ fontSize: 13, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: 0.5 }}>
          overall deviation · {severityLabel(ecart.score_global, 0)}
        </span>
      </div>
      <Meter label="montee" ecart={ecart.par_phase.montee.ecart} detail={ecart.par_phase.montee.detail} />
      <Meter label="croisiere" ecart={ecart.par_phase.croisiere.ecart} detail={ecart.par_phase.croisiere.detail} />
      <Meter label="descente" ecart={ecart.par_phase.descente.ecart} detail={ecart.par_phase.descente.detail} />
    </section>
  );
}
