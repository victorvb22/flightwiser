import { useEffect, useRef, useState } from "react";
import { severityColor } from "../../lib/severity";

const VIEW_W = 320;
const VIEW_H = 150;
const PAD_BOTTOM = 22;
const PAD_TOP = 16;
const GAP = 3;
const GROW_MS = 700;

/** 0 -> 1 over GROW_MS, restarting whenever `key` changes (bar heights are
 * `progress * their real height`, so bars grow from the axis up to their
 * final value instead of just appearing — same "count up, don't snap"
 * language as useCountUp, but driving several bars from one shared clock
 * rather than one number). */
function useGrowth(key: string): number {
  const [progress, setProgress] = useState(0);
  const rafRef = useRef(0);
  useEffect(() => {
    setProgress(0);
    const start = performance.now();
    function tick(now: number) {
      const t = Math.min(1, (now - start) / GROW_MS);
      // ease-out cubic — fast start, settles gently instead of a linear ramp
      setProgress(1 - (1 - t) ** 3);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [key]);
  return progress;
}

/**
 * Histogramme à 10 bins (0-0.1, ..., 0.9-1.0). Chaque barre est colorée sur
 * le même dégradé vert->rouge que le reste de l'app, selon la valeur du bin
 * qu'elle représente — ce n'est pas un double-encodage : c'est la même
 * variable déjà portée par l'axe X, juste rendue plus lisible d'un coup
 * d'œil. `good=1` : bin haut = vert (score_anomalie) ; `good=0` : bin bas =
 * vert (score_ecart).
 *
 * `highlightBin`, when set, dims every bar except that one index (the bin a
 * currently-selected flight's own score falls into) — undefined/null means
 * no selection, every bar shown at full opacity.
 */
export function HistogrammeScores({
  title,
  counts,
  good = 1,
  highlightBin = null,
}: {
  title: string;
  counts: number[];
  good?: 0 | 1;
  highlightBin?: number | null;
}) {
  const max = Math.max(...counts, 1);
  const barW = (VIEW_W - GAP * (counts.length - 1)) / counts.length;
  const plotH = VIEW_H - PAD_TOP - PAD_BOTTOM;
  const growth = useGrowth(counts.join(","));

  return (
    <div>
      <p style={{ margin: "0 0 8px", textAlign: "center" }}>
        <span style={{ fontSize: 13, color: "#fff", opacity: 0.8, textTransform: "uppercase", letterSpacing: 0.6 }}>{title}</span>
      </p>
      <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} role="img" aria-label={`Histogram ${title}`} style={{ width: "100%", height: "auto" }}>
        <line x1={0} y1={VIEW_H - PAD_BOTTOM} x2={VIEW_W} y2={VIEW_H - PAD_BOTTOM} stroke="var(--border-strong)" strokeWidth={1} />
        {counts.map((c, i) => {
          const h = (c / max) * plotH * growth;
          const x = i * (barW + GAP);
          const y = VIEW_H - PAD_BOTTOM - h;
          const color = severityColor((i + 0.5) / counts.length, good);
          const dimmed = highlightBin !== null && i !== highlightBin;
          return (
            <g key={i} style={{ opacity: dimmed ? 0.18 : 1, transition: "opacity 250ms ease" }}>
              <rect x={x} y={y} width={barW} height={Math.max(h, c > 0 ? 2 : 0)} fill={color} rx={2} />
              {c > 0 && growth > 0.9 && (
                <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize={4.5} opacity={0.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                  {c}
                </text>
              )}
              <text x={x + barW / 2} y={VIEW_H - 7} textAnchor="middle" fontSize={9} fontFamily="var(--font-mono)" fill="var(--text-faint)">
                {(i / counts.length).toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
