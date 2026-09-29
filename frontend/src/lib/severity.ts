/**
 * Continuous green -> amber -> red gradient driven by a [0,1] score, rather
 * than fixed steps — a continuous score deserves a continuous color.
 * `good=1` (default): a high score = green (e.g. score_anomalie, where
 * higher = more normal). Pass `good=0` for the reverse (e.g. score_ecart,
 * where lower = better).
 */

function normalized(score: number, good: 0 | 1): number {
  return Math.max(0, Math.min(1, good === 1 ? score : 1 - score));
}

/** HSL hue (degrees): green ~160°, amber ~45°, red ~355°. Passes through
 * orange, never through a "clean" hue halfway that would read as an
 * unintended 3rd color. Exposed separately so other components can vary
 * saturation/lightness on the same hue (e.g. the trajectory encodes altitude
 * as lightness on the status hue). */
export function severityHue(score: number, good: 0 | 1 = 1): number {
  const t = normalized(score, good);
  return t > 0.5 ? 160 - (1 - t) * 2 * (160 - 45) : 45 - (0.5 - t) * 2 * 45;
}

export function severityColor(score: number, good: 0 | 1 = 1): string {
  const t = normalized(score, good);
  return `hsl(${severityHue(score, good).toFixed(0)} 78% ${(52 + t * 6).toFixed(0)}%)`;
}

/** `anomalyAt` (default 0.35, matching the original single shared cutoff)
 * lets one caller tighten just its own "anomaly" boundary without moving
 * the other two callers' (JaugeEcart.tsx, DirectnessGauge.tsx) — cf.
 * ScoreAnomalie.tsx's own comment on why its 0.35 default read as
 * over-eager (roughly a third of genuinely normal flights land below it
 * purely from the percentile-rank scoring having no fixed "normal" rate to
 * calibrate against, not a real fault rate). ecart_trajectoire/directness
 * aren't percentile ranks the same way (a deviation-from-simulation score,
 * a distance ratio) and haven't been shown to have the same problem, so
 * they keep the original default rather than being changed on the same
 * unverified assumption. */
export function severityLabel(score: number, good: 0 | 1 = 1, anomalyAt = 0.35): string {
  const t = normalized(score, good);
  if (t >= 0.65) return "normal";
  if (t >= anomalyAt) return "worth watching";
  return "anomaly";
}

/**
 * Worst of several independently-scaled scores, each with its own "good
 * direction" — compared on a shared 0-1 badness scale (via `normalized`)
 * rather than the raw scores directly, since e.g. a 0.4 anomaly score and a
 * 0.4 deviation score don't mean the same thing. Candidates with a null
 * score (no data for that flight/feature) are skipped; null is returned only
 * if every candidate is null.
 */
export function worstSeverity(
  candidates: { score: number | null; good: 0 | 1 }[]
): { score: number; good: 0 | 1 } | null {
  const scored = candidates.filter((c): c is { score: number; good: 0 | 1 } => c.score !== null);
  if (scored.length === 0) return null;
  return scored.reduce((worst, c) => ((1 - normalized(c.score, c.good)) > (1 - normalized(worst.score, worst.good)) ? c : worst));
}
