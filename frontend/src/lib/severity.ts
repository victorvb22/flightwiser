/**
 * Dégradé continu vert -> ambre -> rouge en fonction d'un score [0,1], plutôt
 * que des paliers fixes — un score continu mérite une couleur continue.
 * `good=1` (par défaut) : score haut = vert (ex. score_anomalie, où plus
 * haut = plus normal). Passer `good=0` pour l'inverse (ex. score_ecart, où
 * plus bas = meilleur).
 */

function normalized(score: number, good: 0 | 1): number {
  return Math.max(0, Math.min(1, good === 1 ? score : 1 - score));
}

/** Teinte HSL (degrés) : vert ~160°, ambre ~45°, rouge ~355°. Passe par
 * l'orange, jamais par un hue "propre" à mi-chemin qui lirait comme une 3e
 * couleur non voulue. Exposée séparément pour que d'autres composants
 * puissent faire varier saturation/luminosité sur la même teinte (ex. la
 * trajectoire encode l'altitude en luminosité sur la teinte du statut). */
export function severityHue(score: number, good: 0 | 1 = 1): number {
  const t = normalized(score, good);
  return t > 0.5 ? 160 - (1 - t) * 2 * (160 - 45) : 45 - (0.5 - t) * 2 * 45;
}

export function severityColor(score: number, good: 0 | 1 = 1): string {
  const t = normalized(score, good);
  return `hsl(${severityHue(score, good).toFixed(0)} 78% ${(52 + t * 6).toFixed(0)}%)`;
}

export function severityLabel(score: number, good: 0 | 1 = 1): string {
  const t = normalized(score, good);
  if (t >= 0.65) return "normal";
  if (t >= 0.35) return "worth watching";
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
