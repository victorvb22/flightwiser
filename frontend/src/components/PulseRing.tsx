/**
 * Same expanding/fading ring used behind the airborne plane marker on the
 * map (VueTrajectoire.tsx) — a small solid dot with a ring growing outward
 * and fading, recoloured and reused here as the search-status indicator
 * next to "FLIGHT SEARCH" instead of the old plain breathing dot. Thicker
 * stroke than the map's version (that one counter-scales to stay hairline
 * thin at any zoom; this one doesn't need to).
 */
export function PulseRing({ color, singlePulse = false }: { color: string; singlePulse?: boolean }) {
  const repeatCount = singlePulse ? "1" : "indefinite";
  // Slightly smaller than before (34x34, was 38x38) — its parent row now has
  // a fixed height regardless (RechercheVol.tsx), so this isn't strictly
  // needed to avoid a layout shift any more, but a smaller pulse next to a
  // 31px title reads better anyway.
  return (
    // display:block — an <svg> is inline by default, which baseline-aligns
    // it within its wrapping span using the ambient line-height's font
    // metrics rather than just filling the span's box. That baseline offset
    // (not any transform) was what shifted the ring a few pixels below the
    // title text's own center despite both boxes being centered identically.
    <svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true" style={{ display: "block" }}>
      <circle cx="17" cy="17" r="4" fill={color} />
      <circle cx="17" cy="17" r="5" fill="none" stroke={color} strokeWidth="2.2">
        <animate attributeName="r" values="5;15" dur="2.6s" repeatCount={repeatCount} />
        <animate attributeName="opacity" values="0.9;0" dur="2.6s" repeatCount={repeatCount} />
      </circle>
    </svg>
  );
}
