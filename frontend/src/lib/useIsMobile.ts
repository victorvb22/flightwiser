import { useEffect, useState } from "react";

// Same idea as index.css's existing 880px dashboard-grid breakpoint, just
// exposed to JS for the handful of things a media query alone can't express
// (different label text, a touch-vs-hover interaction fix) — not real
// device detection, viewport width. Anything gated on this evaluates to the
// exact same branch as before on a desktop-width screen, so nothing here
// can change existing desktop behaviour.
const MOBILE_QUERY = "(max-width: 640px)";

export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(() => window.matchMedia(MOBILE_QUERY).matches);
  useEffect(() => {
    const mql = window.matchMedia(MOBILE_QUERY);
    const listener = (e: MediaQueryListEvent) => setIsMobile(e.matches);
    mql.addEventListener("change", listener);
    return () => mql.removeEventListener("change", listener);
  }, []);
  return isMobile;
}
