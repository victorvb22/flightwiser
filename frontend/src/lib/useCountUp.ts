import { useEffect, useState } from "react";

/**
 * Eases a number up from 0 to `target` over `durationMs`, restarting
 * whenever `target` changes. Used to make the flight-diagnostic gauges
 * fill in on arrival instead of snapping straight to their final reading.
 */
export function useCountUp(target: number, durationMs = 1400): number {
  const [value, setValue] = useState(0);

  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    setValue(0);

    function tick(now: number) {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
      setValue(eased * target);
      if (t < 1) raf = requestAnimationFrame(tick);
    }

    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return value;
}
