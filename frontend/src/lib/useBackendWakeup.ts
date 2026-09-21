import { useEffect, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// Render's free tier (render.yaml) puts the backend to sleep after 15
// minutes with no requests -- the first request after that has to wait for
// the container to spin back up, which can take up to about a minute, far
// beyond what an already-warm backend ever takes (well under a second).
// There's no endpoint that reports "I'm asleep" without itself waking the
// backend up, so a fixed timer raced against a lightweight health probe is
// the only way to tell the two states apart from the browser.
const SLOW_AFTER_MS = 1200;
// Bounds a single attempt: while the container is still starting, a
// request can hang or get refused outright rather than just respond
// slowly -- this keeps retries moving instead of getting stuck on one
// attempt for the whole budget below.
const ATTEMPT_TIMEOUT_MS = 15_000;
const RETRY_DELAY_MS = 3_000;
// Total time to keep retrying before giving up and just letting the page's
// own real request surface whatever error it hits -- comfortably above the
// "up to a minute" a cold start has been observed to take.
const MAX_TOTAL_WAIT_MS = 75_000;

/** True once a `GET /api/v1/health` probe has been pending longer than
 * SLOW_AFTER_MS -- i.e. the backend looks like it's waking up from Render's
 * free-tier sleep, not just a normal request in flight. Flips back to false
 * as soon as the backend actually responds (or after MAX_TOTAL_WAIT_MS of
 * retries, so the banner never lingers forever if something else is
 * actually wrong). Fires once per mount -- called independently by the
 * Search and Aggregate view pages, cf. their own useEffect. */
export function useBackendWakeup(): boolean {
  const [wakingUp, setWakingUp] = useState(false);

  useEffect(() => {
    let cancelled = false;
    // Guards the slow-timer callback below: without it, a probe that
    // succeeds *before* SLOW_AFTER_MS still leaves that timer armed, and it
    // fires anyway a moment later and flips the banner back on for an
    // already-warm backend. Cleared the instant probe() settles either way.
    let settled = false;
    const slowTimer = setTimeout(() => {
      if (!cancelled && !settled) setWakingUp(true);
    }, SLOW_AFTER_MS);

    async function probe() {
      const deadline = Date.now() + MAX_TOTAL_WAIT_MS;
      while (!cancelled && Date.now() < deadline) {
        try {
          const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
            signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
          });
          if (response.ok) {
            settled = true;
            clearTimeout(slowTimer);
            if (!cancelled) setWakingUp(false);
            return;
          }
        } catch {
          // Connection refused/aborted while the container is still
          // starting -- expected mid-wakeup, just retry below.
        }
        if (!cancelled) {
          await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS));
        }
      }
      settled = true;
      clearTimeout(slowTimer);
      if (!cancelled) setWakingUp(false);
    }
    probe();

    return () => {
      cancelled = true;
      clearTimeout(slowTimer);
    };
  }, []);

  return wakingUp;
}
