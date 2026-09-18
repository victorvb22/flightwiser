import { useEffect, useRef, useState } from "react";

const FLICKER_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const DEFAULT_TICK_MS = 45;
const FLICKER_TICKS = 5;
const NBSP = " ";

/**
 * Text that materialises character by character with a brief run of random
 * letters before settling — the classic mechanical "split-flap" departure
 * board look (Solari board), rendered here with plain text (no per-letter
 * tiles) to stay in keeping with the rest of the app's minimalism.
 *
 * Two settle patterns, picked with `sequential`:
 *  - default (false): every column starts flickering at once, only their
 *    *settle time* is staggered left-to-right (or right-to-left in
 *    `reverse`) — the original "FLIGHT SEARCH" title effect, unchanged.
 *  - sequential (true): columns resolve strictly one at a time — the next
 *    only starts once the previous has settled — so "-----" becoming
 *    "10972" visibly passes through "1----", "10---", "109--" and so on.
 *    Continues from whatever's currently on screen rather than restarting
 *    from blank. Used for the flight identifier and characteristics, where
 *    that reveal was explicitly requested; the title was never meant to
 *    change.
 */
export function SplitFlapText({
  text,
  style,
  tickMs = DEFAULT_TICK_MS,
  reverse = false,
  sequential = false,
  onSettled,
}: {
  text: string;
  style?: React.CSSProperties;
  tickMs?: number;
  /** Settle (sequential) or stagger (default) right-to-left instead of
   * left-to-right — used for the "revert to placeholder dashes" direction,
   * mirroring the "reveal real value" sweep rather than repeating it. */
  reverse?: boolean;
  /** One column at a time instead of all columns at once with staggered
   * settling — see comment above. */
  sequential?: boolean;
  /** Fired once every column has settled on its final character
   * (non-sequential mode only — the titles that use this are never
   * sequential). Used where something else should wait for the flicker-in
   * to visually finish before starting, e.g. a status indicator appearing
   * next to a page title only once it's fully readable. */
  onSettled?: () => void;
}) {
  const [display, setDisplay] = useState<string[]>(() => (sequential ? text.split("") : text.split("").map(() => "")));
  // Mirrors `display` synchronously so the sequential effect always knows
  // what's actually on screen right now, including mid-animation if `text`
  // changes again before a previous run finishes — state updates alone are
  // async and would read stale.
  const displayRef = useRef<string[]>(display);

  useEffect(() => {
    const target = text.split("");

    if (!sequential) {
      // Original behaviour: every column's own interval starts immediately;
      // only the tick count at which each one settles is staggered.
      setDisplay(target.map(() => ""));
      const intervals: ReturnType<typeof setInterval>[] = [];
      let maxSettleAt = 0;

      target.forEach((ch, i) => {
        if (!/[a-zA-Z0-9-]/.test(ch)) {
          setDisplay((d) => {
            const next = [...d];
            next[i] = ch;
            return next;
          });
          return;
        }
        let tick = 0;
        const columnIndex = reverse ? target.length - 1 - i : i;
        const settleAt = 5 + columnIndex * 2;
        maxSettleAt = Math.max(maxSettleAt, settleAt);
        const id = setInterval(() => {
          tick += 1;
          setDisplay((d) => {
            const next = [...d];
            next[i] = tick >= settleAt ? ch : FLICKER_CHARS[Math.floor(Math.random() * FLICKER_CHARS.length)];
            return next;
          });
          if (tick >= settleAt) clearInterval(id);
        }, tickMs);
        intervals.push(id);
      });

      const settledTimeout = onSettled ? setTimeout(onSettled, maxSettleAt * tickMs) : null;
      return () => {
        intervals.forEach(clearInterval);
        if (settledTimeout) clearTimeout(settledTimeout);
      };
    }

    // Sequential: chain each column's flicker-then-settle after the
    // previous one, so only one column is ever mid-flicker at a time.
    function setColumn(i: number, ch: string) {
      const next = [...displayRef.current];
      next[i] = ch;
      displayRef.current = next;
      setDisplay(next);
    }

    // Seed every column from whatever's currently shown there — padded out
    // to the target's own length with its final characters, since a column
    // beyond the old text's length has nothing prior to hold. Columns not
    // yet reached in this pass keep showing this seed value untouched
    // (e.g. the placeholder's "-") until their own turn comes.
    const seed = target.map((ch, i) => displayRef.current[i] ?? ch);
    displayRef.current = seed;
    setDisplay(seed);

    const order = target.map((_, i) => i);
    if (reverse) order.reverse();

    let cancelled = false;
    let activeInterval: ReturnType<typeof setInterval> | null = null;

    function runColumn(orderIdx: number) {
      if (cancelled || orderIdx >= order.length) return;
      const i = order[orderIdx];
      const ch = target[i];
      if (!/[a-zA-Z0-9-]/.test(ch)) {
        setColumn(i, ch);
        runColumn(orderIdx + 1);
        return;
      }
      let tick = 0;
      activeInterval = setInterval(() => {
        tick += 1;
        const settled = tick >= FLICKER_TICKS;
        setColumn(i, settled ? ch : FLICKER_CHARS[Math.floor(Math.random() * FLICKER_CHARS.length)]);
        if (settled) {
          clearInterval(activeInterval!);
          runColumn(orderIdx + 1);
        }
      }, tickMs);
    }
    runColumn(0);

    return () => {
      cancelled = true;
      if (activeInterval) clearInterval(activeInterval);
    };
  }, [text, tickMs, reverse, sequential]);

  return (
    <span style={style} aria-label={text}>
      {display.map((ch, i) => (
        // A bare " " as an inline-block's only content can get collapsed by
        // normal HTML whitespace rules — a non-breaking space renders
        // reliably, and the explicit minWidth keeps the gap even before
        // this column's own tick has set anything.
        <span key={i} aria-hidden="true" style={{ display: "inline-block", minWidth: "0.6em" }}>
          {ch === "" || ch === " " ? NBSP : ch}
        </span>
      ))}
    </span>
  );
}
